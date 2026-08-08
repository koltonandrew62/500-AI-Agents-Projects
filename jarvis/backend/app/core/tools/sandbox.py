"""Sandboxed code execution: exec_python and exec_shell.

`exec_python` runs arbitrary Python in a throwaway subprocess with a 5s
timeout and a temp cwd, after an AST-level static check rejects known
dangerous constructs. This is defense-in-depth, not a real sandbox (no
seccomp/chroot/network namespace) — the AST denylist plus a short timeout
and a scratch cwd are the guarantees actually provided.

`exec_shell` runs a single allowlisted command with no shell interpretation
at all (`shell=False`), so shell metacharacters are inert.
"""

from __future__ import annotations

import ast
import asyncio
import shlex
import tempfile
from pathlib import Path
from typing import Any

from app.core.tools.base import register
from app.models.schemas import ToolResult

PYTHON_TIMEOUT_S = 5.0
SHELL_TIMEOUT_S = 5.0

# Whole modules that are never allowed to be imported by sandboxed code —
# each one is a direct route to the network, process control, or bulk
# filesystem destruction.
_BANNED_MODULES = {"subprocess", "socket", "shutil", "ctypes", "importlib", "multiprocessing", "os.path"}

# Dotted attribute paths banned even if the owning module is otherwise fine
# (e.g. `os` itself is allowed for things like `os.getcwd()`).
_BANNED_ATTR_CHAINS = {
    ("os", "system"), ("os", "popen"), ("os", "exec"), ("os", "execv"), ("os", "execve"),
    ("os", "execvp"), ("os", "spawn"), ("os", "spawnl"), ("os", "spawnv"), ("os", "fork"),
    ("os", "kill"), ("os", "remove"), ("os", "unlink"), ("os", "rmdir"),
}

# Bare names that are dangerous regardless of attribute access — these cover
# the classic __import__-tricks path to reaching banned modules dynamically.
_BANNED_CALL_NAMES = {"__import__", "eval", "exec", "compile", "globals", "vars"}


class SandboxViolation(Exception):
    """Raised by the AST checker when code contains a denylisted construct."""


class _SafetyVisitor(ast.NodeVisitor):
    """Walks the AST once, collecting every denylist violation it finds."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if alias.name in _BANNED_MODULES or root in _BANNED_MODULES:
                self.violations.append(f"import of banned module: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if module in _BANNED_MODULES or root in _BANNED_MODULES:
            self.violations.append(f"import from banned module: {module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Bare dangerous builtins: __import__("os").system(...) etc.
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALL_NAMES:
            self.violations.append(f"call to banned builtin: {node.func.id}")

        # Dotted attribute calls like os.system(...) or shutil.rmtree(...).
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            chain = (node.func.value.id, node.func.attr)
            if chain in _BANNED_ATTR_CHAINS:
                self.violations.append(f"call to banned attribute: {chain[0]}.{chain[1]}")
            if chain == ("shutil", "rmtree"):
                self.violations.append("call to banned attribute: shutil.rmtree")

        # open(path, mode) in write/append/exclusive/plus mode with a path
        # literal that looks like it targets outside the scratch cwd.
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self._check_open_call(node)

        self.generic_visit(node)

    def _check_open_call(self, node: ast.Call) -> None:
        mode = ""
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = str(kw.value.value)
        is_write = any(c in mode for c in ("w", "a", "x", "+"))
        if not is_write:
            return
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            path = node.args[0].value
            if path.startswith("/") or path.startswith("~") or ".." in path:
                self.violations.append(f"write-mode open() outside sandbox cwd: {path!r}")


def check_python_source(source: str) -> list[str]:
    """Parse `source` and return a list of denylist violation messages."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    return visitor.violations


@register
class ExecPythonTool:
    name = "exec_python"
    description = (
        "Execute a short Python snippet in an isolated subprocess with a 5s "
        "timeout and a temp working directory. Network/process/bulk-delete "
        "calls are statically rejected before execution."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "minLength": 1, "description": "Python source to execute."},
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    async def run(self, code: str) -> ToolResult:
        violations = check_python_source(code)
        if violations:
            return ToolResult(
                ok=False,
                output="",
                summary=f"Rejected by sandbox denylist: {'; '.join(violations)}",
            )

        with tempfile.TemporaryDirectory(prefix="jarvis-sandbox-") as tmp_dir:
            script_path = Path(tmp_dir) / "snippet.py"
            script_path.write_text(code, encoding="utf-8")

            try:
                proc = await asyncio.create_subprocess_exec(
                    "python3",
                    str(script_path),
                    cwd=tmp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=PYTHON_TIMEOUT_S)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(ok=False, output="", summary=f"exec_python timed out after {PYTHON_TIMEOUT_S}s")

            out_text = stdout.decode("utf-8", errors="replace")
            err_text = stderr.decode("utf-8", errors="replace")
            ok = proc.returncode == 0
            combined = out_text if ok else f"{out_text}\n{err_text}".strip()
            summary = "exec_python succeeded" if ok else f"exec_python exited {proc.returncode}"
            return ToolResult(ok=ok, output=combined, summary=summary, meta={"returncode": proc.returncode})


# Commands considered harmless enough to run with zero arguments-side-effects
# beyond reading already-public system state.
_SHELL_ALLOWLIST = {"ls", "cat", "date", "uname", "df", "ps", "echo", "pwd", "which"}


@register
class ExecShellTool:
    name = "exec_shell"
    description = "Run a single allowlisted read-only shell command (ls, cat, date, uname, df, ps, echo, pwd, which)."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1, "description": "Full command line, e.g. 'ls -la'."},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    async def run(self, command: str) -> ToolResult:
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return ToolResult(ok=False, output="", summary=f"Could not parse command: {exc}")

        if not tokens:
            return ToolResult(ok=False, output="", summary="Empty command")

        program = tokens[0]
        if program not in _SHELL_ALLOWLIST:
            return ToolResult(ok=False, output="", summary=f"Command not allowlisted: {program!r}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *tokens,  # shell=False by construction — no metachar interpretation
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=SHELL_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(ok=False, output="", summary=f"exec_shell timed out after {SHELL_TIMEOUT_S}s")
        except FileNotFoundError:
            return ToolResult(ok=False, output="", summary=f"Command not found: {program!r}")

        out_text = stdout.decode("utf-8", errors="replace")
        err_text = stderr.decode("utf-8", errors="replace")
        ok = proc.returncode == 0
        combined = out_text if ok else f"{out_text}\n{err_text}".strip()
        return ToolResult(
            ok=ok,
            output=combined,
            summary=f"{program} exited {proc.returncode}",
            meta={"returncode": proc.returncode},
        )
