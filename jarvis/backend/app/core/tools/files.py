"""Filesystem tools: read_file, write_file, list_dir, search_files, file_info.

Every path argument is resolved against `settings.workspace_root` and must
stay inside it after resolution — this is the load-bearing security property
of this module. Binary files and anything over 1MB are refused outright.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import settings
from app.core.tools.base import register
from app.models.schemas import ToolResult

MAX_FILE_BYTES = 1_000_000  # 1MB cap on both read and write

# Extensions we consider text and therefore safe to read/write as strings.
# Anything else is refused rather than guessed at, to avoid returning
# mangled bytes decoded with `errors="replace"`.
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".tsx", ".jsx", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".csv", ".tsv", ".html",
    ".htm", ".css", ".xml", ".sh", ".bash", ".sql", ".log", ".rst", "",
}


class PathEscapeError(Exception):
    """Raised whenever a candidate path resolves outside the workspace root."""


def _resolve_in_workspace(relative: str) -> Path:
    """Resolve `relative` against workspace_root, refusing any escape.

    Rejects `..` traversal, absolute-path escapes, and symlinks that resolve
    outside the root — all three are checked against the *resolved* path,
    not the literal string, since string-only checks are trivially bypassed.
    """
    root = settings.workspace_root.resolve()
    candidate = Path(relative)
    if candidate.is_absolute():
        # Absolute paths are only allowed if they were already inside root.
        merged = candidate
    else:
        merged = root / candidate
    resolved = merged.resolve()  # follows symlinks — this is the real check
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PathEscapeError(f"path {relative!r} escapes workspace root") from exc
    return resolved


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES


@register
class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file (max 1MB) from the workspace."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root."}
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    async def run(self, path: str) -> ToolResult:
        try:
            resolved = _resolve_in_workspace(path)
        except PathEscapeError as exc:
            return ToolResult(ok=False, output="", summary=str(exc))

        if not resolved.exists():
            return ToolResult(ok=False, output="", summary=f"No such file: {path!r}")
        if not resolved.is_file():
            return ToolResult(ok=False, output="", summary=f"Not a file: {path!r}")
        if not _is_text_file(resolved):
            return ToolResult(ok=False, output="", summary=f"Refusing to read non-text file: {path!r}")

        size = resolved.stat().st_size
        if size > MAX_FILE_BYTES:
            return ToolResult(ok=False, output="", summary=f"File too large ({size} bytes > 1MB cap)")

        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(ok=False, output="", summary=f"File is not valid UTF-8 text: {path!r}")

        return ToolResult(ok=True, output=text, summary=f"Read {size} bytes from {path!r}")


@register
class WriteFileTool:
    name = "write_file"
    description = "Write UTF-8 text (max 1MB) to a file in the workspace, creating parent dirs as needed."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root."},
            "content": {"type": "string", "description": "Text content to write."},
            "append": {"type": "boolean", "description": "Append instead of overwrite. Default false."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    async def run(self, path: str, content: str, append: bool = False) -> ToolResult:
        try:
            resolved = _resolve_in_workspace(path)
        except PathEscapeError as exc:
            return ToolResult(ok=False, output="", summary=str(exc))

        encoded = content.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            return ToolResult(ok=False, output="", summary="Content exceeds 1MB cap")
        if not _is_text_file(resolved):
            return ToolResult(ok=False, output="", summary=f"Refusing to write non-text file: {path!r}")

        # Existing symlinks must also resolve inside root even before we write
        # through them — re-check post-mkdir to catch a symlinked parent dir.
        resolved.parent.mkdir(parents=True, exist_ok=True)
        real_parent = resolved.parent.resolve()
        try:
            real_parent.relative_to(settings.workspace_root.resolve())
        except ValueError:
            return ToolResult(ok=False, output="", summary=f"path {path!r} escapes workspace root")

        mode = "a" if append else "w"
        with resolved.open(mode, encoding="utf-8") as fh:
            fh.write(content)

        verb = "Appended to" if append else "Wrote"
        return ToolResult(ok=True, output="", summary=f"{verb} {len(encoded)} bytes at {path!r}")


@register
class ListDirTool:
    name = "list_dir"
    description = "List files and subdirectories inside a workspace directory."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to workspace root. Default '.'."},
        },
        "required": [],
        "additionalProperties": False,
    }

    async def run(self, path: str = ".") -> ToolResult:
        try:
            resolved = _resolve_in_workspace(path)
        except PathEscapeError as exc:
            return ToolResult(ok=False, output="", summary=str(exc))

        if not resolved.exists():
            return ToolResult(ok=False, output="", summary=f"No such directory: {path!r}")
        if not resolved.is_dir():
            return ToolResult(ok=False, output="", summary=f"Not a directory: {path!r}")

        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries]
        return ToolResult(
            ok=True,
            output="\n".join(lines),
            summary=f"{len(entries)} entries in {path!r}",
            meta={"count": len(entries)},
        )


@register
class SearchFilesTool:
    name = "search_files"
    description = "Search for a text substring across text files under a workspace directory."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "path": {"type": "string", "description": "Directory to search under. Default '.'."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def run(self, query: str, path: str = ".", max_results: int = 50) -> ToolResult:
        try:
            resolved = _resolve_in_workspace(path)
        except PathEscapeError as exc:
            return ToolResult(ok=False, output="", summary=str(exc))

        if not resolved.is_dir():
            return ToolResult(ok=False, output="", summary=f"Not a directory: {path!r}")

        matches: list[str] = []
        root = settings.workspace_root.resolve()
        for candidate in sorted(resolved.rglob("*")):
            if len(matches) >= max_results:
                break
            if not candidate.is_file() or not _is_text_file(candidate):
                continue
            # Guard against a symlinked file inside the tree escaping root.
            try:
                candidate.resolve().relative_to(root)
            except ValueError:
                continue
            if candidate.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    rel = candidate.relative_to(root)
                    matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(matches) >= max_results:
                        break

        return ToolResult(
            ok=True,
            output="\n".join(matches),
            summary=f"{len(matches)} match(es) for {query!r}",
            meta={"count": len(matches)},
        )


@register
class FileInfoTool:
    name = "file_info"
    description = "Get metadata (size, type, modified time) for a workspace path."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    async def run(self, path: str) -> ToolResult:
        try:
            resolved = _resolve_in_workspace(path)
        except PathEscapeError as exc:
            return ToolResult(ok=False, output="", summary=str(exc))

        if not resolved.exists():
            return ToolResult(ok=False, output="", summary=f"No such path: {path!r}")

        stat = resolved.stat()
        kind = "dir" if resolved.is_dir() else "file"
        info = {
            "type": kind,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        }
        summary = f"{kind} {path!r}: {stat.st_size} bytes"
        return ToolResult(ok=True, output=str(info), summary=summary, meta=info)
