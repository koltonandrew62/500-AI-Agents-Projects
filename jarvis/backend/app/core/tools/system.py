"""System tools: get_time, get_system_info, take_note, set_volume, open_app.

`set_volume` and `open_app` shell out to platform-specific utilities behind
a strict allowlist — never build a command line from unvalidated free text.
"""

from __future__ import annotations

import asyncio
import platform
import time
from datetime import datetime, timezone
from typing import Any

import psutil

from app.config import settings
from app.core.tools.base import register
from app.models.schemas import ToolResult

_PROC_START = time.time()

# Platform-aware, allowlisted app launchers. Only these keys may ever be
# opened — arbitrary strings from the caller are never passed to a shell.
_APP_COMMANDS: dict[str, dict[str, list[str]]] = {
    "Darwin": {
        "browser": ["open", "-a", "Safari"],
        "terminal": ["open", "-a", "Terminal"],
        "notes": ["open", "-a", "Notes"],
        "calculator": ["open", "-a", "Calculator"],
        "finder": ["open", "-a", "Finder"],
    },
    "Linux": {
        "browser": ["xdg-open", "https://"],
        "terminal": ["x-terminal-emulator"],
        "notes": ["gedit"],
        "calculator": ["gnome-calculator"],
        "finder": ["xdg-open", "."],
    },
    "Windows": {
        "browser": ["cmd", "/c", "start", "msedge"],
        "terminal": ["cmd", "/c", "start", "cmd"],
        "notes": ["notepad"],
        "calculator": ["calc"],
        "finder": ["explorer"],
    },
}


@register
class GetTimeTool:
    name = "get_time"
    description = "Get the current local date and time."
    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    async def run(self) -> ToolResult:
        now = datetime.now().astimezone()
        iso = now.isoformat()
        return ToolResult(ok=True, output=iso, summary=now.strftime("%A, %B %d %Y %H:%M:%S %Z"))


@register
class GetSystemInfoTool:
    name = "get_system_info"
    description = "Get CPU, memory, disk, and uptime information for the host machine."
    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    async def run(self) -> ToolResult:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        uptime_s = int(time.time() - psutil.boot_time())
        info = {
            "cpu_percent": cpu,
            "mem_percent": mem.percent,
            "mem_used_gb": round(mem.used / 1e9, 2),
            "mem_total_gb": round(mem.total / 1e9, 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / 1e9, 2),
            "uptime_s": uptime_s,
            "platform": platform.platform(),
        }
        summary = f"CPU {cpu:.0f}% · MEM {mem.percent:.0f}% · DISK {disk.percent:.0f}%"
        return ToolResult(ok=True, output=str(info), summary=summary, meta=info)


@register
class TakeNoteTool:
    name = "take_note"
    description = "Append a timestamped note to the workspace notes file."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1}},
        "required": ["text"],
        "additionalProperties": False,
    }

    async def run(self, text: str) -> ToolResult:
        notes_path = settings.workspace_root / "notes.md"
        settings.workspace_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"- [{timestamp}] {text}\n"
        with notes_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return ToolResult(ok=True, output=line.strip(), summary="Note saved")


_VOLUME_TIMEOUT_S = 5.0


@register
class SetVolumeTool:
    name = "set_volume"
    description = "Set the system output volume (0-100), platform-aware."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": ["level"],
        "additionalProperties": False,
    }

    async def run(self, level: int) -> ToolResult:
        system = platform.system()
        if system == "Darwin":
            cmd = ["osascript", "-e", f"set volume output volume {level}"]
        elif system == "Linux":
            cmd = ["amixer", "-D", "pulse", "sset", "Master", f"{level}%"]
        elif system == "Windows":
            # No stock CLI volume control on Windows; report unsupported
            # rather than guessing at a third-party tool being installed.
            return ToolResult(ok=False, output="", summary="set_volume is not supported on Windows without a third-party tool")
        else:
            return ToolResult(ok=False, output="", summary=f"Unsupported platform: {system}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_VOLUME_TIMEOUT_S)
        except (asyncio.TimeoutError, FileNotFoundError) as exc:
            return ToolResult(ok=False, output="", summary=f"Could not set volume: {exc}")

        if proc.returncode != 0:
            return ToolResult(ok=False, output="", summary=f"Volume command failed: {stderr.decode(errors='replace')}")
        return ToolResult(ok=True, output="", summary=f"Volume set to {level}%")


@register
class OpenAppTool:
    name = "open_app"
    description = "Open an allowlisted application: browser, terminal, notes, calculator, finder."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "app": {
                "type": "string",
                "enum": ["browser", "terminal", "notes", "calculator", "finder"],
            }
        },
        "required": ["app"],
        "additionalProperties": False,
    }

    async def run(self, app: str) -> ToolResult:
        system = platform.system()
        commands = _APP_COMMANDS.get(system)
        if commands is None:
            return ToolResult(ok=False, output="", summary=f"Unsupported platform: {system}")

        cmd = commands.get(app)
        if cmd is None:
            return ToolResult(ok=False, output="", summary=f"App not allowlisted on {system}: {app!r}")

        try:
            await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
        except FileNotFoundError as exc:
            return ToolResult(ok=False, output="", summary=f"Could not launch {app!r}: {exc}")

        return ToolResult(ok=True, output="", summary=f"Opened {app}")
