"""Tool package entrypoint.

Importing this package imports every tool module so their `@register`
decorators run and populate the shared `registry`. The agent loop only ever
needs `registry` (to list specs) and `run_tool` (to execute by name) —
never the individual tool classes.
"""

from __future__ import annotations

from app.core.tools import base, files, sandbox, scheduler, system, web  # noqa: F401
from app.core.tools.base import registry, run_tool
from app.core.tools.scheduler import run_scheduler

__all__ = ["registry", "run_tool", "run_scheduler"]
