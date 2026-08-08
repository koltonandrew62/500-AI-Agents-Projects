"""Tool registry: the `Tool` protocol, `@register` decorator, and the
`registry` singleton the agent loop walks to discover capabilities.

See docs/CONTRACTS.md §4 — this is the authoritative implementation of the
`core/tools/base.py` interface described there. Tools are never hardcoded
into the agent loop; it always calls `registry.all()` / `run_tool(name, args)`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.models.schemas import ToolResult, ToolSpec

logger = logging.getLogger("jarvis.tools")

# Safety-net ceiling for any single tool call. Individual tools may (and the
# sandbox tools do) enforce their own tighter timeouts; this just guarantees
# the agent loop can never be blocked forever by a tool that forgot to.
DEFAULT_TOOL_TIMEOUT_S = 30.0


@runtime_checkable
class Tool(Protocol):
    """Structural contract every tool implements (docs/CONTRACTS.md §4)."""

    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema object describing accepted params

    async def run(self, **kwargs: Any) -> ToolResult: ...


@dataclass
class ToolRegistry:
    """In-memory registry of every tool that has self-registered via `@register`."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> Tool:
        if not getattr(tool, "name", None):
            raise ValueError(f"tool {tool!r} has no non-empty 'name' attribute")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool
        logger.debug("registered tool %r", tool.name)
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        """Every registered tool's public spec, in registration order."""
        return [
            ToolSpec(name=t.name, description=t.description, parameters=t.schema)
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())


registry = ToolRegistry()


def register(cls_or_instance: type[Tool] | Tool) -> Tool:
    """Decorator that instantiates (if given a class) and registers a Tool.

    Usage::

        @register
        class ReadFileTool:
            name = "read_file"
            description = "..."
            schema = {...}
            async def run(self, **kwargs) -> ToolResult: ...

    The module-level name is rebound to the *instance*, matching the
    singleton-per-tool pattern used throughout this package.
    """
    instance = cls_or_instance() if isinstance(cls_or_instance, type) else cls_or_instance
    return registry.register(instance)


# ---------------------------------------------------------------------------
# JSON-Schema-subset validation
# ---------------------------------------------------------------------------
#
# We deliberately do not take a dependency on the `jsonschema` package (it is
# not part of this module's declared dependency surface). This supports the
# subset of JSON Schema actually used by tool authors in this package: type,
# properties, required, enum, minimum/maximum, minLength/maxLength, items,
# and additionalProperties. Unknown keywords are ignored rather than
# rejected — permissive by design so schemas can carry descriptive-only keys.

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _check_type(value: Any, expected: str) -> bool:
    py_type = _JSON_TYPES.get(expected)
    if py_type is None:
        return True  # unrecognized type keyword — don't block on it
    if expected in ("integer", "number") and isinstance(value, bool):
        return False  # bool is an int subclass in Python; JSON Schema disallows this
    return isinstance(value, py_type)


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> str | None:
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: must be one of {schema['enum']!r}"

    expected_type = schema.get("type")
    if expected_type and not _check_type(value, expected_type):
        return f"{path}: expected type {expected_type!r}, got {type(value).__name__}"

    if expected_type == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{path}: shorter than minLength {schema['minLength']}"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{path}: longer than maxLength {schema['maxLength']}"

    if expected_type in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum {schema['maximum']}"

    if expected_type == "array":
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                err = _validate_value(item, item_schema, f"{path}[{i}]")
                if err:
                    return err

    if expected_type == "object":
        err = _validate_object(value, schema, path)
        if err:
            return err

    return None


def _validate_object(args: dict[str, Any], schema: dict[str, Any], path: str = "$") -> str | None:
    if not isinstance(args, dict):
        return f"{path}: expected an object, got {type(args).__name__}"

    for key in schema.get("required", []):
        if key not in args:
            return f"{path}: missing required field {key!r}"

    properties = schema.get("properties", {})
    # Closed-schema guard: when a tool opts into additionalProperties: False,
    # reject any argument the tool doesn't declare — this is what stops a
    # confused/adversarial caller from smuggling extra kwargs into `run()`.
    if schema.get("additionalProperties") is False:
        extra = set(args) - set(properties)
        if extra:
            return f"{path}: unexpected field(s) {sorted(extra)!r}"

    for key, value in args.items():
        prop_schema = properties.get(key)
        if prop_schema is None:
            continue
        err = _validate_value(value, prop_schema, f"{path}.{key}")
        if err:
            return err

    return None


def validate_schema(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Validate `args` against `schema`. Returns an error string, or None if valid."""
    if schema.get("type", "object") != "object":
        return None
    return _validate_object(args, schema)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


async def run_tool(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT_S,
) -> ToolResult:
    """Look up, schema-validate, and execute a tool by name.

    This is the single entrypoint the agent loop should call. It never
    raises: unknown tool names, schema violations, timeouts, and exceptions
    raised by the tool implementation itself are all normalized into
    `ToolResult(ok=False, ...)` so one misbehaving tool can never take down
    a turn.
    """
    call_args = dict(args or {})

    tool = registry.get(name)
    if tool is None:
        return ToolResult(ok=False, output="", summary=f"Unknown tool: {name!r}")

    error = validate_schema(tool.schema, call_args)
    if error is not None:
        return ToolResult(ok=False, output="", summary=f"Invalid arguments for {name!r}: {error}")

    try:
        result = await asyncio.wait_for(tool.run(**call_args), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("tool %r timed out after %.1fs", name, timeout)
        return ToolResult(ok=False, output="", summary=f"Tool {name!r} timed out after {timeout:.0f}s")
    except TypeError as exc:
        # Usually means the schema let through args that don't match run()'s
        # real signature — treat as a validation failure, not a crash.
        logger.exception("tool %r called with a bad signature", name)
        return ToolResult(ok=False, output="", summary=f"Tool {name!r} argument error: {exc}")
    except Exception as exc:  # noqa: BLE001 - tool boundary must never raise
        logger.exception("tool %r raised", name)
        return ToolResult(ok=False, output="", summary=f"Tool {name!r} failed: {exc}")

    if not isinstance(result, ToolResult):
        logger.error("tool %r returned %r instead of a ToolResult", name, type(result))
        return ToolResult(ok=False, output="", summary=f"Tool {name!r} returned an invalid result type")
    return result
