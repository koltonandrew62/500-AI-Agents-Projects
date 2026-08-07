"""The provider-agnostic LLM interface.

CONTRACTS.md section 4 defines this exact shape:

    class LLMProvider(Protocol):
        async def stream(self, messages: list[Message], tools: list[ToolSpec] | None
                         ) -> AsyncIterator[LLMDelta]: ...
        async def complete(self, messages: list[Message]) -> str: ...
        async def vision(self, messages: list[Message], image_b64: str) -> str: ...

Any concrete provider (OpenRouter today, others later) implements this
Protocol structurally -- no inheritance required. The agent loop and the
router depend only on this interface, never on a concrete provider class.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from app.models.schemas import LLMDelta, Message, ToolSpec


@runtime_checkable
class LLMProvider(Protocol):
    """Structural interface every LLM backend must satisfy."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMDelta]:
        """Stream a completion token-by-token.

        Yields :class:`LLMDelta` chunks. Text deltas carry partial text in
        ``.text``. A fully-accumulated tool call is yielded as a single
        ``LLMDelta`` with ``tool_name`` and ``tool_args`` set (never as
        fragments -- accumulation across chunks is the provider's job). The
        final delta of the stream has ``finished=True``.
        """
        ...

    async def complete(self, messages: list[Message]) -> str:
        """Return a single, fully-materialized completion (no streaming)."""
        ...

    async def vision(self, messages: list[Message], image_b64: str) -> str:
        """Return a completion grounded in an image.

        ``image_b64`` is a raw base64-encoded JPEG payload (no data-URI
        prefix required from the caller -- the provider is responsible for
        wrapping it correctly for the wire format it speaks).
        """
        ...
