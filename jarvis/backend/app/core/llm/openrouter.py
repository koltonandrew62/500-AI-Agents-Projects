"""OpenRouter implementation of :class:`LLMProvider`.

Talks to ``https://openrouter.ai/api/v1/chat/completions`` via
``httpx.AsyncClient``. One :class:`OpenRouterProvider` instance is bound to a
single model (the router owns picking *which* model/instance to use and
falling back between them).

Supports:
  * SSE token streaming, including OpenAI-style ``tool_calls`` deltas which
    arrive fragmented (by index, with ``function.arguments`` built up one
    partial JSON chunk at a time) and must be accumulated before we can hand
    a caller a complete, parseable tool call.
  * Vision turns via OpenAI-style multimodal ``content`` arrays
    (``image_url`` parts holding a base64 data URI).
  * The ``HTTP-Referer`` / ``X-Title`` headers OpenRouter uses to attribute
    traffic on https://openrouter.ai/rankings.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.core.llm.errors import BadResponse, NoKey, ProviderDown, RateLimited
from app.models.schemas import LLMDelta, Message, ToolSpec

CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"

# Identifies this app to OpenRouter (shows up in their dashboard/rankings).
# Not secret -- these are informational headers, not credentials.
DEFAULT_HTTP_REFERER = "https://github.com/jarvis-assistant"
DEFAULT_X_TITLE = "J.A.R.V.I.S."

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=15.0, pool=10.0)


def _resolve_api_key(explicit: str | None) -> str:
    """Resolve the OpenRouter API key, raising :class:`NoKey` if unset.

    Imports ``app.config`` lazily (rather than at module import time) so
    that importing this module never hard-fails while ``app/config.py`` is
    still being built by another agent -- the failure is deferred to the
    point where a key is actually needed, and surfaced as a clear ``NoKey``
    instead of a raw ``ImportError``.
    """
    if explicit:
        return explicit
    try:
        from app.config import settings
    except ImportError as exc:  # pragma: no cover - depends on sibling agent
        raise NoKey(
            "No OpenRouter API key configured (app.config could not be "
            "imported). Set OPENROUTER_API_KEY in your .env file (see "
            ".env.example) and restart the backend."
        ) from exc

    key = getattr(settings, "openrouter_api_key", None)
    if not key:
        raise NoKey()
    return key


def _to_data_uri(image: str) -> str:
    """Wrap a raw base64 JPEG payload as a data URI, unless already one."""
    if image.startswith("data:"):
        return image
    return f"data:image/jpeg;base64,{image}"


def _message_to_wire(msg: Message, extra_images: list[str] | None = None) -> dict[str, Any]:
    """Convert a :class:`Message` to OpenAI/OpenRouter chat-completions wire format."""
    images = list(msg.images)
    if extra_images:
        images.extend(extra_images)

    wire: dict[str, Any] = {"role": msg.role.value}
    if images:
        parts: list[dict[str, Any]] = []
        if msg.content:
            parts.append({"type": "text", "text": msg.content})
        for image in images:
            parts.append({"type": "image_url", "image_url": {"url": _to_data_uri(image)}})
        wire["content"] = parts
    else:
        wire["content"] = msg.content

    if msg.name:
        wire["name"] = msg.name
    if msg.tool_call_id:
        wire["tool_call_id"] = msg.tool_call_id
    return wire


def _tools_to_wire(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None  # HTTP-date form; not worth parsing for our backoff purposes


class _ToolCallAccumulator:
    """Accumulates fragmented OpenAI-style streamed tool-call deltas by index.

    OpenRouter (mirroring the OpenAI streaming format) sends tool calls as a
    list of partial ``tool_calls`` entries per chunk, each carrying an
    ``index``. The function ``name`` typically arrives whole on the first
    fragment for that index; ``arguments`` arrives as successive partial
    JSON-string fragments that must be concatenated before parsing. There is
    no explicit "this tool call is done" marker per-call -- completeness is
    only known once the stream reports ``finish_reason`` (or ends).
    """

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, str]] = {}
        self._order: list[int] = []

    def add(self, tool_call_deltas: list[dict[str, Any]]) -> None:
        for delta in tool_call_deltas:
            index = delta.get("index", 0)
            entry = self._by_index.get(index)
            if entry is None:
                entry = {"id": "", "name": "", "arguments": ""}
                self._by_index[index] = entry
                self._order.append(index)

            call_id = delta.get("id")
            if call_id:
                entry["id"] = call_id

            function = delta.get("function") or {}
            name = function.get("name")
            if name:
                entry["name"] += name
            arguments = function.get("arguments")
            if arguments:
                entry["arguments"] += arguments

    def finalize(self, model: str) -> list[LLMDelta]:
        """Parse and yield every accumulated tool call, in first-seen order."""
        deltas: list[LLMDelta] = []
        for index in self._order:
            entry = self._by_index[index]
            if not entry["name"]:
                raise BadResponse(
                    model, detail=f"streamed tool call at index {index} has no function name"
                )
            raw_args = entry["arguments"] or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise BadResponse(
                    model,
                    detail=f"could not parse arguments for tool '{entry['name']}': {exc}",
                ) from exc
            deltas.append(LLMDelta(tool_name=entry["name"], tool_args=args))
        self._by_index.clear()
        self._order.clear()
        return deltas


class OpenRouterProvider:
    """`LLMProvider` backed by a single OpenRouter model.

    One instance == one model. :class:`~app.core.llm.router.ModelRouter`
    holds one instance per model in a fallback chain and tries them in turn.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        http_referer: str = DEFAULT_HTTP_REFERER,
        x_title: str = DEFAULT_X_TITLE,
    ) -> None:
        self.model = model
        self._api_key = api_key  # resolved lazily so construction never raises NoKey early
        self._explicit_api_key = api_key
        self._timeout = timeout
        self._http_referer = http_referer
        self._x_title = x_title

    def _headers(self) -> dict[str, str]:
        key = _resolve_api_key(self._explicit_api_key)
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._http_referer,
            "X-Title": self._x_title,
        }

    def _build_payload(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        stream: bool,
        extra_image_for_last: str | None = None,
    ) -> dict[str, Any]:
        wire_messages: list[dict[str, Any]] = []
        last_index = len(messages) - 1
        for i, msg in enumerate(messages):
            extra = [extra_image_for_last] if extra_image_for_last and i == last_index else None
            wire_messages.append(_message_to_wire(msg, extra))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "stream": stream,
        }
        wire_tools = _tools_to_wire(tools)
        if wire_tools:
            payload["tools"] = wire_tools
            payload["tool_choice"] = "auto"
        return payload

    async def _raise_for_error_status(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            raise RateLimited(self.model, retry_after=_parse_retry_after(response.headers))
        if response.status_code >= 500:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise ProviderDown(self.model, detail=f"HTTP {response.status_code}: {body[:500]}")
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise BadResponse(self.model, detail=f"HTTP {response.status_code}: {body[:500]}")

    # -- LLMProvider protocol -------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMDelta]:
        payload = self._build_payload(messages, tools=tools, stream=True)
        accumulator = _ToolCallAccumulator()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload
                ) as response:
                    await self._raise_for_error_status(response)

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise BadResponse(
                                self.model, detail=f"malformed SSE JSON: {exc}"
                            ) from exc

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        finish_reason = choice.get("finish_reason")

                        text = delta.get("content")
                        if text:
                            yield LLMDelta(text=text)

                        tool_call_deltas = delta.get("tool_calls")
                        if tool_call_deltas:
                            accumulator.add(tool_call_deltas)

                        if finish_reason:
                            for tool_delta in accumulator.finalize(self.model):
                                yield tool_delta

        except httpx.TimeoutException as exc:
            raise ProviderDown(self.model, detail=f"timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderDown(self.model, detail=str(exc)) from exc

        # Flush any tool calls that never got an explicit finish_reason chunk
        # (some providers omit it and just close the stream instead).
        for tool_delta in accumulator.finalize(self.model):
            yield tool_delta

        yield LLMDelta(finished=True)

    async def complete(self, messages: list[Message]) -> str:
        payload = self._build_payload(messages, tools=None, stream=False)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload
                )
        except httpx.TimeoutException as exc:
            raise ProviderDown(self.model, detail=f"timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderDown(self.model, detail=str(exc)) from exc

        await self._raise_for_error_status(response)

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise BadResponse(self.model, detail=f"unexpected response shape: {exc}") from exc

    async def vision(self, messages: list[Message], image_b64: str) -> str:
        if not messages:
            raise BadResponse(self.model, detail="vision() requires at least one message")

        payload = self._build_payload(
            messages, tools=None, stream=False, extra_image_for_last=image_b64
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload
                )
        except httpx.TimeoutException as exc:
            raise ProviderDown(self.model, detail=f"timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderDown(self.model, detail=str(exc)) from exc

        await self._raise_for_error_status(response)

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise BadResponse(self.model, detail=f"unexpected response shape: {exc}") from exc
