"""Free-tier model routing with transparent fallback.

Implements the fallback chains from CONTRACTS.md section 5: three purposes
(``planning``, ``chat``, ``vision``), each an ordered list of OpenRouter free
models. On a rate limit (429) or server error (5xx) the router retries the
current model with exponential backoff + jitter (up to
``max_attempts_per_model``), then transparently falls to the next model in
the chain. A malformed response (:class:`BadResponse`) is treated as
non-transient and moves straight to the next model. A missing API key
(:class:`NoKey`) is a configuration error that no amount of fallback fixes,
so it propagates immediately.

Which model actually served a request is always exposed to the caller:
``RoutedCompletion.model_used`` for one-shot calls, ``RoutedStream.model_used``
for streaming calls (populated as soon as the winning model is known, before
the first chunk is yielded).
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Literal

from app.core.llm.errors import (
    AllModelsExhausted,
    BadResponse,
    NoKey,
    ProviderDown,
    RateLimited,
)
from app.core.llm.openrouter import OpenRouterProvider
from app.models.schemas import LLMDelta, Message, ToolSpec

ChainName = Literal["planning", "chat", "vision"]

# --- Fallback chains --------------------------------------------------------
#
# `planning` is CONTRACTS.md section 5, verbatim and in order.
#
# `chat` and `vision` aren't pinned by CONTRACTS.md to specific model IDs
# (vision only pins "must be a vision-capable free model" with the two
# example IDs given in the build brief); the choices below are this agent's
# assumption -- documented in the summary, easy to override via the `chains`
# constructor kwarg if another agent's expectations differ.

PLANNING_CHAIN: tuple[str, ...] = (
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
)

CHAT_CHAIN: tuple[str, ...] = (
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
)

VISION_CHAIN: tuple[str, ...] = (
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
)

DEFAULT_MAX_ATTEMPTS_PER_MODEL = 3
DEFAULT_BASE_DELAY_S = 0.5
DEFAULT_MAX_DELAY_S = 8.0


@dataclass(frozen=True)
class RoutedCompletion:
    """Result of a non-streaming routed call."""

    text: str
    model_used: str


class RoutedStream:
    """An `AsyncIterator[LLMDelta]` that also exposes which model served it.

    ``model_used`` is ``None`` until the router has committed to a model
    (i.e. that model yielded at least one chunk successfully); it is then
    stable for the remainder of the stream. Checking it only makes sense
    once iteration has started -- it answers "who's serving this", not
    "who will serve this".
    """

    def __init__(self) -> None:
        self.model_used: str | None = None
        self._agen: AsyncIterator[LLMDelta] | None = None

    def __aiter__(self) -> "RoutedStream":
        return self

    async def __anext__(self) -> LLMDelta:
        if self._agen is None:  # pragma: no cover - programmer error guard
            raise RuntimeError("RoutedStream used before its generator was attached")
        return await self._agen.__anext__()


class ModelRouter:
    """Routes calls across a fallback chain of free OpenRouter models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        chains: dict[ChainName, tuple[str, ...]] | None = None,
        max_attempts_per_model: int = DEFAULT_MAX_ATTEMPTS_PER_MODEL,
        base_delay_s: float = DEFAULT_BASE_DELAY_S,
        max_delay_s: float = DEFAULT_MAX_DELAY_S,
    ) -> None:
        self._api_key = api_key
        self._chains: dict[ChainName, tuple[str, ...]] = {
            "planning": PLANNING_CHAIN,
            "chat": CHAT_CHAIN,
            "vision": VISION_CHAIN,
            **(chains or {}),
        }
        self._max_attempts = max(1, max_attempts_per_model)
        self._base_delay = base_delay_s
        self._max_delay = max_delay_s
        self._providers: dict[str, OpenRouterProvider] = {}

        # Best-effort, process-wide "who served last" convenience field.
        # Not concurrency-safe -- under concurrent turns, prefer
        # RoutedCompletion.model_used / RoutedStream.model_used instead.
        self.last_model_used: str | None = None

    def chain_for(self, name: ChainName) -> tuple[str, ...]:
        return self._chains[name]

    def _provider_for(self, model: str) -> OpenRouterProvider:
        provider = self._providers.get(model)
        if provider is None:
            provider = OpenRouterProvider(model, api_key=self._api_key)
            self._providers[model] = provider
        return provider

    async def _backoff(self, attempt: int, retry_after: float | None = None) -> None:
        if retry_after is not None and retry_after > 0:
            delay = min(retry_after, self._max_delay)
        else:
            delay = min(self._max_delay, self._base_delay * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, delay * 0.25)
        await asyncio.sleep(delay + jitter)

    # -- Non-streaming (complete / vision) ------------------------------------

    async def _run_chain(
        self,
        chain_name: ChainName,
        call: Callable[[OpenRouterProvider], Awaitable[str]],
    ) -> RoutedCompletion:
        errors: dict[str, Exception] = {}
        for model in self._chains[chain_name]:
            provider = self._provider_for(model)
            attempt = 1
            while attempt <= self._max_attempts:
                try:
                    text = await call(provider)
                except NoKey:
                    raise  # config error -- no model in any chain will help
                except RateLimited as exc:
                    errors[model] = exc
                    if attempt < self._max_attempts:
                        await self._backoff(attempt, exc.retry_after)
                        attempt += 1
                        continue
                    break
                except ProviderDown as exc:
                    errors[model] = exc
                    if attempt < self._max_attempts:
                        await self._backoff(attempt)
                        attempt += 1
                        continue
                    break
                except BadResponse as exc:
                    errors[model] = exc
                    break  # not transient -- retrying the same model won't help
                else:
                    self.last_model_used = model
                    return RoutedCompletion(text=text, model_used=model)
        raise AllModelsExhausted(chain_name, errors)

    async def complete_chat(self, messages: list[Message]) -> RoutedCompletion:
        """Route a fast conversational completion through the `chat` chain."""
        return await self._run_chain("chat", lambda p: p.complete(messages))

    async def complete_planning(self, messages: list[Message]) -> RoutedCompletion:
        """Route a reasoning-heavy completion through the `planning` chain."""
        return await self._run_chain("planning", lambda p: p.complete(messages))

    async def vision(self, messages: list[Message], image_b64: str) -> RoutedCompletion:
        """Route a vision call through the `vision` chain (vision-capable models only)."""
        return await self._run_chain("vision", lambda p: p.vision(messages, image_b64))

    # -- Streaming -------------------------------------------------------------

    async def _stream_chain(
        self,
        chain_name: ChainName,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        routed: RoutedStream,
    ) -> AsyncIterator[LLMDelta]:
        errors: dict[str, Exception] = {}
        for model in self._chains[chain_name]:
            provider = self._provider_for(model)
            attempt = 1
            while attempt <= self._max_attempts:
                gen = provider.stream(messages, tools)
                try:
                    first_delta = await gen.__anext__()
                except StopAsyncIteration:
                    errors[model] = BadResponse(model, detail="stream produced no output")
                    break
                except NoKey:
                    raise  # config error -- no model in any chain will help
                except RateLimited as exc:
                    errors[model] = exc
                    if attempt < self._max_attempts:
                        await self._backoff(attempt, exc.retry_after)
                        attempt += 1
                        continue
                    break
                except ProviderDown as exc:
                    errors[model] = exc
                    if attempt < self._max_attempts:
                        await self._backoff(attempt)
                        attempt += 1
                        continue
                    break
                except BadResponse as exc:
                    errors[model] = exc
                    break
                else:
                    # First chunk arrived -- this model is live. Commit to it:
                    # no further fallback once any output has reached the
                    # caller, since we can't un-yield partial tokens.
                    routed.model_used = model
                    self.last_model_used = model
                    yield first_delta
                    async for delta in gen:
                        yield delta
                    return
        raise AllModelsExhausted(chain_name, errors)

    def stream_chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> RoutedStream:
        """Route a streaming fast-chat completion through the `chat` chain."""
        routed = RoutedStream()
        routed._agen = self._stream_chain("chat", messages, tools, routed)
        return routed

    def stream_planning(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> RoutedStream:
        """Route a streaming planning completion through the `planning` chain."""
        routed = RoutedStream()
        routed._agen = self._stream_chain("planning", messages, tools, routed)
        return routed
