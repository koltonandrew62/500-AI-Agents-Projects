"""Typed exceptions for the LLM layer.

Every failure mode that can occur while talking to an LLM provider is
represented by a dedicated exception class so callers (the agent loop, the
router, API routes) can catch precisely what they know how to handle instead
of doing string-matching on generic ``Exception`` messages.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-layer errors."""


class NoKey(LLMError):
    """Raised when no API key is configured for a provider.

    This is a configuration error, not a transient failure -- it will not
    resolve itself by retrying.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "No OpenRouter API key configured. Set OPENROUTER_API_KEY "
                "in your .env file (see .env.example) and restart the "
                "backend."
            )
        )


class RateLimited(LLMError):
    """Raised on HTTP 429 (or an equivalent rate-limit signal) from a provider.

    Carries the model that was rate limited and, when the provider supplied
    one, the number of seconds the caller should wait before retrying.
    """

    def __init__(self, model: str, retry_after: float | None = None) -> None:
        self.model = model
        self.retry_after = retry_after
        suffix = f" (retry after {retry_after:.1f}s)" if retry_after else ""
        super().__init__(f"Rate limited by model '{model}'{suffix}")


class ProviderDown(LLMError):
    """Raised on HTTP 5xx, timeouts, or connection failures.

    Distinct from :class:`RateLimited` because it signals the provider (or
    the specific model backend) is unavailable rather than throttling us.
    """

    def __init__(self, model: str, detail: str = "") -> None:
        self.model = model
        self.detail = detail
        msg = f"Provider unavailable for model '{model}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class BadResponse(LLMError):
    """Raised when a provider returns a 2xx response that we cannot parse.

    Examples: malformed JSON, missing ``choices``, an SSE stream that never
    terminates cleanly, or a response shape that doesn't match what the
    OpenRouter API is documented to return.
    """

    def __init__(self, model: str, detail: str = "") -> None:
        self.model = model
        self.detail = detail
        msg = f"Bad response from model '{model}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class AllModelsExhausted(LLMError):
    """Raised by the router when every model in a chain has failed.

    Carries the per-model errors so the caller can log or surface a useful
    diagnostic instead of a single opaque failure.
    """

    def __init__(self, chain_name: str, errors: dict[str, Exception]) -> None:
        self.chain_name = chain_name
        self.errors = errors
        detail = "; ".join(f"{model}: {err}" for model, err in errors.items())
        super().__init__(
            f"All models in chain '{chain_name}' failed: {detail}"
        )
