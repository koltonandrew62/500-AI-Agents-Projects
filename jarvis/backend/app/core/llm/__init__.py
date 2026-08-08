"""LLM layer: provider protocol, OpenRouter implementation, and the router.

Everything downstream (the agent loop, API routes) should depend on this
package's public surface -- ``ModelRouter`` for normal use, ``LLMProvider``
for typing, ``OpenRouterProvider`` only if a single fixed model is needed
without fallback -- and never reach into submodules directly.
"""

from app.core.llm.errors import (
    AllModelsExhausted,
    BadResponse,
    LLMError,
    NoKey,
    ProviderDown,
    RateLimited,
)
from app.core.llm.openrouter import OpenRouterProvider
from app.core.llm.provider import LLMProvider
from app.core.llm.router import (
    CHAT_CHAIN,
    PLANNING_CHAIN,
    VISION_CHAIN,
    ChainName,
    ModelRouter,
    RoutedCompletion,
    RoutedStream,
)

__all__ = [
    # Protocol
    "LLMProvider",
    # Providers
    "OpenRouterProvider",
    # Router
    "ModelRouter",
    "RoutedCompletion",
    "RoutedStream",
    "ChainName",
    "PLANNING_CHAIN",
    "CHAT_CHAIN",
    "VISION_CHAIN",
    # Errors
    "LLMError",
    "NoKey",
    "RateLimited",
    "ProviderDown",
    "BadResponse",
    "AllModelsExhausted",
]
