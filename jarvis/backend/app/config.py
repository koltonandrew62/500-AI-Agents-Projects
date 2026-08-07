"""Application configuration for J.A.R.V.I.S.

Single source of truth for runtime settings, loaded from environment
variables / a `.env` file via pydantic-settings. This module MUST be
import-safe with no required environment variables present — every other
module does `from app.config import settings` at import time (including
during test collection), so a missing `.env` or missing API key must never
raise here. The LLM layer is responsible for raising a clear error the
first time it actually tries to call OpenRouter without a key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# ---------------------------------------------------------------------------
# Free-tier model chains (see docs/CONTRACTS.md section 5)
# ---------------------------------------------------------------------------

DEFAULT_PLANNING_MODELS: list[str] = [
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

DEFAULT_CHAT_MODELS: list[str] = [
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# OpenRouter free vision-capable models. Kept separate from the text chains
# since not every free-tier text model accepts image inputs.
DEFAULT_VISION_MODELS: list[str] = [
    "qwen/qwen2.5-vl-32b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
]


class Settings(BaseSettings):
    """Runtime configuration, populated from environment / `.env`.

    Every field has a safe default so the module can be imported (and the
    app can boot far enough to answer health checks) even with no `.env`
    file at all. `openrouter_api_key` defaults to an empty string rather
    than `None` so downstream code can do simple truthiness checks; the LLM
    provider layer is what actually enforces "you need a key to call this".
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- LLM provider -------------------------------------------------
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key. Required at call time, not at import time.",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter-compatible API base URL.",
    )

    # -- Model chains (ordered: primary -> fallback -> last resort) ---
    # NoDecode: pydantic-settings otherwise tries json.loads() on env/dotenv
    # string values for list-typed fields before our validator ever runs,
    # which blows up on plain comma-separated values like "a,b,c". NoDecode
    # hands the raw string straight to `_split_csv` below instead.
    planning_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_PLANNING_MODELS)
    )
    chat_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_CHAT_MODELS)
    )
    vision_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_VISION_MODELS)
    )

    # -- Storage --------------------------------------------------------
    workspace_root: Path = Field(default=Path.home() / "jarvis-workspace")
    memory_db_path: Path = Field(default=Path.home() / "jarvis-workspace" / "memory.db")
    tasks_db_path: Path = Field(default=Path.home() / "jarvis-workspace" / "tasks.db")

    # -- Server -----------------------------------------------------------
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # -- Sensing / features --------------------------------------------
    telemetry_interval_ms: int = Field(default=1000)
    vision_enabled: bool = Field(default=True)
    voice_enabled: bool = Field(default=True)

    # -- Memory / embeddings --------------------------------------------
    embed_model: str = Field(default="all-MiniLM-L6-v2")

    # -- Agent loop -------------------------------------------------------
    max_tool_iterations: int = Field(default=8)

    # -- Logging ------------------------------------------------------
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    # Validators / coercion helpers — allow comma-separated env strings
    # for list-typed fields (`FOO=a,b,c`) in addition to JSON arrays,
    # since plain env files rarely quote JSON cleanly.
    # ------------------------------------------------------------------

    @field_validator(
        "planning_models", "chat_models", "vision_models", "cors_origins", mode="before"
    )
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                import json

                return json.loads(stripped)  # explicit JSON array syntax
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("workspace_root", "memory_db_path", "tasks_db_path", mode="before")
    @classmethod
    def _expand_path(cls, value: object) -> object:
        if isinstance(value, str) and value:
            return Path(value).expanduser()
        return value

    def ensure_dirs(self) -> None:
        """Create the workspace root and parent dirs of DB paths if missing.

        Safe to call multiple times; never raises on a pre-existing directory.
        """
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks_db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
