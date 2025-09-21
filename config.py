"""Centralized configuration for the Naexya Docs AI application.

This module defines provider metadata, persona prompt templates, specification
categories, and export rendering configuration in a single location. Keeping
these values together makes it easier to maintain consistent behaviour across
modules such as ``ai_client.py`` and ``app.py``.

The dictionaries below are intentionally verbose and heavily commented so that
future contributors can understand every field without cross-referencing API
documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Loading .env files is optional but convenient for local development.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency may be missing in some envs.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """Fallback stub when python-dotenv is not installed."""

        return False

# ---------------------------------------------------------------------------
# AI Provider configuration
# ---------------------------------------------------------------------------
# ``AI_PROVIDERS`` captures the details required to interact with each
# third-party large language model. Each entry explains the authentication
# header, supported models, and default parameter choices that the application
# should use. Additional providers can be added by following the same schema.
AI_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "display_name": "OpenAI",
        # Base endpoint for Chat Completions. Individual modules append
        # provider-specific paths as needed.
        "base_url": "https://api.openai.com/v1",
        "chat_endpoint": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-5",
        "available_models": ["gpt-5"],
        # The provider requires a Bearer token with the ``Authorization`` header.
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
        # Conservative defaults to balance quality with latency and cost.
        "default_params": {"temperature": 0.7, "max_tokens": 2048},
        # Basic rate-limit guidance for UI messaging and back-off strategies.
        "rate_limits": {
            "requests_per_minute": 500,
            "tokens_per_minute": 600000,
        },
    },
    "anthropic": {
        "display_name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "chat_endpoint": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-4-sonnet",
        "available_models": ["claude-4-sonnet"],
        # Anthropic expects both ``x-api-key`` and ``anthropic-version`` headers.
        "headers": {
            "x-api-key": "{api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        "default_params": {"temperature": 0.7, "max_tokens": 2048},
        "rate_limits": {
            "requests_per_minute": 400,
            "tokens_per_minute": 480000,
        },
    },
    "google": {
        "display_name": "Google",
        "base_url": "https://generativelanguage.googleapis.com/v1",
        "chat_endpoint": "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-pro:generateContent",
        "default_model": "gemini-2.5-pro",
        "available_models": ["gemini-2.5-pro"],
        # Gemini uses a query parameter for the API key; headers remain JSON.
        "headers": {"Content-Type": "application/json"},
        "default_params": {"temperature": 0.7, "max_output_tokens": 2048},
        "rate_limits": {
            "requests_per_minute": 300,
            "tokens_per_minute": 360000,
        },
    },
    "xai": {
        "display_name": "xAI",
        "base_url": "https://api.x.ai/v1",
        "chat_endpoint": "https://api.x.ai/v1/chat/completions",
        "default_model": "grok-4-fast",
        "available_models": ["grok-4-fast"],
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "default_params": {"temperature": 0.7, "max_tokens": 2048},
        "rate_limits": {
            "requests_per_minute": 200,
            "tokens_per_minute": 240000,
        },
    },
    "moonshot": {
        "display_name": "Moonshot",
        "base_url": "https://api.moonshot.ai/v1",
        "chat_endpoint": "https://api.moonshot.ai/v1/chat/completions",
        "default_model": "kimi-k2",
        "available_models": ["kimi-k2"],
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "default_params": {"temperature": 0.7, "max_tokens": 2048},
        "rate_limits": {
            "requests_per_minute": 150,
            "tokens_per_minute": 180000,
        },
    },
    "qwen": {
        "display_name": "Qwen",
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "chat_endpoint": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "default_model": "qwen3-next",
        "available_models": ["qwen3-next"],
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "default_params": {"temperature": 0.7, "max_tokens": 2048},
        "rate_limits": {
            "requests_per_minute": 250,
            "tokens_per_minute": 300000,
        },
    },
}

# ---------------------------------------------------------------------------
# Persona configuration
# ---------------------------------------------------------------------------
# Personas determine how AI assistants respond to users. Providing rich,
# descriptive prompts ensures that conversations remain on-topic and that the
# extracted specifications are actionable.
AI_PERSONAS: Dict[str, Dict[str, str]] = {
    "requirements_specialist": {
        "display_name": "Requirements Specialist",
        "prompt": (
            "You are an expert business analyst specializing in gathering and "
            "documenting software requirements. Focus on user stories, business "
            "features, workflows, and functional requirements. Always ask "
            "clarifying questions and provide structured output."
        ),
    },
    "technical_architect": {
        "display_name": "Technical Architect",
        "prompt": (
            "You are a senior technical architect specializing in system design "
            "and implementation. Focus on API specifications, database schemas, "
            "system architecture, and technical implementation details. Provide "
            "detailed technical specifications."
        ),
    },
}

# ---------------------------------------------------------------------------
# Specification taxonomy
# ---------------------------------------------------------------------------
# ``SPECIFICATION_TYPES`` controls the categories displayed in the UI when
# reviewing and exporting specifications.
SPECIFICATION_TYPES = [
    "User Stories",
    "Features",
    "API Endpoints",
    "Database Design",
    "System Architecture",
]

# ---------------------------------------------------------------------------
# Export template configuration
# ---------------------------------------------------------------------------
# Each export format references template files stored under ``templates/``. The
# metadata here describes how those templates should be used by the export
# helpers in ``utils.py`` or ``app.py``.
EXPORT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "html": {
        "path": "templates/export_html.html",
        "content_type": "text/html",
        "description": "Rich HTML report suitable for sharing with stakeholders.",
    },
    "markdown": {
        "path": "templates/export_markdown.md",
        "content_type": "text/markdown",
        "description": "Lightweight Markdown export for version control or wikis.",
    },
}

# ---------------------------------------------------------------------------
# Application configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProviderCredential:
    """Runtime view of provider configuration resolved from the environment."""

    provider: str
    env_var: str
    api_key: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Return the human-friendly name defined in ``AI_PROVIDERS``."""

        provider_meta = AI_PROVIDERS.get(self.provider, {})
        return provider_meta.get("display_name", self.provider.title())


@dataclass
class AppConfig:
    """Container holding runtime configuration for the Gradio interface."""

    database_path: Optional[Path]
    providers: Dict[str, ProviderCredential] = field(default_factory=dict)
    default_provider: str = "openai"
    demo_mode: bool = False
    space_id: Optional[str] = None
    persistence_enabled: bool = True

    @classmethod
    def from_environment(cls) -> "AppConfig":
        """Build an :class:`AppConfig` instance using environment variables."""

        load_dotenv()
        validate_configuration()

        env = os.environ
        is_spaces = any(env.get(var) for var in ("SPACE_ID", "HF_SPACE_ID", "HF_HOME"))

        def _is_truthy(value: Optional[str]) -> bool:
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        disable_storage = _is_truthy(env.get("NAEXYA_DISABLE_STORAGE"))
        enable_storage = _is_truthy(env.get("NAEXYA_ENABLE_STORAGE"))

        if disable_storage:
            persistence_enabled = False
        elif enable_storage:
            persistence_enabled = True
        else:
            # Hugging Face Spaces mount a read-only filesystem for the repository.
            # Default to in-memory storage there unless explicitly overridden.
            persistence_enabled = not is_spaces

        if persistence_enabled:
            data_dir = Path(
                env.get("NAEXYA_DATA_DIR")
                or ("/data" if is_spaces else Path(__file__).resolve().parent)
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            database_path: Optional[Path] = (
                data_dir / env.get("NAEXYA_DB_FILENAME", "naexya_docs_ai.db")
            ).resolve()
        else:
            database_path = None

        provider_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "xai": "XAI_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "qwen": "QWEN_API_KEY",
        }

        providers = {
            name: ProviderCredential(
                provider=name,
                env_var=env_var,
                api_key=env.get(env_var) or None,
            )
            for name, env_var in provider_env_map.items()
        }

        # Choose a sensible default provider, preferring explicit environment configuration.
        configured = [key for key, cred in providers.items() if cred.api_key]
        requested_default = (env.get("NAEXYA_DEFAULT_PROVIDER") or "openai").lower()
        if requested_default not in providers:
            requested_default = "openai"
        default_provider = requested_default if (configured and requested_default in configured) else (configured[0] if configured else "openai")

        demo_mode = not bool(configured)

        return cls(
            database_path=database_path,
            providers=providers,
            default_provider=default_provider,
            demo_mode=demo_mode,
            space_id=env.get("SPACE_ID") or env.get("HF_SPACE_ID"),
            persistence_enabled=persistence_enabled,
        )

    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve the configured API key for ``provider`` if available."""

        credential = self.providers.get(provider.lower())
        return credential.api_key if credential else None

    def configured_providers(self) -> Dict[str, ProviderCredential]:
        """Return only the providers that currently have API keys configured."""

        return {name: cred for name, cred in self.providers.items() if cred.api_key}


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------
# The functions below provide quick sanity checks that configuration dictionaries
# contain the expected fields. They raise ``ValueError`` with descriptive
# messages so callers can fail fast during application start-up.

def validate_provider_config(provider_key: str) -> None:
    """Validate a single provider configuration entry.

    Args:
        provider_key: The dictionary key identifying the provider (e.g. ``"openai"``).

    Raises:
        ValueError: If required fields are missing or improperly formatted.
    """

    config = AI_PROVIDERS.get(provider_key)
    if config is None:
        raise ValueError(f"Provider '{provider_key}' is not defined in AI_PROVIDERS.")

    required_fields = [
        "display_name",
        "base_url",
        "chat_endpoint",
        "default_model",
        "headers",
        "default_params",
        "rate_limits",
    ]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(
            f"Provider '{provider_key}' is missing required fields: {', '.join(missing)}"
        )

    if "Authorization" in config["headers"] and "{api_key}" not in config["headers"]["Authorization"]:
        raise ValueError(
            f"Provider '{provider_key}' Authorization header must include '{{api_key}}' placeholder."
        )


def validate_all_providers() -> None:
    """Validate every provider configuration entry."""

    for provider_key in AI_PROVIDERS:
        validate_provider_config(provider_key)


def validate_personas() -> None:
    """Ensure persona definitions include prompts for consistent behaviour."""

    for key, persona in AI_PERSONAS.items():
        if "prompt" not in persona or not persona["prompt"].strip():
            raise ValueError(f"Persona '{key}' must include a non-empty prompt.")


def validate_specification_types() -> None:
    """Verify specification types are unique and non-empty."""

    if not SPECIFICATION_TYPES:
        raise ValueError("SPECIFICATION_TYPES must contain at least one entry.")

    normalized = [spec.strip() for spec in SPECIFICATION_TYPES if spec.strip()]
    if len(normalized) != len(SPECIFICATION_TYPES):
        raise ValueError("SPECIFICATION_TYPES must not contain blank values.")

    if len(set(normalized)) != len(normalized):
        raise ValueError("SPECIFICATION_TYPES entries must be unique.")


def validate_export_templates() -> None:
    """Confirm export template metadata includes expected fields."""

    required_fields = {"path", "content_type", "description"}
    for key, template in EXPORT_TEMPLATES.items():
        missing = required_fields - template.keys()
        if missing:
            raise ValueError(
                f"Export template '{key}' is missing fields: {', '.join(sorted(missing))}"
            )


def validate_configuration() -> None:
    """Run all configuration validators.

    This helper is convenient during application start-up to ensure environment
    configuration issues are detected early rather than failing deep inside the
    request cycle.
    """

    validate_all_providers()
    validate_personas()
    validate_specification_types()
    validate_export_templates()


__all__ = [
    "AI_PROVIDERS",
    "AI_PERSONAS",
    "SPECIFICATION_TYPES",
    "EXPORT_TEMPLATES",
    "ProviderCredential",
    "AppConfig",
    "validate_configuration",
]
