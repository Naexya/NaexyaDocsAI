"""Unified AI client abstraction for Naexya Docs AI.

This module centralises the integration logic for every supported AI provider,
so the rest of the application can request completions without knowing anything
about HTTP payload formats or authentication details.  The implementation
favours readability and extensive inline documentation over brevity because it
serves as both reference material and an onboarding resource for new
contributors.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import requests

from config import AI_PROVIDERS, AppConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider configuration metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """Container describing the static details for a provider.

    Attributes
    ----------
    name:
        Human friendly label used in logs and error messages.
    endpoint:
        HTTPS endpoint for the chat or text generation API.
    default_model:
        Suggested model identifier when a caller does not provide one.
    supports_streaming:
        Flag documenting whether the HTTP API provides a streaming interface.
    """

    name: str
    endpoint: str
    default_model: str
    supports_streaming: bool = True


PROVIDERS: Dict[str, ProviderConfig] = {
    # OpenAI's Chat Completions endpoint.  Authentication is handled with a
    # Bearer token header, and the request payload is expressed as JSON.
    "openai": ProviderConfig(
        name="OpenAI GPT-5",
        endpoint="https://api.openai.com/v1/chat/completions",
        default_model="gpt-5",
        supports_streaming=True,
    ),
    # Anthropic's Messages API.  This API expects a slightly different JSON
    # schema compared to OpenAI, including an explicit "messages" array with
    # role/content pairs.  It uses X-API-Key and Anthropic-Version headers.
    "anthropic": ProviderConfig(
        name="Anthropic Claude-4-Sonnet",
        endpoint="https://api.anthropic.com/v1/messages",
        default_model="claude-4-sonnet",
        supports_streaming=True,
    ),
    # Google Gemini's Generative Language API.  It expects the content in a
    # "contents" array that contains "parts" objects.  Authentication is
    # performed via a query parameter instead of HTTP headers.
    "google": ProviderConfig(
        name="Google Gemini-2.5-Pro",
        endpoint="https://generativelanguage.googleapis.com/v1/models/"
        "gemini-2.5-pro:generateContent",
        default_model="gemini-2.5-pro",
        supports_streaming=False,
    ),
    # xAI's Grok models mimic the OpenAI schema but use their own endpoint and
    # versioned Accept header.
    "xai": ProviderConfig(
        name="xAI Grok-4-Fast",
        endpoint="https://api.x.ai/v1/chat/completions",
        default_model="grok-4-fast",
        supports_streaming=True,
    ),
    # Moonshot's Kimi API is also compatible with the chat completions format
    # yet includes an "X-Api-Key" header.
    "moonshot": ProviderConfig(
        name="Moonshot Kimi-K2",
        endpoint="https://api.moonshot.ai/v1/chat/completions",
        default_model="kimi-k2",
        supports_streaming=True,
    ),
    # Alibaba's Qwen DashScope endpoint accepts JSON requests with a "model"
    # field and a "input" object.  Streaming requires a special accept header.
    "qwen": ProviderConfig(
        name="Qwen3-Next",
        endpoint="https://dashscope.aliyuncs.com/api/v1/services/"
        "aigc/text-generation/generation",
        default_model="qwen3-next",
        supports_streaming=True,
    ),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_provider_headers(provider: str, api_key: str) -> Dict[str, str]:
    """Return the HTTP headers required for a specific provider.

    Parameters
    ----------
    provider:
        Provider identifier (e.g. ``"openai"``).  Case insensitive.
    api_key:
        Secret token used for authentication.  The function does *not* validate
        the contents but will raise a :class:`ValueError` when missing.

    Notes
    -----
    Each vendor has its own header requirements, therefore the logic is kept in
    a dedicated helper so new providers can be added without modifying the rest
    of the codebase.
    """

    if not api_key:
        raise ValueError("API key is required for provider headers")

    provider_key = provider.lower()
    if provider_key == "openai":
        return {"Authorization": f"Bearer {api_key}"}
    if provider_key == "anthropic":
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    if provider_key == "google":
        # Google uses query string authentication, but we still provide content
        # type for completeness.
        return {"Content-Type": "application/json"}
    if provider_key == "xai":
        return {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
    if provider_key == "moonshot":
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if provider_key == "qwen":
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    raise ValueError(f"Unsupported provider '{provider}'")


def _build_payload(provider: str, model: Optional[str], messages: List[Dict[str, str]]) -> Dict[str, object]:
    """Construct the HTTP payload matching the provider's schema."""

    provider_key = provider.lower()
    if provider_key == "openai" or provider_key == "xai" or provider_key == "moonshot":
        return {
            "model": model,
            "messages": messages,
            "stream": False,
        }
    if provider_key == "anthropic":
        return {
            "model": model,
            "messages": [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in messages
            ],
            "max_tokens": 4096,
            "stream": False,
        }
    if provider_key == "google":
        # Gemini expects nested "contents" with parts containing text payloads.
        return {
            "model": model,
            "contents": [
                {
                    "role": message["role"],
                    "parts": [{"text": message["content"]}],
                }
                for message in messages
            ],
        }
    if provider_key == "qwen":
        return {
            "model": model,
            "input": {
                "messages": messages,
            },
            "parameters": {"enable_search": False},
        }

    raise ValueError(f"Unsupported provider '{provider}'")


# ---------------------------------------------------------------------------
# Core API interaction helpers
# ---------------------------------------------------------------------------

def call_ai_provider(
    provider: str,
    model: Optional[str],
    messages: List[Dict[str, str]],
    api_key: str,
    timeout: int = 60,
) -> Dict[str, object]:
    """Send a chat completion request to the specified provider.

    The helper translates a generic ``messages`` list into the JSON body expected
    by each API.  It returns the parsed JSON response so higher level code can
    extract relevant fields.

    Error handling is intentionally defensive: network errors, non-successful
    HTTP responses, and JSON parsing failures are all logged with context and
    re-raised as :class:`RuntimeError` to keep calling code consistent.
    """

    provider_key = provider.lower()
    if provider_key not in PROVIDERS:
        raise ValueError(f"Unsupported provider '{provider}'")

    config = PROVIDERS[provider_key]
    resolved_model = model or config.default_model
    headers = get_provider_headers(provider_key, api_key)
    payload = _build_payload(provider_key, resolved_model, messages)

    try:
        if provider_key == "google":
            # Google requires the API key as a query parameter rather than header.
            response = requests.post(
                config.endpoint,
                params={"key": api_key},
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
        else:
            response = requests.post(
                config.endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
    except requests.RequestException as exc:  # pragma: no cover - network errors
        logger.exception("Network failure when calling %s", config.name)
        raise RuntimeError(f"Failed to reach {config.name}: {exc}") from exc

    if not response.ok:
        logger.error(
            "Provider %s responded with status %s: %s",
            config.name,
            response.status_code,
            response.text,
        )
        raise RuntimeError(
            f"{config.name} returned {response.status_code}: {response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - unexpected payloads
        logger.exception("Invalid JSON from %s", config.name)
        raise RuntimeError(f"Invalid JSON response from {config.name}") from exc

    return payload


def handle_streaming_response(response: Iterable[bytes]) -> Generator[str, None, None]:
    """Convert a streaming HTTP response into decoded text chunks.

    Some providers (OpenAI, Anthropic, xAI, Moonshot, Qwen) support streaming
    tokens over an HTTP connection.  Gradio primarily expects plain text, so
    this utility yields decoded strings one by one.  Callers can combine the
    chunks or surface them progressively in the UI.
    """

    for chunk in response:
        if not chunk:
            continue
        try:
            decoded = chunk.decode("utf-8")
        except UnicodeDecodeError:  # pragma: no cover - unexpected encoding
            logger.warning("Received non UTF-8 chunk from streaming response")
            continue
        yield decoded


# ---------------------------------------------------------------------------
# Response post-processing utilities
# ---------------------------------------------------------------------------

def extract_specifications_from_response(response_text: str) -> List[Dict[str, str]]:
    """Extract structured specification blocks from the raw model output.

    The helper searches for Markdown style headings and bullet lists describing
    requirements.  The format is intentionally permissive because different
    models may return subtly different layouts.  The result is a list of
    dictionaries so higher level code can serialise or store it easily.
    """

    specs: List[Dict[str, str]] = []
    if not response_text:
        return specs

    pattern = re.compile(r"^#+\\s*(?P<title>.+)$", re.MULTILINE)
    matches = list(pattern.finditer(response_text))

    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response_text)
        body = response_text[start:end].strip()

        if not body:
            continue

        specs.append(
            {
                "title": title,
                "content": body,
                "status": "pending",
            }
        )

    # Fallback: if no headings were found, treat the whole message as a single
    # specification for manual review.
    if not specs:
        specs.append({"title": "Generated Specification", "content": response_text.strip(), "status": "pending"})

    return specs


# ---------------------------------------------------------------------------
# Demo utilities
# ---------------------------------------------------------------------------

def mock_ai_response(persona_type: str, user_message: str) -> str:
    """Return a deterministic response for demo sessions without API keys."""

    persona = persona_type.lower()
    if persona == "business":
        return (
            "# Business Requirement Summary\n"
            f"Customer input: {user_message}\n\n"
            "- Objective: Deliver clear stakeholder value.\n"
            "- Success Criteria: Measure impact using agreed KPIs.\n"
            "- Constraints: Respect budget and compliance limits."
        )
    if persona == "technical":
        return (
            "# Technical Solution Outline\n"
            f"Key request: {user_message}\n\n"
            "- Architecture: Propose modular microservices with shared auth.\n"
            "- Integrations: Connect to existing analytics platform via REST.\n"
            "- Risks: Validate performance under peak concurrency."
        )
    return (
        "# General Response\n"
        f"Prompt echoed: {user_message}\n\n"
        "This persona is not defined yet, but the placeholder keeps the UI\n"
        "functional during demos."
    )


class AIClient:
    """Convenience wrapper that routes prompts to configured AI providers."""

    def __init__(self, config: AppConfig):
        self.config = config

    def _resolve_provider(self) -> Tuple[str, str]:
        """Return the provider identifier and API key to use for requests."""

        preferred = self.config.default_provider.lower()
        api_key = self.config.get_api_key(preferred)
        if api_key:
            return preferred, api_key

        for name, credential in self.config.configured_providers().items():
            if credential.api_key:
                return name, credential.api_key

        raise RuntimeError(
            "No AI provider API keys are configured. Supply a key or enable demo mode."
        )

    @staticmethod
    def _extract_text(provider: str, payload: Dict[str, object]) -> str:
        """Normalise provider responses to a plain text string."""

        try:
            if provider in {"openai", "xai", "moonshot"}:
                return str(payload["choices"][0]["message"]["content"]).strip()
            if provider == "anthropic":
                return str(payload["content"][0]["text"]).strip()
            if provider == "google":
                return str(payload["candidates"][0]["content"]["parts"][0]["text"]).strip()
            if provider == "qwen":
                output = payload.get("output") or payload.get("data") or {}
                if isinstance(output, dict) and "text" in output:
                    return str(output["text"]).strip()
                if "result" in payload and isinstance(payload["result"], dict):
                    maybe_text = payload["result"].get("output_text")
                    if maybe_text:
                        return str(maybe_text).strip()
        except (IndexError, KeyError, TypeError):  # pragma: no cover - defensive
            logger.exception("Unexpected response schema from provider %s", provider)

        return str(payload)

    def generate_specification(
        self,
        *,
        prompt: str,
        persona: str = "general",
        user_message: Optional[str] = None,
    ) -> str:
        """Send ``prompt`` to a provider or return a deterministic demo response."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        if self.config.demo_mode:
            demo_persona = (
                "business"
                if persona == "requirements"
                else "technical" if persona == "technical" else persona
            )
            return mock_ai_response(demo_persona, user_message or prompt)

        provider, api_key = self._resolve_provider()
        payload = call_ai_provider(
            provider=provider,
            model=None,
            messages=[{"role": "user", "content": prompt.strip()}],
            api_key=api_key,
        )
        return self._extract_text(provider, payload)


__all__ = [
    "ProviderConfig",
    "PROVIDERS",
    "call_ai_provider",
    "extract_specifications_from_response",
    "get_provider_headers",
    "handle_streaming_response",
    "mock_ai_response",
    "AIClient",
]
