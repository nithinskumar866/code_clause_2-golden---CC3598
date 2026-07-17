"""
Provider-agnostic LLM factory.

The reasoning engine must never be tied to a specific model or vendor: switching
provider or model is a configuration change (env vars), not a code change. All LLM
instantiation flows through `get_llm()`, which returns a llama-index LLM exposing a
uniform interface (`.complete(...)`), so reasoning code never branches on provider.

Config (app/core/config.py, from env):
    LLM_PROVIDER : openai | anthropic | google   (aliases: gemini->google, claude->anthropic)
    LLM_MODEL    : explicit model id; blank -> per-provider default below
    <PROVIDER>_API_KEY : OPENAI_API_KEY | ANTHROPIC_API_KEY | GOOGLE_API_KEY (or GEMINI_API_KEY)

Returns None when no provider/key is configured (or on init failure) so the caller
falls back to the deterministic engine — the platform always works without an LLM.
Integration packages are imported lazily per provider, so an uninstalled integration
for an unused provider never breaks import.
"""
from typing import Optional
from app.core.config import settings
from app.core.logging import logger

# Canonicalize user-facing provider names / aliases.
_ALIASES = {
    "openai": "openai",
    "anthropic": "anthropic", "claude": "anthropic",
    "google": "google", "gemini": "google", "googlegenai": "google", "google_genai": "google",
}

# Sensible defaults when LLM_MODEL is blank. Overridable purely via config.
_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "google": "gemini-1.5-pro",
}

_API_KEY = {
    "openai": lambda: settings.OPENAI_API_KEY,
    "anthropic": lambda: settings.ANTHROPIC_API_KEY,
    "google": lambda: settings.GOOGLE_API_KEY,
}


def resolve_provider() -> str:
    """Canonical provider name from config (empty string if unrecognized)."""
    raw = (settings.LLM_PROVIDER or "").strip().lower()
    return _ALIASES.get(raw, raw)


def resolve_model(provider: str) -> str:
    """Explicit LLM_MODEL if set, else the per-provider default."""
    return (settings.LLM_MODEL or "").strip() or _DEFAULT_MODEL.get(provider, "")


def get_llm() -> Optional[object]:
    """
    Build the configured LLM, or return None to signal the deterministic fallback.
    """
    provider = resolve_provider()
    if provider not in _API_KEY:
        logger.warning(f"Unknown or unset LLM_PROVIDER '{settings.LLM_PROVIDER}'. Using deterministic engine.")
        return None

    api_key = _API_KEY[provider]()
    if not api_key:
        logger.warning(f"No API key configured for provider '{provider}'. Using deterministic engine.")
        return None

    model = resolve_model(provider)
    try:
        if provider == "openai":
            from llama_index.llms.openai import OpenAI
            llm = OpenAI(model=model, api_key=api_key)
        elif provider == "anthropic":
            from llama_index.llms.anthropic import Anthropic
            llm = Anthropic(model=model, api_key=api_key)
        elif provider == "google":
            from llama_index.llms.google_genai import GoogleGenAI
            llm = GoogleGenAI(model=model, api_key=api_key)
        else:  # unreachable (guarded above), kept for clarity
            return None
        logger.info(f"LLM ready: provider='{provider}', model='{model}'.")
        return llm
    except Exception as e:
        logger.error(
            f"Failed to initialize LLM (provider='{provider}', model='{model}'): {e}. "
            f"Using deterministic engine.",
            exc_info=True,
        )
        return None
