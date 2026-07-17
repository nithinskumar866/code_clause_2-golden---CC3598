"""
Tests for the provider-agnostic LLM factory. These are config-resolution tests —
no network calls, no real model instantiation.
"""
from app.core.config import settings
from app.services.ai import llm_service


def test_provider_aliases_canonicalize(monkeypatch):
    for raw, canonical in [("gemini", "google"), ("claude", "anthropic"),
                            ("OpenAI", "openai"), ("google_genai", "google")]:
        monkeypatch.setattr(settings, "LLM_PROVIDER", raw)
        assert llm_service.resolve_provider() == canonical


def test_model_default_then_override(monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODEL", "")
    assert llm_service.resolve_model("google") == "gemini-1.5-pro"
    assert llm_service.resolve_model("anthropic") == "claude-sonnet-5"
    assert llm_service.resolve_model("openai") == "gpt-4o-mini"
    # LLM_MODEL always wins, regardless of provider — pure config, no code change.
    monkeypatch.setattr(settings, "LLM_MODEL", "gemini-2.0-flash")
    assert llm_service.resolve_model("google") == "gemini-2.0-flash"


def test_get_llm_returns_none_without_key(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "google")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    assert llm_service.get_llm() is None


def test_get_llm_returns_none_for_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "some_unsupported_llm")
    assert llm_service.get_llm() is None
