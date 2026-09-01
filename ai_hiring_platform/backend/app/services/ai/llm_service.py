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
# google default is a fast, low-cost, strong-reasoning model; set LLM_MODEL to
# 'gemini-1.5-pro' / 'gemini-2.5-pro' for maximum quality, or to whatever your key
# supports. A model your key cannot serve simply falls back to the deterministic
# engine at call time (get_llm never raises for a bad model id).
_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "google": "gemini-2.0-flash",
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


class _CompletionResponse:
    """Minimal stand-in for a llama-index CompletionResponse (`.text`)."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.text


class OpenAICompatibleLLM:
    """
    Client for any OpenAI-compatible `/chat/completions` endpoint — Ollama, vLLM,
    LM Studio, LocalAI, a RunPod proxy, or OpenAI itself.

    Exposes the same `.complete(prompt) -> obj.text` surface as the llama-index LLMs,
    so reasoning code never learns where the model is hosted. Implemented directly on
    httpx (already a dependency) rather than through a provider integration, because
    self-hosted model ids (`llama3.1:8b`) are not in any vendor's model registry and
    would otherwise trip context-window lookups.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _endpoint(self) -> str:
        base = self.base_url
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2) -> _CompletionResponse:
        import httpx

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = httpx.post(
            self._endpoint,
            headers=headers,
            json={"model": self.model, "messages": messages, "temperature": temperature},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        try:
            return _CompletionResponse(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected completion payload from {self._endpoint}: {payload}") from e

    def health(self) -> bool:
        """True when the endpoint answers a trivial prompt — used by /chat/health."""
        try:
            return bool(self.complete("ping", temperature=0.0).text)
        except Exception as e:
            logger.warning(f"LLM health check failed for {self._endpoint}: {e}")
            return False


def _named_runtime() -> Optional[OpenAICompatibleLLM]:
    """
    The Ollama / RunPod endpoint selected by `LLM_RUNTIME`, if one is configured.

    Both are configured at the same time and chosen by name, because the GPU box and
    the local daemon serve the same OpenAI-compatible API and the only real question
    is which one to point at today. `auto` prefers RunPod (the faster box) and falls
    back to Ollama; an EXPLICIT choice never falls through to the other, since a
    benchmark that quietly ran on the wrong runtime is worse than one that fails.
    """
    runtime = (settings.LLM_RUNTIME or "auto").strip().lower()
    if runtime == "api":
        return None

    def runpod() -> Optional[OpenAICompatibleLLM]:
        if not (settings.RUNPOD_BASE_URL and settings.RUNPOD_MODEL):
            return None
        return OpenAICompatibleLLM(
            base_url=settings.RUNPOD_BASE_URL, model=settings.RUNPOD_MODEL,
            api_key=settings.RUNPOD_API_KEY, timeout=settings.LLM_TIMEOUT_SECONDS,
        )

    def ollama() -> Optional[OpenAICompatibleLLM]:
        if not (settings.OLLAMA_BASE_URL and settings.OLLAMA_MODEL):
            return None
        return OpenAICompatibleLLM(
            base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL,
            api_key="", timeout=settings.LLM_TIMEOUT_SECONDS,
        )

    if runtime == "runpod":
        llm = runpod()
        if llm is None:
            logger.warning("LLM_RUNTIME=runpod but RUNPOD_BASE_URL/RUNPOD_MODEL are unset.")
        return llm
    if runtime == "ollama":
        llm = ollama()
        if llm is None:
            logger.warning("LLM_RUNTIME=ollama but OLLAMA_MODEL is unset.")
        return llm

    return runpod() or ollama()


def get_llm() -> Optional[object]:
    """
    Build the configured LLM, or return None to signal the deterministic fallback.

    Resolution order: a named runtime (`LLM_RUNTIME` -> Ollama/RunPod), then an
    explicit `LLM_BASE_URL`, then a hosted provider. Self-hosted endpoints commonly
    need no API key, which is why they are selected on URL+model rather than on a key.
    """
    named = _named_runtime()
    if named is not None:
        logger.info(f"LLM ready: runtime='{settings.LLM_RUNTIME}', "
                    f"endpoint='{named.base_url}', model='{named.model}'.")
        return named

    if settings.LLM_BASE_URL:
        model = (settings.LLM_MODEL or "").strip()
        if not model:
            logger.warning("LLM_BASE_URL is set but LLM_MODEL is blank. Using deterministic engine.")
            return None
        logger.info(f"LLM ready: OpenAI-compatible endpoint '{settings.LLM_BASE_URL}', model='{model}'.")
        return OpenAICompatibleLLM(
            base_url=settings.LLM_BASE_URL,
            model=model,
            api_key=settings.LLM_API_KEY,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

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
