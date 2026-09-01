"""
Configuration for company_intel.

One settings object, read once from the environment at import time. Every threshold,
limit and cadence in the system is here rather than inline, so tuning the crawler or
the retrieval floor is an .env edit and not a code change.

Qdrant is the only datastore. There is deliberately no second-database setting to
reach for when something feels awkward to model as points and payloads.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_file() -> str:
    """The .env next to the project root, whatever directory uvicorn was started in."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    # --- Qdrant ------------------------------------------------------------
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_TIMEOUT_SECONDS: float = 30.0
    QDRANT_COMPANIES_COLLECTION: str = "ci_companies"
    QDRANT_SOURCES_COLLECTION: str = "ci_sources"
    QDRANT_CONTENT_COLLECTION: str = "ci_content"

    # --- Embeddings --------------------------------------------------------
    # 'fastembed' runs locally with no key; 'ollama' calls a remote GPU endpoint.
    # NOT interchangeable at runtime: they emit different dimensions, so switching
    # provider means rebuilding the content collection and re-crawling.
    EMBED_PROVIDER: str = "fastembed"
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"   # used when provider is fastembed
    EMBED_OLLAMA_URL: str = ""
    EMBED_OLLAMA_MODEL: str = "nomic-embed-text"
    EMBED_DIM: int = 384
    # 16 measured fastest against RunPod: the round trip dominates, so bigger batches
    # win until the request itself gets slow.
    EMBED_BATCH: int = 16
    EMBED_TIMEOUT_SECONDS: float = 120.0
    EMBED_MAX_RETRIES: int = 3
    # The retrieval floor is PER MODEL, and leaving it blank is the safe choice: the
    # default below is calibrated for whichever model is configured.
    #
    # Sharing one floor across models is a real trap. Measured on the same live corpus,
    # off-topic questions top out at 0.43 with bge-small but 0.55 with nomic, because
    # the two models simply use different parts of the cosine range. A floor of 0.55 is
    # correct for bge and lets "who won the 2018 world cup" (0.553) through on nomic.
    EMBED_MIN_SIMILARITY: float = 0.0  # 0 means "use the calibrated default"

    # --- Crawler -----------------------------------------------------------
    CRAWL_USER_AGENT: str = "company-intel-bot/0.1"
    CRAWL_TIMEOUT_SECONDS: float = 15.0
    CRAWL_MAX_PAGES: int = 60
    CRAWL_MAX_DEPTH: int = 3
    CRAWL_CONCURRENCY_PER_HOST: int = 2
    CRAWL_DEFAULT_DELAY_SECONDS: float = 1.0
    CRAWL_MAX_RETRIES: int = 3
    CRAWL_MAX_BYTES: int = 3_000_000
    CRAWL_MAX_FAILURES: int = 5
    CRAWL_RESPECT_ROBOTS: bool = True

    # --- Rendering (headless browser) --------------------------------------
    # Opt-in PER COMPANY (`render: true` in the registry), never global: rendering costs
    # a browser process and seconds per page, and forfeits conditional GET, so a site
    # that serves real HTML must not pay for it.
    BROWSER_TIMEOUT_SECONDS: float = 30.0
    # 'domcontentloaded' | 'load' | 'networkidle'. networkidle is the most reliable for
    # single-page apps and the slowest; it is the right default for the only sites that
    # need rendering at all.
    BROWSER_WAIT_UNTIL: str = "networkidle"
    # A settle after the network quiets — frameworks often paint one tick later.
    BROWSER_SETTLE_MS: int = 1200
    # Lazy content loads on scroll, so a browser that never scrolls reads only the hero
    # section. Bounded: a company page is not an infinite feed.
    BROWSER_SCROLL_STEPS: int = 10
    BROWSER_SCROLL_PAUSE_MS: int = 300
    BROWSER_SCROLL_BUDGET_SECONDS: float = 6.0
    # How long to WAIT for network quiet after the DOM is ready. Bounded, because
    # analytics and chat widgets poll indefinitely and the page is usually ready long
    # before they stop.
    BROWSER_IDLE_MS: int = 4000
    # Below this, a rendered page is treated as a failed render and the plain fetch is
    # used instead — rendering must never return less than not rendering.
    BROWSER_MIN_TEXT_CHARS: int = 400
    # When this share of a crawl's pages turn out to be duplicates of one another, the
    # site is almost certainly serving one shell for every path. Reported, and used to
    # suggest rendering.
    SPA_DUPLICATE_RATIO: float = 0.5

    # --- Chunking ----------------------------------------------------------
    # Character budgets rather than token counts, to keep chunking free of a tokenizer
    # dependency that would have to stay in sync with the embedding model.
    #
    # The target MUST stay below the model's input limit. At 2400 chars a measured 14%
    # of chunks reached BGE's 512-token ceiling, so their tails were silently dropped
    # from the vector while the full text was still stored and cited — the chunk claimed
    # to contain text the vector had never seen. ~4.6 chars per token here, so 512
    # tokens is roughly 2350 chars; 1800 leaves real headroom.
    CHUNK_TARGET_CHARS: int = 1800
    CHUNK_OVERLAP_CHARS: int = 240
    CHUNK_MIN_CHARS: int = 200

    # --- Refresh cadence ---------------------------------------------------
    REFRESH_DAYS_NEWS: int = 2
    REFRESH_DAYS_PRODUCTS: int = 14
    REFRESH_DAYS_ABOUT: int = 30
    REFRESH_DAYS_DEFAULT: int = 30
    REFRESH_BATCH: int = 50

    # --- Answering ---------------------------------------------------------
    # 'none' quotes the sources directly; 'ollama' and 'openai' phrase them. Unlike the
    # embedding provider this one IS a safe fallback — a model that cannot be reached
    # costs prose, not correctness, because the facts come from retrieval either way.
    ANSWER_PROVIDER: str = "none"
    ANSWER_OLLAMA_URL: str = ""
    ANSWER_MODEL: str = "llama3.1:8b"
    ANSWER_TIMEOUT_SECONDS: float = 180.0
    OPENAI_API_KEY: str = ""
    ANSWER_MAX_CONTEXT_CHUNKS: int = 8

    # --- Derived -----------------------------------------------------------
    @property
    def qdrant_configured(self) -> bool:
        return bool(self.QDRANT_URL)

    @property
    def llm_configured(self) -> bool:
        provider = (self.ANSWER_PROVIDER or "none").strip().lower()
        if provider == "ollama":
            return bool(self.ANSWER_OLLAMA_URL)
        if provider == "openai":
            return bool(self.OPENAI_API_KEY)
        return False

    @property
    def embed_provider(self) -> str:
        return (self.EMBED_PROVIDER or "fastembed").strip().lower()

    # Measured on a live corpus of Indian IT company sites: the value is the midpoint
    # between the lowest-scoring genuine answer and the highest-scoring off-topic one.
    #   bge-small : on-topic >= 0.62, off-topic <= 0.43  -> 0.55
    #   nomic     : on-topic >= 0.689, off-topic <= 0.553 -> 0.62
    _CALIBRATED_FLOORS = {"bge": 0.55, "nomic": 0.62}

    @property
    def min_similarity(self) -> float:
        """The retrieval floor for the configured model."""
        if self.EMBED_MIN_SIMILARITY > 0:
            return float(self.EMBED_MIN_SIMILARITY)
        name = (
            self.EMBED_OLLAMA_MODEL if self.embed_provider == "ollama" else self.EMBED_MODEL
        ).lower()
        for family, floor in self._CALIBRATED_FLOORS.items():
            if family in name:
                return floor
        # An unrecognised model gets the stricter of the calibrated floors rather than a
        # guess: refusing a real question is recoverable, answering nonsense is not.
        return max(self._CALIBRATED_FLOORS.values())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
