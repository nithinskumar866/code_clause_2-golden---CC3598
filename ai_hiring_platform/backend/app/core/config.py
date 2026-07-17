import os
from dotenv import load_dotenv

# Load .env file
# Root directory of the backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Settings:
    PROJECT_NAME: str = "AI Hiring Intelligence Platform"
    API_V1_STR: str = "/api/v1"
    
    # LLM configurations — provider/model-agnostic. Switching provider or model is a
    # config change only (never code). See services/ai/llm_service.py.
    #   LLM_PROVIDER: openai | anthropic | google (aliases: gemini, claude)
    #   LLM_MODEL:    explicit model id; blank -> a sensible per-provider default
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Accept either GOOGLE_API_KEY or GEMINI_API_KEY for the Google Gemini provider.
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'hiring_platform.db')}")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Upload folder
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "temp_uploads")
    
    # Compatibility score weights
    WEIGHT_COVERAGE: float = 0.35
    WEIGHT_EXPERIENCE: float = 0.25
    WEIGHT_PROJECTS: float = 0.20
    WEIGHT_CONFIDENCE: float = 0.15
    WEIGHT_QUALITY: float = 0.05

    # Retrieval ranking weights (Must sum to 1.0)
    RETRIEVAL_WEIGHT_SIMILARITY: float = 0.70
    RETRIEVAL_WEIGHT_SECTION: float = 0.15
    RETRIEVAL_WEIGHT_DENSITY: float = 0.05
    RETRIEVAL_WEIGHT_TECH_SPECIFICITY: float = 0.10

    # Minimum raw cosine similarity (0..1) a retrieved chunk must clear to count as
    # evidence for a requirement. Matches below this are dropped, so a requirement
    # with no genuinely-relevant chunk yields zero matches and is correctly reported
    # as "Missing" rather than a weak "Partial" backed by an unrelated chunk.
    RETRIEVAL_MIN_SIMILARITY: float = float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.30"))

    # Dashboard analytics decision buckets (over overall_score 0-100).
    # Selected: score >= SELECTED_MIN; Rejected: score < BORDERLINE_MIN;
    # Borderline: everything in between. Trends window is DASHBOARD_TRENDS_DAYS days.
    DASHBOARD_SELECTED_MIN: int = int(os.getenv("DASHBOARD_SELECTED_MIN", "80"))
    DASHBOARD_BORDERLINE_MIN: int = int(os.getenv("DASHBOARD_BORDERLINE_MIN", "60"))
    DASHBOARD_TRENDS_DAYS: int = int(os.getenv("DASHBOARD_TRENDS_DAYS", "30"))

    # Analytics windows/limits and optional result cache.
    ANALYTICS_DAILY_DAYS: int = int(os.getenv("ANALYTICS_DAILY_DAYS", "30"))
    ANALYTICS_WEEKLY_WEEKS: int = int(os.getenv("ANALYTICS_WEEKLY_WEEKS", "12"))
    ANALYTICS_MONTHLY_MONTHS: int = int(os.getenv("ANALYTICS_MONTHLY_MONTHS", "12"))
    ANALYTICS_TOP_LIMIT: int = int(os.getenv("ANALYTICS_TOP_LIMIT", "5"))
    ANALYTICS_RECENT_LIMIT: int = int(os.getenv("ANALYTICS_RECENT_LIMIT", "10"))
    # Seconds to cache analytics aggregates across requests. 0 disables caching
    # (default) so results are always fresh; set >0 in single-DB deployments.
    ANALYTICS_CACHE_TTL_SECONDS: int = int(os.getenv("ANALYTICS_CACHE_TTL_SECONDS", "0"))

    # Skill-semantics reasoning (category classification + transferability)
    # Minimum centroid cosine similarity to accept a category; below this a
    # requirement is treated as an unknown/general skill.
    CATEGORY_MIN_SIMILARITY: float = 0.50
    # Bonus added to a transfer score when the missing skill and the candidate's
    # related skill fall in the same category (reflects strong conceptual overlap).
    TRANSFER_SAME_CATEGORY_BOOST: float = 0.10

    # --- Authenticity / keyword-stuffing detection (deterministic) ---
    # A claimed skill is "over-claimed" when it appears only in listing sections
    # (Skills/Summary) with no supporting Experience/Project evidence. Risk bands
    # are keyed on the fraction of claimed skills that are over-claimed.
    STUFFING_MEDIUM_FRACTION: float = float(os.getenv("STUFFING_MEDIUM_FRACTION", "0.25"))
    STUFFING_HIGH_FRACTION: float = float(os.getenv("STUFFING_HIGH_FRACTION", "0.50"))
    # Resume quality_score blend: how well claims are substantiated (corroboration)
    # vs. how detailed the demonstrating evidence is (depth). Must sum to 1.0.
    QUALITY_WEIGHT_CORROBORATION: float = 0.60
    QUALITY_WEIGHT_DEPTH: float = 0.40

    # --- Requirement prioritization (must-have vs nice-to-have) ---
    # Coverage scoring weights each requirement by how essential the JD makes it,
    # so missing a must-have costs far more than missing a nice-to-have.
    REQUIREMENT_WEIGHT_MUST: float = 1.0
    REQUIREMENT_WEIGHT_NICE: float = float(os.getenv("REQUIREMENT_WEIGHT_NICE", "0.4"))

settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
