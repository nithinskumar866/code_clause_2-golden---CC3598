import os
from dotenv import load_dotenv

# Load .env file
# Root directory of the backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Settings:
    PROJECT_NAME: str = "AI Hiring Intelligence Platform"
    API_V1_STR: str = "/api/v1"
    
    # LLM configurations
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
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

    # Skill-semantics reasoning (category classification + transferability)
    # Minimum centroid cosine similarity to accept a category; below this a
    # requirement is treated as an unknown/general skill.
    CATEGORY_MIN_SIMILARITY: float = 0.50
    # Bonus added to a transfer score when the missing skill and the candidate's
    # related skill fall in the same category (reflects strong conceptual overlap).
    TRANSFER_SAME_CATEGORY_BOOST: float = 0.10

settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
