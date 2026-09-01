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
    # Optional OpenAI-COMPATIBLE base URL (Ollama, vLLM, LM Studio, RunPod, LocalAI...).
    # When set, reasoning is served by that endpoint instead of a hosted provider —
    # a self-hosted Llama needs no API key, so LLM_API_KEY may stay blank.
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "").rstrip("/")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    # Wall-clock ceiling for one reasoning call. Self-hosted GPUs cold-start slowly,
    # so this is generous; the caller always falls back to the deterministic engine.
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    # --- Named self-hosted runtimes (Ollama now, RunPod later) ------------------
    # Both endpoints are configured at once and selected by LLM_RUNTIME, so moving
    # between a laptop's Ollama and a rented GPU is one env var and no redeploy.
    #   LLM_RUNTIME: auto | ollama | runpod | api
    #   'auto' prefers runpod, then ollama, then the hosted provider — whichever is
    #   actually configured. Explicit values never silently fall through, because a
    #   benchmark that quietly ran on the wrong runtime is worse than one that fails.
    LLM_RUNTIME: str = os.getenv("LLM_RUNTIME", "auto").strip().lower()
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")
    RUNPOD_BASE_URL: str = os.getenv("RUNPOD_BASE_URL", "").rstrip("/")
    RUNPOD_MODEL: str = os.getenv("RUNPOD_MODEL", "")
    RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Accept either GOOGLE_API_KEY or GEMINI_API_KEY for the Google Gemini provider.
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    
    # --- Embedding engine (see services/ai/embedding_engines.py) ----------------
    # 'bge'  : local BAAI/bge-large-en-v1.5, in-process, 1024 dims. Default; offline.
    # 'gpu'  : a model served from an Ollama-compatible endpoint, higher fidelity.
    # Vectors from the two are NOT interchangeable, so each engine owns its own
    # corpus index. Selecting 'gpu' with an unreachable endpoint falls back to 'bge'
    # and says so — it never silently searches the wrong index.
    EMBEDDING_ENGINE: str = os.getenv("EMBEDDING_ENGINE", "bge")
    EMBEDDING_GPU_URL: str = os.getenv("EMBEDDING_GPU_URL", "").rstrip("/")
    EMBEDDING_GPU_MODEL: str = os.getenv("EMBEDDING_GPU_MODEL", "")
    # Must match the model's real output size, or every search silently misaligns.
    EMBEDDING_GPU_DIM: int = int(os.getenv("EMBEDDING_GPU_DIM", "768"))
    EMBEDDING_GPU_TIMEOUT_SECONDS: float = float(os.getenv("EMBEDDING_GPU_TIMEOUT_SECONDS", "60"))
    EMBEDDING_GPU_BATCH: int = int(os.getenv("EMBEDDING_GPU_BATCH", "32"))
    # Cosine floor for the GPU model. Every model has its own similarity distribution,
    # so this CANNOT be shared with RETRIEVAL_MIN_SIMILARITY. Measured on
    # nomic-embed-text: relevant pairs >= 0.592, unrelated <= 0.426; 0.51 sits in that
    # gap. Reusing BGE's 0.62 here silently discarded real matches.
    EMBEDDING_GPU_MIN_SIMILARITY: float = float(os.getenv("EMBEDDING_GPU_MIN_SIMILARITY", "0.51"))

    # --- mxbai-embed-large-v1 (local, 1024 dims) ---------------------------
    # Floor calibrated to EQUAL SELECTIVITY with BGE rather than picked by feel: 0.537
    # is where mxbai admits the same fraction of query-chunk pairs that BGE admits at
    # 0.62, measured over 279 chunks of this project's real resumes. Comparing models
    # on a shared threshold would reward whichever one scores higher in absolute terms,
    # which says nothing about ranking quality.
    EMBEDDING_MXBAI_MIN_SIMILARITY: float = float(os.getenv("EMBEDDING_MXBAI_MIN_SIMILARITY", "0.537"))
    # 8, not FastEmbed's default 256: a 335M-parameter model at 512 tokens asked ONNX
    # Runtime for a 1.15 GB activation buffer and died. This is a correctness setting,
    # and it costs nothing — measured on real resume chunks (~450 chars), throughput is
    # flat at ~1.9 chunks/s from batch 8 through 64, because the work is CPU-bound
    # rather than batching-bound. Budget roughly 45 minutes to index 600 resumes with
    # this model the first time; afterwards only new resumes are embedded.
    # (FastEmbed's `parallel=` multiprocessing would help, but it deadlocks on Windows
    # and is unsafe inside a request worker, so it is deliberately not used.)
    EMBEDDING_MXBAI_BATCH: int = int(os.getenv("EMBEDDING_MXBAI_BATCH", "8"))

    # --- Automatic indexing -------------------------------------------------
    # A resume becomes searchable by every available model as soon as it is uploaded,
    # on a background thread. This is what makes one store serve every screen: nothing
    # asks the recruiter to "index" on a second page, and no section re-embeds what
    # another section already embedded.
    #
    # Turn OFF only when you want to control embedding cost explicitly (e.g. loading a
    # few thousand CVs before a demo, then indexing once). The Model Lab indexing
    # controls remain available either way.
    AUTO_INDEX_ON_UPLOAD: bool = os.getenv("AUTO_INDEX_ON_UPLOAD", "true").lower() == "true"
    # One incremental pass at boot, so a store that predates a model catches up without
    # anyone pressing a button. Costs a fingerprint scan when everything is current.
    AUTO_INDEX_ON_STARTUP: bool = os.getenv("AUTO_INDEX_ON_STARTUP", "true").lower() == "true"
    # How many files an upload may contain before indexing stops being automatic.
    #
    # A handful of CVs should just work — waiting for someone to press a button is the
    # friction that made the pool look empty. A 300-file backlog is a different thing
    # entirely: mxbai embeds at ~1.9 chunks/second, so a full pass is around 40 minutes,
    # and which of those 300 are worth that cost is a decision only the recruiter can
    # make. Above this threshold the files are stored and left OUT of the working set,
    # for selection on the Documents screen.
    AUTO_INDEX_MAX_BATCH: int = int(os.getenv("AUTO_INDEX_MAX_BATCH", "20"))

    # --- Company intelligence module (services/company/*) ----------------------
    # A SEPARATE knowledge base: employer/company records in Qdrant Cloud, reached
    # only when the recruiter flips the chat into company mode. It shares nothing
    # with the candidate pool — different store, different collection, different
    # retrieval path — so nothing here can affect resume search.
    #
    # The collection's vector size is fixed at creation time to the GPU engine's
    # dimension (nomic-embed-text, 768). That is why this module never falls back to
    # the local 384-dim engine: a fallback would either be rejected by Qdrant or,
    # worse, silently search a 768-dim space with 384-dim meaning.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "").rstrip("/")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COMPANIES_COLLECTION: str = os.getenv("QDRANT_COMPANIES_COLLECTION", "companies")
    QDRANT_TIMEOUT_SECONDS: float = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30"))
    # Cosine floor for company retrieval. Held apart from the resume floors because it
    # is calibrated on a different corpus: company prose is short, dense and written in
    # marketing register, so its similarity distribution is not the resume one.
    COMPANY_MIN_SIMILARITY: float = float(os.getenv("COMPANY_MIN_SIMILARITY", "0.45"))
    COMPANY_TOP_K: int = int(os.getenv("COMPANY_TOP_K", "24"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'hiring_platform.db')}")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Upload folder
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "temp_uploads")
    
    # --- Match Score weights (must sum to 1.00) --------------------------------
    # The nine dimensions a candidate is scored on against a specific JD. This is
    # the headline number the platform reports, and the weights are the agreed
    # specification — change them only by agreement, because two people using
    # different weights cannot compare a candidate at all.
    #
    # Ordering reflects what actually decides a hire: what they can do (Skill 20 +
    # Technology 14 = 34%), how far along they are (Experience 15 + Designation 14
    # = 29%), then context (Industry 9, Education 8, Location 8, Availability 7),
    # with Freshness at 5% acting as a tie-breaker rather than a real signal.
    MATCH_WEIGHT_SKILL: float = 0.20
    MATCH_WEIGHT_EXPERIENCE: float = 0.15
    MATCH_WEIGHT_TECHNOLOGY: float = 0.14
    MATCH_WEIGHT_DESIGNATION: float = 0.14
    MATCH_WEIGHT_INDUSTRY: float = 0.09
    MATCH_WEIGHT_EDUCATION: float = 0.08
    MATCH_WEIGHT_LOCATION: float = 0.08
    MATCH_WEIGHT_AVAILABILITY: float = 0.07
    MATCH_WEIGHT_FRESHNESS: float = 0.05

    # Neutral scores, used when the JD or the resume is SILENT about a dimension.
    # A requirement nobody stated must not be able to fail a candidate, so silence
    # scores mid-range rather than zero. These are not defaults for a *failed*
    # comparison — a candidate in the wrong city still scores low, they just are
    # not treated the same as a candidate whose city is unknown.
    MATCH_NEUTRAL_SKILL: float = 50.0
    MATCH_NEUTRAL_FRESHNESS: float = 50.0
    MATCH_NEUTRAL_LOCATION: float = 50.0
    MATCH_NEUTRAL_AVAILABILITY: float = 50.0
    MATCH_NEUTRAL_INDUSTRY: float = 50.0
    MATCH_NEUTRAL_EDUCATION: float = 50.0
    MATCH_NEUTRAL_DESIGNATION: float = 50.0
    MATCH_NEUTRAL_EXPERIENCE: float = 50.0
    # The resume, not the JD, is silent about when the candidate can start. That is
    # a small negative signal rather than a neutral one: the employer asked and the
    # candidate did not say.
    MATCH_MISSING_AVAILABILITY: float = 40.0
    # A candidate in a different city is still reachable — relocation happens — so
    # the floor is 20, not 0.
    MATCH_LOCATION_DIFFERENT: float = 20.0

    # Penalty slopes.
    MATCH_EXPERIENCE_PENALTY_PER_YEAR: float = 15.0
    MATCH_AVAILABILITY_PENALTY_PER_WEEK: float = 10.0
    # Freshness decays from 100 at upload to this floor over MATCH_FRESHNESS_DAYS.
    MATCH_FRESHNESS_FLOOR: float = 40.0
    MATCH_FRESHNESS_DAYS: float = 365.0
    # Credit for a requirement matched only through a spelling mistake.
    MATCH_PARTIAL_SKILL_CREDIT: float = 0.6

    # Interpretation bands over the final Match Score. (label, inclusive minimum),
    # highest first — the first band the score reaches wins.
    MATCH_SCORE_BANDS = [
        ("Excellent fit", 85.0),
        ("Strong fit", 70.0),
        ("Moderate fit", 50.0),
        ("Weak fit", 0.0),
    ]

    # Compatibility score weights
    # Coverage is the primary signal (does the candidate have the skills?).
    # Experience measures narrative evidence depth (not just section presence).
    # Projects is reduced — many roles (sysadmin, ops) have no Projects section.
    # Quality (authenticity/keyword-stuffing) is given more influence.
    WEIGHT_COVERAGE: float = 0.40
    WEIGHT_EXPERIENCE: float = 0.25
    WEIGHT_PROJECTS: float = 0.10
    WEIGHT_CONFIDENCE: float = 0.15
    WEIGHT_QUALITY: float = 0.10

    # Retrieval ranking weights (Must sum to 1.0)
    RETRIEVAL_WEIGHT_SIMILARITY: float = 0.70
    RETRIEVAL_WEIGHT_SECTION: float = 0.15
    RETRIEVAL_WEIGHT_DENSITY: float = 0.05
    RETRIEVAL_WEIGHT_TECH_SPECIFICITY: float = 0.10

    # Sentence-aware chunking (resume_structuring_service). Chunks respect sentence
    # boundaries and target ~CHUNK_TARGET_CHARS, never exceeding CHUNK_MAX_CHARS; a
    # small sentence overlap preserves context across chunk edges for retrieval recall.
    CHUNK_TARGET_CHARS: int = int(os.getenv("CHUNK_TARGET_CHARS", "350"))
    CHUNK_MAX_CHARS: int = int(os.getenv("CHUNK_MAX_CHARS", "600"))
    CHUNK_SENTENCE_OVERLAP: int = int(os.getenv("CHUNK_SENTENCE_OVERLAP", "1"))

    # Hybrid retrieval: fuse dense (vector/cosine) and sparse (BM25 keyword) rankings
    # via Reciprocal Rank Fusion. Weights bias the fusion toward semantic vs lexical
    # matches; RRF_K damps the contribution of low-ranked items.
    HYBRID_RETRIEVAL_ENABLED: bool = os.getenv("HYBRID_RETRIEVAL_ENABLED", "true").lower() == "true"
    HYBRID_WEIGHT_DENSE: float = float(os.getenv("HYBRID_WEIGHT_DENSE", "0.6"))
    HYBRID_WEIGHT_SPARSE: float = float(os.getenv("HYBRID_WEIGHT_SPARSE", "0.4"))
    HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))

    # --- LLM-assisted retrieval (see services/ai/chat_llm_retrieval.py) ---------
    # A THIRD stage on top of dense+sparse: the model judges evidence and re-ranks
    # the shortlist. Both default OFF, so the deterministic funnel is unchanged
    # until a GPU is actually available — and remains the fallback on any failure.
    #
    # The model is never allowed to introduce a candidate or a passage. It only
    # judges text that deterministic retrieval already returned, so a wrong verdict
    # can reorder or demote a real candidate but can never invent one.
    RETRIEVAL_LLM_VERIFY_ENABLED: bool = os.getenv("RETRIEVAL_LLM_VERIFY_ENABLED", "false").lower() == "true"
    RETRIEVAL_LLM_RERANK_ENABLED: bool = os.getenv("RETRIEVAL_LLM_RERANK_ENABLED", "false").lower() == "true"
    # Only the top slice is re-ranked: recall is the bi-encoder's job, precision is
    # the model's, and paying for a judgement on candidate 300 buys nothing.
    RETRIEVAL_LLM_RERANK_TOP_K: int = int(os.getenv("RETRIEVAL_LLM_RERANK_TOP_K", "50"))
    # Share of the final score the model's relevance judgement carries. It enters as
    # one more weighted component alongside Skill/Technology/Location/Experience, so
    # it is renormalised with them and shows up in the recruiter's breakdown rather
    # than silently moving the number.
    MATCH_WEIGHT_LLM_RELEVANCE: float = float(os.getenv("MATCH_WEIGHT_LLM_RELEVANCE", "0.18"))
    # Calls run concurrently — the .NET portal shipped a sequential per-item loop and
    # it cost ~13 serial round trips per turn. Do not repeat that here.
    RETRIEVAL_LLM_CONCURRENCY: int = int(os.getenv("RETRIEVAL_LLM_CONCURRENCY", "8"))
    # Items per prompt. Batching cuts round trips; too large and small models start
    # dropping entries from the JSON array.
    RETRIEVAL_LLM_BATCH: int = int(os.getenv("RETRIEVAL_LLM_BATCH", "8"))
    # Hard wall-clock ceiling for the WHOLE assisted stage. When it expires the
    # deterministic ordering stands and the answer still goes out on time.
    RETRIEVAL_LLM_DEADLINE_SECONDS: float = float(os.getenv("RETRIEVAL_LLM_DEADLINE_SECONDS", "20"))
    RETRIEVAL_LLM_CACHE_SIZE: int = int(os.getenv("RETRIEVAL_LLM_CACHE_SIZE", "4096"))

    # Minimum raw cosine similarity (0..1) a retrieved chunk must clear to count as
    # evidence for a requirement. Matches below this are dropped, so a requirement
    # with no genuinely-relevant chunk yields zero matches and is correctly reported
    # as "Missing" rather than a weak "Partial" backed by an unrelated chunk.
    #
    # Calibrated for BGE-large: genuine same-domain matches score ~0.62+, whereas
    # loosely-related cross-domain tech (e.g. a React resume vs a "python" query)
    # sits ~0.56-0.61. A 0.30 floor let unrelated tech pass, so an off-domain resume
    # spuriously "matched" every requirement. 0.62 separates real evidence from noise.
    RETRIEVAL_MIN_SIMILARITY: float = float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.62"))

    # Dashboard analytics decision buckets (over overall_score 0-100).
    # Selected: score >= SELECTED_MIN; Rejected: score < BORDERLINE_MIN;
    # Borderline: everything in between. Trends window is DASHBOARD_TRENDS_DAYS days.
    #
    # Aligned to MATCH_SCORE_BANDS: Selected covers Strong and Excellent (>= 70),
    # Borderline is Moderate (>= 50), Rejected is Weak. These MUST track the bands —
    # a dashboard that calls a candidate "Rejected" while their report reads
    # "Strong fit" is reporting two different opinions as one fact.
    DASHBOARD_SELECTED_MIN: int = int(os.getenv("DASHBOARD_SELECTED_MIN", "70"))
    DASHBOARD_BORDERLINE_MIN: int = int(os.getenv("DASHBOARD_BORDERLINE_MIN", "50"))
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

    # Skill relationship reasoning (evaluation_service transfer/equivalence logic).
    # Calibrated on BGE-large skill-name cosine: near-synonyms (SQL/MySQL/PostgreSQL)
    # sit ~0.81-0.84, adjacent-but-distinct skills (Docker/Kubernetes 0.70,
    # PyTorch/TensorFlow 0.72, Python/Django 0.73) sit ~0.68-0.75.
    #   >= EQUIVALENCE : candidate effectively already has the skill (rescue a Missing
    #                    requirement — e.g. MySQL satisfied by SQL). Raw cosine only.
    #   >= RELATED     : adjacent skill — note the transfer, but never claim the skill.
    SKILL_EQUIVALENCE_MIN: float = float(os.getenv("SKILL_EQUIVALENCE_MIN", "0.80"))
    SKILL_RELATED_MIN: float = float(os.getenv("SKILL_RELATED_MIN", "0.68"))

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
