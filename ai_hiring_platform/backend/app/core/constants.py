import os

# Root directory of storage relative to the backend workspace
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
STORAGE_DIR = os.path.join(PROJECT_DIR, "storage")

# Upload subdirectories
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
RESUME_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "resumes")
JOB_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "jobs")

# Processing subdirectories
VECTOR_STORE_DIR = os.path.join(STORAGE_DIR, "vectors")
REPORT_DIR = os.path.join(STORAGE_DIR, "reports")
EXPORT_DIR = os.path.join(STORAGE_DIR, "exports")

# Ensure all storage paths exist
for path in [STORAGE_DIR, UPLOAD_DIR, RESUME_UPLOAD_DIR, JOB_UPLOAD_DIR, VECTOR_STORE_DIR, REPORT_DIR, EXPORT_DIR]:
    os.makedirs(path, exist_ok=True)

# Allowed file specifications
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}

# AI configurations
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Entity Statuses
STATUS_UPLOADED = "Uploaded"
STATUS_INDEXED = "Indexed"
STATUS_ANALYSED = "Analysed"
STATUS_FAILED = "Failed"

# Predefined Software Engineering Keyword Taxonomy
# For extracting job requirements from Job Descriptions in jd_requirement_extractor.py
TECH_TAXONOMY = {
    "Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang", "rust", "ruby", "php", "swift", "kotlin", "scala"
    ],
    "Frontend": [
        "react", "angular", "vue", "next.js", "nextjs", "nuxt", "svelte", "tailwind", "css", "html", "bootstrap", "sass", "webpack", "vite"
    ],
    "Backend": [
        "fastapi", "django", "flask", "express", "node.js", "nodejs", "spring boot", "laravel", "rails", "asp.net", "nest.js", "nestjs"
    ],
    "Databases": [
        "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis", "cassandra", "dynamodb", "elasticsearch", "neo4j", "mariadb"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "git", "github", "gitlab", "ci/cd"
    ],
    "AI & Data Science": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "numpy", "pandas", "scipy", "llm", "llama", "rag", "langchain", "llamaindex", "huggingface", "bert", "openai", "claude", "nlp", "computer vision", "opencv"
    ],
    "Methodologies & Practices": [
        "agile", "scrum", "tdd", "unit testing", "rest api", "graphql", "microservices", "system design"
    ]
}

# Human-friendly display labels for each TECH_TAXONOMY category.
# The Hiring Decision Agent classifies every requirement (including skills not
# present in the taxonomy) into the nearest category centroid, then presents it
# using these labels. Keys MUST match TECH_TAXONOMY keys.
CATEGORY_DISPLAY_NAMES = {
    "Languages": "Programming Language",
    "Frontend": "Frontend",
    "Backend": "Backend Framework",
    "Databases": "Database",
    "Cloud & DevOps": "Cloud & DevOps",
    "AI & Data Science": "AI & Machine Learning",
    "Methodologies & Practices": "Methodology & Practice",
}

# Category assigned when a requirement is not confidently close to any centroid.
FALLBACK_CATEGORY = "General Technical Skill"

# Learning-effort bands for missing skills, selected by transferability score.
# Ordered strongest transfer first; the first band whose threshold is met wins.
# Each entry: (min_transfer_score, estimated_time, strength_label)
LEARNING_TRANSFER_BANDS = [
    (0.78, "5-7 days", "strong"),
    (0.62, "10-14 days", "moderate"),
    (0.0, "21-28 days", "weak"),
]

# Learning effort for a partially-evidenced skill (listed but not demonstrated).
PARTIAL_SKILL_LEARNING_TIME = "2-4 days"
