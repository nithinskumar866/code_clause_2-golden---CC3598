using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace JobPortal.Api.Data;

// ---------------------------------------------------------------------------
// Tables OWNED by the Python platform. Mapped read-only so the portal can see
// the JDs and resumes the recruiter side already holds. Nothing here is ever
// written or migrated from .NET — the Python backend owns their shape, and a
// schema change made from this side would break its pytest suite.
// ---------------------------------------------------------------------------

[Table("job_descriptions")]
public class PythonJobDescription
{
    [Column("id")] public int Id { get; set; }
    [Column("filename")] public string Filename { get; set; } = "";
    [Column("upload_time")] public DateTime UploadTime { get; set; }
    [Column("status")] public string Status { get; set; } = "";
}

[Table("resumes")]
public class PythonResume
{
    [Column("id")] public int Id { get; set; }
    [Column("filename")] public string Filename { get; set; } = "";
    [Column("upload_time")] public DateTime UploadTime { get; set; }
    [Column("status")] public string Status { get; set; } = "";
    [Column("content_hash")] public string? ContentHash { get; set; }
}

// ---------------------------------------------------------------------------
// Tables OWNED by this portal. All prefixed `portal_` and purely additive, so
// they can live inside the same SQLite file without colliding with anything
// SQLAlchemy declares. This is the whole of "sharing the Python DB": one file,
// two owners, disjoint table sets.
// ---------------------------------------------------------------------------

/// <summary>
/// A job posting as the portal understands it: the structured content extracted
/// from an uploaded JD document (or typed in directly), ready to be embedded,
/// searched and shown on the board.
/// </summary>
[Table("portal_jobs")]
public class PortalJob
{
    public int Id { get; set; }

    /// <summary>
    /// The `job_descriptions.id` this posting was extracted from, when it came
    /// from a document the Python side already stored. Null for postings created
    /// directly through the portal, which is why it is not a foreign key —
    /// the portal must keep working if the recruiter platform's rows are pruned.
    /// </summary>
    public int? SourceJobDescriptionId { get; set; }

    /// <summary>Path on disk of the uploaded document, when there was one.</summary>
    public string? SourceFilePath { get; set; }

    [MaxLength(300)] public string Title { get; set; } = "";
    [MaxLength(200)] public string Company { get; set; } = "";
    [MaxLength(200)] public string Location { get; set; } = "";

    /// <summary>Remote | Hybrid | Onsite | Unspecified. Free text, not an enum,
    /// because it is extracted from prose and must survive whatever the JD says.</summary>
    [MaxLength(40)] public string WorkMode { get; set; } = "Unspecified";

    [MaxLength(60)] public string EmploymentType { get; set; } = "Unspecified";
    [MaxLength(80)] public string SeniorityLevel { get; set; } = "Unspecified";

    public double? MinYearsExperience { get; set; }
    public double? MaxYearsExperience { get; set; }

    public decimal? SalaryMin { get; set; }
    public decimal? SalaryMax { get; set; }
    [MaxLength(10)] public string? SalaryCurrency { get; set; }

    /// <summary>Newline-separated. Kept as text rather than a child table: these
    /// are extracted phrases with no identity of their own, and every consumer
    /// wants the whole list at once.</summary>
    public string RequiredSkills { get; set; } = "";
    public string PreferredSkills { get; set; } = "";
    public string Responsibilities { get; set; } = "";
    public string Qualifications { get; set; } = "";

    public string Summary { get; set; } = "";

    /// <summary>Full extracted document text. The source of truth every derived
    /// field can be re-derived from, so a better extractor can be re-run later
    /// without asking the recruiter to re-upload.</summary>
    public string RawText { get; set; } = "";

    /// <summary>The exact text that was embedded. Stored so a vector can always
    /// be explained, and so a change to how the embedding text is composed is
    /// detectable rather than silent.</summary>
    public string EmbeddedText { get; set; } = "";

    [MaxLength(40)] public string ExtractionMode { get; set; } = "deterministic";

    public string? ApplyUrl { get; set; }

    public bool IsPublished { get; set; } = true;

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    public List<PortalJobVector> Vectors { get; set; } = new();
}

/// <summary>
/// One job's embedding under one model. Keyed by model AND dimension because
/// vectors from different models are not comparable — mixing them produces a
/// cosine that is arithmetically valid and semantically meaningless.
/// </summary>
[Table("portal_job_vectors")]
public class PortalJobVector
{
    public int Id { get; set; }

    public int JobId { get; set; }
    public PortalJob? Job { get; set; }

    [MaxLength(120)] public string Model { get; set; } = "";
    public int Dimension { get; set; }

    /// <summary>L2-normalised float32 vector, little-endian. Normalised at write
    /// time so search is a dot product rather than a per-query renormalisation.</summary>
    public byte[] Vector { get; set; } = Array.Empty<byte>();

    /// <summary>SHA-256 of the embedded text. Lets re-indexing skip jobs whose
    /// text has not changed, which is what keeps a re-index cheap.</summary>
    [MaxLength(64)] public string TextHash { get; set; } = "";

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// A resume uploaded by a candidate through the portal chat. Distinct from the
/// Python `resumes` table on purpose: those are recruiter-sourced CVs in the
/// screening pool, these are job-seekers who walked in the front door. Merging
/// the two would silently put every job applicant into the recruiter's pool.
/// </summary>
[Table("portal_resumes")]
public class PortalResume
{
    public int Id { get; set; }

    [MaxLength(300)] public string Filename { get; set; } = "";
    public string StoredPath { get; set; } = "";
    [MaxLength(64)] public string ContentHash { get; set; } = "";

    public string RawText { get; set; } = "";

    [MaxLength(200)] public string CandidateName { get; set; } = "";
    [MaxLength(200)] public string Email { get; set; } = "";
    [MaxLength(60)] public string Phone { get; set; } = "";
    [MaxLength(200)] public string Location { get; set; } = "";
    [MaxLength(300)] public string CurrentTitle { get; set; } = "";

    public double? YearsExperience { get; set; }

    /// <summary>Newline-separated, as with the job side.</summary>
    public string Skills { get; set; } = "";
    public string Titles { get; set; } = "";
    public string Education { get; set; } = "";

    public string Summary { get; set; } = "";

    [MaxLength(40)] public string ExtractionMode { get; set; } = "deterministic";

    [MaxLength(120)] public string EmbeddingModel { get; set; } = "";
    public int EmbeddingDimension { get; set; }
    public byte[]? Embedding { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// One candidate's conversation. Holds the resume under discussion and the
/// filters the conversation has narrowed to, so a follow-up like "remote only"
/// refines the previous result instead of starting a new search.
/// </summary>
[Table("portal_chat_sessions")]
public class PortalChatSession
{
    [MaxLength(64)] public string Id { get; set; } = "";

    public int? ResumeId { get; set; }
    public PortalResume? Resume { get; set; }

    /// <summary>JSON-serialised <c>JobFilters</c>. Stored as a blob of JSON
    /// rather than columns because the filter set is expected to grow, and a
    /// migration per new filter would be pure friction for zero benefit.</summary>
    public string FiltersJson { get; set; } = "{}";

    /// <summary>JSON-serialised list of the last match result. An "explain"
    /// follow-up is answered from this, never by re-running retrieval — re-running
    /// would let the same question produce two different answers.</summary>
    public string LastMatchesJson { get; set; } = "[]";

    /// <summary>
    /// The posting the conversation is currently about.
    ///
    /// Without it "is it remote?" and "what do they pay?" have no subject, and the
    /// deterministic router had no choice but to treat them as a fresh search — which
    /// is why a follow-up used to silently answer a different question.
    /// </summary>
    public int? FocusJobId { get; set; }

    /// <summary>
    /// The fact package the last answer was built from, as JSON.
    ///
    /// "Explain that" is answered from this rather than by rebuilding it. Rebuilding
    /// would let the same question produce two different answers, and defending a
    /// score by recomputing it is how the recruiter assistant once re-scored a
    /// candidate mid-explanation.
    /// </summary>
    public string LastFactsJson { get; set; } = "{}";

    /// <summary>
    /// Previous filter states, oldest first — the undo stack behind "go back" and
    /// "actually, make that hybrid".
    /// </summary>
    public string FilterHistoryJson { get; set; } = "[]";

    /// <summary>
    /// Which scorer this conversation is using — see <c>ScoringModes</c>.
    ///
    /// On the session rather than on the request because it is a setting, not a
    /// property of one question: a candidate who switches to reasoned scoring
    /// expects the next answer to use it too. It is also read by the background
    /// pass, which runs after the request that set it is long gone.
    /// </summary>
    [MaxLength(16)] public string ScoringMode { get; set; } = "computed";

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    public List<PortalChatMessage> Messages { get; set; } = new();
}

[Table("portal_chat_messages")]
public class PortalChatMessage
{
    public int Id { get; set; }

    [MaxLength(64)] public string SessionId { get; set; } = "";
    public PortalChatSession? Session { get; set; }

    /// <summary>user | assistant | system</summary>
    [MaxLength(20)] public string Role { get; set; } = "user";

    public string Content { get; set; } = "";

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// A candidate's application to one posting on this board.
///
/// The scores are COPIED here rather than re-derived on read. A recruiter opening
/// an application months later must see what the candidate saw when they applied:
/// re-scoring would quietly move the number as models, thresholds or the posting
/// itself change, and the applicant would be judged against a bar that did not
/// exist when they applied.
///
/// Deliberately separate from the Python `analyses` table. An application is a
/// candidate's act; an analysis is the recruiter's evaluation. They become linked
/// only when a recruiter explicitly accepts the applicant, which is a write the
/// PYTHON side performs — see <see cref="AcceptedAnalysisId"/>.
/// </summary>
[Table("portal_applications")]
public class PortalApplication
{
    public int Id { get; set; }

    public int JobId { get; set; }
    public PortalJob? Job { get; set; }

    public int ResumeId { get; set; }
    public PortalResume? Resume { get; set; }

    /// <summary>
    /// Copied from the resume so it survives that row being replaced, and so the
    /// unique index can key on the person's actual document rather than on
    /// whichever row happened to hold it.
    /// </summary>
    [MaxLength(64)] public string ResumeContentHash { get; set; } = "";

    /// <summary>Submitted | Accepted | Declined.</summary>
    [MaxLength(20)] public string Status { get; set; } = ApplicationStatus.Submitted;

    public string CoverLetter { get; set; } = "";

    /// <summary>`llm` or `deterministic` — which path wrote the letter. Recorded
    /// because a recruiter reading it is entitled to know.</summary>
    [MaxLength(40)] public string LetterMode { get; set; } = "deterministic";

    // -- the fit, frozen at submission ---------------------------------------
    public double FitScore { get; set; }
    [MaxLength(40)] public string FitBand { get; set; } = "";
    public double SemanticScore { get; set; }
    public double SkillScore { get; set; }
    public double TitleScore { get; set; }
    public double ExperienceScore { get; set; }

    /// <summary>The per-skill verdicts as scored at submission, JSON.</summary>
    public string SkillsJson { get; set; } = "[]";
    public string StrengthsJson { get; set; } = "[]";
    public string GapsJson { get; set; } = "[]";
    public string RecruiterNote { get; set; } = "";

    /// <summary>Whether the score came from real semantic matching or the
    /// deterministic fallback. Without it a 46% from the hashing embedder is
    /// indistinguishable from a 46% that means something.</summary>
    public bool SemanticMatching { get; set; }
    [MaxLength(120)] public string EmbeddingModel { get; set; } = "";

    /// <summary>
    /// Model version that produced the fit score (e.g., "gpt-oss:20b").
    /// </summary>
    [MaxLength(120)] public string ModelVersion { get; set; } = "";

    /// <summary>
    /// Prompt version hash for the scoring pipeline. Bumped when scoring logic changes.
    /// </summary>
    [MaxLength(64)] public string PromptVersion { get; set; } = "";

    /// <summary>
    /// JSON array of skill-verdict cache keys used to produce this fit snapshot.
    /// Allows full reconstruction of the reasoning later.
    /// </summary>
    public string SkillVerdictCacheKeysJson { get; set; } = "[]";

    /// <summary>
    /// Set once a recruiter accepts this applicant into the hiring pipeline. The
    /// analysis row itself is created by the PYTHON backend — this side never
    /// writes its tables — and the id is reported back here so the application
    /// knows it has been taken up.
    /// </summary>
    public int? AcceptedAnalysisId { get; set; }
    public DateTime? AcceptedAt { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Cache for skill transferability verdicts.
/// Keyed by (jobId, candidateSkillSetHash, requiredSkill) to avoid re-deriving
/// the same LLM judgment for the same job/skill pair across sessions.
/// </summary>
[Table("portal_skill_verdict_cache")]
public class PortalSkillVerdictCache
{
    public int Id { get; set; }

    public int JobId { get; set; }
    public PortalJob? Job { get; set; }

    /// <summary>SHA-256 of the candidate's skill list (sorted, joined).</summary>
    [MaxLength(64)] public string CandidateSkillSetHash { get; set; } = "";

    /// <summary>The required skill from the job posting.</summary>
    [MaxLength(200)] public string RequiredSkill { get; set; } = "";

    /// <summary>Have | Transferable | Missing</summary>
    [MaxLength(20)] public string Verdict { get; set; } = "";

    /// <summary>If Transferable, the candidate skill that transfers.</summary>
    [MaxLength(200)] public string? EvidenceSkill { get; set; }

    /// <summary>Confidence 0.0-1.0.</summary>
    public double Confidence { get; set; }

    /// <summary>Model version that produced this verdict (e.g., "gpt-oss:20b").</summary>
    [MaxLength(120)] public string ModelVersion { get; set; } = "";

    /// <summary>Prompt version hash. Bumped when the skill-transfer prompt changes.</summary>
    [MaxLength(64)] public string PromptVersion { get; set; } = "";

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>The closed set of application states.</summary>
public static class ApplicationStatus
{
    public const string Submitted = "Submitted";
    public const string Accepted = "Accepted";
    public const string Declined = "Declined";

    public static bool IsKnown(string? value) =>
        value is Submitted or Accepted or Declined;
}
