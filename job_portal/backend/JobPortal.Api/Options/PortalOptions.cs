namespace JobPortal.Api.Options;

/// <summary>
/// Where the portal lives relative to the Python platform it shares a database with.
/// </summary>
public class PortalOptions
{
    public const string SectionName = "Portal";

    /// <summary>
    /// The shared SQLite file. Defaults to the Python backend's
    /// <c>hiring_platform.db</c>; point it elsewhere to run the portal standalone.
    /// </summary>
    public string DatabasePath { get; set; } = "";

    /// <summary>Root of the Python backend's <c>storage/</c> directory — the portal
    /// reads recruiter-uploaded JD documents from <c>uploads/jobs/</c> beneath it.</summary>
    public string PythonStorageDir { get; set; } = "";

    /// <summary>Where candidate resumes uploaded through the portal are written.
    /// Deliberately NOT the Python resume folder: those are the recruiter's
    /// screening pool, these are walk-in applicants.</summary>
    public string ResumeUploadDir { get; set; } = "";

    /// <summary>Origins allowed to open a SignalR connection. SignalR with
    /// credentials cannot use a wildcard origin, so these must be explicit.</summary>
    public string[] AllowedOrigins { get; set; } = { "http://localhost:5174", "http://localhost:5173" };

    /// <summary>Largest upload accepted, in megabytes.</summary>
    public int MaxUploadMegabytes { get; set; } = 15;

    /// <summary>
    /// Built frontend to serve from this same process.
    ///
    /// When present, the API and the SPA share one origin — which removes CORS
    /// entirely, removes the need for the browser to know a second base URL, and
    /// leaves exactly one port to expose when the app is shared or tunnelled.
    /// Blank or missing means API-only, which is what the Vite dev server wants.
    /// </summary>
    public string SpaPath { get; set; } = "";
}

/// <summary>
/// Connection details for the Ollama-compatible endpoint used for both chat
/// reasoning and embeddings. Every field is configuration: switching from Ollama
/// Cloud to a self-hosted box, or from one model tag to another, is never a code
/// change.
/// </summary>
public class OllamaOptions
{
    public const string SectionName = "Ollama";

    /// <summary>e.g. <c>https://ollama.com</c> for Ollama Cloud, or a RunPod proxy URL.</summary>
    public string BaseUrl { get; set; } = "";

    /// <summary>Bearer token. Blank for an unauthenticated self-hosted endpoint.</summary>
    public string ApiKey { get; set; } = "";

    /// <summary>Chat/reasoning model tag, e.g. <c>glm-5.2</c>.</summary>
    public string ChatModel { get; set; } = "";

    /// <summary>Embedding model tag, e.g. <c>nomic-embed-text</c>.</summary>
    public string EmbedModel { get; set; } = "";

    /// <summary>
    /// Separate base URL for embeddings. Chat-oriented cloud endpoints do not
    /// always serve embedding models, and being forced to choose one host for both
    /// would mean losing semantic matching to keep the chat, or the reverse. Blank
    /// means "use <see cref="BaseUrl"/>".
    /// </summary>
    public string EmbedBaseUrl { get; set; } = "";

    public string EmbedApiKey { get; set; } = "";

    /// <summary>Must match the embedding model's real output width. A wrong value
    /// is caught at first use rather than silently misaligning every search.</summary>
    public int EmbedDimension { get; set; } = 768;

    public int TimeoutSeconds { get; set; } = 120;

    /// <summary>Chat temperature. Low by default: this model extracts structured
    /// fields and writes recruiter notes over given evidence, neither of which
    /// benefits from sampling diversity.</summary>
    public double Temperature { get; set; } = 0.2;

    /// <summary>
    /// How long the endpoint should hold the model in memory after a call, in
    /// Ollama's duration format ("30m", "1h", "-1" for indefinitely).
    ///
    /// Blank leaves the server default (5 minutes), which is not enough when several
    /// models share one box: measured on this deployment, weights were evicted between
    /// calls minutes apart and every request paid ~8.9s to reload them against ~0.7s
    /// of real inference.
    /// </summary>
    public string KeepAlive { get; set; } = "30m";

    public bool Enabled => !string.IsNullOrWhiteSpace(BaseUrl) && !string.IsNullOrWhiteSpace(ChatModel);

    public string ResolvedEmbedBaseUrl => string.IsNullOrWhiteSpace(EmbedBaseUrl) ? BaseUrl : EmbedBaseUrl;
    public string ResolvedEmbedApiKey => string.IsNullOrWhiteSpace(EmbedBaseUrl) ? ApiKey : EmbedApiKey;

    public bool EmbeddingsEnabled =>
        !string.IsNullOrWhiteSpace(ResolvedEmbedBaseUrl) && !string.IsNullOrWhiteSpace(EmbedModel);
}

/// <summary>
/// Who an applicant should talk to, by the kind of role they are asking about.
///
/// Config rather than constants because these are PEOPLE. They change teams,
/// leave, and hand over — and none of that should require a rebuild to reflect.
/// </summary>
public class ScreeningOptions
{
    public const string SectionName = "Screening";

    public List<ScreeningContactOption> Contacts { get; set; } = new();
}

public class ScreeningContactOption
{
    /// <summary>
    /// Which work mode this person covers: <c>Remote</c>, <c>Onsite</c> or
    /// <c>Hybrid</c>. Matched case-insensitively against the posting's own mode;
    /// an entry marked <c>Any</c> is offered when nothing else fits.
    /// </summary>
    public string WorkMode { get; set; } = "Any";

    public string Name { get; set; } = "";

    /// <summary>Profile or contact URL. Anything that is not http(s) is dropped —
    /// a config typo must not become a link the app renders.</summary>
    public string Url { get; set; } = "";

    /// <summary>What they cover, in the applicant's words: "remote roles".</summary>
    public string Handles { get; set; } = "";
}

/// <summary>Matching and conversation thresholds. Config, never constants in code.</summary>
public class MatchingOptions
{
    public const string SectionName = "Matching";

    /// <summary>Cosine floor a job must clear against the resume to be shown at all.
    /// Below this the similarity is noise, and surfacing it would be the vector
    /// equivalent of a keyword false positive.</summary>
    public double MinSimilarity { get; set; } = 0.45;

    /// <summary>How many jobs the semantic stage retrieves before scoring.</summary>
    public int CandidatePoolSize { get; set; } = 50;

    /// <summary>How many matches are shown in one reply.</summary>
    public int TopN { get; set; } = 5;

    /// <summary>Above this many matches, the bot asks a narrowing question instead
    /// of dumping the list.</summary>
    public int ConversationalFilterThreshold { get; set; } = 8;

    /// <summary>
    /// How many roles the REASONED scorer judges before the turn is handed back.
    ///
    /// Small on purpose. A reasoned judgement costs a model call of ten to twenty
    /// seconds, so a shortlist of ten would leave the candidate watching a spinner
    /// for three minutes. Three arrive while they are still reading the opening
    /// sentence; the rest fill in behind them.
    /// </summary>
    public int ReasonedFirstBatch { get; set; } = 3;

    /// <summary>
    /// The most roles one question will ever be reasoned about, first batch
    /// included. The ceiling exists because the work continues after the turn ends
    /// and nothing else would stop it.
    /// </summary>
    public int ReasonedMaxJobs { get; set; } = 10;

    // Fit score composition. Semantic similarity carries the most weight because
    // it is the signal that survives vocabulary differences; explicit skill
    // overlap is a strong but brittle confirmation; title and experience are
    // context. Must sum to 1.0.
    public double WeightSemantic { get; set; } = 0.50;
    public double WeightSkillOverlap { get; set; } = 0.25;
    public double WeightTitle { get; set; } = 0.15;
    public double WeightExperience { get; set; } = 0.10;

    /// <summary>
    /// Cosine at which two skill names are treated as the same skill ("React"
    /// satisfying "React.js").
    ///
    /// MUST be recalibrated per embedding model — a threshold is a property of the
    /// model's similarity distribution, not of the domain. Measured on
    /// nomic-embed-text over hand-labelled pairs, the equivalent and non-equivalent
    /// bands OVERLAP: the highest non-equivalent pair ("Modern JavaScript UI
    /// framework" ~ "JavaScript", 0.701) scores above the lowest genuinely
    /// equivalent one ("CSS" ~ "SCSS", 0.430). No threshold separates them.
    ///
    /// 0.72 is therefore set above the highest observed FALSE positive rather than
    /// below the lowest true one. That deliberately demotes some real equivalences
    /// ("K8s" ~ "Kubernetes", 0.481) to Missing, where the transfer reasoner picks
    /// them up as transferable. The asymmetry is the point: an understated
    /// "transferable" is truthful, whereas a false "Have" tells a candidate they
    /// are covered for something they cannot do. At 0.55 this reported "CSS"
    /// satisfied by "JavaScript".
    /// </summary>
    public double SkillEquivalenceMin { get; set; } = 0.72;

    /// <summary>
    /// Cosine at which a skill would count as merely adjacent.
    ///
    /// Defaults EQUAL to <see cref="SkillEquivalenceMin"/>, which collapses the band
    /// to nothing — deliberately. Measured on this deployment's models, the adjacent
    /// and unrelated distributions overlap almost exactly (nomic bare: adjacent
    /// median 0.379 vs unrelated 0.383), so no threshold in that region separates
    /// them. Any value below equivalence would put confident falsehoods in front of
    /// candidates in both directions.
    ///
    /// Transferability is instead decided by <see cref="Services.Matching.ISkillTransferService"/>,
    /// which reasons over the candidate's actual skill list. Lower this only if you
    /// have measured a real gap for YOUR embedding model.
    /// </summary>
    public double SkillRelatedMin { get; set; } = 0.72;
}
