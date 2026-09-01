namespace JobPortal.Api.Services.Rag;

/// <summary>
/// Everything the RAG scoring mode can be tuned by.
///
/// One options class for the whole feature, because the feature is one folder.
/// Credentials are the exception: <see cref="PostgresConnectionString"/> and
/// <see cref="QdrantApiKey"/> come from the environment, never from
/// appsettings.json, so a checked-in config file never carries a secret.
/// </summary>
public class RagOptions
{
    public const string SectionName = "Rag";

    /// <summary>
    /// Whether the mode is offered at all.
    ///
    /// False leaves every existing scoring mode untouched and the third button
    /// hidden — which is what should happen on a deployment with no Qdrant and no
    /// Postgres, rather than a button that fails when pressed.
    /// </summary>
    public bool Enabled { get; set; } = true;

    // -- storage ------------------------------------------------------------

    /// <summary>Npgsql connection string. Set via <c>Rag__PostgresConnectionString</c>.</summary>
    public string PostgresConnectionString { get; set; } = "";

    /// <summary>gRPC host for Qdrant. The .NET client speaks gRPC, not HTTP.</summary>
    public string QdrantHost { get; set; } = "localhost";

    public int QdrantPort { get; set; } = 6334;

    public bool QdrantUseTls { get; set; } = false;

    /// <summary>Set via <c>Rag__QdrantApiKey</c>. Blank means an unauthenticated endpoint.</summary>
    public string QdrantApiKey { get; set; } = "";

    /// <summary>
    /// Collection names. These are ALIASES, never the physical collections.
    ///
    /// A re-embedding run builds a new physical collection beside the live one
    /// and repoints the alias when it finishes, so changing embedding model costs
    /// no downtime and a failed run leaves the old index serving.
    /// </summary>
    public string ResumeCollection { get; set; } = "resume_chunks";

    public string JobCollection { get; set; } = "job_chunks";

    // -- chunking -----------------------------------------------------------

    /// <summary>
    /// Chunk sizing, in characters. Ported from the Python platform, where these
    /// were tuned against real résumés.
    ///
    /// A chunk accumulates whole sentences until it reaches <see cref="ChunkTargetChars"/>
    /// and the next sentence would push it past <see cref="ChunkMaxChars"/>.
    /// Chunks never span a section, because the section is what distinguishes a
    /// skill someone used from one they merely listed.
    /// </summary>
    public int ChunkTargetChars { get; set; } = 350;

    public int ChunkMaxChars { get; set; } = 600;

    /// <summary>
    /// Sentences (not characters) carried from the end of one chunk into the next.
    ///
    /// Overlap in sentences rather than a character window means a chunk boundary
    /// never lands mid-sentence, so a quote pulled from either side of it is still
    /// something the candidate actually wrote.
    /// </summary>
    public int ChunkSentenceOverlap { get; set; } = 1;

    // -- ingestion ----------------------------------------------------------

    /// <summary>
    /// Chunks per embedding call.
    ///
    /// The single biggest lever on ingestion throughput. The existing reindex path
    /// calls the one-string overload in a loop, which is one HTTP round trip per
    /// chunk; at 1.5M chunks that is the difference between hours and weeks.
    /// </summary>
    public int EmbedBatchSize { get; set; } = 64;

    /// <summary>
    /// How many documents may wait in the queue before uploads start blocking.
    ///
    /// Bounded on purpose: an unbounded channel turns a burst of uploads into
    /// unbounded memory, and the failure arrives long after the cause.
    /// </summary>
    public int IngestQueueCapacity { get; set; } = 10_000;

    /// <summary>Attempts before a document is parked as failed rather than retried forever.</summary>
    public int MaxIngestAttempts { get; set; } = 3;

    // -- retrieval ----------------------------------------------------------

    /// <summary>
    /// STAGE 1 — how many chunks the vector search returns before grouping.
    ///
    /// Deliberately far larger than the document shortlist: several chunks of one
    /// résumé will rank highly together, so retrieving only 20 chunks might yield
    /// only four distinct people.
    /// </summary>
    public int ShortlistChunkDepth { get; set; } = 200;

    /// <summary>STAGE 1 — documents handed to the judge.</summary>
    public int ShortlistSize { get; set; } = 15;

    /// <summary>
    /// How a document's score is built from its chunks: the best chunk carries
    /// most of it, the mean of the top few carries the rest.
    ///
    /// A single excellent passage should win — that is what a specialist looks
    /// like — but corroboration across several passages should beat one lucky
    /// sentence, which is what the mean term buys.
    /// </summary>
    public double BestChunkWeight { get; set; } = 0.7;

    public int MeanChunkCount { get; set; } = 3;

    /// <summary>
    /// Cosine floor. Below this a chunk is noise, not weak evidence.
    ///
    /// 0.51 is the MEASURED value for nomic-embed-text on this corpus: relevant
    /// pairs sit at or above 0.592 and unrelated ones at or below 0.426, so 0.51
    /// sits in the gap between them. It is a property of the embedding model's
    /// similarity distribution and NOT of hiring — recalibrate it whenever the
    /// embedding model changes, or it silently discards genuine matches.
    /// </summary>
    public double MinSimilarity { get; set; } = 0.51;

    /// <summary>
    /// STAGE 2 — chunks retrieved per requirement, within the shortlist.
    ///
    /// Small: the judge is being asked about one requirement, and handing it
    /// twenty passages to weigh is how the evaluation drifts back toward reading
    /// the whole document.
    /// </summary>
    public int ChunksPerRequirement { get; set; } = 4;

    /// <summary>
    /// Whether a requirement that finds nothing is retried with expanded terms
    /// before being recorded MISSING.
    ///
    /// This is the agentic step. It is the difference between "the résumé does not
    /// contain this phrase" and "the résumé does not evidence this requirement",
    /// which is the distinction the whole evaluation turns on.
    /// </summary>
    public bool ExpandOnEmpty { get; set; } = true;
}
