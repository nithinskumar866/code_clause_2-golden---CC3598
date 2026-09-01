using JobPortal.Api.Services.Rag.Data;
using JobPortal.Api.Services.Rag.Vectors;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Rag.Retrieval;

/// <summary>One passage that matched, with everything needed to quote it.</summary>
public record RetrievedChunk(long ChunkId, int ParentId, string Section, string Text, double Score);

/// <summary>
/// One document from STAGE 1, with the passages that put it there.
///
/// <see cref="Score"/> is a retrieval score, not a fit score. It orders the
/// shortlist and nothing else — the number a person sees comes from the judge in
/// stage 2, because similarity cannot tell that leading three engineers differs
/// from leading three hundred.
/// </summary>
public record DocumentMatch(int ParentId, double Score, IReadOnlyList<RetrievedChunk> Chunks);

/// <summary>
/// The two retrieval stages, as a seam.
///
/// An interface so the agent above it can be tested without Qdrant and Postgres
/// running. The agent's job — turning passages into verdicts — is where the
/// decisions are, and it should not need a database to exercise.
/// </summary>
public interface IRagRetriever
{
    Task<IReadOnlyList<DocumentMatch>> ShortlistAsync(
        string query, string parentType, IReadOnlySet<int>? restrictTo = null,
        int? take = null, CancellationToken ct = default);

    Task<IReadOnlyList<RetrievedChunk>> ForRequirementAsync(
        string requirement, string parentType, int parentId, CancellationToken ct = default);
}

/// <summary>
/// Both retrieval stages.
///
///   STAGE 1 — <see cref="ShortlistAsync"/>. Milliseconds, no model. One embedding,
///   one filtered vector search, grouped by document. Its whole job is to discard
///   the 99,980 documents nobody should read and hand over ten or twenty.
///
///   STAGE 2 — <see cref="ForRequirementAsync"/>. Also milliseconds, still no
///   model, but pointed: the passages of ONE document that bear on ONE
///   requirement. This is what replaces pasting an entire CV into a prompt and
///   truncating it at nine thousand characters.
///
/// Neither stage judges anything. That is deliberate and it is the line the whole
/// architecture rests on: math sieves, the model decides.
/// </summary>
public class RagRetriever : IRagRetriever
{
    private readonly IRagVectorStore _vectors;
    private readonly IEmbeddingService _embeddings;
    private readonly RagDbContext _rag;
    private readonly RagOptions _options;
    private readonly ILogger<RagRetriever> _logger;

    public RagRetriever(
        IRagVectorStore vectors,
        IEmbeddingService embeddings,
        RagDbContext rag,
        IOptions<RagOptions> options,
        ILogger<RagRetriever> logger)
    {
        _vectors = vectors;
        _embeddings = embeddings;
        _rag = rag;
        _options = options.Value;
        _logger = logger;
    }

    /// <summary>
    /// STAGE 1 — the sieve.
    /// </summary>
    /// <param name="query">What to search for: a CV's text when suggesting jobs,
    /// a posting's text when finding candidates.</param>
    /// <param name="parentType">Which side to search — see <see cref="RagParentTypes"/>.</param>
    /// <param name="restrictTo">
    /// Ids the caller has already narrowed to, usually by a structured filter.
    /// Applied INSIDE the index, so a constraint removes documents before the
    /// nearest-neighbour cut rather than after it. Filtering afterwards is how a
    /// pool of fifty becomes an empty result the moment "remote only" is added.
    /// </param>
    public async Task<IReadOnlyList<DocumentMatch>> ShortlistAsync(
        string query,
        string parentType,
        IReadOnlySet<int>? restrictTo = null,
        int? take = null,
        CancellationToken ct = default)
    {
        var vector = await EmbedAsync(query, ct);
        if (vector.Length == 0) return Array.Empty<DocumentMatch>();

        var hits = await _vectors.SearchAsync(
            CollectionFor(parentType),
            vector,
            // Far more chunks than documents wanted: several passages of one strong
            // document rank together, so asking for twenty chunks can yield only
            // four distinct people.
            _options.ShortlistChunkDepth,
            _options.MinSimilarity,
            restrictTo,
            ct);

        if (hits.Count == 0) return Array.Empty<DocumentMatch>();

        var texts = await TextsForAsync(hits.Select(h => h.ChunkId), ct);

        var grouped = hits
            .GroupBy(h => h.ParentId)
            .Select(group =>
            {
                var chunks = group
                    .OrderByDescending(h => h.Score)
                    .Select(h => new RetrievedChunk(
                        h.ChunkId, h.ParentId, h.Section,
                        texts.TryGetValue(h.ChunkId, out var text) ? text : "", h.Score))
                    // A hit whose text is missing means the two stores disagree.
                    // Dropping it is right: a quote nobody can read is not evidence.
                    .Where(c => c.Text.Length > 0)
                    .ToList();

                return new DocumentMatch(group.Key, Aggregate(chunks), chunks);
            })
            .Where(d => d.Chunks.Count > 0)
            .OrderByDescending(d => d.Score)
            .Take(take ?? _options.ShortlistSize)
            .ToList();

        _logger.LogDebug(
            "Stage 1: {Chunks} chunks -> {Documents} documents (floor {Floor:0.00}).",
            hits.Count, grouped.Count, _options.MinSimilarity);

        return grouped;
    }

    /// <summary>
    /// STAGE 2 — the passages of one document that bear on one requirement.
    ///
    /// Returns few, deliberately. The judge is being asked about a single
    /// requirement, and handing it twenty passages to weigh is how an evaluation
    /// drifts back toward reading the whole document — which is the thing this
    /// replaces.
    /// </summary>
    public async Task<IReadOnlyList<RetrievedChunk>> ForRequirementAsync(
        string requirement,
        string parentType,
        int parentId,
        CancellationToken ct = default)
    {
        var vector = await EmbedAsync(requirement, ct);
        if (vector.Length == 0) return Array.Empty<RetrievedChunk>();

        var hits = await _vectors.SearchAsync(
            CollectionFor(parentType),
            vector,
            _options.ChunksPerRequirement,
            _options.MinSimilarity,
            new HashSet<int> { parentId },
            ct);

        if (hits.Count == 0) return Array.Empty<RetrievedChunk>();

        var texts = await TextsForAsync(hits.Select(h => h.ChunkId), ct);

        return hits
            .Select(h => new RetrievedChunk(
                h.ChunkId, h.ParentId, h.Section,
                texts.TryGetValue(h.ChunkId, out var text) ? text : "", h.Score))
            .Where(c => c.Text.Length > 0)
            .ToList();
    }

    /// <summary>
    /// A document's score from its chunks.
    ///
    /// The best passage carries most of it — a single excellent match is what a
    /// specialist looks like — and the mean of the next few carries the rest, so
    /// corroboration across several passages beats one lucky sentence.
    /// </summary>
    private double Aggregate(IReadOnlyList<RetrievedChunk> chunks)
    {
        if (chunks.Count == 0) return 0;

        var best = chunks[0].Score;
        var mean = chunks.Take(Math.Max(1, _options.MeanChunkCount)).Average(c => c.Score);

        return _options.BestChunkWeight * best + (1 - _options.BestChunkWeight) * mean;
    }

    /// <summary>
    /// The text of the hit chunks, in ONE query.
    ///
    /// Text lives in Postgres rather than in the vector payload so there is one
    /// place that knows what a chunk says. The cost is this lookup, and it is a
    /// single batched SELECT rather than one per hit.
    /// </summary>
    private async Task<Dictionary<long, string>> TextsForAsync(
        IEnumerable<long> chunkIds, CancellationToken ct)
    {
        var ids = chunkIds.Distinct().ToList();

        return await _rag.Chunks
            .AsNoTracking()
            .Where(c => ids.Contains(c.ChunkId))
            .ToDictionaryAsync(c => c.ChunkId, c => c.Text, ct);
    }

    private async Task<float[]> EmbedAsync(string text, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<float>();

        var batch = await _embeddings.EmbedAsync(Documents.TextStructure.Clip(text, 6000), ct);

        if (!batch.IsSemantic || batch.Vectors.Count == 0)
        {
            // The fallback embedder's vectors are not comparable with the indexed
            // ones. Searching with them returns confident nonsense, so return
            // nothing and let the caller report the mode unavailable.
            _logger.LogWarning("Semantic embeddings unavailable; RAG retrieval cannot run.");
            return Array.Empty<float>();
        }

        return VectorMath.Normalize(batch.Vectors[0]);
    }

    private string CollectionFor(string parentType) =>
        parentType == RagParentTypes.Resume ? _options.ResumeCollection : _options.JobCollection;
}
