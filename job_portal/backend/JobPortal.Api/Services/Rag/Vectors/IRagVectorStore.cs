namespace JobPortal.Api.Services.Rag.Vectors;

/// <summary>
/// One chunk's vector, ready to store.
///
/// <see cref="ChunkId"/> is the Postgres row id and becomes the Qdrant point id,
/// so a search result maps back to its text with no translation table. The text
/// itself is deliberately NOT here: it lives in Postgres, and duplicating it into
/// the vector store would give two places to disagree about what a chunk says.
/// </summary>
public record RagPoint(
    long ChunkId,
    float[] Vector,
    int ParentId,
    string ParentType,
    string Section,
    int Ordinal,
    int Page);

/// <summary>A search hit: which chunk, how close, and enough payload to group by
/// document without a second round trip.</summary>
public record RagHit(long ChunkId, double Score, int ParentId, string Section);

/// <summary>
/// The vector side of the RAG store.
///
/// Chunk-level, unlike the portal's existing <c>IVectorStore</c>, which is keyed
/// by job id and has nowhere to put a résumé or a passage. Deliberately narrow:
/// upsert, search, delete a document's chunks, count, and say whether the backend
/// is reachable at all.
/// </summary>
public interface IRagVectorStore
{
    /// <summary>Writes or replaces points. Batched — one call per embedding batch,
    /// not one per chunk.</summary>
    Task UpsertAsync(string collection, IReadOnlyList<RagPoint> points, CancellationToken ct = default);

    /// <summary>
    /// Nearest chunks above <paramref name="minScore"/>.
    ///
    /// <paramref name="restrictToParents"/> is applied as a payload filter INSIDE
    /// the index, not afterwards. That distinction is the reason for using a
    /// vector database at all: filtering after retrieval is how a shortlist of 50
    /// becomes an empty result the moment a constraint is added.
    /// </summary>
    Task<IReadOnlyList<RagHit>> SearchAsync(
        string collection,
        float[] query,
        int limit,
        double minScore,
        IReadOnlySet<int>? restrictToParents = null,
        CancellationToken ct = default);

    /// <summary>Removes every chunk of one document. Called before re-indexing it,
    /// so a document that shrank does not leave orphaned vectors behind.</summary>
    Task DeleteParentAsync(string collection, int parentId, CancellationToken ct = default);

    Task<long> CountAsync(string collection, CancellationToken ct = default);

    /// <summary>Whether the store can serve. The RAG mode reports itself
    /// unavailable rather than silently falling back to a different scorer.</summary>
    Task<bool> IsAvailableAsync(CancellationToken ct = default);
}
