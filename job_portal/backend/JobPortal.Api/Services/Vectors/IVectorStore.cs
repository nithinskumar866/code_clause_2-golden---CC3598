namespace JobPortal.Api.Services.Vectors;

public record VectorHit(int JobId, double Score);

/// <summary>
/// Persistence and search for job vectors.
///
/// Kept behind an interface with no SQL in its signature so the storage decision
/// stays reversible: brute-force cosine over SQLite blobs is exactly right at this
/// size and needs no infrastructure, but a corpus that outgrows it should be able
/// to move to Qdrant or SQL Server vectors by writing one more implementation
/// rather than by touching matching, chat or indexing.
/// </summary>
public interface IVectorStore
{
    /// <summary>Writes or replaces a job's vector for one model.</summary>
    Task UpsertAsync(int jobId, string model, int dimension, float[] vector, string textHash,
        CancellationToken ct = default);

    /// <summary>
    /// Nearest jobs by cosine, highest first. Only vectors from <paramref name="model"/>
    /// are considered: comparing across models produces a number that is arithmetically
    /// valid and semantically meaningless.
    /// </summary>
    Task<IReadOnlyList<VectorHit>> SearchAsync(float[] query, string model, int topK,
        double minScore, IReadOnlySet<int>? restrictTo = null, CancellationToken ct = default);

    /// <summary>Text hash per job for one model, so re-indexing can skip jobs whose
    /// embedded text has not changed.</summary>
    Task<IReadOnlyDictionary<int, string>> GetIndexedHashesAsync(string model,
        CancellationToken ct = default);

    Task<int> CountAsync(string model, CancellationToken ct = default);

    /// <summary>Vector counts per model. Two populated models mean an index was
    /// built partly before and partly after an embedder change, which is worth
    /// showing rather than hiding.</summary>
    Task<IReadOnlyDictionary<string, int>> CountsByModelAsync(CancellationToken ct = default);

    Task RemoveAsync(int jobId, CancellationToken ct = default);
}
