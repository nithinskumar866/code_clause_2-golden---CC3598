using System.Collections.Concurrent;
using JobPortal.Api.Data;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Services.Vectors;

/// <summary>
/// Holds decoded vectors in memory, per model, so a search does not re-read and
/// re-decode the whole table on every keystroke.
///
/// Registered as a singleton and invalidated on write. At portal scale the whole
/// index is a few megabytes, and the alternative (a fresh query per search) turns
/// every conversational follow-up into a full table scan plus a full float decode.
/// </summary>
public class JobVectorCache
{
    private readonly ConcurrentDictionary<string, IReadOnlyList<(int JobId, float[] Vector)>> _byModel = new();
    private readonly SemaphoreSlim _lock = new(1, 1);

    public void Invalidate() => _byModel.Clear();

    public async Task<IReadOnlyList<(int JobId, float[] Vector)>> GetAsync(
        string model, PortalDbContext db, CancellationToken ct)
    {
        if (_byModel.TryGetValue(model, out var cached)) return cached;

        await _lock.WaitAsync(ct);
        try
        {
            // Another caller may have populated it while this one waited.
            if (_byModel.TryGetValue(model, out cached)) return cached;

            var rows = await db.JobVectors
                .AsNoTracking()
                .Where(v => v.Model == model)
                .Select(v => new { v.JobId, v.Vector })
                .ToListAsync(ct);

            var decoded = rows
                .Select(r => (r.JobId, Vector: VectorMath.FromBytes(r.Vector)))
                .ToList();

            _byModel[model] = decoded;
            return decoded;
        }
        finally
        {
            _lock.Release();
        }
    }
}

public class SqliteVectorStore : IVectorStore
{
    private readonly PortalDbContext _db;
    private readonly JobVectorCache _cache;
    private readonly ILogger<SqliteVectorStore> _logger;

    public SqliteVectorStore(PortalDbContext db, JobVectorCache cache, ILogger<SqliteVectorStore> logger)
    {
        _db = db;
        _cache = cache;
        _logger = logger;
    }

    public async Task UpsertAsync(int jobId, string model, int dimension, float[] vector,
        string textHash, CancellationToken ct = default)
    {
        // Normalise on the way in so search is a dot product rather than a full
        // cosine over the whole corpus on every query.
        var normalised = VectorMath.Normalize(vector.ToArray());
        var bytes = VectorMath.ToBytes(normalised);

        var existing = await _db.JobVectors
            .FirstOrDefaultAsync(v => v.JobId == jobId && v.Model == model, ct);

        if (existing is null)
        {
            _db.JobVectors.Add(new PortalJobVector
            {
                JobId = jobId,
                Model = model,
                Dimension = dimension,
                Vector = bytes,
                TextHash = textHash,
                CreatedAt = DateTime.UtcNow,
            });
        }
        else
        {
            existing.Dimension = dimension;
            existing.Vector = bytes;
            existing.TextHash = textHash;
            existing.CreatedAt = DateTime.UtcNow;
        }

        await _db.SaveChangesAsync(ct);
        _cache.Invalidate();
    }

    public async Task<IReadOnlyList<VectorHit>> SearchAsync(
        float[] query, string model, int topK, double minScore,
        IReadOnlySet<int>? restrictTo = null, CancellationToken ct = default)
    {
        if (query.Length == 0 || topK <= 0) return Array.Empty<VectorHit>();

        var vectors = await _cache.GetAsync(model, _db, ct);
        if (vectors.Count == 0) return Array.Empty<VectorHit>();

        var normalisedQuery = VectorMath.Normalize(query.ToArray());

        var hits = new List<VectorHit>(Math.Min(vectors.Count, 256));
        foreach (var (jobId, vector) in vectors)
        {
            if (restrictTo is not null && !restrictTo.Contains(jobId)) continue;

            // A width mismatch inside one model id means the configured dimension
            // changed under a populated index. Skipping is right: the alternative
            // is a similarity of zero presented as a real ranking.
            if (vector.Length != normalisedQuery.Length) continue;

            var score = VectorMath.Dot(normalisedQuery, vector);
            if (score < minScore) continue;

            hits.Add(new VectorHit(jobId, score));
        }

        return hits.OrderByDescending(h => h.Score).Take(topK).ToList();
    }

    public async Task<IReadOnlyDictionary<int, string>> GetIndexedHashesAsync(
        string model, CancellationToken ct = default)
    {
        var rows = await _db.JobVectors
            .AsNoTracking()
            .Where(v => v.Model == model)
            .Select(v => new { v.JobId, v.TextHash })
            .ToListAsync(ct);

        return rows.ToDictionary(r => r.JobId, r => r.TextHash);
    }

    public Task<int> CountAsync(string model, CancellationToken ct = default) =>
        _db.JobVectors.CountAsync(v => v.Model == model, ct);

    public async Task<IReadOnlyDictionary<string, int>> CountsByModelAsync(CancellationToken ct = default)
    {
        var rows = await _db.JobVectors
            .AsNoTracking()
            .GroupBy(v => v.Model)
            .Select(g => new { Model = g.Key, Count = g.Count() })
            .ToListAsync(ct);

        return rows.ToDictionary(r => r.Model, r => r.Count);
    }

    public async Task RemoveAsync(int jobId, CancellationToken ct = default)
    {
        var rows = await _db.JobVectors.Where(v => v.JobId == jobId).ToListAsync(ct);
        if (rows.Count == 0) return;

        _db.JobVectors.RemoveRange(rows);
        await _db.SaveChangesAsync(ct);
        _cache.Invalidate();
        _logger.LogInformation("Removed {Count} vector(s) for job {JobId}", rows.Count, jobId);
    }
}
