using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace JobPortal.Api.Services.Rag.Vectors;

/// <summary>
/// Qdrant behind <see cref="IRagVectorStore"/>.
///
/// Every call names a COLLECTION ALIAS rather than a physical collection, so
/// re-embedding under a new model can build a replacement beside the live one and
/// repoint the alias atomically. Nothing here needs to know that happened.
/// </summary>
public class QdrantRagVectorStore : IRagVectorStore
{
    private readonly QdrantClient _client;
    private readonly ILogger<QdrantRagVectorStore> _logger;

    public QdrantRagVectorStore(QdrantClient client, ILogger<QdrantRagVectorStore> logger)
    {
        _client = client;
        _logger = logger;
    }

    public async Task UpsertAsync(
        string collection, IReadOnlyList<RagPoint> points, CancellationToken ct = default)
    {
        if (points.Count == 0) return;

        var structs = points.Select(p =>
        {
            var point = new PointStruct
            {
                // The Postgres chunk id IS the point id. One key across both
                // stores means a hit needs no translation to become text.
                Id = new PointId { Num = (ulong)p.ChunkId },
                Vectors = p.Vector,
            };

            point.Payload[RagPayload.ParentId] = p.ParentId;
            point.Payload[RagPayload.ParentType] = p.ParentType;
            point.Payload[RagPayload.Section] = p.Section;
            point.Payload[RagPayload.Ordinal] = p.Ordinal;
            point.Payload[RagPayload.Page] = p.Page;

            return point;
        }).ToList();

        await _client.UpsertAsync(collection, structs, cancellationToken: ct);
    }

    public async Task<IReadOnlyList<RagHit>> SearchAsync(
        string collection,
        float[] query,
        int limit,
        double minScore,
        IReadOnlySet<int>? restrictToParents = null,
        CancellationToken ct = default)
    {
        if (query.Length == 0 || limit <= 0) return Array.Empty<RagHit>();

        Filter? filter = null;
        if (restrictToParents is { Count: > 0 })
        {
            // Applied inside the index traversal, not after it. Retrieving a
            // fixed pool and filtering afterwards is how "remote only" empties a
            // result set that had plenty of remote roles further down.
            filter = new Filter();
            filter.Must.Add(Conditions.Match(
                RagPayload.ParentId, restrictToParents.Select(id => (long)id).ToList()));
        }

        var hits = await _client.SearchAsync(
            collection,
            query,
            filter: filter,
            limit: (ulong)limit,
            scoreThreshold: (float)minScore,
            payloadSelector: true,
            cancellationToken: ct);

        return hits.Select(h => new RagHit(
            ChunkId: (long)h.Id.Num,
            Score: h.Score,
            ParentId: (int)(h.Payload.TryGetValue(RagPayload.ParentId, out var pid) ? pid.IntegerValue : 0),
            Section: h.Payload.TryGetValue(RagPayload.Section, out var sec) ? sec.StringValue : "")).ToList();
    }

    public async Task DeleteParentAsync(string collection, int parentId, CancellationToken ct = default)
    {
        var filter = new Filter();
        filter.Must.Add(Conditions.Match(RagPayload.ParentId, parentId));

        await _client.DeleteAsync(collection, filter, cancellationToken: ct);
    }

    public async Task<long> CountAsync(string collection, CancellationToken ct = default)
    {
        try
        {
            return (long)await _client.CountAsync(collection, cancellationToken: ct);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not count '{Collection}'.", collection);
            return -1;
        }
    }

    public async Task<bool> IsAvailableAsync(CancellationToken ct = default)
    {
        try
        {
            await _client.ListCollectionsAsync(ct);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Qdrant is not reachable.");
            return false;
        }
    }
}
