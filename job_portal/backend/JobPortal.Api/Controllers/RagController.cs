using JobPortal.Api.Contracts;
using JobPortal.Api.Services.Rag;
using JobPortal.Api.Services.Rag.Data;
using JobPortal.Api.Services.Rag.Ingestion;
using JobPortal.Api.Services.Rag.Vectors;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Controllers;

/// <summary>
/// Operating the RAG index: what is in it, and re-indexing when something is
/// wrong with it.
///
/// Read-only apart from the re-index triggers, which exist because ingestion is
/// otherwise entirely automatic — the reconciliation sweep catches new documents
/// on its own, so these endpoints are for the cases automation cannot cover: a
/// document parked as failed, or a deliberate rebuild after a chunking change.
/// </summary>
[ApiController]
[Route("api/rag")]
public class RagController : ControllerBase
{
    private readonly RagDbContext _rag;
    private readonly IRagVectorStore _vectors;
    private readonly RagIngestionQueue _queue;
    private readonly RagOptions _options;

    public RagController(
        RagDbContext rag,
        IRagVectorStore vectors,
        RagIngestionQueue queue,
        IOptions<RagOptions> options)
    {
        _rag = rag;
        _vectors = vectors;
        _queue = queue;
        _options = options.Value;
    }

    public record CollectionStatus(
        string Name, long Documents, long Chunks, long Vectors, bool Aligned);

    public record RagStatus(
        bool Available,
        int QueueDepth,
        IReadOnlyList<CollectionStatus> Collections,
        IReadOnlyDictionary<string, int> ByStatus,
        IReadOnlyList<string> RecentFailures);

    /// <summary>
    /// What the index holds, and whether it agrees with itself.
    ///
    /// <c>Aligned</c> is the number that matters: chunk rows and vector points
    /// must match exactly. They diverge when ingestion was interrupted between the
    /// two writes, and a divergence means searches can return a chunk id whose
    /// text is missing — or miss text that was never embedded.
    /// </summary>
    [HttpGet("status")]
    public async Task<ActionResult<ApiResponse<RagStatus>>> Status(CancellationToken ct)
    {
        var available = await _vectors.IsAvailableAsync(ct);

        var collections = new List<CollectionStatus>();
        foreach (var (type, name) in new[]
                 {
                     (RagParentTypes.Resume, _options.ResumeCollection),
                     (RagParentTypes.Job, _options.JobCollection),
                 })
        {
            var chunks = await _rag.Chunks.CountAsync(c => c.ParentType == type, ct);
            var documents = await _rag.Chunks.Where(c => c.ParentType == type)
                .Select(c => c.ParentId).Distinct().CountAsync(ct);
            var vectors = available ? await _vectors.CountAsync(name, ct) : -1;

            collections.Add(new CollectionStatus(name, documents, chunks, vectors, vectors == chunks));
        }

        var byStatus = await _rag.IngestState
            .GroupBy(s => s.Status)
            .Select(g => new { g.Key, Count = g.Count() })
            .ToDictionaryAsync(g => g.Key, g => g.Count, ct);

        var failures = await _rag.IngestState
            .Where(s => s.Status == RagIngestStatuses.Failed)
            .OrderByDescending(s => s.UpdatedAt)
            .Take(10)
            .Select(s => $"{s.ParentType} {s.ParentId}: {s.LastError}")
            .ToListAsync(ct);

        return Ok(ApiResponse<RagStatus>.Ok(
            new RagStatus(available, _queue.Depth, collections, byStatus, failures),
            available ? "RAG index reachable." : "Qdrant is not reachable."));
    }

    /// <summary>
    /// Queues one document for re-indexing.
    ///
    /// Clears the ingest-state row rather than deleting chunks: the ingester
    /// replaces them itself, and removing them here would leave the document
    /// unsearchable for however long the queue takes to reach it.
    /// </summary>
    [HttpPost("reindex/{parentType}/{parentId:int}")]
    public async Task<ActionResult<ApiResponse<string>>> Reindex(
        string parentType, int parentId, CancellationToken ct)
    {
        var type = Normalise(parentType);
        if (type is null) return BadRequest(ApiResponse<string>.Fail("parentType must be 'resume' or 'job'."));

        await _rag.IngestState
            .Where(s => s.ParentType == type && s.ParentId == parentId)
            .ExecuteDeleteAsync(ct);

        await _queue.EnqueueAsync(new IngestJob(parentId, type), ct);

        return Ok(ApiResponse<string>.Ok($"{type} {parentId} queued.", "Queued for re-indexing."));
    }

    /// <summary>
    /// Forgets every ingestion verdict so the next sweep re-reads everything.
    ///
    /// The rebuild path after a chunking or embedding-model change. Chunks and
    /// vectors are left in place until each document is re-ingested, so the index
    /// keeps answering throughout — a rebuild that empties the index first is a
    /// rebuild with an outage in the middle of it.
    /// </summary>
    [HttpPost("reindex-all")]
    public async Task<ActionResult<ApiResponse<string>>> ReindexAll(
        [FromQuery] bool includeFailed = true, CancellationToken ct = default)
    {
        var removed = includeFailed
            ? await _rag.IngestState.ExecuteDeleteAsync(ct)
            : await _rag.IngestState.Where(s => s.Status != RagIngestStatuses.Failed)
                .ExecuteDeleteAsync(ct);

        return Ok(ApiResponse<string>.Ok(
            $"cleared {removed}",
            "Every document will be re-read on the next reconciliation sweep."));
    }

    private static string? Normalise(string value) => value.ToLowerInvariant() switch
    {
        "resume" or "resumes" => RagParentTypes.Resume,
        "job" or "jobs" => RagParentTypes.Job,
        _ => null,
    };
}
