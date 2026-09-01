using System.Security.Cryptography;
using System.Text;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Rag.Data;
using JobPortal.Api.Services.Rag.Vectors;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Rag.Ingestion;

/// <summary>What one document's ingestion did. Returned so a backfill can report
/// totals without re-querying.</summary>
public record IngestOutcome(bool Indexed, bool Skipped, int Chunks, string? Error = null);

/// <summary>
/// Ingests one document: read it, chunk it, embed the chunks, store both halves.
///
/// Separate from the worker that calls it so the whole pipeline can be tested
/// without a hosted service, and so a single document can be re-ingested on
/// demand.
///
/// The two stores are written in an order chosen for what survives a crash:
/// Postgres first, then Qdrant. A crash between them leaves chunks with no
/// vectors — invisible to search, and repaired by the next run because the
/// ingest-state row was never marked indexed. The reverse order would leave
/// vectors pointing at chunk ids that do not exist, which surfaces as a search
/// hit whose text cannot be found.
/// </summary>
public class RagIngestionService
{
    private readonly PortalDbContext _portal;
    private readonly RagDbContext _rag;
    private readonly DocumentChunker _chunker;
    private readonly IEmbeddingService _embeddings;
    private readonly IRagVectorStore _vectors;
    private readonly RagOptions _options;
    private readonly ILogger<RagIngestionService> _logger;

    public RagIngestionService(
        PortalDbContext portal,
        RagDbContext rag,
        DocumentChunker chunker,
        IEmbeddingService embeddings,
        IRagVectorStore vectors,
        IOptions<RagOptions> options,
        ILogger<RagIngestionService> logger)
    {
        _portal = portal;
        _rag = rag;
        _chunker = chunker;
        _embeddings = embeddings;
        _vectors = vectors;
        _options = options.Value;
        _logger = logger;
    }

    public async Task<IngestOutcome> IngestAsync(IngestJob job, CancellationToken ct = default)
    {
        var (text, found) = await ReadSourceAsync(job, ct);
        if (!found) return await FailAsync(job, "the source document no longer exists", ct);
        if (string.IsNullOrWhiteSpace(text)) return await FailAsync(job, "the document has no readable text", ct);

        var hash = Hash(text);

        var state = await _rag.IngestState
            .FirstOrDefaultAsync(s => s.ParentType == job.ParentType && s.ParentId == job.ParentId, ct);

        // Unchanged content: nothing to do. This is what makes a re-run of a
        // hundred-thousand-document backfill free rather than a second full pass.
        if (state is { Status: RagIngestStatuses.Indexed } && state.SourceHash == hash)
        {
            return new IngestOutcome(Indexed: false, Skipped: true, Chunks: state.ChunkCount);
        }

        var chunks = _chunker.Chunk(text, isResume: job.ParentType == RagParentTypes.Resume);
        if (chunks.Count == 0) return await FailAsync(job, "the document produced no chunks", ct);

        var collection = CollectionFor(job.ParentType);

        // Replace rather than merge. A document that shrank would otherwise keep
        // its old tail as orphaned chunks that still answer searches.
        await _vectors.DeleteParentAsync(collection, job.ParentId, ct);
        await _rag.Chunks
            .Where(c => c.ParentType == job.ParentType && c.ParentId == job.ParentId)
            .ExecuteDeleteAsync(ct);

        var rows = chunks.Select(c => new RagChunk
        {
            ParentId = job.ParentId,
            ParentType = job.ParentType,
            Section = Trim(c.Section, 64),
            Ordinal = c.Ordinal,
            Page = c.Page,
            Text = c.Text,
            SourceHash = hash,
        }).ToList();

        // Saved first so the database assigns the chunk ids the vectors will use.
        _rag.Chunks.AddRange(rows);
        await _rag.SaveChangesAsync(ct);

        var embedded = await EmbedAndStoreAsync(collection, rows, ct);
        if (embedded < rows.Count)
        {
            return await FailAsync(job,
                $"only {embedded} of {rows.Count} chunks embedded", ct);
        }

        await MarkIndexedAsync(job, state, hash, rows.Count, ct);
        return new IngestOutcome(Indexed: true, Skipped: false, Chunks: rows.Count);
    }

    /// <summary>
    /// Embeds in batches and writes each batch straight through.
    ///
    /// The batch size is the single biggest lever on ingestion throughput: the
    /// portal's existing reindex calls the one-string overload in a loop, which is
    /// one network round trip per chunk. At 1.5M chunks that is the difference
    /// between hours and weeks.
    /// </summary>
    private async Task<int> EmbedAndStoreAsync(
        string collection, IReadOnlyList<RagChunk> rows, CancellationToken ct)
    {
        var stored = 0;

        foreach (var batch in rows.Chunk(Math.Max(1, _options.EmbedBatchSize)))
        {
            ct.ThrowIfCancellationRequested();

            var result = await _embeddings.EmbedAsync(batch.Select(r => r.Text).ToList(), ct);

            if (!result.IsSemantic)
            {
                // The deterministic fallback embedder produces vectors that are
                // not comparable with the real model's. Indexing them would
                // poison the collection with points that can never match.
                _logger.LogError(
                    "Embedding fell back to the deterministic embedder; refusing to index " +
                    "{Count} chunks that would be incomparable with the rest of the collection.",
                    batch.Length);
                return stored;
            }

            if (result.Vectors.Count != batch.Length)
            {
                _logger.LogError(
                    "Embedder returned {Got} vectors for {Wanted} chunks.",
                    result.Vectors.Count, batch.Length);
                return stored;
            }

            var points = batch.Select((row, i) => new RagPoint(
                row.ChunkId,
                VectorMath.Normalize(result.Vectors[i]),
                row.ParentId,
                row.ParentType,
                row.Section,
                row.Ordinal,
                row.Page)).ToList();

            await _vectors.UpsertAsync(collection, points, ct);
            stored += points.Count;
        }

        return stored;
    }

    private async Task<(string Text, bool Found)> ReadSourceAsync(IngestJob job, CancellationToken ct)
    {
        if (job.ParentType == RagParentTypes.Resume)
        {
            var resume = await _portal.Resumes.AsNoTracking()
                .FirstOrDefaultAsync(r => r.Id == job.ParentId, ct);
            return resume is null ? ("", false) : (resume.RawText ?? "", true);
        }

        var posting = await _portal.Jobs.AsNoTracking()
            .FirstOrDefaultAsync(j => j.Id == job.ParentId, ct);

        if (posting is null) return ("", false);

        // The raw posting, not the curated embedding text: chunking wants the
        // document as written, headings intact, because the headings are what the
        // chunker splits on.
        var text = string.IsNullOrWhiteSpace(posting.RawText) ? posting.EmbeddedText : posting.RawText;
        return (text ?? "", true);
    }

    private async Task MarkIndexedAsync(
        IngestJob job, RagIngestState? state, string hash, int chunks, CancellationToken ct)
    {
        if (state is null)
        {
            _rag.IngestState.Add(new RagIngestState
            {
                ParentId = job.ParentId,
                ParentType = job.ParentType,
                Status = RagIngestStatuses.Indexed,
                SourceHash = hash,
                ChunkCount = chunks,
                Attempts = 0,
                UpdatedAt = DateTime.UtcNow,
            });
        }
        else
        {
            state.Status = RagIngestStatuses.Indexed;
            state.SourceHash = hash;
            state.ChunkCount = chunks;
            state.Attempts = 0;
            state.LastError = null;
            state.UpdatedAt = DateTime.UtcNow;
        }

        await _rag.SaveChangesAsync(ct);
    }

    /// <summary>
    /// Records a failure and decides whether it is worth retrying.
    ///
    /// A document is parked as failed only after exhausting its attempts, and it
    /// is never retried automatically after that — a document that cannot be read
    /// will not become readable, and retrying it forever hides the working
    /// backlog behind a wall of the same error.
    /// </summary>
    private async Task<IngestOutcome> FailAsync(IngestJob job, string reason, CancellationToken ct)
    {
        var state = await _rag.IngestState
            .FirstOrDefaultAsync(s => s.ParentType == job.ParentType && s.ParentId == job.ParentId, ct);

        if (state is null)
        {
            state = new RagIngestState { ParentId = job.ParentId, ParentType = job.ParentType };
            _rag.IngestState.Add(state);
        }

        state.Attempts++;
        state.LastError = Trim(reason, 500);
        state.UpdatedAt = DateTime.UtcNow;
        state.Status = state.Attempts >= _options.MaxIngestAttempts
            ? RagIngestStatuses.Failed
            : RagIngestStatuses.Pending;

        await _rag.SaveChangesAsync(ct);

        _logger.Log(
            state.Status == RagIngestStatuses.Failed ? LogLevel.Warning : LogLevel.Debug,
            "Ingestion of {Type} {Id} failed ({Attempt}/{Max}): {Reason}",
            job.ParentType, job.ParentId, state.Attempts, _options.MaxIngestAttempts, reason);

        return new IngestOutcome(Indexed: false, Skipped: false, Chunks: 0, Error: reason);
    }

    public string CollectionFor(string parentType) =>
        parentType == RagParentTypes.Resume ? _options.ResumeCollection : _options.JobCollection;

    private static string Trim(string value, int max) =>
        value.Length <= max ? value : value[..max];

    private static string Hash(string text) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
}
