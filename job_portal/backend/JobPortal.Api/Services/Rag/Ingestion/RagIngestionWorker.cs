using JobPortal.Api.Data;
using JobPortal.Api.Services.Rag.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Rag.Ingestion;

/// <summary>
/// Runs the ingestion backlog, and keeps the index in step with the portal.
///
/// Two jobs in one hosted service:
///
///   1. DRAIN — take documents off the queue and ingest them, one at a time. Not
///      parallel: the embedding endpoint is one GPU, and several concurrent
///      batches do not finish in a fraction of the time, they finish at roughly
///      the same moment while making eviction more likely.
///
///   2. RECONCILE — periodically compare what the portal holds against what has
///      been indexed, and queue the difference.
///
/// The sweep is why no existing controller had to change. An upload could have
/// enqueued directly, but that means editing the endpoints the two working scoring
/// modes depend on. A sweep needs none of that, is self-healing after any outage,
/// and collapses backfill and steady-state into one mechanism. The cost is
/// latency: a new CV becomes searchable within one interval rather than instantly.
/// </summary>
public class RagIngestionWorker : BackgroundService
{
    /// <summary>
    /// How often the portal is compared against the index.
    ///
    /// Short enough that a newly uploaded CV is searchable while the candidate is
    /// still in the session; long enough that the query costs nothing at rest.
    /// </summary>
    private static readonly TimeSpan SweepInterval = TimeSpan.FromSeconds(60);

    /// <summary>Documents queued per sweep. A first run against 1L documents must
    /// feed the queue steadily rather than in one enormous burst.</summary>
    private const int SweepBatch = 500;

    private readonly IServiceScopeFactory _scopes;
    private readonly RagIngestionQueue _queue;
    private readonly RagOptions _options;
    private readonly ILogger<RagIngestionWorker> _logger;

    public RagIngestionWorker(
        IServiceScopeFactory scopes,
        RagIngestionQueue queue,
        IOptions<RagOptions> options,
        ILogger<RagIngestionWorker> logger)
    {
        _scopes = scopes;
        _queue = queue;
        _options = options.Value;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!_options.Enabled) return;

        _logger.LogInformation("RAG ingestion worker started.");

        // Both loops run for the life of the process. Neither may take the other
        // down: a reconciliation query that fails must not stop the drain.
        var drain = DrainAsync(stoppingToken);
        var reconcile = ReconcileLoopAsync(stoppingToken);

        await Task.WhenAll(drain, reconcile);
    }

    private async Task DrainAsync(CancellationToken ct)
    {
        try
        {
            await foreach (var job in _queue.ReadAllAsync(ct))
            {
                try
                {
                    // A fresh scope per document: the DbContexts are scoped, and
                    // one held for the life of the worker would accumulate every
                    // entity it ever tracked.
                    using var scope = _scopes.CreateScope();
                    var ingest = scope.ServiceProvider.GetRequiredService<RagIngestionService>();

                    var outcome = await ingest.IngestAsync(job, ct);

                    if (outcome.Indexed)
                    {
                        _logger.LogDebug("Indexed {Type} {Id} as {Chunks} chunks.",
                            job.ParentType, job.ParentId, outcome.Chunks);
                    }
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    // One poisonous document must not stop the backlog. The
                    // attempt count in the ingest-state row is what eventually
                    // parks it.
                    _logger.LogError(ex, "Ingesting {Type} {Id} threw; continuing.",
                        job.ParentType, job.ParentId);
                }
            }
        }
        catch (OperationCanceledException) { /* shutting down */ }
    }

    private async Task ReconcileLoopAsync(CancellationToken ct)
    {
        // A short delay so the first sweep does not race the startup migration.
        try { await Task.Delay(TimeSpan.FromSeconds(5), ct); }
        catch (OperationCanceledException) { return; }

        while (!ct.IsCancellationRequested)
        {
            try
            {
                var queued = await ReconcileAsync(ct);
                if (queued > 0) _logger.LogInformation("Reconciliation queued {Count} document(s).", queued);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { return; }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Reconciliation sweep failed; retrying next interval.");
            }

            try { await Task.Delay(SweepInterval, ct); }
            catch (OperationCanceledException) { return; }
        }
    }

    /// <summary>
    /// Queues everything the portal holds that the index does not.
    ///
    /// Compares ids rather than content: a document with no ingest-state row has
    /// never been seen, and one whose row is <c>pending</c> was interrupted. A row
    /// marked <c>failed</c> is deliberately skipped — it exhausted its attempts,
    /// and re-queuing it every minute would bury the working backlog under the
    /// same error.
    ///
    /// Content changes are caught by the hash comparison inside the ingester, not
    /// here: this pass would otherwise have to read every document's text on every
    /// sweep, which at 1L documents is a scan nobody needs once a minute.
    /// </summary>
    public async Task<int> ReconcileAsync(CancellationToken ct)
    {
        using var scope = _scopes.CreateScope();
        var portal = scope.ServiceProvider.GetRequiredService<PortalDbContext>();
        var rag = scope.ServiceProvider.GetRequiredService<RagDbContext>();

        var queued = 0;

        queued += await SweepAsync(
            RagParentTypes.Job,
            await portal.Jobs.AsNoTracking().Where(j => j.IsPublished).Select(j => j.Id).ToListAsync(ct),
            rag, ct);

        queued += await SweepAsync(
            RagParentTypes.Resume,
            await portal.Resumes.AsNoTracking().Select(r => r.Id).ToListAsync(ct),
            rag, ct);

        return queued;
    }

    private async Task<int> SweepAsync(
        string parentType, List<int> portalIds, RagDbContext rag, CancellationToken ct)
    {
        if (portalIds.Count == 0) return 0;

        var settled = await rag.IngestState
            .AsNoTracking()
            .Where(s => s.ParentType == parentType &&
                        (s.Status == RagIngestStatuses.Indexed || s.Status == RagIngestStatuses.Failed))
            .Select(s => s.ParentId)
            .ToListAsync(ct);

        var outstanding = portalIds.Except(settled).Take(SweepBatch).ToList();

        var queued = 0;
        foreach (var id in outstanding)
        {
            if (ct.IsCancellationRequested) break;
            if (await _queue.EnqueueAsync(new IngestJob(id, parentType), ct)) queued++;
        }

        return queued;
    }
}
