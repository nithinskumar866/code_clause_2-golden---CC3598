using System.Threading.Channels;
using JobPortal.Api.Services.Rag.Data;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Rag.Ingestion;

/// <summary>
/// One document waiting to be chunked and embedded.
///
/// Carries only ids. The text is read at processing time, not captured here: a
/// document that changed between being queued and being processed should be
/// indexed as it is NOW, and holding a megabyte of text per queued item would put
/// the backlog in memory instead of in the queue.
/// </summary>
public record IngestJob(int ParentId, string ParentType)
{
    public static IngestJob Resume(int id) => new(id, RagParentTypes.Resume);
    public static IngestJob Job(int id) => new(id, RagParentTypes.Job);
}

/// <summary>
/// The ingestion backlog.
///
/// Bounded, and that is the point: an unbounded channel converts a burst of
/// uploads into unbounded memory, and the failure surfaces long after the cause.
/// At capacity the writer waits, which pushes back on whatever is producing —
/// the correct behaviour for a backfill of a hundred thousand documents.
///
/// Deduplicated by (type, id) while an item is still waiting, so re-queuing a
/// document that has not been processed yet does not make the worker do it twice.
/// </summary>
public class RagIngestionQueue
{
    private readonly Channel<IngestJob> _channel;
    private readonly HashSet<IngestJob> _pending = new();
    private readonly object _gate = new();

    public RagIngestionQueue(IOptions<RagOptions> options)
    {
        _channel = Channel.CreateBounded<IngestJob>(new BoundedChannelOptions(
            Math.Max(16, options.Value.IngestQueueCapacity))
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.Wait,
        });
    }

    /// <summary>How many documents are waiting. Surfaced on the health endpoint —
    /// a queue that only grows is the first sign ingestion cannot keep up.</summary>
    public int Depth
    {
        get { lock (_gate) return _pending.Count; }
    }

    /// <summary>Queues a document, unless the same one is already waiting.</summary>
    public async ValueTask<bool> EnqueueAsync(IngestJob job, CancellationToken ct = default)
    {
        lock (_gate)
        {
            if (!_pending.Add(job)) return false;
        }

        try
        {
            await _channel.Writer.WriteAsync(job, ct);
            return true;
        }
        catch
        {
            // Never leave a phantom entry behind: it would block this document
            // from ever being queued again.
            lock (_gate) _pending.Remove(job);
            throw;
        }
    }

    public async IAsyncEnumerable<IngestJob> ReadAllAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var job in _channel.Reader.ReadAllAsync(ct))
        {
            // Removed on DEQUEUE rather than on completion, so a document that
            // changes while it is being processed can be queued again for the
            // newer content instead of being swallowed as a duplicate.
            lock (_gate) _pending.Remove(job);
            yield return job;
        }
    }
}
