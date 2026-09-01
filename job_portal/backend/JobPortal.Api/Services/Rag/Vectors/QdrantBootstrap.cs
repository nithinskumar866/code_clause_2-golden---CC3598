using JobPortal.Api.Services.Rag;
using Microsoft.Extensions.Options;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace JobPortal.Api.Services.Rag.Vectors;

/// <summary>
/// Creates the Qdrant collections the RAG mode needs, if they are absent.
///
/// Runs once at startup and is idempotent — an existing collection is left
/// exactly as it is, because re-creating one would silently discard every vector
/// in it.
///
/// **Everything addresses an ALIAS, never a physical collection.** The alias
/// <c>resume_chunks</c> points at <c>resume_chunks_v1</c>; re-embedding under a
/// new model builds <c>resume_chunks_v2</c> beside it and repoints the alias in
/// one atomic operation when the build finishes. That is what makes changing
/// embedding model a zero-downtime operation instead of a maintenance window, and
/// it means a build that fails half-way leaves the old index serving rather than
/// a half-populated one.
/// </summary>
public class QdrantBootstrap
{
    private readonly QdrantClient _client;
    private readonly RagOptions _options;
    private readonly IEmbeddingDimensionSource _dimension;
    private readonly ILogger<QdrantBootstrap> _logger;

    public QdrantBootstrap(
        QdrantClient client,
        IOptions<RagOptions> options,
        IEmbeddingDimensionSource dimension,
        ILogger<QdrantBootstrap> logger)
    {
        _client = client;
        _options = options.Value;
        _dimension = dimension;
        _logger = logger;
    }

    public async Task EnsureCollectionsAsync(CancellationToken ct = default)
    {
        await EnsureAsync(_options.ResumeCollection, ct);
        await EnsureAsync(_options.JobCollection, ct);
    }

    private async Task EnsureAsync(string alias, CancellationToken ct)
    {
        var aliases = await _client.ListAliasesAsync(cancellationToken: ct);
        if (aliases.Any(a => a.AliasName == alias))
        {
            _logger.LogDebug("Qdrant alias '{Alias}' already exists.", alias);
            return;
        }

        var physical = $"{alias}_v1";
        var dimension = (ulong)_dimension.Dimension;

        if (!await _client.CollectionExistsAsync(physical, ct))
        {
            await _client.CreateCollectionAsync(
                physical,
                new VectorParams
                {
                    Size = dimension,
                    Distance = Distance.Cosine,

                    // Scalar int8 quantization: ~4x less memory for a recall cost
                    // small enough to be invisible at this dimensionality. 1.5M
                    // vectors at 768d is 4.6 GB raw and ~1.2 GB quantized, which
                    // is the difference between "fits in RAM" and "does not".
                    QuantizationConfig = new QuantizationConfig
                    {
                        Scalar = new ScalarQuantization
                        {
                            Type = QuantizationType.Int8,
                            AlwaysRam = true,
                        },
                    },
                },
                // m=16 / ef_construct=128 are Qdrant's balanced defaults, stated
                // explicitly so a future change is a decision rather than a drift.
                hnswConfig: new HnswConfigDiff { M = 16, EfConstruct = 128 },
                cancellationToken: ct);

            _logger.LogInformation(
                "Created Qdrant collection '{Collection}' ({Dimension}d, cosine, int8).",
                physical, dimension);
        }

        await _client.CreateAliasAsync(alias, physical, cancellationToken: ct);
        _logger.LogInformation("Qdrant alias '{Alias}' -> '{Collection}'.", alias, physical);

        // Payload indexes. Without these a filter is a scan of every point's
        // payload; with them the filter runs inside the HNSW traversal, which is
        // the entire reason for choosing a vector database over the in-memory
        // scan the portal does today.
        //
        // The schema type has to match how the value is stored, or the filter
        // silently matches nothing: parent_id is written as a number, the rest as
        // strings.
        await _client.CreatePayloadIndexAsync(
            physical, RagPayload.ParentId, PayloadSchemaType.Integer, cancellationToken: ct);

        foreach (var field in new[] { RagPayload.ParentType, RagPayload.Section })
        {
            await _client.CreatePayloadIndexAsync(
                physical, field, PayloadSchemaType.Keyword, cancellationToken: ct);
        }
    }
}

/// <summary>
/// The payload keys stored beside every vector.
///
/// Named constants because they are written by the ingester and read by the
/// retriever, and a typo between the two produces a filter that silently matches
/// nothing rather than an error.
/// </summary>
public static class RagPayload
{
    public const string ParentId = "parent_id";
    public const string ParentType = "parent_type";
    public const string Section = "section";
    public const string Ordinal = "ordinal";
    public const string Page = "page";
}

/// <summary>
/// Where the vector width comes from.
///
/// An interface rather than a config value because the width is a property of the
/// embedding model, and the collection must be created with the width the model
/// actually returns — a mismatch is only discovered on first search, as every
/// query scoring zero.
/// </summary>
public interface IEmbeddingDimensionSource
{
    int Dimension { get; }
}
