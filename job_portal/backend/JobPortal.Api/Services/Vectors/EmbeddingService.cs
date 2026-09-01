using JobPortal.Api.Options;
using JobPortal.Api.Services.Llm;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Vectors;

/// <summary>
/// The embedder the rest of the portal talks to. Prefers the configured embedding
/// model and falls back to <see cref="HashingEmbeddingService"/> when it is absent
/// or unreachable.
///
/// The fallback is announced, never disguised: the returned batch says which model
/// produced it and whether that model is semantic, and the health endpoint and the
/// chat both surface that. A recruiter who is getting lexical matches deserves to
/// know why a candidate with "React" did not surface for "frontend framework".
/// </summary>
public class EmbeddingService : IEmbeddingService
{
    private readonly IOllamaClient _ollama;
    private readonly HashingEmbeddingService _fallback;
    private readonly OllamaOptions _options;
    private readonly ILogger<EmbeddingService> _logger;

    /// <summary>Batch size for the remote endpoint. Small enough that one slow or
    /// oversized batch cannot exhaust the request timeout for the whole index.</summary>
    private const int RemoteBatchSize = 16;

    public EmbeddingService(
        IOllamaClient ollama,
        HashingEmbeddingService fallback,
        IOptions<OllamaOptions> options,
        ILogger<EmbeddingService> logger)
    {
        _ollama = ollama;
        _fallback = fallback;
        _options = options.Value;
        _logger = logger;
    }

    public string PreferredModelId =>
        _ollama.EmbeddingsAvailable ? _ollama.EmbedModel : HashingEmbeddingService.Id;

    public bool SemanticAvailable => _ollama.EmbeddingsAvailable;

    public async Task<EmbeddingBatch> EmbedAsync(string text, CancellationToken ct = default) =>
        await EmbedAsync(new[] { text }, ct);

    public async Task<EmbeddingBatch> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        if (texts.Count == 0)
            return new EmbeddingBatch(PreferredModelId, 0, SemanticAvailable, Array.Empty<float[]>());

        if (_ollama.EmbeddingsAvailable)
        {
            var remote = await EmbedRemoteAsync(texts, ct);
            if (remote is not null)
            {
                return new EmbeddingBatch(
                    _ollama.EmbedModel, _options.EmbedDimension, IsSemantic: true, remote);
            }

            _logger.LogWarning(
                "Embedding model '{Model}' was unavailable for this batch; using the deterministic " +
                "embedder. Matching is lexical until the endpoint returns.", _ollama.EmbedModel);
        }

        var vectors = _fallback.EmbedBatch(texts);
        return new EmbeddingBatch(
            HashingEmbeddingService.Id, _fallback.Dimension, IsSemantic: false, vectors);
    }

    private async Task<IReadOnlyList<float[]>?> EmbedRemoteAsync(
        IReadOnlyList<string> texts, CancellationToken ct)
    {
        var all = new List<float[]>(texts.Count);

        for (var offset = 0; offset < texts.Count; offset += RemoteBatchSize)
        {
            var slice = texts.Skip(offset).Take(RemoteBatchSize).ToList();
            var vectors = await _ollama.EmbedAsync(slice, ct);

            // A partial success is not a success. Half a batch from the model and
            // half from the fallback would put two vector spaces in one index, and
            // every comparison across that boundary would be meaningless.
            if (vectors is null) return null;

            all.AddRange(vectors.Select(VectorMath.Normalize));
        }

        return all;
    }
}
