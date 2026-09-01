namespace JobPortal.Api.Services.Vectors;

/// <summary>
/// A batch of vectors together with the identity of whatever actually produced
/// them.
///
/// The model id travels with the vectors rather than sitting on the service
/// because the service can fall back mid-run: if the embedding endpoint dies
/// halfway through indexing, the jobs embedded before and after the failure came
/// from different models and their vectors are not comparable. Returning the id
/// per call is what lets the store keep them apart instead of silently mixing two
/// vector spaces into one ranking.
/// </summary>
public record EmbeddingBatch(
    string ModelId,
    int Dimension,
    bool IsSemantic,
    IReadOnlyList<float[]> Vectors);

public interface IEmbeddingService
{
    /// <summary>The model the service will try first. Not a promise: a call can
    /// still come back from the fallback.</summary>
    string PreferredModelId { get; }

    /// <summary>True when the preferred model is a real embedding model, i.e. when
    /// semantic (rather than merely lexical) matching is available.</summary>
    bool SemanticAvailable { get; }

    Task<EmbeddingBatch> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default);

    Task<EmbeddingBatch> EmbedAsync(string text, CancellationToken ct = default);
}
