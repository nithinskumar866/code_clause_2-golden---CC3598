namespace JobPortal.Api.Services.Llm;

public record ChatTurn(string Role, string Content);

/// <summary>
/// Access to an Ollama-compatible endpoint (Ollama Cloud, a self-hosted box,
/// RunPod). Deliberately narrow: the portal needs streamed chat, whole-response
/// chat, and embeddings — nothing else.
/// </summary>
public interface IOllamaClient
{
    bool ChatAvailable { get; }
    bool EmbeddingsAvailable { get; }
    string ChatModel { get; }
    string EmbedModel { get; }

    /// <summary>Streams the reply token by token, as the model produces it.</summary>
    IAsyncEnumerable<string> StreamChatAsync(
        IReadOnlyList<ChatTurn> messages,
        CancellationToken ct = default);

    /// <summary>
    /// The whole reply at once. Returns null on any failure — every caller has a
    /// deterministic path to fall back to, and a null is how they learn to take it.
    /// </summary>
    /// <param name="temperature">
    /// Overrides the configured sampling temperature for this call only. Passed by
    /// callers whose output is a JUDGEMENT rather than prose: the same résumé and
    /// the same posting must not score 41% on one ask and 67% on the next, and at
    /// the shared 0.2 they do.
    /// </param>
    Task<string?> ChatAsync(
        IReadOnlyList<ChatTurn> messages,
        bool jsonMode = false,
        double? temperature = null,
        CancellationToken ct = default);

    /// <summary>Embeds a batch. Returns null if the endpoint is unavailable or
    /// disagrees with the configured dimension.</summary>
    Task<IReadOnlyList<float[]>?> EmbedAsync(
        IReadOnlyList<string> inputs,
        CancellationToken ct = default);

    /// <summary>Model tags the endpoint actually serves, for the health endpoint.
    /// Empty if it cannot be reached.</summary>
    Task<IReadOnlyList<string>> ListModelsAsync(CancellationToken ct = default);

    /// <summary>Returns the chat model name for cache versioning.</summary>
    string GetModelName();
}
