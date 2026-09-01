using JobPortal.Api.Services.Llm;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace JobPortal.Api.Tests;

/// <summary>
/// An <see cref="IOllamaClient"/> that is never available.
///
/// Every test here runs the deterministic path deliberately. That path has to
/// work on its own — it is what the portal falls back to whenever the endpoint is
/// missing — and testing against a live model would make these tests depend on a
/// network and on whatever a model felt like returning that day.
/// </summary>
internal class OfflineLlm : IOllamaClient
{
    public bool ChatAvailable => false;
    public bool EmbeddingsAvailable => false;
    public string ChatModel => "";
    public string EmbedModel => "";

    public async IAsyncEnumerable<string> StreamChatAsync(
        IReadOnlyList<ChatTurn> messages,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.CompletedTask;
        yield break;
    }

    public Task<string?> ChatAsync(IReadOnlyList<ChatTurn> m, bool jsonMode = false, double? temperature = null, CancellationToken ct = default)
        => Task.FromResult<string?>(null);

    public Task<IReadOnlyList<float[]>?> EmbedAsync(IReadOnlyList<string> i, CancellationToken ct = default)
        => Task.FromResult<IReadOnlyList<float[]>?>(null);

    public Task<IReadOnlyList<string>> ListModelsAsync(CancellationToken ct = default)
        => Task.FromResult<IReadOnlyList<string>>(Array.Empty<string>());

    public string GetModelName() => "offline";
}

/// <summary>An LLM that answers with whatever the test scripted.</summary>
internal class ScriptedLlm : IOllamaClient
{
    private readonly string? _reply;
    public ScriptedLlm(string? reply) => _reply = reply;

    public bool ChatAvailable => true;
    public bool EmbeddingsAvailable => false;
    public string ChatModel => "scripted";
    public string EmbedModel => "";

    public async IAsyncEnumerable<string> StreamChatAsync(
        IReadOnlyList<ChatTurn> messages,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.CompletedTask;
        yield break;
    }

    public Task<string?> ChatAsync(IReadOnlyList<ChatTurn> m, bool jsonMode = false, double? temperature = null, CancellationToken ct = default)
        => Task.FromResult(_reply);

    public Task<IReadOnlyList<float[]>?> EmbedAsync(IReadOnlyList<string> i, CancellationToken ct = default)
        => Task.FromResult<IReadOnlyList<float[]>?>(null);

    public Task<IReadOnlyList<string>> ListModelsAsync(CancellationToken ct = default)
        => Task.FromResult<IReadOnlyList<string>>(Array.Empty<string>());

    public string GetModelName() => "scripted";
}