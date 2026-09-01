using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using JobPortal.Api.Options;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Llm;

/// <summary>
/// Talks the Ollama HTTP API. Chat and embeddings may live on different hosts —
/// a cloud endpoint that serves GLM will not necessarily serve an embedding model,
/// and the portal must not lose semantic matching because of that.
///
/// Every method fails soft. The portal's matching, extraction and reply generation
/// all have deterministic implementations; the LLM makes them better, and is never
/// allowed to make them unavailable.
/// </summary>
public class OllamaClient : IOllamaClient
{
    public const string ChatHttpClient = "ollama-chat";
    public const string EmbedHttpClient = "ollama-embed";

    private readonly IHttpClientFactory _factory;
    private readonly OllamaOptions _options;
    private readonly ILogger<OllamaClient> _logger;

    public OllamaClient(
        IHttpClientFactory factory,
        IOptions<OllamaOptions> options,
        ILogger<OllamaClient> logger)
    {
        _factory = factory;
        _options = options.Value;
        _logger = logger;
    }

    // -- the breaker --------------------------------------------------------
    //
    // Static, so it is shared by every scope and every request in the process.
    // The thing being tracked is the state of ONE remote endpoint, which is not a
    // per-request fact, and a breaker rebuilt with each scoped client would never
    // see the second failure.

    /// <summary>
    /// When the endpoint may be tried again after a timeout.
    ///
    /// A reasoned shortlist calls the model once per role, and with the endpoint
    /// hung EVERY call waited the full timeout before failing — three roles meant
    /// six minutes of silence where the honest answer was available after the
    /// first two. Measured against a pod whose GPU was held by another model:
    /// each call sat for 120s and returned nothing.
    ///
    /// So the first timeout is believed. Calls inside the cooldown fail instantly
    /// and their callers take the deterministic path, which is exactly what they
    /// would have done two minutes later anyway.
    /// </summary>
    private static long _retryAfterTicks;

    /// <summary>
    /// How long the endpoint is presumed down after a timeout.
    ///
    /// Short: a pod that was evicting a model recovers in well under a minute, and
    /// a breaker that stays open for five would keep answering deterministically
    /// long after the model came back.
    /// </summary>
    private static readonly TimeSpan BreakerCooldown = TimeSpan.FromSeconds(45);

    private static bool BreakerOpen => DateTime.UtcNow.Ticks < Interlocked.Read(ref _retryAfterTicks);

    private void TripBreaker(string reason)
    {
        Interlocked.Exchange(ref _retryAfterTicks, DateTime.UtcNow.Add(BreakerCooldown).Ticks);
        _logger.LogWarning(
            "LLM endpoint marked unavailable for {Seconds}s ({Reason}). " +
            "Callers fall back to the deterministic path until then.",
            BreakerCooldown.TotalSeconds, reason);
    }

    /// <summary>Cleared as soon as any call succeeds, so recovery needs no timer.</summary>
    private static void ResetBreaker() => Interlocked.Exchange(ref _retryAfterTicks, 0);

    public bool ChatAvailable => _options.Enabled && !BreakerOpen;
    public bool EmbeddingsAvailable => _options.EmbeddingsEnabled;
    public string ChatModel => _options.ChatModel;
    public string EmbedModel => _options.EmbedModel;

    public string GetModelName() => _options.ChatModel;

    // -- chat ---------------------------------------------------------------

    public async IAsyncEnumerable<string> StreamChatAsync(
        IReadOnlyList<ChatTurn> messages,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (!ChatAvailable) yield break;

        var payload = BuildChatPayload(messages, stream: true, jsonMode: false);
        var client = _factory.CreateClient(ChatHttpClient);

        HttpResponseMessage response;
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, "api/chat")
            {
                Content = new StringContent(payload, Encoding.UTF8, "application/json"),
            };
            response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Ollama chat stream could not be opened; caller falls back.");
            TripBreaker("stream could not be opened");
            yield break;
        }

        using (response)
        {
            if (!response.IsSuccessStatusCode)
            {
                var body = await SafeReadAsync(response, ct);
                _logger.LogWarning("Ollama chat stream returned {Status}: {Body}", (int)response.StatusCode, body);
                if ((int)response.StatusCode is 502 or 503 or 504 or 524)
                    TripBreaker($"stream endpoint returned {(int)response.StatusCode}");
                yield break;
            }

            await using var stream = await response.Content.ReadAsStreamAsync(ct);
            using var reader = new StreamReader(stream);

            // NDJSON: one JSON object per line, each carrying the next fragment.
            while (!reader.EndOfStream)
            {
                ct.ThrowIfCancellationRequested();

                string? line;
                try { line = await reader.ReadLineAsync(ct); }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    _logger.LogWarning(ex, "Ollama chat stream broke mid-response.");
                    yield break;
                }

                if (string.IsNullOrWhiteSpace(line)) continue;

                string? fragment = null;
                var done = false;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    if (root.TryGetProperty("error", out var error))
                    {
                        _logger.LogWarning("Ollama chat stream error: {Error}", error.ToString());
                        yield break;
                    }
                    if (root.TryGetProperty("message", out var message) &&
                        message.TryGetProperty("content", out var content))
                    {
                        fragment = content.GetString();
                    }
                    done = root.TryGetProperty("done", out var d) && d.ValueKind == JsonValueKind.True;
                }
                catch (JsonException)
                {
                    // A partial line is not worth aborting the whole reply over.
                    continue;
                }

                if (!string.IsNullOrEmpty(fragment)) yield return fragment;
                if (done) yield break;
            }
        }
    }

    public async Task<string?> ChatAsync(
        IReadOnlyList<ChatTurn> messages,
        bool jsonMode = false,
        double? temperature = null,
        CancellationToken ct = default)
    {
        if (!ChatAvailable) return null;

        try
        {
            var client = _factory.CreateClient(ChatHttpClient);
            var payload = BuildChatPayload(messages, stream: false, jsonMode, temperature);
            using var content = new StringContent(payload, Encoding.UTF8, "application/json");
            using var response = await client.PostAsync("api/chat", content, ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("Ollama chat returned {Status}: {Body}",
                    (int)response.StatusCode, await SafeReadAsync(response, ct));

                // A gateway timeout is the proxy giving up on a request the model
                // never answered — the same condition as our own timeout, reported
                // by whoever gave up first. 524 is Cloudflare's; RunPod fronts the
                // pod with one, and it fires at ~125s against our 120s.
                if ((int)response.StatusCode is 502 or 503 or 504 or 524)
                    TripBreaker($"endpoint returned {(int)response.StatusCode}");

                return null;
            }

            var body = await response.Content.ReadAsStringAsync(ct);
            using var doc = JsonDocument.Parse(body);
            if (doc.RootElement.TryGetProperty("message", out var message) &&
                message.TryGetProperty("content", out var text))
            {
                ResetBreaker();
                return text.GetString();
            }
            return null;
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            TripBreaker($"chat timed out after {_options.TimeoutSeconds}s");
            // An HttpClient TIMEOUT throws TaskCanceledException, which is an
            // OperationCanceledException — so the filter below deliberately let it
            // through, on the assumption that a cancellation always meant the caller
            // had gone away. It does not. A slow endpoint then escaped every
            // fallback in the stack and surfaced as "The resume could not be
            // processed", which is why enabling the model broke ingestion that had
            // worked fine while it was switched off.
            //
            // The caller's token is the discriminator: if IT was not cancelled, the
            // deadline was ours and this is an ordinary failure to fall back from.
            _logger.LogWarning(
                "Ollama chat timed out after {Timeout}s; caller falls back to the deterministic path.",
                _options.TimeoutSeconds);
            return null;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Ollama chat call failed; caller falls back to the deterministic path.");
            return null;
        }
    }

    private string BuildChatPayload(
        IReadOnlyList<ChatTurn> messages, bool stream, bool jsonMode, double? temperature = null)
    {
        var buffer = new MemoryStream();
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            w.WriteString("model", _options.ChatModel);
            w.WriteBoolean("stream", stream);
            if (jsonMode) w.WriteString("format", "json");

            // Keep the model resident between turns.
            //
            // Measured on the RunPod box: two calls minutes apart BOTH reported a
            // load_duration of ~8.9s against ~0.7s of actual inference — the weights
            // were evicted in between, so 93% of the wall clock was reloading a model
            // that had just been used. Three models share that box's memory, so this
            // happens on every call unless we ask for otherwise. Without it the
            // endpoint feels slow for reasons that have nothing to do with its speed.
            if (!string.IsNullOrWhiteSpace(_options.KeepAlive))
                w.WriteString("keep_alive", _options.KeepAlive);

            w.WriteStartArray("messages");
            foreach (var m in messages)
            {
                w.WriteStartObject();
                w.WriteString("role", m.Role);
                w.WriteString("content", m.Content);
                w.WriteEndObject();
            }
            w.WriteEndArray();

            w.WriteStartObject("options");
            w.WriteNumber("temperature", temperature ?? _options.Temperature);
            w.WriteEndObject();

            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    // -- embeddings ---------------------------------------------------------

    public async Task<IReadOnlyList<float[]>?> EmbedAsync(
        IReadOnlyList<string> inputs,
        CancellationToken ct = default)
    {
        if (!EmbeddingsAvailable || inputs.Count == 0) return null;

        var client = _factory.CreateClient(EmbedHttpClient);

        // Current API: /api/embed takes a batch. Older builds only expose
        // /api/embeddings, one prompt per call — worth supporting, because the
        // alternative is losing semantic matching against an older server.
        var batched = await TryBatchEmbedAsync(client, inputs, ct);
        if (batched is not null) return batched;

        return await TryLegacyEmbedAsync(client, inputs, ct);
    }

    private async Task<IReadOnlyList<float[]>?> TryBatchEmbedAsync(
        HttpClient client, IReadOnlyList<string> inputs, CancellationToken ct)
    {
        try
        {
            var payload = JsonSerializer.Serialize(new { model = _options.EmbedModel, input = inputs });
            using var content = new StringContent(payload, Encoding.UTF8, "application/json");
            using var response = await client.PostAsync("api/embed", content, ct);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogDebug("/api/embed returned {Status}; trying the legacy endpoint.",
                    (int)response.StatusCode);
                return null;
            }

            using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
            if (!doc.RootElement.TryGetProperty("embeddings", out var arr) ||
                arr.ValueKind != JsonValueKind.Array)
            {
                return null;
            }

            var vectors = arr.EnumerateArray().Select(ReadVector).ToList();
            return Validate(vectors, inputs.Count);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            // A timeout, not the caller leaving. See the note in ChatAsync.
            _logger.LogDebug("/api/embed timed out; trying the legacy endpoint.");
            return null;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogDebug(ex, "/api/embed failed; trying the legacy endpoint.");
            return null;
        }
    }

    private async Task<IReadOnlyList<float[]>?> TryLegacyEmbedAsync(
        HttpClient client, IReadOnlyList<string> inputs, CancellationToken ct)
    {
        try
        {
            var vectors = new List<float[]>(inputs.Count);
            foreach (var input in inputs)
            {
                var payload = JsonSerializer.Serialize(new { model = _options.EmbedModel, prompt = input });
                using var content = new StringContent(payload, Encoding.UTF8, "application/json");
                using var response = await client.PostAsync("api/embeddings", content, ct);

                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogWarning("Embedding endpoint returned {Status}: {Body}. " +
                        "Falling back to the deterministic embedder.",
                        (int)response.StatusCode, await SafeReadAsync(response, ct));
                    return null;
                }

                using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
                if (!doc.RootElement.TryGetProperty("embedding", out var arr)) return null;
                vectors.Add(ReadVector(arr));
            }
            return Validate(vectors, inputs.Count);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            _logger.LogWarning(
                "Embedding call timed out after {Timeout}s; falling back to the deterministic embedder.",
                _options.TimeoutSeconds);
            return null;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Embedding call failed; falling back to the deterministic embedder.");
            return null;
        }
    }

    private static float[] ReadVector(JsonElement element) =>
        element.ValueKind == JsonValueKind.Array
            ? element.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray()
            : Array.Empty<float>();

    /// <summary>
    /// A configured dimension that disagrees with the model's real width means
    /// every stored vector would be compared against differently-shaped queries.
    /// Refusing the batch here turns a silently wrong ranking into a visible
    /// fallback.
    /// </summary>
    private IReadOnlyList<float[]>? Validate(List<float[]> vectors, int expectedCount)
    {
        if (vectors.Count != expectedCount || vectors.Any(v => v.Length == 0)) return null;

        var width = vectors[0].Length;
        if (vectors.Any(v => v.Length != width))
        {
            _logger.LogWarning("Embedding endpoint returned vectors of differing widths; rejecting the batch.");
            return null;
        }

        if (width != _options.EmbedDimension)
        {
            _logger.LogWarning(
                "Embedding model '{Model}' returned {Actual} dimensions but Ollama:EmbedDimension is {Configured}. " +
                "Update the configured dimension — the portal is using the deterministic embedder until you do.",
                _options.EmbedModel, width, _options.EmbedDimension);
            return null;
        }

        return vectors;
    }

    // -- diagnostics --------------------------------------------------------

    public async Task<IReadOnlyList<string>> ListModelsAsync(CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(_options.BaseUrl)) return Array.Empty<string>();
        try
        {
            var client = _factory.CreateClient(ChatHttpClient);
            using var response = await client.GetAsync("api/tags", ct);
            if (!response.IsSuccessStatusCode) return Array.Empty<string>();

            using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
            if (!doc.RootElement.TryGetProperty("models", out var models)) return Array.Empty<string>();

            return models.EnumerateArray()
                .Select(m => m.TryGetProperty("name", out var n) ? n.GetString() : null)
                .Where(n => !string.IsNullOrWhiteSpace(n))
                .Select(n => n!)
                .ToList();
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            _logger.LogDebug("Listing models timed out.");
            return Array.Empty<string>();
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogDebug(ex, "Could not list models from the Ollama endpoint.");
            return Array.Empty<string>();
        }
    }

    private static async Task<string> SafeReadAsync(HttpResponseMessage response, CancellationToken ct)
    {
        try
        {
            var body = await response.Content.ReadAsStringAsync(ct);
            return body.Length > 400 ? body[..400] : body;
        }
        catch { return "<unreadable>"; }
    }

    /// <summary>Applies auth and timeout to a named client. Called from Program.cs
    /// so the wiring for both endpoints stays in one place.</summary>
    public static void Configure(HttpClient client, string baseUrl, string apiKey, int timeoutSeconds)
    {
        if (!string.IsNullOrWhiteSpace(baseUrl))
        {
            // A base address without a trailing slash silently drops its last
            // path segment when a relative URI is appended.
            client.BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/");
        }
        client.Timeout = TimeSpan.FromSeconds(timeoutSeconds);
        if (!string.IsNullOrWhiteSpace(apiKey))
        {
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        }
    }
}
