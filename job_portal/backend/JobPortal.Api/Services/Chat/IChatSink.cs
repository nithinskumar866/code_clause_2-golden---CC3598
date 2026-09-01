using JobPortal.Api.Contracts;

namespace JobPortal.Api.Services.Chat;

/// <summary>
/// Where a reply goes as it is produced.
///
/// The orchestrator writes to this rather than to SignalR directly, so the
/// conversation logic has no idea whether it is talking to a WebSocket, a test,
/// or an HTTP response. That matters: streaming is the hardest part of this
/// feature to test, and a pipeline that can only run inside a hub cannot be
/// tested at all.
/// </summary>
public interface IChatSink
{
    /// <summary>A step in the visible reasoning ("Reading your resume...").</summary>
    Task ThoughtAsync(string stage, string text, CancellationToken ct = default);

    /// <summary>The next fragment of the reply text.</summary>
    Task TokenAsync(string token, CancellationToken ct = default);

    /// <summary>The structured match result, sent alongside the prose so the UI can
    /// render real cards instead of parsing markdown.</summary>
    Task MatchesAsync(MatchResultDto matches, CancellationToken ct = default);

    /// <summary>
    /// Postings offered to someone with no CV yet.
    ///
    /// A separate channel from <see cref="MatchesAsync"/> on purpose: these carry
    /// no fit score, and sending them as matches with zeroed scores would put a
    /// number on screen that is about nobody.
    /// </summary>
    Task SuggestionsAsync(JobSuggestionResultDto suggestions, CancellationToken ct = default);

    /// <summary>
    /// Things worth asking next. Distinct from <see cref="SuggestionsAsync"/>, which
    /// offers JOB POSTINGS — these are QUESTIONS, and a client that confused the two
    /// would render roles as prompts.
    /// </summary>
    Task FollowUpsAsync(IReadOnlyList<FollowUpDto> followUps, CancellationToken ct = default);

    /// <summary>A command for the UI: apply filters, open a job, switch view.</summary>
    Task ActionAsync(UiActionDto action, CancellationToken ct = default);

    /// <summary>The turn is over; carries the assembled reply for the transcript.</summary>
    Task CompleteAsync(string fullText, CancellationToken ct = default);

    Task ErrorAsync(string message, CancellationToken ct = default);
}

/// <summary>Collects everything into memory. Used by the REST fallback endpoint
/// and by tests.</summary>
public class BufferingChatSink : IChatSink
{
    private readonly System.Text.StringBuilder _text = new();

    public List<ThoughtDto> Thoughts { get; } = new();
    public List<UiActionDto> Actions { get; } = new();
    public MatchResultDto? Matches { get; private set; }
    public JobSuggestionResultDto? Suggestions { get; private set; }
    public string? Error { get; private set; }
    public string Text => _text.ToString();

    public Task ThoughtAsync(string stage, string text, CancellationToken ct = default)
    {
        Thoughts.Add(new ThoughtDto(stage, text));
        return Task.CompletedTask;
    }

    public Task TokenAsync(string token, CancellationToken ct = default)
    {
        _text.Append(token);
        return Task.CompletedTask;
    }

    public Task MatchesAsync(MatchResultDto matches, CancellationToken ct = default)
    {
        Matches = matches;
        return Task.CompletedTask;
    }

    public Task SuggestionsAsync(JobSuggestionResultDto suggestions, CancellationToken ct = default)
    {
        Suggestions = suggestions;
        return Task.CompletedTask;
    }

    public IReadOnlyList<FollowUpDto> FollowUps { get; private set; } = Array.Empty<FollowUpDto>();

    public Task FollowUpsAsync(IReadOnlyList<FollowUpDto> followUps, CancellationToken ct = default)
    {
        FollowUps = followUps;
        return Task.CompletedTask;
    }

    public Task ActionAsync(UiActionDto action, CancellationToken ct = default)
    {
        // BufferingChatSink accepts all actions for testing purposes
        Actions.Add(action);
        return Task.CompletedTask;
    }

    public Task CompleteAsync(string fullText, CancellationToken ct = default) => Task.CompletedTask;

    public Task ErrorAsync(string message, CancellationToken ct = default)
    {
        Error = message;
        return Task.CompletedTask;
    }
}
