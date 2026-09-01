using JobPortal.Api.Contracts;
using JobPortal.Api.Services.Chat;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Logging;

namespace JobPortal.Api.Hubs;

/// <summary>
/// Pushes a reply to one browser as it is produced.
///
/// Everything is addressed to <see cref="HubCallerContext.ConnectionId"/> rather
/// than broadcast: two candidates on the same board must never see each other's
/// resumes, matches or messages, and a hub that defaults to Clients.All makes
/// that a one-word mistake away.
/// </summary>
public class ChatHub : Hub
{
    public const string Route = "/hubs/chat";

    private readonly IChatOrchestrator _orchestrator;
    private readonly ILogger<ChatHub> _logger;

    public ChatHub(IChatOrchestrator orchestrator, ILogger<ChatHub> logger)
    {
        _orchestrator = orchestrator;
        _logger = logger;
    }

    /// <summary>
    /// Validates and sends a UI action to the client. Invalid commands are rejected
    /// and logged but not sent to the client.
    /// </summary>
    private async Task SendValidatedActionAsync(UiActionDto action, string sessionId, CancellationToken ct = default)
    {
        var validated = CopilotCommandValidator.Validate(action, sessionId, _logger);
        if (validated is not null)
        {
            await Clients.Caller.SendAsync("UiAction", action, ct);
        }
        // Invalid commands are silently dropped but logged by the validator
    }

    /// <summary>
    /// Sends a message and streams the reply back.
    /// </summary>
    /// <param name="scoringMode">
    /// Which scorer to use for this turn — "computed" or "reasoned". Optional so an
    /// older client that does not send it keeps working; SignalR supplies null and
    /// the session's existing setting stands.
    /// </param>
    public async Task SendMessage(string sessionId, string message, string? scoringMode = null)
    {
        if (string.IsNullOrWhiteSpace(sessionId))
        {
            await Clients.Caller.SendAsync("Error", "No session id was supplied.");
            return;
        }

        var sink = new HubChatSink(Clients.Caller, Context.ConnectionAborted, sessionId, _logger);
        try
        {
            await _orchestrator.HandleMessageAsync(
                sessionId, message ?? "", sink, scoringMode, Context.ConnectionAborted);
        }
        catch (OperationCanceledException)
        {
            // The user closed the tab or navigated away mid-reply. Normal.
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Chat turn failed for session {SessionId}", sessionId);
            await sink.ErrorAsync("Something went wrong handling that message.");
        }
        finally
        {
            await sink.EndAsync();
        }
    }

    /// <summary>
    /// Runs the matching turn for a resume that was just uploaded over HTTP.
    ///
    /// The file goes over REST because a multi-megabyte PDF does not belong in a
    /// WebSocket frame; the narration comes back over the hub because that is the
    /// part the user watches. The client posts the file, then calls this with the
    /// id it got back.
    /// </summary>
    public async Task AnalyzeResume(string sessionId, int resumeId, string? scoringMode = null)
    {
        if (string.IsNullOrWhiteSpace(sessionId))
        {
            await Clients.Caller.SendAsync("Error", "No session id was supplied.");
            return;
        }

        var sink = new HubChatSink(Clients.Caller, Context.ConnectionAborted, sessionId, _logger);
        try
        {
            await _orchestrator.HandleResumeAsync(
                sessionId, resumeId, sink, scoringMode, Context.ConnectionAborted);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Resume analysis failed for session {SessionId}", sessionId);
            await sink.ErrorAsync("Something went wrong analysing that resume.");
        }
        finally
        {
            await sink.EndAsync();
        }
    }

    public override Task OnConnectedAsync()
    {
        _logger.LogDebug("Chat client {ConnectionId} connected", Context.ConnectionId);
        return base.OnConnectedAsync();
    }
}

/// <summary>
/// Adapts the orchestrator's sink onto one SignalR caller.
///
/// The event names here are the client contract, mirrored in the React hook.
/// </summary>
public class HubChatSink : IChatSink
{
    private readonly IClientProxy _client;
    private readonly CancellationToken _connectionAborted;
    private readonly string _sessionId;
    private readonly ILogger _logger;

    public HubChatSink(IClientProxy client, CancellationToken connectionAborted, string sessionId, ILogger logger)
    {
        _client = client;
        _connectionAborted = connectionAborted;
        _sessionId = sessionId;
        _logger = logger;
    }

    public Task ThoughtAsync(string stage, string text, CancellationToken ct = default) =>
        _client.SendAsync("Thought", new ThoughtDto(stage, text), Link(ct));

    public Task TokenAsync(string token, CancellationToken ct = default) =>
        _client.SendAsync("Token", token, Link(ct));

    public Task MatchesAsync(MatchResultDto matches, CancellationToken ct = default) =>
        _client.SendAsync("Matches", matches, Link(ct));

    public Task SuggestionsAsync(JobSuggestionResultDto suggestions, CancellationToken ct = default) =>
        _client.SendAsync("Suggestions", suggestions, Link(ct));

    public Task FollowUpsAsync(IReadOnlyList<FollowUpDto> followUps, CancellationToken ct = default) =>
        _client.SendAsync("FollowUps", followUps, Link(ct));

    public Task ActionAsync(UiActionDto action, CancellationToken ct = default)
    {
        var validated = CopilotCommandValidator.Validate(action, _sessionId, _logger);
        if (validated is not null)
        {
            return _client.SendAsync("UiAction", action, Link(ct));
        }
        // Invalid commands are silently dropped but logged by the validator
        return Task.CompletedTask;
    }

    public Task CompleteAsync(string fullText, CancellationToken ct = default) =>
        _client.SendAsync("Complete", fullText, Link(ct));

    public Task ErrorAsync(string message, CancellationToken ct = default) =>
        _client.SendAsync("Error", message, Link(ct));

    /// <summary>
    /// Signals the end of the turn regardless of how it ended.
    ///
    /// Without this a client that lost the connection mid-reply keeps its typing
    /// indicator up forever, which reads as a hang rather than a dropped
    /// connection.
    /// </summary>
    public Task EndAsync() => _client.SendAsync("End", _connectionAborted);

    private CancellationToken Link(CancellationToken ct) =>
        ct == default ? _connectionAborted : ct;
}
