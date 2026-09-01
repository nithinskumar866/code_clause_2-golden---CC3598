using JobPortal.Api.Contracts;
using JobPortal.Api.Mapping;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Chat;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Resumes;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Controllers;

[ApiController]
[Route("api")]
public class ChatController : ControllerBase
{
    private readonly IResumeService _resumes;
    private readonly IChatOrchestrator _chat;
    private readonly IMatchingService _matching;
    private readonly PortalOptions _portal;
    private readonly ILogger<ChatController> _logger;

    public ChatController(
        IResumeService resumes,
        IChatOrchestrator chat,
        IMatchingService matching,
        IOptions<PortalOptions> portal,
        ILogger<ChatController> logger)
    {
        _resumes = resumes;
        _chat = chat;
        _matching = matching;
        _portal = portal.Value;
        _logger = logger;
    }

    /// <summary>
    /// Uploads a candidate resume.
    ///
    /// Over REST rather than the hub: a multi-megabyte PDF pushed through a
    /// WebSocket would have to be chunked by hand for no benefit. The client then
    /// calls the hub's AnalyzeResume with the returned id, and watches the
    /// narration come back.
    /// </summary>
    [HttpPost("resumes/upload")]
    [RequestSizeLimit(32 * 1024 * 1024)]
    public async Task<ActionResult<ApiResponse<ResumeProfileDto>>> UploadResume(
        IFormFile file, CancellationToken ct)
    {
        if (file is null || file.Length == 0)
            return BadRequest(ApiResponse<ResumeProfileDto>.Fail("No file was uploaded."));

        if (file.Length > _portal.MaxUploadMegabytes * 1024L * 1024L)
        {
            return BadRequest(ApiResponse<ResumeProfileDto>.Fail(
                $"That file is larger than the {_portal.MaxUploadMegabytes} MB limit."));
        }

        try
        {
            await using var stream = file.OpenReadStream();
            var resume = await _resumes.IngestAsync(stream, file.FileName, ct);
            return Ok(ApiResponse<ResumeProfileDto>.Ok(resume.ToDto(), "Resume read."));
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException)
        {
            return BadRequest(ApiResponse<ResumeProfileDto>.Fail(ex.Message));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Resume upload failed for {Filename}", file.FileName);
            return StatusCode(500, ApiResponse<ResumeProfileDto>.Fail("The resume could not be processed."));
        }
    }

    /// <summary>
    /// Starts a fresh conversation on the same session id.
    ///
    /// The CV survives deliberately — see <see cref="IChatOrchestrator.ResetSessionAsync"/>.
    /// </summary>
    [HttpDelete("chat/{sessionId}/messages")]
    public async Task<ActionResult<ApiResponse<ChatSessionDto>>> ResetSession(
        string sessionId, CancellationToken ct) =>
        Ok(ApiResponse<ChatSessionDto>.Ok(
            await _chat.ResetSessionAsync(sessionId, ct), "Conversation cleared."));

    [HttpGet("chat/{sessionId}")]
    public async Task<ActionResult<ApiResponse<ChatSessionDto>>> GetSession(
        string sessionId, CancellationToken ct) =>
        Ok(ApiResponse<ChatSessionDto>.Ok(await _chat.GetOrCreateSessionAsync(sessionId, ct)));

    /// <param name="ScoringMode">
    /// "computed" or "reasoned". Omitted leaves the session's current setting alone.
    /// </param>
    public record ChatMessageRequest(string Message, string? ScoringMode = null);

    public record ChatTurnDto(
        string Reply,
        IReadOnlyList<ThoughtDto> Thoughts,
        IReadOnlyList<UiActionDto> Actions,
        MatchResultDto? Matches);

    /// <summary>
    /// The non-streaming path.
    ///
    /// Kept because a WebSocket is not always available — a corporate proxy, a
    /// server-side integration, a test — and the conversation should not be
    /// reachable only through SignalR. It runs the identical pipeline and buffers
    /// what the hub would have streamed.
    /// </summary>
    [HttpPost("chat/{sessionId}/message")]
    public async Task<ActionResult<ApiResponse<ChatTurnDto>>> SendMessage(
        string sessionId, [FromBody] ChatMessageRequest request, CancellationToken ct)
    {
        var sink = new BufferingChatSink();
        await _chat.HandleMessageAsync(sessionId, request.Message ?? "", sink, request.ScoringMode, ct);

        if (sink.Error is not null)
            return StatusCode(500, ApiResponse<ChatTurnDto>.Fail(sink.Error));

        return Ok(ApiResponse<ChatTurnDto>.Ok(
            new ChatTurnDto(sink.Text, sink.Thoughts, sink.Actions, sink.Matches)));
    }

    [HttpPost("chat/{sessionId}/analyze/{resumeId:int}")]
    public async Task<ActionResult<ApiResponse<ChatTurnDto>>> Analyze(
        string sessionId, int resumeId, CancellationToken ct,
        [FromQuery] string? scoringMode = null)
    {
        var sink = new BufferingChatSink();
        await _chat.HandleResumeAsync(sessionId, resumeId, sink, scoringMode, ct);

        if (sink.Error is not null)
            return StatusCode(500, ApiResponse<ChatTurnDto>.Fail(sink.Error));

        return Ok(ApiResponse<ChatTurnDto>.Ok(
            new ChatTurnDto(sink.Text, sink.Thoughts, sink.Actions, sink.Matches)));
    }

    /// <summary>Matching on its own, with no conversation around it. Useful for a
    /// "see all matches" view and for verifying the pipeline directly.</summary>
    [HttpPost("resumes/{resumeId:int}/matches")]
    public async Task<ActionResult<ApiResponse<MatchResultDto>>> Matches(
        int resumeId, [FromBody] JobFilters? filters, [FromQuery] int? top, CancellationToken ct)
    {
        var resume = await _resumes.GetAsync(resumeId, ct);
        if (resume is null) return NotFound(ApiResponse<MatchResultDto>.Fail($"No resume with id {resumeId}."));

        var result = await _matching.MatchAsync(resume, filters ?? new JobFilters(), top, ct);
        return Ok(ApiResponse<MatchResultDto>.Ok(result, $"{result.Matches.Count} match(es)."));
    }
}
