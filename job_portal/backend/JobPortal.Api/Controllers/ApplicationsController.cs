using JobPortal.Api.Contracts;
using JobPortal.Api.Services.Applications;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.StaticFiles;

namespace JobPortal.Api.Controllers;

/// <summary>
/// Applying to postings on this board, and the recruiter's view of who applied.
///
/// Preview and submit are separate endpoints on purpose. Applying is
/// outward-facing and cannot be taken back, so the expensive, revealing half
/// (match, letters, duplicates) is a GET-shaped operation the candidate reviews,
/// and only the reviewed result is posted.
/// </summary>
[ApiController]
[Route("api/applications")]
public class ApplicationsController : ControllerBase
{
    private readonly IApplicationService _applications;
    private readonly ILogger<ApplicationsController> _logger;

    public ApplicationsController(
        IApplicationService applications, ILogger<ApplicationsController> logger)
    {
        _applications = applications;
        _logger = logger;
    }

    /// <summary>
    /// What applying would send: one entry per posting, with its letter, its fit
    /// and whether this CV has already been sent there. Nothing is stored.
    /// </summary>
    [HttpPost("preview")]
    public async Task<ActionResult<ApiResponse<ApplicationPreviewDto>>> Preview(
        [FromBody] ApplyPreviewRequest request, CancellationToken ct)
    {
        try
        {
            var preview = await _applications.PreviewAsync(request, ct);
            return Ok(ApiResponse<ApplicationPreviewDto>.Ok(
                preview,
                preview.EligibleCount == 0
                    ? "Nothing new to apply to."
                    : $"{preview.EligibleCount} application(s) ready to review."));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ApiResponse<ApplicationPreviewDto>.Fail(ex.Message));
        }
    }

    /// <summary>
    /// Writes one cover letter, for a role already on screen in the review.
    ///
    /// Separate from preview because a local model takes tens of seconds per
    /// letter: the review opens immediately on composed letters, and this upgrades
    /// them one at a time without holding the panel shut.
    /// </summary>
    [HttpPost("draft-letter")]
    public async Task<ActionResult<ApiResponse<DraftLetterDto>>> DraftLetter(
        [FromBody] DraftLetterRequest request, CancellationToken ct)
    {
        try
        {
            return Ok(ApiResponse<DraftLetterDto>.Ok(
                await _applications.DraftLetterAsync(request, ct), "Letter written."));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ApiResponse<DraftLetterDto>.Fail(ex.Message));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Drafting a letter failed for job {JobId}", request.JobId);
            return StatusCode(500, ApiResponse<DraftLetterDto>.Fail("The letter could not be written."));
        }
    }

    /// <summary>Submits the reviewed applications.</summary>
    [HttpPost]
    public async Task<ActionResult<ApiResponse<ApplySubmitResultDto>>> Submit(
        [FromBody] ApplySubmitRequest request, CancellationToken ct)
    {
        try
        {
            var result = await _applications.SubmitAsync(request, ct);
            return Ok(ApiResponse<ApplySubmitResultDto>.Ok(
                result,
                result.SubmittedCount == 1
                    ? "Application sent."
                    : $"{result.SubmittedCount} application(s) sent."));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ApiResponse<ApplySubmitResultDto>.Fail(ex.Message));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Submitting applications failed for resume {ResumeId}", request.ResumeId);
            return StatusCode(500, ApiResponse<ApplySubmitResultDto>.Fail(
                "The applications could not be sent."));
        }
    }

    /// <summary>The recruiter's inbox: who applied, to what, with what fit.</summary>
    [HttpGet]
    public async Task<ActionResult<ApiResponse<IReadOnlyList<ApplicationDto>>>> List(
        [FromQuery] int? jobId, [FromQuery] string? status, CancellationToken ct)
    {
        var rows = await _applications.ListAsync(jobId, status, ct);
        return Ok(ApiResponse<IReadOnlyList<ApplicationDto>>.Ok(rows, $"{rows.Count} application(s)."));
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<ApiResponse<ApplicationDto>>> Get(int id, CancellationToken ct)
    {
        var row = await _applications.GetAsync(id, ct);
        return row is null
            ? NotFound(ApiResponse<ApplicationDto>.Fail($"No application with id {id}."))
            : Ok(ApiResponse<ApplicationDto>.Ok(row));
    }

    /// <summary>
    /// Records a decision.
    ///
    /// Accepting carries the id of the analysis the PYTHON backend created when the
    /// recruiter took this applicant into the pipeline. This side never writes the
    /// recruiter platform's tables, so the id arrives here rather than being
    /// produced here.
    /// </summary>
    [HttpPatch("{id:int}/status")]
    public async Task<ActionResult<ApiResponse<ApplicationDto>>> SetStatus(
        int id, [FromBody] ApplicationStatusRequest request, CancellationToken ct)
    {
        try
        {
            var row = await _applications.SetStatusAsync(id, request, ct);
            return row is null
                ? NotFound(ApiResponse<ApplicationDto>.Fail($"No application with id {id}."))
                : Ok(ApiResponse<ApplicationDto>.Ok(row, $"Marked {row.Status.ToLowerInvariant()}."));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ApiResponse<ApplicationDto>.Fail(ex.Message));
        }
    }

    /// <summary>
    /// The CV as the candidate uploaded it.
    ///
    /// Served from the stored path rather than a path supplied by the caller — the
    /// only file this can ever return is the one this application points at.
    /// </summary>
    [HttpGet("{id:int}/resume")]
    public async Task<IActionResult> DownloadResume(int id, CancellationToken ct)
    {
        var row = await _applications.GetEntityAsync(id, ct);
        if (row?.Resume is null) return NotFound();

        var path = row.Resume.StoredPath;
        if (string.IsNullOrWhiteSpace(path) || !System.IO.File.Exists(path))
            return NotFound();

        if (!new FileExtensionContentTypeProvider().TryGetContentType(path, out var contentType))
            contentType = "application/octet-stream";

        var bytes = await System.IO.File.ReadAllBytesAsync(path, ct);
        return File(bytes, contentType, row.Resume.Filename);
    }
}
