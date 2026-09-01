using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Screening;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Controllers;

/// <summary>
/// Pre-application screening, as HTTP.
///
/// Two calls, matching the two moments in the flow: the applicant opens "Apply"
/// and is shown four questions; they answer, and the gate says whether they may
/// proceed.
///
/// Kept in its own controller alongside its own service so the whole feature is
/// two files and no edits to anything that already works.
/// </summary>
[ApiController]
[Route("api/jobs/{jobId:int}/screening")]
public class ScreeningController : ControllerBase
{
    private readonly PortalDbContext _db;
    private readonly IApplicationScreeningService _screening;

    public ScreeningController(PortalDbContext db, IApplicationScreeningService screening)
    {
        _db = db;
        _screening = screening;
    }

    /// <summary>The four questions to put in front of the applicant.</summary>
    [HttpGet]
    public async Task<ActionResult<ApiResponse<ScreeningQuestionSet>>> Questions(
        int jobId, CancellationToken ct)
    {
        var job = await Job(jobId, ct);
        if (job is null) return NotFound(ApiResponse<ScreeningQuestionSet>.Fail($"No job with id {jobId}."));

        var set = await _screening.BuildAsync(job, ct);
        return Ok(ApiResponse<ScreeningQuestionSet>.Ok(set, $"{set.Questions.Count} question(s)."));
    }

    public record SubmitRequest(IReadOnlyList<ScreeningAnswer> Answers);

    /// <summary>
    /// Whether these answers clear the gate.
    ///
    /// Returns 200 with <c>passed: false</c> rather than a 4xx when the applicant
    /// declines something: a considered "no" is a valid outcome of asking, not a
    /// malformed request, and the body carries what to tell them.
    /// </summary>
    [HttpPost]
    public async Task<ActionResult<ApiResponse<ScreeningResult>>> Submit(
        int jobId, [FromBody] SubmitRequest request, CancellationToken ct)
    {
        var job = await Job(jobId, ct);
        if (job is null) return NotFound(ApiResponse<ScreeningResult>.Fail($"No job with id {jobId}."));

        var result = await _screening.EvaluateAsync(
            job, request?.Answers ?? Array.Empty<ScreeningAnswer>(), ct);

        return Ok(ApiResponse<ScreeningResult>.Ok(result, result.Message));
    }

    private Task<PortalJob?> Job(int jobId, CancellationToken ct) =>
        _db.Jobs.AsNoTracking().FirstOrDefaultAsync(j => j.Id == jobId && j.IsPublished, ct);
}
