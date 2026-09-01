using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Mapping;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Jobs;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Vectors;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Controllers;

[ApiController]
[Route("api/jobs")]
public class JobsController : ControllerBase
{
    /// <summary>
    /// Files accepted in one bulk upload.
    ///
    /// Extraction may call the model per document, so a batch is minutes of work, not
    /// milliseconds. Capping it keeps one request from holding a connection open long
    /// enough to be killed by a proxy — and the uploader gets a clear "split this up"
    /// instead of a timeout with no results at all.
    /// </summary>
    public const int MaxBulkFiles = 50;

    private readonly PortalDbContext _db;
    private readonly IJobService _jobs;
    private readonly IJobSuggestionService _suggestions;
    private readonly IVectorStore _vectors;
    private readonly IEmbeddingService _embeddings;
    private readonly IDocumentTextExtractor _extractor;
    private readonly ILogger<JobsController> _logger;

    public JobsController(
        PortalDbContext db,
        IJobService jobs,
        IJobSuggestionService suggestions,
        IVectorStore vectors,
        IEmbeddingService embeddings,
        IDocumentTextExtractor extractor,
        ILogger<JobsController> logger)
    {
        _db = db;
        _jobs = jobs;
        _suggestions = suggestions;
        _vectors = vectors;
        _embeddings = embeddings;
        _extractor = extractor;
        _logger = logger;
    }

    /// <summary>
    /// Judges one résumé against one posting with the model doing the reasoning.
    ///
    /// The alternative scoring path, offered alongside the computed one rather than
    /// replacing it: this returns a verdict with its evidence quoted from the CV,
    /// where the computed score returns four weighted dimensions. They are allowed
    /// to disagree — one is arithmetic over structure, the other reading
    /// comprehension — and seeing both is how you find out which to trust.
    /// </summary>
    [HttpPost("evaluate")]
    public async Task<ActionResult<ApiResponse<JobEvaluationDto>>> Evaluate(
        [FromBody] EvaluateRequest request,
        [FromServices] IJobEvaluationService evaluator,
        [FromServices] Services.Resumes.IResumeService resumes,
        CancellationToken ct)
    {
        var resume = await resumes.GetAsync(request.ResumeId, ct);
        if (resume is null)
            return NotFound(ApiResponse<JobEvaluationDto>.Fail($"No resume with id {request.ResumeId}."));

        var job = await _db.Jobs.AsNoTracking().FirstOrDefaultAsync(j => j.Id == request.JobId, ct);
        if (job is null)
            return NotFound(ApiResponse<JobEvaluationDto>.Fail($"No job with id {request.JobId}."));

        var evaluation = await evaluator.EvaluateAsync(resume, job, ct);

        return evaluation is null
            ? StatusCode(503, ApiResponse<JobEvaluationDto>.Fail(
                "No chat model is available, so this role cannot be reasoned about. The computed score still applies."))
            : Ok(ApiResponse<JobEvaluationDto>.Ok(evaluation, $"{evaluation.OverallMatch}% — {evaluation.Category}"));
    }

    /// <summary>
    /// Roles to open a conversation with, before any CV exists.
    ///
    /// Deliberately carries no fit score: a percentage is a claim about a person,
    /// and there is nobody to make it about until a CV is uploaded.
    /// </summary>
    [HttpGet("suggestions")]
    public async Task<ActionResult<ApiResponse<JobSuggestionResultDto>>> Suggestions(
        [FromQuery] string? query, [FromQuery] int top, CancellationToken ct)
    {
        var result = await _suggestions.SuggestAsync(query, new JobFilters(), top <= 0 ? 4 : top, ct);
        return Ok(ApiResponse<JobSuggestionResultDto>.Ok(result, $"{result.Jobs.Count} suggestion(s)."));
    }

    /// <summary>The job board. Filters here are exact and deterministic; semantic
    /// ranking belongs to the match endpoint, not to browsing.</summary>
    [HttpGet]
    public async Task<ActionResult<ApiResponse<IReadOnlyList<JobSummaryDto>>>> List(
        [FromQuery] string? search,
        [FromQuery] string? workMode,
        [FromQuery] string? location,
        [FromQuery] string? seniority,
        [FromQuery] decimal? minSalary,
        [FromQuery] int limit = 100,
        CancellationToken ct = default)
    {
        var query = _db.Jobs.AsNoTracking().Where(j => j.IsPublished);

        if (!string.IsNullOrWhiteSpace(workMode))
            query = query.Where(j => j.WorkMode == workMode);
        if (!string.IsNullOrWhiteSpace(seniority))
            query = query.Where(j => j.SeniorityLevel == seniority);
        if (!string.IsNullOrWhiteSpace(location))
            query = query.Where(j => j.Location.Contains(location));
        if (minSalary is { } floor)
            query = query.Where(j => j.SalaryMax == null || j.SalaryMax >= floor);
        if (!string.IsNullOrWhiteSpace(search))
        {
            query = query.Where(j =>
                j.Title.Contains(search) ||
                j.Company.Contains(search) ||
                j.RequiredSkills.Contains(search) ||
                j.Summary.Contains(search));
        }

        var jobs = await query
            .OrderByDescending(j => j.CreatedAt)
            .Take(Math.Clamp(limit, 1, 500))
            .ToListAsync(ct);

        var indexed = (await _vectors.GetIndexedHashesAsync(_embeddings.PreferredModelId, ct)).Keys.ToHashSet();

        var data = jobs.Select(j => j.ToSummaryDto(indexed.Contains(j.Id))).ToList();
        return Ok(ApiResponse<IReadOnlyList<JobSummaryDto>>.Ok(data, $"{data.Count} job(s)."));
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<ApiResponse<JobDetailDto>>> Get(int id, CancellationToken ct)
    {
        var job = await _db.Jobs.AsNoTracking().FirstOrDefaultAsync(j => j.Id == id, ct);
        if (job is null) return NotFound(ApiResponse<JobDetailDto>.Fail($"No job with id {id}."));

        var indexed = (await _vectors.GetIndexedHashesAsync(_embeddings.PreferredModelId, ct)).ContainsKey(id);
        return Ok(ApiResponse<JobDetailDto>.Ok(job.ToDetailDto(indexed)));
    }

    /// <summary>Uploads a JD document. The posting is extracted and indexed before
    /// this returns, so a job is searchable the moment it appears on the board.</summary>
    [HttpPost("upload")]
    [RequestSizeLimit(32 * 1024 * 1024)]
    public async Task<ActionResult<ApiResponse<JobUploadResultDto>>> Upload(
        IFormFile file, CancellationToken ct)
    {
        if (file is null || file.Length == 0)
            return BadRequest(ApiResponse<JobUploadResultDto>.Fail("No file was uploaded."));

        if (!_extractor.IsSupported(file.FileName))
        {
            return BadRequest(ApiResponse<JobUploadResultDto>.Fail(
                $"'{Path.GetExtension(file.FileName)}' is not supported. Upload a PDF, DOCX, TXT or MD file."));
        }

        try
        {
            await using var stream = file.OpenReadStream();
            var job = await _jobs.CreateFromDocumentAsync(stream, file.FileName, null, null, ct);
            var indexed = (await _vectors.GetIndexedHashesAsync(_embeddings.PreferredModelId, ct))
                .ContainsKey(job.Id);

            return Ok(ApiResponse<JobUploadResultDto>.Ok(
                new JobUploadResultDto(
                    job.ToDetailDto(indexed), indexed,
                    _embeddings.PreferredModelId, _embeddings.SemanticAvailable),
                $"'{job.Title}' posted."));
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException)
        {
            // A document we genuinely cannot read: the uploader can act on this.
            return BadRequest(ApiResponse<JobUploadResultDto>.Fail(ex.Message));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Job upload failed for {Filename}", file.FileName);
            return StatusCode(500, ApiResponse<JobUploadResultDto>.Fail("The job could not be processed."));
        }
    }

    /// <summary>
    /// Uploads several JD documents in one request.
    ///
    /// Each file is processed INDEPENDENTLY and the response carries a per-file
    /// outcome, because in a batch of thirty a couple of unreadable scans are normal
    /// and must not discard the twenty-eight that parsed. A single pass/fail would
    /// force the uploader to re-send everything to retry two files.
    ///
    /// Processing is sequential on purpose. `PortalDbContext` is a scoped EF Core
    /// context and is not safe for concurrent use, so a Task.WhenAll here would be a
    /// data-corruption bug rather than a speed-up. Extraction is also the expensive
    /// part (it may call the model), so the batch is capped rather than parallelised.
    /// </summary>
    [HttpPost("upload-bulk")]
    [RequestSizeLimit(256 * 1024 * 1024)]
    public async Task<ActionResult<ApiResponse<JobBulkUploadResultDto>>> UploadBulk(
        [FromForm] IFormFileCollection files, CancellationToken ct)
    {
        if (files is null || files.Count == 0)
            return BadRequest(ApiResponse<JobBulkUploadResultDto>.Fail("No files were uploaded."));

        if (files.Count > MaxBulkFiles)
        {
            return BadRequest(ApiResponse<JobBulkUploadResultDto>.Fail(
                $"{files.Count} files is over the {MaxBulkFiles}-file limit for one batch. " +
                "Upload them in smaller batches."));
        }

        var items = new List<JobUploadItemDto>(files.Count);
        var indexedHashes = await _vectors.GetIndexedHashesAsync(_embeddings.PreferredModelId, ct);

        foreach (var file in files)
        {
            // The client disconnected or the caller cancelled: stop starting new work,
            // but still return what already succeeded.
            if (ct.IsCancellationRequested) break;

            var name = Path.GetFileName(file.FileName) ?? "(unnamed)";

            if (file.Length == 0)
            {
                items.Add(new JobUploadItemDto(name, false, Error: "The file is empty."));
                continue;
            }
            if (!_extractor.IsSupported(file.FileName))
            {
                items.Add(new JobUploadItemDto(name, false,
                    Error: $"'{Path.GetExtension(file.FileName)}' is not supported. Use PDF, DOCX, TXT or MD."));
                continue;
            }

            try
            {
                await using var stream = file.OpenReadStream();
                var job = await _jobs.CreateFromDocumentAsync(stream, file.FileName, null, null, ct);
                items.Add(new JobUploadItemDto(
                    name, true, job.ToSummaryDto(indexedHashes.ContainsKey(job.Id)),
                    indexedHashes.ContainsKey(job.Id)));
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                // One bad document must never sink the batch. Log the detail, hand the
                // uploader something they can act on.
                _logger.LogWarning(ex, "Bulk job upload failed for {Filename}", name);
                var reason = ex is InvalidOperationException or NotSupportedException
                    ? ex.Message
                    : "The document could not be read.";
                items.Add(new JobUploadItemDto(name, false, Error: reason));
            }
        }

        var succeeded = items.Count(i => i.Success);
        var failed = items.Count - succeeded;

        return Ok(ApiResponse<JobBulkUploadResultDto>.Ok(
            new JobBulkUploadResultDto(
                items.Count, succeeded, failed, items,
                _embeddings.PreferredModelId, _embeddings.SemanticAvailable),
            failed == 0
                ? $"{succeeded} job(s) posted."
                : $"{succeeded} of {items.Count} posted; {failed} could not be read."));
    }

    /// <summary>Creates a posting from the form rather than a document.</summary>
    [HttpPost]
    public async Task<ActionResult<ApiResponse<JobDetailDto>>> Create(
        [FromBody] CreateJobRequest request, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Title))
            return BadRequest(ApiResponse<JobDetailDto>.Fail("A job title is required."));

        var job = await _jobs.CreateFromRequestAsync(request, ct);
        var indexed = (await _vectors.GetIndexedHashesAsync(_embeddings.PreferredModelId, ct)).ContainsKey(job.Id);

        return Ok(ApiResponse<JobDetailDto>.Ok(job.ToDetailDto(indexed), $"'{job.Title}' posted."));
    }

    [HttpDelete("{id:int}")]
    public async Task<ActionResult<ApiResponse<string>>> Delete(int id, CancellationToken ct)
    {
        var job = await _db.Jobs.FirstOrDefaultAsync(j => j.Id == id, ct);
        if (job is null) return NotFound(ApiResponse<string>.Fail($"No job with id {id}."));

        // Vectors first: a job row removed while its vectors survive leaves the
        // index returning ids that no longer resolve to anything.
        await _vectors.RemoveAsync(id, ct);
        _db.Jobs.Remove(job);
        await _db.SaveChangesAsync(ct);

        return Ok(ApiResponse<string>.Ok(job.Title, "Job removed."));
    }

    /// <summary>Pulls the recruiter platform's existing JDs onto the board.</summary>
    [HttpPost("import-from-platform")]
    public async Task<ActionResult<ApiResponse<ImportResultDto>>> Import(CancellationToken ct)
    {
        var result = await _jobs.ImportFromPythonAsync(ct);
        return Ok(ApiResponse<ImportResultDto>.Ok(
            result, $"Imported {result.Imported}, skipped {result.Skipped}, failed {result.Failed}."));
    }

    [HttpGet("index-status")]
    public async Task<ActionResult<ApiResponse<IndexStatusDto>>> IndexStatus(CancellationToken ct) =>
        Ok(ApiResponse<IndexStatusDto>.Ok(await _jobs.GetIndexStatusAsync(ct)));

    /// <summary>
    /// Re-embeds the board. Needed after switching embedding model, since vectors
    /// from the previous one cannot be compared against the new one's queries.
    /// </summary>
    [HttpPost("reindex")]
    public async Task<ActionResult<ApiResponse<IndexStatusDto>>> Reindex(
        [FromQuery] bool force = false, CancellationToken ct = default)
    {
        var count = await _jobs.ReindexAllAsync(force, ct);
        return Ok(ApiResponse<IndexStatusDto>.Ok(
            await _jobs.GetIndexStatusAsync(ct), $"{count} job(s) embedded."));
    }
}
