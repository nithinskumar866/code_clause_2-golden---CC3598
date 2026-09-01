using System.Security.Cryptography;
using System.Text;
using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Jobs;

public interface IJobService
{
    Task<PortalJob> CreateFromDocumentAsync(Stream file, string filename, string? storedPath, int? sourceJdId,
        CancellationToken ct = default);
    Task<PortalJob> CreateFromRequestAsync(CreateJobRequest request, CancellationToken ct = default);
    Task<bool> IndexAsync(PortalJob job, CancellationToken ct = default);
    Task<int> ReindexAllAsync(bool force, CancellationToken ct = default);
    Task<ImportResultDto> ImportFromPythonAsync(CancellationToken ct = default);
    Task<IndexStatusDto> GetIndexStatusAsync(CancellationToken ct = default);
}

public class JobService : IJobService
{
    private readonly PortalDbContext _db;
    private readonly IDocumentTextExtractor _extractor;
    private readonly IJobExtractionService _extraction;
    private readonly IEmbeddingService _embeddings;
    private readonly IVectorStore _vectors;
    private readonly PortalOptions _portal;
    private readonly ILogger<JobService> _logger;

    public JobService(
        PortalDbContext db,
        IDocumentTextExtractor extractor,
        IJobExtractionService extraction,
        IEmbeddingService embeddings,
        IVectorStore vectors,
        IOptions<PortalOptions> portal,
        ILogger<JobService> logger)
    {
        _db = db;
        _extractor = extractor;
        _extraction = extraction;
        _embeddings = embeddings;
        _vectors = vectors;
        _portal = portal.Value;
        _logger = logger;
    }

    public async Task<PortalJob> CreateFromDocumentAsync(
        Stream file, string filename, string? storedPath, int? sourceJdId, CancellationToken ct = default)
    {
        var rawText = _extractor.Extract(file, filename);
        var extracted = await _extraction.ExtractAsync(rawText, filename, ct);

        var job = new PortalJob
        {
            SourceJobDescriptionId = sourceJdId,
            SourceFilePath = storedPath,
            RawText = rawText,
        };
        Apply(job, extracted);

        _db.Jobs.Add(job);
        await _db.SaveChangesAsync(ct);

        await IndexAsync(job, ct);
        return job;
    }

    public async Task<PortalJob> CreateFromRequestAsync(CreateJobRequest request, CancellationToken ct = default)
    {
        var description = request.Description ?? "";

        // A typed-in posting still goes through extraction, on the description the
        // employer wrote. Otherwise a job posted through the form would carry only
        // the fields someone bothered to fill in, and would rank worse than the
        // identical job uploaded as a PDF.
        var extracted = await _extraction.ExtractAsync(
            BuildSyntheticDocument(request), request.Title, ct);

        var job = new PortalJob { RawText = description };
        Apply(job, extracted);

        // Explicit form input always beats anything inferred from prose: the
        // employer typing "Hybrid" into the field is a statement, not a hint.
        job.Title = Fallback(request.Title, job.Title);
        job.Company = Fallback(request.Company, job.Company);
        job.Location = Fallback(request.Location, job.Location);
        job.WorkMode = Fallback(request.WorkMode, job.WorkMode);
        job.EmploymentType = Fallback(request.EmploymentType, job.EmploymentType);
        job.SeniorityLevel = Fallback(request.SeniorityLevel, job.SeniorityLevel);
        job.MinYearsExperience = request.MinYearsExperience ?? job.MinYearsExperience;
        job.MaxYearsExperience = request.MaxYearsExperience ?? job.MaxYearsExperience;
        job.SalaryMin = request.SalaryMin ?? job.SalaryMin;
        job.SalaryMax = request.SalaryMax ?? job.SalaryMax;
        job.SalaryCurrency = request.SalaryCurrency ?? job.SalaryCurrency;
        job.ApplyUrl = request.ApplyUrl;

        if (request.RequiredSkills is { Count: > 0 })
            job.RequiredSkills = TextStructure.JoinLines(request.RequiredSkills);
        if (request.PreferredSkills is { Count: > 0 })
            job.PreferredSkills = TextStructure.JoinLines(request.PreferredSkills);

        _db.Jobs.Add(job);
        await _db.SaveChangesAsync(ct);

        await IndexAsync(job, ct);
        return job;
    }

    /// <summary>
    /// Renders form input as a document so the same extractor reads both paths.
    /// One extraction implementation means a posting cannot behave differently
    /// depending on how it arrived.
    /// </summary>
    private static string BuildSyntheticDocument(CreateJobRequest request)
    {
        var builder = new StringBuilder();
        builder.AppendLine(request.Title);
        if (!string.IsNullOrWhiteSpace(request.Company)) builder.AppendLine($"Company: {request.Company}");
        if (!string.IsNullOrWhiteSpace(request.Location)) builder.AppendLine($"Location: {request.Location}");
        if (!string.IsNullOrWhiteSpace(request.WorkMode)) builder.AppendLine($"Work Mode: {request.WorkMode}");
        if (!string.IsNullOrWhiteSpace(request.EmploymentType))
            builder.AppendLine($"Employment Type: {request.EmploymentType}");
        builder.AppendLine();

        if (!string.IsNullOrWhiteSpace(request.Description)) builder.AppendLine(request.Description);

        if (request.RequiredSkills is { Count: > 0 })
        {
            builder.AppendLine();
            builder.AppendLine("Requirements");
            foreach (var skill in request.RequiredSkills) builder.AppendLine("- " + skill);
        }
        if (request.PreferredSkills is { Count: > 0 })
        {
            builder.AppendLine();
            builder.AppendLine("Nice to Have");
            foreach (var skill in request.PreferredSkills) builder.AppendLine("- " + skill);
        }

        return builder.ToString();
    }

    private static void Apply(PortalJob job, ExtractedJob extracted)
    {
        job.Title = extracted.Title;
        job.Company = extracted.Company;
        job.Location = extracted.Location;
        job.WorkMode = extracted.WorkMode;
        job.EmploymentType = extracted.EmploymentType;
        job.SeniorityLevel = extracted.SeniorityLevel;
        job.MinYearsExperience = extracted.MinYearsExperience;
        job.MaxYearsExperience = extracted.MaxYearsExperience;
        job.SalaryMin = extracted.SalaryMin;
        job.SalaryMax = extracted.SalaryMax;
        job.SalaryCurrency = extracted.SalaryCurrency;
        job.RequiredSkills = TextStructure.JoinLines(extracted.RequiredSkills);
        job.PreferredSkills = TextStructure.JoinLines(extracted.PreferredSkills);
        job.Responsibilities = TextStructure.JoinLines(extracted.Responsibilities);
        job.Qualifications = TextStructure.JoinLines(extracted.Qualifications);
        job.Summary = extracted.Summary;
        job.ExtractionMode = extracted.Mode;
        job.UpdatedAt = DateTime.UtcNow;
    }

    private static string Fallback(string? preferred, string current) =>
        string.IsNullOrWhiteSpace(preferred) ? current : preferred.Trim();

    // -- indexing -----------------------------------------------------------

    public async Task<bool> IndexAsync(PortalJob job, CancellationToken ct = default)
    {
        var text = _extraction.BuildEmbeddingText(job);
        if (string.IsNullOrWhiteSpace(text)) return false;

        var batch = await _embeddings.EmbedAsync(text, ct);
        if (batch.Vectors.Count == 0) return false;

        job.EmbeddedText = text;
        await _db.SaveChangesAsync(ct);

        await _vectors.UpsertAsync(job.Id, batch.ModelId, batch.Dimension, batch.Vectors[0], Hash(text), ct);

        if (!batch.IsSemantic)
        {
            _logger.LogInformation(
                "Job {JobId} indexed with the deterministic embedder. Matching for it is lexical " +
                "until an embedding model is reachable.", job.Id);
        }
        return true;
    }

    public async Task<int> ReindexAllAsync(bool force, CancellationToken ct = default)
    {
        var model = _embeddings.PreferredModelId;
        var known = force
            ? new Dictionary<int, string>()
            : (Dictionary<int, string>)await _vectors.GetIndexedHashesAsync(model, ct);

        var jobs = await _db.Jobs.ToListAsync(ct);
        var indexed = 0;

        foreach (var job in jobs)
        {
            ct.ThrowIfCancellationRequested();

            var text = _extraction.BuildEmbeddingText(job);
            var hash = Hash(text);

            // Skip only when this exact text is already indexed under this exact
            // model. Anything else and the stored vector no longer describes the
            // job it is attached to.
            if (!force && known.TryGetValue(job.Id, out var existing) && existing == hash) continue;

            var batch = await _embeddings.EmbedAsync(text, ct);
            if (batch.Vectors.Count == 0) continue;

            job.EmbeddedText = text;
            await _vectors.UpsertAsync(job.Id, batch.ModelId, batch.Dimension, batch.Vectors[0], hash, ct);
            indexed++;
        }

        await _db.SaveChangesAsync(ct);
        _logger.LogInformation("Re-index complete: {Indexed} of {Total} jobs embedded with {Model}",
            indexed, jobs.Count, model);
        return indexed;
    }

    // -- import from the recruiter platform ---------------------------------

    /// <summary>
    /// Pulls JDs the Python platform already holds into the portal as postings.
    ///
    /// One-way and additive by design. The portal never writes to the recruiter
    /// platform's tables, and re-running this is safe: a JD already imported is
    /// recognised by its source id and skipped rather than duplicated.
    /// </summary>
    public async Task<ImportResultDto> ImportFromPythonAsync(CancellationToken ct = default)
    {
        var notes = new List<string>();

        if (!await PortalSchemaInitializer.TableExistsAsync(_db, "job_descriptions", ct))
        {
            return new ImportResultDto(0, 0, 0, new[]
            {
                "The shared database has no 'job_descriptions' table yet. Run the Python backend " +
                "once so it creates its schema, then import again.",
            });
        }

        var jobsDir = Path.Combine(_portal.PythonStorageDir, "uploads", "jobs");
        if (!Directory.Exists(jobsDir))
        {
            return new ImportResultDto(0, 0, 0, new[]
            {
                $"The recruiter platform's JD folder was not found at '{jobsDir}'. " +
                "Check Portal:PythonStorageDir.",
            });
        }

        var alreadyImported = await _db.Jobs
            .Where(j => j.SourceJobDescriptionId != null)
            .Select(j => j.SourceJobDescriptionId!.Value)
            .ToListAsync(ct);
        var seen = alreadyImported.ToHashSet();

        var descriptions = await _db.PythonJobDescriptions.AsNoTracking().ToListAsync(ct);

        int imported = 0, skipped = 0, failed = 0;

        foreach (var jd in descriptions)
        {
            ct.ThrowIfCancellationRequested();

            if (seen.Contains(jd.Id)) { skipped++; continue; }

            // The Python side stores uploads as "{id}_{original filename}".
            var path = Path.Combine(jobsDir, $"{jd.Id}_{jd.Filename}");
            if (!File.Exists(path))
            {
                failed++;
                notes.Add($"JD {jd.Id} ('{jd.Filename}'): file missing on disk.");
                continue;
            }

            if (!_extractor.IsSupported(jd.Filename))
            {
                skipped++;
                notes.Add($"JD {jd.Id} ('{jd.Filename}'): unsupported file type.");
                continue;
            }

            try
            {
                await using var stream = File.OpenRead(path);
                await CreateFromDocumentAsync(stream, jd.Filename, path, jd.Id, ct);
                imported++;
            }
            catch (Exception ex)
            {
                failed++;
                notes.Add($"JD {jd.Id} ('{jd.Filename}'): {ex.Message}");
                _logger.LogWarning(ex, "Failed to import JD {JdId}", jd.Id);
            }
        }

        _logger.LogInformation("Imported {Imported}, skipped {Skipped}, failed {Failed} recruiter JDs",
            imported, skipped, failed);
        return new ImportResultDto(imported, skipped, failed, notes);
    }

    public async Task<IndexStatusDto> GetIndexStatusAsync(CancellationToken ct = default)
    {
        var model = _embeddings.PreferredModelId;
        return new IndexStatusDto(
            TotalJobs: await _db.Jobs.CountAsync(ct),
            IndexedJobs: await _vectors.CountAsync(model, ct),
            ActiveModel: model,
            SemanticMatching: _embeddings.SemanticAvailable,
            VectorsByModel: await _vectors.CountsByModelAsync(ct));
    }

    public static string Hash(string text) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
}
