using System.Text.Json;
using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Mapping;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Services.Applications;

public interface IApplicationService
{
    Task<ApplicationPreviewDto> PreviewAsync(ApplyPreviewRequest request, CancellationToken ct = default);
    Task<ApplySubmitResultDto> SubmitAsync(ApplySubmitRequest request, CancellationToken ct = default);
    Task<IReadOnlyList<ApplicationDto>> ListAsync(int? jobId, string? status, CancellationToken ct = default);
    Task<ApplicationDto?> GetAsync(int id, CancellationToken ct = default);
    Task<PortalApplication?> GetEntityAsync(int id, CancellationToken ct = default);
    Task<ApplicationDto?> SetStatusAsync(int id, ApplicationStatusRequest request, CancellationToken ct = default);

    /// <summary>Writes one letter for a role already under review.</summary>
    Task<DraftLetterDto> DraftLetterAsync(DraftLetterRequest request, CancellationToken ct = default);
}

/// <summary>
/// Applying to postings on this board.
///
/// Two rules shape the whole service.
///
/// <b>Nothing is sent without a review.</b> <see cref="PreviewAsync"/> does all the
/// expensive work — matching, letter writing, duplicate detection — and returns it
/// for a human to look at; <see cref="SubmitAsync"/> only persists what came back.
/// Applying is outward-facing and cannot be undone from here, so the split is not
/// a UI nicety, it is where the consent lives.
///
/// <b>The scores are frozen.</b> What is stored is what the candidate saw. Models
/// change, thresholds get recalibrated, postings get edited; re-deriving a fit
/// score on read would quietly re-judge an applicant against a bar that did not
/// exist when they applied.
/// </summary>
public class ApplicationService : IApplicationService
{
    /// <summary>
    /// The most postings one bulk apply can reach.
    ///
    /// A fit floor with no cap is how "apply to anything above 40%" turns into a
    /// hundred employers hearing from someone who typed one sentence. The cap is
    /// reported rather than silently applied.
    /// </summary>
    public const int MaxBulkApply = 25;

    /// <summary>
    /// How many cover letters are written at once.
    ///
    /// Above one so a five-role review is not five model calls end to end; bounded
    /// because the thing answering them is usually a single small model server, and
    /// twenty-five simultaneous requests turn a slow review into a failed one.
    /// </summary>
    private const int MaxConcurrentLetters = 4;

    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private readonly PortalDbContext _db;
    private readonly IMatchingService _matching;
    private readonly ICoverLetterService _letters;
    private readonly IVectorStore _vectors;
    private readonly IEmbeddingService _embeddings;
    private readonly ILogger<ApplicationService> _logger;

    public ApplicationService(
        PortalDbContext db,
        IMatchingService matching,
        ICoverLetterService letters,
        IVectorStore vectors,
        IEmbeddingService embeddings,
        ILogger<ApplicationService> logger)
    {
        _db = db;
        _matching = matching;
        _letters = letters;
        _vectors = vectors;
        _embeddings = embeddings;
        _logger = logger;
    }

    /// <summary>
    /// Which postings are searchable under the active model.
    ///
    /// Read once per request and passed down rather than asked per row: the store
    /// answers with the whole map, so calling it inside a loop turns one query into
    /// one per application.
    /// </summary>
    private async Task<HashSet<int>> IndexedJobIdsAsync(CancellationToken ct) =>
        (await _vectors.GetIndexedHashesAsync(_embeddings.PreferredModelId, ct)).Keys.ToHashSet();

    public async Task<ApplicationPreviewDto> PreviewAsync(
        ApplyPreviewRequest request, CancellationToken ct = default)
    {
        var resume = await _db.Resumes.FirstOrDefaultAsync(r => r.Id == request.ResumeId, ct)
            ?? throw new InvalidOperationException($"No resume with id {request.ResumeId}.");

        // Ranked by the same pipeline the chat uses. Applying must never disagree
        // with the score the candidate was shown a moment earlier.
        var matches = await _matching.MatchAsync(resume, request.Filters ?? new JobFilters(), null, ct);

        var wanted = matches.Matches.AsEnumerable();

        if (request.JobIds is { Count: > 0 })
        {
            var ids = request.JobIds.ToHashSet();
            wanted = wanted.Where(m => ids.Contains(m.Job.Id));
        }

        if (request.MinFitScore is { } floor)
            wanted = wanted.Where(m => m.FitScore >= floor);

        var cap = Math.Clamp(request.Limit ?? MaxBulkApply, 1, MaxBulkApply);
        var selected = wanted.Take(cap).ToList();

        var alreadyApplied = await AppliedJobIdsAsync(resume.ContentHash, ct);

        // Every posting in one query. Fetching them inside the loop below was an
        // N+1, and worse, it interleaved database calls with the DbContext while
        // the letters ran — which is what made parallelising them unsafe.
        var wantedIds = selected.Select(m => m.Job.Id).ToList();
        var jobsById = await _db.Jobs
            .AsNoTracking()
            .Where(j => wantedIds.Contains(j.Id))
            .ToDictionaryAsync(j => j.Id, ct);

        var pending = selected
            .Where(match => jobsById.ContainsKey(match.Job.Id))
            .Select(match => new
            {
                Match = match,
                Job = jobsById[match.Job.Id],
                // A duplicate still gets a row in the review, marked, rather than
                // being dropped: "3 of your 5 were already sent" is information the
                // candidate needs, and a silently shorter list looks like a bug.
                Duplicate = alreadyApplied.Contains(match.Job.Id),
            })
            .ToList();

        // Written concurrently when asked for. Each letter is an independent model
        // call taking tens of seconds on a local model, so doing them in sequence
        // made a five-role review take five times longer than it needed to.
        // Bounded because the endpoint behind this is one small model server, and
        // firing twenty-five requests at it at once is how a slow review becomes a
        // failed one.
        //
        // Not drafted by default: the review panel has to open now, and a composed
        // letter is a real letter. Each row is upgraded on request instead.
        var letters = new CoverLetter[pending.Count];
        using (var slots = new SemaphoreSlim(MaxConcurrentLetters))
        {
            await Task.WhenAll(pending.Select(async (entry, index) =>
            {
                if (entry.Duplicate)
                {
                    letters[index] = new CoverLetter("", CoverLetter.Deterministic);
                    return;
                }

                if (!request.DraftLetters)
                {
                    letters[index] = _letters.Compose(resume, entry.Job, entry.Match);
                    return;
                }

                await slots.WaitAsync(ct);
                try
                {
                    letters[index] = await _letters.WriteAsync(resume, entry.Job, entry.Match, ct);
                }
                finally
                {
                    slots.Release();
                }
            }));
        }

        var items = pending
            .Select((entry, index) => new ApplicationPreviewItemDto(
                entry.Match.Job, entry.Match.FitScore, entry.Match.FitBand,
                letters[index].Text, letters[index].Mode,
                entry.Match.Skills, entry.Match.Strengths, entry.Match.Gaps, entry.Duplicate))
            .ToList();

        var deterministicLetters = pending
            .Where((entry, index) => !entry.Duplicate && letters[index].Mode == CoverLetter.Deterministic)
            .Count();

        var eligible = items.Count(i => !i.AlreadyApplied);

        return new ApplicationPreviewDto(
            resume.ToDto(),
            items,
            eligible,
            items.Count - eligible,
            matches.SemanticMatching,
            matches.EmbeddingModel,
            eligible > 0 && deterministicLetters == eligible);
    }

    public async Task<ApplySubmitResultDto> SubmitAsync(
        ApplySubmitRequest request, CancellationToken ct = default)
    {
        var resume = await _db.Resumes.FirstOrDefaultAsync(r => r.Id == request.ResumeId, ct)
            ?? throw new InvalidOperationException($"No resume with id {request.ResumeId}.");

        var requested = (request.Items ?? Array.Empty<ApplySubmitItem>())
            .GroupBy(item => item.JobId)
            .Select(group => group.First())
            .Take(MaxBulkApply)
            .ToList();

        if (requested.Count == 0)
            return new ApplySubmitResultDto(Array.Empty<ApplicationDto>(), Array.Empty<string>(), 0, 0);

        var alreadyApplied = await AppliedJobIdsAsync(resume.ContentHash, ct);

        // Re-scored once here rather than trusting numbers posted by the client:
        // the fit snapshot is what a recruiter will judge this person by, so it has
        // to come from the scorer, not from whatever the browser sent.
        var matches = await _matching.MatchAsync(resume, new JobFilters(), null, ct);
        var byJob = matches.Matches.ToDictionary(m => m.Job.Id);

        var submitted = new List<PortalApplication>();
        var skipped = new List<string>();

        foreach (var item in requested)
        {
            var job = await _db.Jobs.FirstOrDefaultAsync(j => j.Id == item.JobId, ct);
            if (job is null)
            {
                skipped.Add($"Job {item.JobId} no longer exists.");
                continue;
            }

            if (alreadyApplied.Contains(job.Id))
            {
                skipped.Add($"Already applied to {job.Title}.");
                continue;
            }

            if (!byJob.TryGetValue(job.Id, out var match))
            {
                // Scoring is what makes an application reviewable; without it the
                // recruiter gets a name and nothing to judge it by.
                skipped.Add($"{job.Title} could not be scored against this CV.");
                continue;
            }

            // The letter the candidate approved, not a fresh one. Regenerating here
            // would send text nobody read, and a model does not repeat itself.
            var letter = item.CoverLetter?.Trim();
            string mode;
            if (string.IsNullOrWhiteSpace(letter))
            {
                var written = await _letters.WriteAsync(resume, job, match, ct);
                letter = written.Text;
                mode = written.Mode;
            }
            else
            {
                mode = "reviewed";
            }

            // Collect skill verdict cache keys for this application
            var skillVerdictCacheKeys = match.Skills
                .Where(s => s.Status == "Transferable" && !string.IsNullOrWhiteSpace(s.EvidenceSkill))
                .Select(s => $"{job.Id}|{s.Skill}|{s.EvidenceSkill}")
                .ToList();

            submitted.Add(new PortalApplication
            {
                JobId = job.Id,
                ResumeId = resume.Id,
                ResumeContentHash = resume.ContentHash,
                Status = ApplicationStatus.Submitted,
                CoverLetter = letter,
                LetterMode = mode,
                FitScore = match.FitScore,
                FitBand = match.FitBand,
                SemanticScore = match.SemanticScore,
                SkillScore = match.SkillScore,
                TitleScore = match.TitleScore,
                ExperienceScore = match.ExperienceScore,
                SkillsJson = JsonSerializer.Serialize(match.Skills, Json),
                StrengthsJson = JsonSerializer.Serialize(match.Strengths, Json),
                GapsJson = JsonSerializer.Serialize(match.Gaps, Json),
                RecruiterNote = match.RecruiterNote,
                SemanticMatching = matches.SemanticMatching,
                EmbeddingModel = matches.EmbeddingModel,
                ModelVersion = _embeddings.PreferredModelId, // Use embedding model as proxy for model version
                PromptVersion = "v1", // Bump when scoring pipeline changes
                SkillVerdictCacheKeysJson = JsonSerializer.Serialize(skillVerdictCacheKeys, Json),
            });
        }

        if (submitted.Count > 0)
        {
            _db.Applications.AddRange(submitted);
            try
            {
                await _db.SaveChangesAsync(ct);
            }
            catch (DbUpdateException ex)
            {
                // The unique index is the real guarantee; the check above is only an
                // optimisation. Losing that race means someone applied twice at once,
                // and the right outcome is still exactly one application.
                _logger.LogWarning(ex, "A concurrent apply collided with the duplicate index; retrying one at a time.");
                foreach (var entry in submitted) _db.Entry(entry).State = EntityState.Detached;
                return await SubmitOneByOneAsync(submitted, skipped, ct);
            }
        }

        return await BuildResultAsync(submitted, skipped, ct);
    }

    /// <summary>Falls back to per-row inserts so one duplicate cannot lose the batch.</summary>
    private async Task<ApplySubmitResultDto> SubmitOneByOneAsync(
        List<PortalApplication> candidates, List<string> skipped, CancellationToken ct)
    {
        var stored = new List<PortalApplication>();

        foreach (var application in candidates)
        {
            try
            {
                _db.Applications.Add(application);
                await _db.SaveChangesAsync(ct);
                stored.Add(application);
            }
            catch (DbUpdateException)
            {
                _db.Entry(application).State = EntityState.Detached;
                var title = await _db.Jobs.Where(j => j.Id == application.JobId)
                    .Select(j => j.Title).FirstOrDefaultAsync(ct) ?? $"job {application.JobId}";
                skipped.Add($"Already applied to {title}.");
            }
        }

        return await BuildResultAsync(stored, skipped, ct);
    }

    private async Task<ApplySubmitResultDto> BuildResultAsync(
        List<PortalApplication> stored, List<string> skipped, CancellationToken ct)
    {
        var indexed = await IndexedJobIdsAsync(ct);

        var dtos = new List<ApplicationDto>(stored.Count);
        foreach (var application in stored)
        {
            var row = await GetEntityAsync(application.Id, ct);
            if (row is not null) dtos.Add(await ToDtoAsync(row, indexed, ct));
        }

        return new ApplySubmitResultDto(dtos, skipped, dtos.Count, skipped.Count);
    }

    public async Task<IReadOnlyList<ApplicationDto>> ListAsync(
        int? jobId, string? status, CancellationToken ct = default)
    {
        var query = _db.Applications
            .Include(a => a.Job)
            .Include(a => a.Resume)
            .AsQueryable();

        if (jobId is { } id) query = query.Where(a => a.JobId == id);
        if (!string.IsNullOrWhiteSpace(status)) query = query.Where(a => a.Status == status);

        var rows = await query.OrderByDescending(a => a.CreatedAt).ToListAsync(ct);
        var indexed = await IndexedJobIdsAsync(ct);

        var result = new List<ApplicationDto>(rows.Count);
        foreach (var row in rows) result.Add(await ToDtoAsync(row, indexed, ct));
        return result;
    }

    public async Task<ApplicationDto?> GetAsync(int id, CancellationToken ct = default)
    {
        var row = await GetEntityAsync(id, ct);
        return row is null ? null : await ToDtoAsync(row, await IndexedJobIdsAsync(ct), ct);
    }

    public Task<PortalApplication?> GetEntityAsync(int id, CancellationToken ct = default) =>
        _db.Applications
            .Include(a => a.Job)
            .Include(a => a.Resume)
            .FirstOrDefaultAsync(a => a.Id == id, ct);

    public async Task<ApplicationDto?> SetStatusAsync(
        int id, ApplicationStatusRequest request, CancellationToken ct = default)
    {
        if (!ApplicationStatus.IsKnown(request.Status))
            throw new InvalidOperationException($"'{request.Status}' is not an application status.");

        var row = await GetEntityAsync(id, ct);
        if (row is null) return null;

        row.Status = request.Status;
        row.UpdatedAt = DateTime.UtcNow;

        if (request.Status == ApplicationStatus.Accepted)
        {
            // The analysis row itself was created by the Python backend — this side
            // never writes the recruiter platform's tables. Recording the id here is
            // how the application learns it was taken up.
            row.AcceptedAnalysisId = request.AnalysisId;
            row.AcceptedAt = DateTime.UtcNow;
        }
        else
        {
            row.AcceptedAnalysisId = null;
            row.AcceptedAt = null;
        }

        await _db.SaveChangesAsync(ct);
        return await ToDtoAsync(row, await IndexedJobIdsAsync(ct), ct);
    }

    public async Task<DraftLetterDto> DraftLetterAsync(
        DraftLetterRequest request, CancellationToken ct = default)
    {
        var resume = await _db.Resumes.FirstOrDefaultAsync(r => r.Id == request.ResumeId, ct)
            ?? throw new InvalidOperationException($"No resume with id {request.ResumeId}.");

        var job = await _db.Jobs.AsNoTracking().FirstOrDefaultAsync(j => j.Id == request.JobId, ct)
            ?? throw new InvalidOperationException($"No job with id {request.JobId}.");

        // Scored here rather than taken from the client: the letter is written
        // against skill verdicts, and verdicts posted by a browser are not evidence.
        var matches = await _matching.MatchAsync(resume, new JobFilters(), null, ct);
        var match = matches.Matches.FirstOrDefault(m => m.Job.Id == request.JobId)
            ?? throw new InvalidOperationException($"'{job.Title}' could not be scored against this CV.");

        var letter = await _letters.WriteAsync(resume, job, match, ct);
        return new DraftLetterDto(job.Id, letter.Text, letter.Mode);
    }

    /// <summary>Postings this CV has already been sent to, by content hash.</summary>
    private async Task<HashSet<int>> AppliedJobIdsAsync(string contentHash, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(contentHash)) return new HashSet<int>();

        var ids = await _db.Applications
            .Where(a => a.ResumeContentHash == contentHash)
            .Select(a => a.JobId)
            .ToListAsync(ct);

        return ids.ToHashSet();
    }

    private async Task<ApplicationDto> ToDtoAsync(
        PortalApplication row, IReadOnlySet<int> indexedJobIds, CancellationToken ct)
    {
        var job = row.Job ?? await _db.Jobs.FirstAsync(j => j.Id == row.JobId, ct);
        var resume = row.Resume ?? await _db.Resumes.FirstAsync(r => r.Id == row.ResumeId, ct);

        return new ApplicationDto(
            row.Id,
            job.ToSummaryDto(indexedJobIds.Contains(job.Id)),
            resume.ToDto(),
            row.Status,
            row.CoverLetter,
            row.LetterMode,
            row.FitScore,
            row.FitBand,
            row.SemanticScore,
            row.SkillScore,
            row.TitleScore,
            row.ExperienceScore,
            Read<List<SkillAssessmentDto>>(row.SkillsJson) ?? new List<SkillAssessmentDto>(),
            Read<List<string>>(row.StrengthsJson) ?? new List<string>(),
            Read<List<string>>(row.GapsJson) ?? new List<string>(),
            row.RecruiterNote,
            row.SemanticMatching,
            row.EmbeddingModel,
            row.AcceptedAnalysisId,
            row.AcceptedAt,
            !string.IsNullOrWhiteSpace(resume.StoredPath) && File.Exists(resume.StoredPath),
            row.CreatedAt);
    }

    /// <summary>A snapshot that will not deserialise must not take the row with it.</summary>
    private T? Read<T>(string json) where T : class
    {
        try
        {
            return JsonSerializer.Deserialize<T>(json, Json);
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "An application snapshot field could not be read; showing it as empty.");
            return null;
        }
    }
}
