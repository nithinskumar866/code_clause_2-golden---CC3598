using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Options;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Matching;

/// <summary>
/// A shortlist part-way through being judged.
///
/// Carries ids rather than entities because the rest of the work happens in a
/// different DI scope — the turn that started it has ended, its DbContext is
/// disposed, and an entity from it would throw on first touch.
/// </summary>
public record ReasonedShortlist(
    int ResumeId,
    JobFilters Filters,
    string BatchId,
    int TotalCandidates,
    bool SemanticMatching,
    string EmbeddingModel,
    IReadOnlyList<JobMatchDto> Judged,
    IReadOnlyList<JobMatchDto> Remaining)
{
    public bool HasMore => Remaining.Count > 0;
}

public interface IReasonedMatchingService
{
    /// <summary>
    /// Retrieves the shortlist and judges the first few roles, reporting each one
    /// the moment it lands. Returns what is left for a background pass.
    /// </summary>
    Task<ReasonedShortlist> StartAsync(
        PortalResume resume,
        JobFilters filters,
        string batchId,
        Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct = default);

    /// <summary>Judges everything <see cref="StartAsync"/> left behind.</summary>
    Task ContinueAsync(
        ReasonedShortlist state,
        Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct = default);
}

/// <summary>
/// Scores a shortlist by asking a model to judge each role, rather than by
/// computing a fit from measured dimensions.
///
/// Retrieval is unchanged and still deterministic: vector search picks WHICH roles
/// are worth looking at, filters remove the ones the candidate ruled out, and the
/// computed score orders what survives. That stage is cheap, reproducible, and the
/// only reason an 8B model is never asked to read fifty postings.
///
/// What changes is the verdict. Under <see cref="MatchingService"/> the percentage
/// is arithmetic over four dimensions, capped by an evidence gate; here it is the
/// model's own judgement of the résumé against the posting's stated requirements,
/// weighted the way that particular role warrants. The measured dimensions are kept
/// on the match and shown beside it, so the two readings can be compared rather than
/// one silently replacing the other.
///
/// Results are reported one at a time as they are judged. A reasoned evaluation
/// costs ten to twenty seconds, and a shortlist of ten delivered atomically would be
/// three minutes of nothing.
/// </summary>
public class ReasonedMatchingService : IReasonedMatchingService
{
    private readonly PortalDbContext _db;
    private readonly IMatchingService _matching;
    private readonly IJobEvaluationService _evaluator;
    private readonly MatchingOptions _options;
    private readonly ILogger<ReasonedMatchingService> _logger;

    public ReasonedMatchingService(
        PortalDbContext db,
        IMatchingService matching,
        IJobEvaluationService evaluator,
        IOptions<MatchingOptions> options,
        ILogger<ReasonedMatchingService> logger)
    {
        _db = db;
        _matching = matching;
        _evaluator = evaluator;
        _options = options.Value;
        _logger = logger;
    }

    public async Task<ReasonedShortlist> StartAsync(
        PortalResume resume,
        JobFilters filters,
        string batchId,
        Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct = default)
    {
        var maxJobs = Math.Max(1, _options.ReasonedMaxJobs);
        var prescored = await _matching.MatchAsync(resume, filters, topN: maxJobs, ct: ct);

        var state = new ReasonedShortlist(
            resume.Id, filters, batchId,
            prescored.TotalCandidates, prescored.SemanticMatching, prescored.EmbeddingModel,
            Judged: Array.Empty<JobMatchDto>(),
            Remaining: prescored.Matches);

        if (prescored.Matches.Count == 0)
        {
            await emit(Compose(state, complete: true), ct);
            return state with { Remaining = Array.Empty<JobMatchDto>() };
        }

        var firstBatch = Math.Clamp(_options.ReasonedFirstBatch, 1, prescored.Matches.Count);
        return await JudgeAsync(resume, state, firstBatch, emit, ct);
    }

    public async Task ContinueAsync(
        ReasonedShortlist state,
        Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct = default)
    {
        if (!state.HasMore) return;

        var resume = await _db.Resumes.AsNoTracking().FirstOrDefaultAsync(r => r.Id == state.ResumeId, ct);
        if (resume is null)
        {
            // The CV was replaced while the background pass was running. Close the
            // batch off rather than leaving the UI counting down forever.
            _logger.LogInformation(
                "Résumé {ResumeId} disappeared mid-evaluation; closing batch {BatchId}.",
                state.ResumeId, state.BatchId);
            await emit(Compose(state with { Remaining = Array.Empty<JobMatchDto>() }, complete: true), ct);
            return;
        }

        await JudgeAsync(resume, state, state.Remaining.Count, emit, ct);
    }

    /// <summary>
    /// Judges the next <paramref name="count"/> roles one at a time, reporting the
    /// shortlist after each.
    ///
    /// Sequential rather than parallel: the model runs on one box, and three
    /// concurrent evaluations do not finish in a third of the time — they finish at
    /// roughly the same moment, which is precisely the delay this design exists to
    /// avoid. One at a time means the first card appears after one evaluation.
    /// </summary>
    private async Task<ReasonedShortlist> JudgeAsync(
        PortalResume resume,
        ReasonedShortlist state,
        int count,
        Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct)
    {
        var judged = state.Judged.ToList();
        var remaining = state.Remaining.ToList();

        for (var i = 0; i < count && remaining.Count > 0; i++)
        {
            ct.ThrowIfCancellationRequested();

            var computed = remaining[0];
            remaining.RemoveAt(0);
            judged.Add(await JudgeOneAsync(resume, computed, ct));

            state = state with { Judged = judged.ToList(), Remaining = remaining.ToList() };
            await emit(Compose(state, complete: remaining.Count == 0), ct);
        }

        return state;
    }

    /// <summary>
    /// One role, judged. Falls back to the computed match — flagged as computed, so
    /// nothing claims a judgement that was never made — when the model is
    /// unreachable or its answer was unusable.
    /// </summary>
    private async Task<JobMatchDto> JudgeOneAsync(
        PortalResume resume, JobMatchDto computed, CancellationToken ct)
    {
        var job = await _db.Jobs.AsNoTracking().FirstOrDefaultAsync(j => j.Id == computed.Job.Id, ct);
        if (job is null) return computed;

        JobEvaluationDto? evaluation;
        try
        {
            evaluation = await _evaluator.EvaluateAsync(resume, job, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Reasoned evaluation of job {JobId} threw; keeping the computed score.", job.Id);
            evaluation = null;
        }

        if (evaluation is null) return computed;

        var strengths = evaluation.HighConfidence.Select(m => m.Requirement).Take(6).ToList();
        var gaps = evaluation.Gaps.Select(g => g.Requirement).Take(6).ToList();

        return computed with
        {
            // The model's number, as asked for. The four measured dimensions below
            // it are left exactly as computed: they are real measurements, and the
            // card shows them beside the judgement so the two can be compared.
            FitScore = evaluation.OverallMatch,
            FitBand = evaluation.Category,
            Skills = JobEvaluationService.ToSkillAssessments(evaluation),
            Strengths = strengths,
            Gaps = gaps,
            RecruiterNote = string.IsNullOrWhiteSpace(evaluation.ExecutiveSummary)
                ? evaluation.Reasoning
                : evaluation.ExecutiveSummary,
            ScoringMode = ScoringModes.Reasoned,
            Evaluation = evaluation,
        };
    }

    /// <summary>
    /// The shortlist as it stands, ordered by the judgement.
    ///
    /// Re-sorted on every emission, so a role judged fourth but scored highest
    /// climbs to the top rather than staying where retrieval happened to put it.
    /// Cards moving as results land is the honest behaviour: the order is a claim
    /// about fit, and holding a stale one to keep the list still would mean showing
    /// a ranking that is known to be wrong.
    /// </summary>
    private MatchResultDto Compose(ReasonedShortlist state, bool complete) =>
        new(
            state.Judged.OrderByDescending(m => m.FitScore).ToList(),
            state.TotalCandidates,
            state.Filters,
            state.SemanticMatching,
            state.EmbeddingModel,
            ScoringModes.Reasoned,
            state.BatchId,
            state.Remaining.Count,
            complete);
}
