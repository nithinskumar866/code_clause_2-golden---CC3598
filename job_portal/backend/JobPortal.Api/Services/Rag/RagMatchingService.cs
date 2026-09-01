using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Mapping;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Rag.Agent;
using JobPortal.Api.Services.Rag.Data;
using JobPortal.Api.Services.Rag.Retrieval;
using JobPortal.Api.Services.Rag.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Rag;

/// <summary>A shortlist part-way through being judged. Ids only — the background
/// pass runs in a different DI scope from the turn that started it.</summary>
public record RagShortlist(
    int ResumeId,
    JobFilters Filters,
    string BatchId,
    int TotalCandidates,
    IReadOnlyList<JobMatchDto> Judged,
    IReadOnlyList<int> RemainingJobIds)
{
    public bool HasMore => RemainingJobIds.Count > 0;
}

public interface IRagMatchingService
{
    /// <summary>Whether the mode can run at all right now.</summary>
    Task<bool> IsAvailableAsync(CancellationToken ct = default);

    Task<RagShortlist> StartAsync(
        PortalResume resume, JobFilters filters, string batchId,
        Func<MatchResultDto, CancellationToken, Task> emit, CancellationToken ct = default);

    Task ContinueAsync(
        RagShortlist state, Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct = default);
}

/// <summary>
/// The RAG scoring mode.
///
/// The two stages, joined:
///
///   STAGE 1 — the CV's text goes into one vector search over the chunked job
///   index, filtered inside the index. Milliseconds. Its output is an ORDER, not
///   a score anybody sees: similarity cannot tell that leading three engineers
///   differs from leading three hundred.
///
///   STAGE 2 — each shortlisted posting is handed to the agent, which retrieves
///   the CV passages bearing on each of that posting's requirements and judges
///   them. Seconds per posting, so the first few are emitted immediately and the
///   rest follow after the turn ends.
///
/// The distinction from the AI-evaluation mode is not the model or the prompt: it
/// is that the model never sees a whole document. It sees the passages that
/// matched, per requirement, with their section labels.
/// </summary>
public class RagMatchingService : IRagMatchingService
{
    private readonly PortalDbContext _portal;
    private readonly IRagRetriever _retriever;
    private readonly RagEvaluationAgent _agent;
    private readonly IRagVectorStore _vectors;
    private readonly RagOptions _options;
    private readonly MatchingOptions _matching;
    private readonly ILogger<RagMatchingService> _logger;

    public RagMatchingService(
        PortalDbContext portal,
        IRagRetriever retriever,
        RagEvaluationAgent agent,
        IRagVectorStore vectors,
        IOptions<RagOptions> options,
        IOptions<MatchingOptions> matching,
        ILogger<RagMatchingService> logger)
    {
        _portal = portal;
        _retriever = retriever;
        _agent = agent;
        _vectors = vectors;
        _options = options.Value;
        _matching = matching.Value;
        _logger = logger;
    }

    public Task<bool> IsAvailableAsync(CancellationToken ct = default) =>
        _vectors.IsAvailableAsync(ct);

    public async Task<RagShortlist> StartAsync(
        PortalResume resume,
        JobFilters filters,
        string batchId,
        Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct = default)
    {
        // The filters are resolved against the portal first, and the surviving ids
        // are pushed INTO the vector search rather than applied to its output.
        var allowed = await AllowedJobIdsAsync(filters, ct);

        var query = string.IsNullOrWhiteSpace(resume.RawText) ? resume.Summary : resume.RawText;

        var shortlist = await _retriever.ShortlistAsync(
            query ?? "", RagParentTypes.Job, allowed, _options.ShortlistSize, ct);

        var state = new RagShortlist(
            resume.Id, filters, batchId, shortlist.Count,
            Judged: Array.Empty<JobMatchDto>(),
            RemainingJobIds: shortlist.Select(s => s.ParentId).ToList());

        if (shortlist.Count == 0)
        {
            await emit(Compose(state, complete: true), ct);
            return state with { RemainingJobIds = Array.Empty<int>() };
        }

        _logger.LogInformation(
            "RAG stage 1: {Count} postings shortlisted for résumé {ResumeId}.",
            shortlist.Count, resume.Id);

        var first = Math.Clamp(_matching.ReasonedFirstBatch, 1, shortlist.Count);
        return await JudgeAsync(resume, state, first, emit, ct);
    }

    public async Task ContinueAsync(
        RagShortlist state,
        Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct = default)
    {
        if (!state.HasMore) return;

        var resume = await _portal.Resumes.AsNoTracking()
            .FirstOrDefaultAsync(r => r.Id == state.ResumeId, ct);

        if (resume is null)
        {
            await emit(Compose(state with { RemainingJobIds = Array.Empty<int>() }, complete: true), ct);
            return;
        }

        await JudgeAsync(resume, state, state.RemainingJobIds.Count, emit, ct);
    }

    /// <summary>
    /// Judges the next <paramref name="count"/> postings, reporting after each.
    ///
    /// Sequential: the model runs on one GPU, and concurrent judgements do not
    /// finish sooner, they finish together — which is precisely the delay that
    /// emitting one at a time exists to avoid.
    /// </summary>
    private async Task<RagShortlist> JudgeAsync(
        PortalResume resume,
        RagShortlist state,
        int count,
        Func<MatchResultDto, CancellationToken, Task> emit,
        CancellationToken ct)
    {
        var judged = state.Judged.ToList();
        var remaining = state.RemainingJobIds.ToList();

        for (var i = 0; i < count && remaining.Count > 0; i++)
        {
            ct.ThrowIfCancellationRequested();

            var jobId = remaining[0];
            remaining.RemoveAt(0);

            var match = await JudgeOneAsync(resume, jobId, ct);
            if (match is not null) judged.Add(match);

            state = state with { Judged = judged.ToList(), RemainingJobIds = remaining.ToList() };
            await emit(Compose(state, complete: remaining.Count == 0), ct);
        }

        return state;
    }

    private async Task<JobMatchDto?> JudgeOneAsync(PortalResume resume, int jobId, CancellationToken ct)
    {
        var job = await _portal.Jobs.AsNoTracking().FirstOrDefaultAsync(j => j.Id == jobId, ct);
        if (job is null) return null;

        JobEvaluationDto? evaluation;
        try
        {
            evaluation = await _agent.EvaluateAsync(resume, job, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "RAG evaluation of job {JobId} threw.", jobId);
            evaluation = null;
        }

        if (evaluation is null) return null;

        return new JobMatchDto(
            job.ToSummaryDto(isIndexed: true),
            evaluation.OverallMatch,
            evaluation.Category,
            // The four measured dimensions belong to the computed scorer and were
            // never calculated here. Reporting zeroes would put four fabricated
            // measurements on the card, so they stay zero and the card's
            // "measured signals" fold has nothing to show for this mode.
            SemanticScore: 0, SkillScore: 0, TitleScore: 0, ExperienceScore: 0,
            JobEvaluationService.ToSkillAssessments(evaluation),
            evaluation.HighConfidence.Select(m => m.Requirement).Take(6).ToList(),
            evaluation.Gaps.Select(g => g.Requirement).Take(6).ToList(),
            string.IsNullOrWhiteSpace(evaluation.ExecutiveSummary)
                ? evaluation.Reasoning
                : evaluation.ExecutiveSummary,
            ScoringModes.Rag,
            evaluation);
    }

    /// <summary>
    /// The postings the conversation's filters allow, as ids for the vector search.
    ///
    /// Resolved here rather than after retrieval so the constraint reaches the
    /// index. Null when nothing is filtered, which lets the search run unrestricted
    /// instead of against a set containing every id in the database.
    /// </summary>
    private async Task<IReadOnlySet<int>?> AllowedJobIdsAsync(JobFilters filters, CancellationToken ct)
    {
        if (filters.IsEmpty) return null;

        var candidates = await _portal.Jobs.AsNoTracking()
            .Where(j => j.IsPublished)
            .ToListAsync(ct);

        // Reuses the predicates the other two modes filter with, so "remote only"
        // means the same thing whichever button is pressed.
        return candidates.Where(j => JobFilterRules.Passes(j, filters)).Select(j => j.Id).ToHashSet();
    }

    private MatchResultDto Compose(RagShortlist state, bool complete) =>
        new(
            state.Judged.OrderByDescending(m => m.FitScore).ToList(),
            state.TotalCandidates,
            state.Filters,
            SemanticMatching: true,
            EmbeddingModel: "",
            ScoringModes.Rag,
            state.BatchId,
            state.RemainingJobIds.Count,
            complete);
}
