using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Mapping;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Matching;

public interface IMatchingService
{
    Task<MatchResultDto> MatchAsync(
        PortalResume resume, JobFilters filters, int? topN = null, CancellationToken ct = default);

    /// <summary>
    /// How many jobs clear the semantic floor and the filters, without doing the
    /// per-job skill work. Used to decide whether to narrow the conversation
    /// before presenting a list.
    /// </summary>
    Task<int> CountCandidatesAsync(PortalResume resume, JobFilters filters, CancellationToken ct = default);
}

/// <summary>
/// Semantic matching between a candidate and the job board.
///
/// The pipeline, in order:
///   1. Retrieve by vector similarity. This is the stage that sees past
///      vocabulary: a resume saying "React" is near a posting asking for a
///      "modern frontend framework" because the vectors are near, with no
///      synonym list anywhere.
///   2. Apply the conversation's filters, deterministically. A candidate who says
///      "remote only" means it, and no similarity score should be able to
///      overrule that.
///   3. Score what survives on four dimensions and combine them.
///
/// Retrieval comes before filtering because filters are cheap and exact while
/// similarity is the expensive part, and because a filter must never be able to
/// reach into the ranking and change the order of what it did not remove.
/// </summary>
public class MatchingService : IMatchingService
{
    private readonly PortalDbContext _db;
    private readonly IEmbeddingService _embeddings;
    private readonly IVectorStore _vectors;
    private readonly ISkillSemanticsService _skills;
    private readonly ISkillTransferService _transfers;
    private readonly IResumeEvidenceService _evidence;
    private readonly MatchingOptions _options;
    private readonly ILogger<MatchingService> _logger;

    public MatchingService(
        PortalDbContext db,
        IEmbeddingService embeddings,
        IVectorStore vectors,
        ISkillSemanticsService skills,
        ISkillTransferService transfers,
        IResumeEvidenceService evidence,
        IOptions<MatchingOptions> options,
        ILogger<MatchingService> logger)
    {
        _db = db;
        _embeddings = embeddings;
        _vectors = vectors;
        _skills = skills;
        _transfers = transfers;
        _evidence = evidence;
        _options = options.Value;
        _logger = logger;
    }

    public async Task<int> CountCandidatesAsync(
        PortalResume resume, JobFilters filters, CancellationToken ct = default)
    {
        var (hits, jobs) = await RetrieveAsync(resume, filters, ct);
        _ = jobs;
        return hits.Count;
    }

    public async Task<MatchResultDto> MatchAsync(
        PortalResume resume, JobFilters filters, int? topN = null, CancellationToken ct = default)
    {
        var model = _embeddings.PreferredModelId;
        var (hits, jobsById) = await RetrieveAsync(resume, filters, ct);

        var take = topN ?? _options.TopN;
        var shortlist = hits.Take(take).ToList();

        if (shortlist.Count == 0)
        {
            return new MatchResultDto(
                Array.Empty<JobMatchDto>(), hits.Count, filters, _embeddings.SemanticAvailable, model);
        }

        var candidateSkills = TextStructure.ReadLines(resume.Skills);

        // Every skill name across the shortlist plus the candidate's own, embedded
        // in a single round trip. Per-job embedding would multiply the network
        // cost by the size of the shortlist for no extra information.
        var vocabulary = shortlist
            .Select(h => jobsById[h.JobId])
            .SelectMany(j => TextStructure.ReadLines(j.RequiredSkills))
            .Concat(candidateSkills)
            .ToList();

        var skillVectors = await _skills.EmbedSkillsAsync(vocabulary, ct);

        var matches = new List<JobMatchDto>(shortlist.Count);
        foreach (var hit in shortlist)
        {
            ct.ThrowIfCancellationRequested();
            var job = jobsById[hit.JobId];
            var scored = Score(job, hit.Score, resume, candidateSkills, skillVectors);
            var gate = EvidenceGateFor(job, resume);
            matches.Add(await ApplyTransfersAsync(scored, candidateSkills, gate, ct));
        }

        // Re-ordered by the composite score: vector order got us the right
        // candidates, but it only knows about overall similarity, not about
        // whether the specific must-have skills are actually present.
        matches = matches.OrderByDescending(m => m.FitScore).ToList();

        return new MatchResultDto(matches, hits.Count, filters, _embeddings.SemanticAvailable, model);
    }

    /// <summary>
    /// Re-judges the Missing requirements with the transfer reasoner, then
    /// rebuilds the parts of the match that depend on the verdicts.
    ///
    /// The score is recomputed rather than left alone: a transferable skill is
    /// worth partial credit, and a fit percentage that ignored the upgrade would
    /// disagree with the note printed directly beneath it.
    /// </summary>
    private async Task<JobMatchDto> ApplyTransfersAsync(
        JobMatchDto match, IReadOnlyList<string> candidateSkills, double evidenceGate, CancellationToken ct)
    {
        var missing = match.Skills
            .Where(s => s.Status == "Missing")
            .Select(s => s.Skill)
            .ToList();

        if (missing.Count == 0) return match;

        IReadOnlyList<SkillTransfer> transfers;
        try
        {
            transfers = await _transfers.FindTransfersAsync(match.Job.Id, match.Job.Title, missing, candidateSkills, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // The deterministic verdicts already in hand are a complete answer.
            _logger.LogWarning(ex, "Transfer reasoning failed for job {JobId}; keeping Have/Missing only.",
                match.Job.Id);
            return match;
        }

        if (transfers.Count == 0) return match;

        var byRequirement = transfers.ToDictionary(t => t.Requirement, StringComparer.OrdinalIgnoreCase);

        var skills = match.Skills
            .Select(s => byRequirement.TryGetValue(s.Skill, out var transfer) && s.Status == "Missing"
                // Similarity stays as measured. Overwriting it with a number the
                // model implied would put a fake measurement in the UI tooltip.
                ? s with { Status = "Transferable", EvidenceSkill = transfer.CandidateSkill }
                : s)
            .ToList();

        var skillScore = SkillScore(skills);
        var fit = _options.WeightSemantic * (match.SemanticScore / 100.0)
                + _options.WeightSkillOverlap * skillScore
                + _options.WeightTitle * (match.TitleScore / 100.0)
                + _options.WeightExperience * (match.ExperienceScore / 100.0);

        // Same gate as the first pass. Without it, a transferable upgrade would
        // quietly lift a role back above the ceiling its evidence never cleared.
        var fitPercent = Math.Round(Math.Clamp(fit * evidenceGate, 0, 1) * 100, 1);

        return match with
        {
            Skills = skills,
            SkillScore = Math.Round(skillScore * 100, 1),
            FitScore = fitPercent,
            FitBand = Band(fitPercent),
            Gaps = skills.Where(s => s.Status == "Missing").Select(s => s.Skill).Take(6).ToList(),
            RecruiterNote = RecruiterNoteWithTransfers(match, skills, transfers),
        };
    }

    /// <summary>
    /// The recruiter note, using the reasoner's own justification for each
    /// transfer. This is the "you have heavy AWS, so frame it as general cloud
    /// knowledge" advice: it is specific because the reason came from comparing
    /// two named skills, not from a template.
    /// </summary>
    private static string RecruiterNoteWithTransfers(
        JobMatchDto match, IReadOnlyList<SkillAssessmentDto> skills, IReadOnlyList<SkillTransfer> transfers)
    {
        var have = skills.Where(s => s.Status == "Have").Select(s => s.Skill).ToList();
        var missing = skills.Where(s => s.Status == "Missing").Select(s => s.Skill).ToList();

        var parts = new List<string>();

        if (have.Count > 0)
        {
            parts.Add($"You match on {string.Join(", ", have.Take(4))}" +
                      (have.Count > 4 ? $" and {have.Count - 4} more." : "."));
        }

        foreach (var transfer in transfers.Take(2))
        {
            var reason = string.IsNullOrWhiteSpace(transfer.Reason)
                ? $"your {transfer.CandidateSkill} experience is closely related"
                : transfer.Reason;

            parts.Add($"They ask for {transfer.Requirement}, which you have not listed — " +
                      $"lead with your {transfer.CandidateSkill}: {reason}");
        }

        if (missing.Count > 0)
        {
            parts.Add($"Genuinely absent: {string.Join(", ", missing.Take(4))}. " +
                      "If you have touched any of these, add them explicitly.");
        }

        return parts.Count > 0 ? string.Join(" ", parts) : match.RecruiterNote;
    }

    // -- retrieval ----------------------------------------------------------

    private async Task<(IReadOnlyList<VectorHit> Hits, Dictionary<int, PortalJob> Jobs)> RetrieveAsync(
        PortalResume resume, JobFilters filters, CancellationToken ct)
    {
        var model = _embeddings.PreferredModelId;
        var query = await GetResumeVectorAsync(resume, model, ct);

        if (query.Length == 0)
        {
            _logger.LogWarning("Resume {ResumeId} has no usable vector; no matches can be produced.", resume.Id);
            return (Array.Empty<VectorHit>(), new Dictionary<int, PortalJob>());
        }

        // Retrieve wide, then filter. The pool is deliberately larger than what
        // will be shown: filters remove jobs after retrieval, and retrieving only
        // TopN would leave nothing behind once "remote only" is applied.
        var poolSize = Math.Max(_options.CandidatePoolSize, _options.TopN * 4);

        var hits = await _vectors.SearchAsync(
            query, model, poolSize, _options.MinSimilarity, restrictTo: null, ct);

        if (hits.Count == 0) return (hits, new Dictionary<int, PortalJob>());

        var ids = hits.Select(h => h.JobId).ToList();
        var jobs = await _db.Jobs
            .AsNoTracking()
            .Where(j => ids.Contains(j.Id) && j.IsPublished)
            .ToDictionaryAsync(j => j.Id, ct);

        var kept = hits
            .Where(h => jobs.ContainsKey(h.JobId))
            .Where(h => Passes(jobs[h.JobId], filters))
            .ToList();

        return (kept, jobs);
    }

    /// <summary>
    /// The candidate's vector, reused from storage when it was produced by the
    /// model currently in use and recomputed when it was not. A vector from a
    /// different model is not stale, it is incomparable, and using it would score
    /// every job against the wrong space.
    /// </summary>
    private async Task<float[]> GetResumeVectorAsync(PortalResume resume, string model, CancellationToken ct)
    {
        if (resume.Embedding is { Length: > 0 } && resume.EmbeddingModel == model)
        {
            return VectorMath.FromBytes(resume.Embedding);
        }

        var text = string.IsNullOrWhiteSpace(resume.RawText) ? resume.Summary : resume.RawText;
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<float>();

        var batch = await _embeddings.EmbedAsync(TextStructure.Clip(text, 4000), ct);
        return batch.Vectors.Count > 0 ? batch.Vectors[0] : Array.Empty<float>();
    }

    // -- filters ------------------------------------------------------------

    /// <summary>Shared with the resume-less browse path via <see cref="JobFilterRules"/>.</summary>
    private static bool Passes(PortalJob job, JobFilters filters) => JobFilterRules.Passes(job, filters);

    /// <summary>
    /// The most a role can score on nothing but wording.
    ///
    /// Not zero, deliberately: a role the candidate has no evidence for may still
    /// be worth showing — that is exactly the "you know AWS, this is Azure" case —
    /// and hiding it would be as wrong as recommending it. It appears, near the
    /// bottom, with the reason attached.
    /// </summary>
    private const double EvidenceFloor = 0.20;

    /// <summary>
    /// The ceiling this candidate's evidence puts on one role, 0.20 to 1.0.
    ///
    /// Computed in one place so the first scoring pass and the transfer re-score
    /// cannot disagree — they did, and the effect was that upgrading a single
    /// skill to Transferable lifted an unevidenced role straight back over the bar.
    /// </summary>
    private double EvidenceGateFor(PortalJob job, PortalResume resume)
    {
        // Every stated requirement, not just the tidy ones. Filtering to
        // skill-shaped entries here meant a posting whose requirements were all
        // extracted as sentences had nothing left to measure, scored full marks by
        // default, and out-ranked postings that were parsed properly.
        var required = TextStructure.ReadLines(job.RequiredSkills);
        var coverage = _evidence.Build(resume).CoverageOf(required);
        return EvidenceFloor + (1 - EvidenceFloor) * coverage;
    }

    /// <summary>
    /// Whether a stated requirement is a skill rather than a whole responsibility
    /// sentence the extractor put in the skills array. Nothing can evidence
    /// "Thorough knowledge of employment laws", so counting it as an unmet
    /// requirement would penalise every candidate equally and mean nothing.
    /// </summary>
    private static bool IsSkillShaped(string value)
    {
        var trimmed = TextStructure.Collapse(value ?? "");
        if (trimmed.Length is < 2 or > 40) return false;
        return trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length <= 4;
    }

    // -- scoring ------------------------------------------------------------

    private JobMatchDto Score(
        PortalJob job,
        double semanticScore,
        PortalResume resume,
        IReadOnlyList<string> candidateSkills,
        IReadOnlyDictionary<string, float[]> skillVectors)
    {
        var required = TextStructure.ReadLines(job.RequiredSkills);
        var assessments = _skills.Assess(required, candidateSkills, skillVectors);

        var skillScore = SkillScore(assessments);
        var titleScore = TitleScore(job, resume, skillVectors);
        var experienceScore = ExperienceScore(job, resume);

        // Cosine runs -1..1 but is effectively 0..1 for text; clamping keeps a
        // stray negative from dragging a composite percentage below zero.
        var semantic = Math.Clamp(semanticScore, 0, 1);

        var fit = _options.WeightSemantic * semantic
                + _options.WeightSkillOverlap * skillScore
                + _options.WeightTitle * titleScore
                + _options.WeightExperience * experienceScore;

        // -- the evidence gate --------------------------------------------------
        //
        // Everything above this line can be moved by words alone. A résumé that
        // says "I would like to move into HR" drifts toward HR postings as a
        // vector, matches their vocabulary, and comes back with a confident
        // percentage behind which there is nothing at all.
        //
        // So the composite is capped by what the document can actually back up:
        // where each requirement appears, and how deeply it is corroborated. With
        // no evidence the ceiling is a fifth of the raw score — enough for the role
        // to still be visible and explained, nowhere near enough to be recommended.
        var fitPercent = Math.Round(Math.Clamp(fit * EvidenceGateFor(job, resume), 0, 1) * 100, 1);

        var strengths = assessments
            .Where(a => a.Status == "Have")
            .Select(a => a.Skill)
            .Take(6)
            .ToList();

        var gaps = assessments
            .Where(a => a.Status == "Missing")
            .Select(a => a.Skill)
            .Take(6)
            .ToList();

        return new JobMatchDto(
            job.ToSummaryDto(isIndexed: true),
            fitPercent,
            Band(fitPercent),
            Math.Round(semantic * 100, 1),
            Math.Round(skillScore * 100, 1),
            Math.Round(titleScore * 100, 1),
            Math.Round(experienceScore * 100, 1),
            assessments,
            strengths,
            gaps,
            RecruiterNote(job, assessments, resume));
    }

    /// <summary>
    /// Fraction of required skills the candidate covers, with partial credit for
    /// adjacent ones. A transferable skill is worth real but not full credit:
    /// someone with AWS applying to an Azure role genuinely is closer than someone
    /// with neither, and genuinely is not the same as someone with Azure.
    /// </summary>
    private static double SkillScore(IReadOnlyList<SkillAssessmentDto> assessments)
    {
        // A posting whose requirements could not be parsed should not be scored at
        // zero on this dimension; that would punish the candidate for the document.
        if (assessments.Count == 0) return 0.5;

        var earned = assessments.Sum(a => a.Status switch
        {
            "Have" => 1.0,
            "Transferable" => 0.5,
            _ => 0.0,
        });

        return earned / assessments.Count;
    }

    private static double TitleScore(
        PortalJob job, PortalResume resume, IReadOnlyDictionary<string, float[]> skillVectors)
    {
        if (string.IsNullOrWhiteSpace(job.Title) || string.IsNullOrWhiteSpace(resume.CurrentTitle)) return 0.5;

        // Titles are not in the skill vocabulary, so this stays lexical: a shared
        // significant word ("Engineer", "Analyst", "Designer") is weak evidence of
        // the same kind of work. Cheap, and it never claims more than it knows.
        var jobWords = Words(job.Title);
        var resumeWords = Words(resume.CurrentTitle);
        if (jobWords.Count == 0 || resumeWords.Count == 0) return 0.5;

        var overlap = jobWords.Intersect(resumeWords, StringComparer.OrdinalIgnoreCase).Count();
        var score = (double)overlap / Math.Max(jobWords.Count, resumeWords.Count);

        _ = skillVectors;
        // Never below neutral: a different title is not evidence against a
        // candidate, only an absence of evidence for them.
        return Math.Max(0.5, score);
    }

    private static List<string> Words(string title) =>
        title.Split(new[] { ' ', '-', '/', ',', '(', ')' }, StringSplitOptions.RemoveEmptyEntries)
             .Select(w => w.Trim().ToLowerInvariant())
             .Where(w => w.Length > 2)
             .Distinct()
             .ToList();

    private static double ExperienceScore(PortalJob job, PortalResume resume)
    {
        // Neither side stated it: no signal, so neutral rather than a penalty.
        if (job.MinYearsExperience is not { } required || resume.YearsExperience is not { } actual) return 0.5;

        if (actual >= required)
        {
            // Comfortably over the bar is fine, but far over it is a real signal
            // that the role is below the candidate's level, so it eases off rather
            // than rewarding indefinitely.
            var excess = actual - required;
            return excess <= 5 ? 1.0 : Math.Max(0.7, 1.0 - (excess - 5) * 0.03);
        }

        // Under the bar: a year short is close, five years short is not. Linear
        // to zero over five years keeps the penalty proportionate.
        var shortfall = required - actual;
        return Math.Max(0, 1.0 - shortfall / 5.0);
    }

    private static string Band(double fitPercent) => fitPercent switch
    {
        >= 85 => "Excellent fit",
        >= 70 => "Strong fit",
        >= 50 => "Moderate fit",
        _ => "Worth a look",
    };

    /// <summary>
    /// The deterministic recruiter note.
    ///
    /// Always produced, so the portal says something useful with no LLM at all.
    /// The chat replaces it with a written-out version when a model is available,
    /// but the substance — which skills are covered, which are transferable from
    /// what, which are genuinely absent — is decided here, from the assessment,
    /// and the model only gets to phrase it.
    /// </summary>
    private static string RecruiterNote(
        PortalJob job, IReadOnlyList<SkillAssessmentDto> assessments, PortalResume resume)
    {
        var have = assessments.Where(a => a.Status == "Have").Select(a => a.Skill).ToList();
        var transferable = assessments.Where(a => a.Status == "Transferable").ToList();
        var missing = assessments.Where(a => a.Status == "Missing").Select(a => a.Skill).ToList();

        var parts = new List<string>();

        if (have.Count > 0)
        {
            parts.Add($"You match on {string.Join(", ", have.Take(4))}" +
                      (have.Count > 4 ? $" and {have.Count - 4} more." : "."));
        }

        foreach (var skill in transferable.Take(2))
        {
            parts.Add($"They ask for {skill.Skill}, which you have not listed, but your " +
                      $"{skill.EvidenceSkill} experience is closely related. Frame it as transferable " +
                      "rather than leaving the gap unaddressed.");
        }

        if (missing.Count > 0)
        {
            parts.Add($"Not evidenced on your resume: {string.Join(", ", missing.Take(4))}. " +
                      "If you have touched any of these, add them explicitly.");
        }

        if (job.MinYearsExperience is { } required && resume.YearsExperience is { } actual && actual < required)
        {
            parts.Add($"The posting asks for {required:0.#} years and your resume shows about " +
                      $"{actual:0.#}. Worth applying if your recent work is senior in scope.");
        }

        return parts.Count > 0
            ? string.Join(" ", parts)
            : "Your profile is a broad match for this role. Add specific technologies to your resume " +
              "to get a sharper read.";
    }
}
