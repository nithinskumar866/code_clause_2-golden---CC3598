using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Chat;

/// <summary>
/// The executors. Each turns a validated <see cref="TurnPlan"/> into the
/// <see cref="FactPackage"/> the answer will be written from.
///
/// No model runs in this file. That is the whole point of the split: the planner
/// decides WHAT was asked, this decides WHAT IS TRUE, and only then does a model get
/// to phrase it. Every number the candidate eventually reads is computed here.
///
/// Every method returns a package or null. Null means "no anchor" — the posting was
/// not found, or the board is empty — and the caller turns that into a clarify rather
/// than an answer, because <see cref="FactPackage"/> refuses to be built unanchored.
/// </summary>
public class ChatCapabilities
{
    private readonly PortalDbContext _db;
    private readonly ISkillSemanticsService _skills;
    private readonly MatchingOptions _options;
    private readonly ILogger<ChatCapabilities> _logger;

    public ChatCapabilities(
        PortalDbContext db,
        ISkillSemanticsService skills,
        IOptions<MatchingOptions> options,
        ILogger<ChatCapabilities> logger)
    {
        _db = db;
        _skills = skills;
        _options = options.Value;
        _logger = logger;
    }

    private Task<List<PortalJob>> LoadAsync(IEnumerable<int> ids, CancellationToken ct)
    {
        var wanted = ids.Distinct().ToList();
        return _db.Jobs.AsNoTracking()
            .Where(j => wanted.Contains(j.Id))
            .ToListAsync(ct);
    }

    /// <summary>
    /// Facts about one or more postings — pay, location, hours, requirements.
    /// </summary>
    public async Task<FactPackage?> FactAboutAsync(
        IReadOnlyList<int> jobIds, IReadOnlyList<JobMatchDto> lastMatches, CancellationToken ct)
    {
        var jobs = await LoadAsync(jobIds, ct);
        if (jobs.Count == 0) return null;

        return new FactPackage(
            jobs.Select(JobFact.From).ToList(),
            jobs.ToDictionary(j => j.Id, CompanyFacts.From),
            Coverage(jobs.Select(j => j.Id), lastMatches));
    }

    /// <summary>
    /// What the JD says about the employer — and only that.
    ///
    /// The package deliberately still carries the posting, so a company question that
    /// drifts into "and what does it pay" is answerable from the same facts rather
    /// than needing a second turn.
    /// </summary>
    public async Task<FactPackage?> CompanyFactAsync(IReadOnlyList<int> jobIds, CancellationToken ct)
    {
        var jobs = await LoadAsync(jobIds, ct);
        if (jobs.Count == 0) return null;

        var companies = jobs.ToDictionary(j => j.Id, CompanyFacts.From);
        var notes = new Dictionary<string, string>();

        // Stated explicitly so the composer has something true to say instead of
        // reaching for what it knows about a real employer.
        if (companies.Values.All(c => !c.SaysAnything))
            notes["NOTE"] = "The posting says nothing about the employer beyond its name and location.";

        return new FactPackage(jobs.Select(JobFact.From).ToList(), companies, notes: notes);
    }

    /// <summary>Two or more postings side by side, with the candidate's coverage of each.</summary>
    public async Task<FactPackage?> CompareAsync(
        IReadOnlyList<int> jobIds, IReadOnlyList<JobMatchDto> lastMatches, CancellationToken ct)
    {
        var jobs = await LoadAsync(jobIds, ct);
        if (jobs.Count == 0) return null;

        var notes = new Dictionary<string, string>();
        foreach (var match in lastMatches.Where(m => jobs.Any(j => j.Id == m.Job.Id)))
            notes[$"FIT job {match.Job.Id}"] = $"{match.FitScore:0.#}% ({match.FitBand})";

        return new FactPackage(
            jobs.Select(JobFact.From).ToList(),
            jobs.ToDictionary(j => j.Id, CompanyFacts.From),
            Coverage(jobs.Select(j => j.Id), lastMatches),
            notes);
    }

    /// <summary>
    /// What to learn for ONE posting — "what should I study for the data science job".
    /// </summary>
    public async Task<FactPackage?> GapPlanAsync(
        IReadOnlyList<int> jobIds, IReadOnlyList<JobMatchDto> lastMatches, CancellationToken ct)
    {
        var jobs = await LoadAsync(jobIds.Take(1), ct);
        if (jobs.Count == 0) return null;

        var job = jobs[0];
        var coverage = Coverage(new[] { job.Id }, lastMatches);

        var notes = new Dictionary<string, string>();
        var missing = coverage.Where(c => c.Status is "Missing").Select(c => c.Requirement).ToList();
        notes["TO CLOSE"] = missing.Count > 0
            ? string.Join(", ", missing)
            // No match record: the CV was never scored against this posting, so the
            // requirements are stated plainly rather than dressed up as gaps.
            : coverage.Count == 0
                ? $"not yet assessed against this posting; its requirements are: " +
                  string.Join(", ", TextStructure.ReadLines(job.RequiredSkills))
                : "nothing — every stated requirement is evidenced";

        return new FactPackage(
            new[] { JobFact.From(job) },
            new Dictionary<int, CompanyFacts> { [job.Id] = CompanyFacts.From(job) },
            coverage, notes);
    }

    /// <summary>What the candidate lacks across the roles currently on screen.</summary>
    public async Task<FactPackage?> GapsAcrossShortlistAsync(
        IReadOnlyList<JobMatchDto> lastMatches, CancellationToken ct)
    {
        if (lastMatches.Count == 0) return null;

        var jobs = await LoadAsync(lastMatches.Select(m => m.Job.Id), ct);
        if (jobs.Count == 0) return null;

        var coverage = Coverage(jobs.Select(j => j.Id), lastMatches);

        // Ranked by how many of the shortlisted roles want it: a gap that blocks four
        // roles is worth more of the candidate's evening than one that blocks one.
        var ranked = coverage
            .Where(c => c.Status == "Missing")
            .GroupBy(c => c.Requirement, StringComparer.OrdinalIgnoreCase)
            .OrderByDescending(g => g.Count())
            .Select(g => $"{g.Key} ({g.Count()} of {jobs.Count} roles)")
            .Take(10)
            .ToList();

        var notes = new Dictionary<string, string>
        {
            ["ROLES ON SCREEN"] = jobs.Count.ToString(),
            ["MISSING MOST OFTEN"] = ranked.Count > 0 ? string.Join("; ", ranked) : "nothing across these roles",
        };

        return new FactPackage(
            jobs.Select(JobFact.From).ToList(),
            jobs.ToDictionary(j => j.Id, CompanyFacts.From),
            coverage, notes);
    }

    /// <summary>
    /// What to learn next, ranked by demand across every open role.
    ///
    /// The answer to "what should I skill up on" comes from THIS BOARD's demand, not
    /// from the resume alone and not from what a model believes is hot right now.
    /// "Kubernetes appears in 7 of 13 open roles and is not evidenced on your CV" is
    /// a defensible sentence; "learn Kubernetes, it's in demand" is not.
    /// </summary>
    public async Task<FactPackage?> SkillUpAcrossBoardAsync(
        IReadOnlyList<JobMatchDto> lastMatches, CancellationToken ct)
    {
        var published = await _db.Jobs.AsNoTracking().Where(j => j.IsPublished).ToListAsync(ct);
        if (published.Count == 0) return null;

        // What the CV already evidences, learned from any scoring already done.
        var evidenced = lastMatches
            .SelectMany(m => m.Skills)
            .Where(s => s.Status is "Have" or "Transferable")
            .Select(s => s.Skill)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var requirements = published
            .SelectMany(j => TextStructure.ReadLines(j.RequiredSkills))
            .Where(skill => !string.IsNullOrWhiteSpace(skill))
            .ToList();

        var demand = (await RankDemandAsync(requirements, evidenced, published.Count, ct))
            .Take(10)
            .ToList();

        var notes = new Dictionary<string, string>
        {
            ["OPEN ROLES"] = published.Count.ToString(),
            ["MOST DEMANDED AND NOT EVIDENCED"] = demand.Count > 0
                ? string.Join("; ", demand)
                : "nothing — the CV evidences every requirement the board asks for",
        };

        // Anchored to the roles that drive the ranking, so the advice stays traceable
        // to real openings rather than becoming free-floating career guidance.
        var anchors = published.Take(8).Select(JobFact.From).ToList();
        return new FactPackage(anchors, notes: notes);
    }

    /// <summary>
    /// Why a score is what it is — answered from the STORED result.
    ///
    /// Never re-scored. Recomputing while defending a number is how the recruiter
    /// assistant once produced a different score inside its own explanation.
    /// </summary>
    public async Task<FactPackage?> ExplainAsync(
        IReadOnlyList<int> jobIds, IReadOnlyList<JobMatchDto> lastMatches, CancellationToken ct)
    {
        var wanted = jobIds.Count > 0 ? jobIds : lastMatches.Take(1).Select(m => m.Job.Id).ToList();
        var jobs = await LoadAsync(wanted, ct);
        if (jobs.Count == 0) return null;

        var notes = new Dictionary<string, string>();
        foreach (var match in lastMatches.Where(m => wanted.Contains(m.Job.Id)))
        {
            notes[$"FIT job {match.Job.Id}"] =
                $"{match.FitScore:0.#}% ({match.FitBand}) — relevance {match.SemanticScore:0.#}%, " +
                $"skills {match.SkillScore:0.#}%, title {match.TitleScore:0.#}%, " +
                $"experience {match.ExperienceScore:0.#}%";
            if (match.Strengths.Count > 0)
                notes[$"STRENGTHS job {match.Job.Id}"] = string.Join(", ", match.Strengths);
            if (match.Gaps.Count > 0)
                notes[$"GAPS job {match.Job.Id}"] = string.Join(", ", match.Gaps);
        }

        return new FactPackage(
            jobs.Select(JobFact.From).ToList(),
            jobs.ToDictionary(j => j.Id, CompanyFacts.From),
            Coverage(wanted, lastMatches), notes);
    }

    /// <summary>
    /// Anything else. Everything known about the anchored postings goes in, and the
    /// answer either comes out of it or does not come out at all.
    /// </summary>
    public async Task<FactPackage?> OpenQuestionAsync(
        IReadOnlyList<int> jobIds, IReadOnlyList<JobMatchDto> lastMatches, CancellationToken ct)
    {
        var jobs = await LoadAsync(jobIds, ct);
        if (jobs.Count == 0) return null;

        var notes = new Dictionary<string, string>();
        foreach (var job in jobs)
        {
            if (!string.IsNullOrWhiteSpace(job.Responsibilities))
                notes[$"RESPONSIBILITIES job {job.Id}"] = TextStructure.Collapse(job.Responsibilities);
            if (!string.IsNullOrWhiteSpace(job.Qualifications))
                notes[$"QUALIFICATIONS job {job.Id}"] = TextStructure.Collapse(job.Qualifications);
        }

        return new FactPackage(
            jobs.Select(JobFact.From).ToList(),
            jobs.ToDictionary(j => j.Id, CompanyFacts.From),
            Coverage(jobs.Select(j => j.Id), lastMatches), notes);
    }

    /// <summary>
    /// Ranks what the board asks for and the CV does not evidence — SEMANTICALLY.
    ///
    /// Grouping requirement strings by equality was the obvious implementation and the
    /// wrong one. "React", "React.js" and "ReactJS" are one demand written three ways;
    /// counting them separately understates all three, and a CV that says "ReactJS"
    /// would never cancel a JD that says "React". That is an algorithm tuned to one
    /// spelling of one CV, which is exactly what must not happen here — a different
    /// candidate or a differently-worded posting silently gets a worse answer.
    ///
    /// So requirements are embedded and clustered by cosine, and "does the CV already
    /// have this" is a cosine test too. The threshold is the same
    /// <see cref="MatchingOptions.SkillEquivalenceMin"/> the matcher uses, which was
    /// calibrated ABOVE the highest observed false positive: a wrong "you already have
    /// this" tells someone not to study something they cannot do.
    ///
    /// Falls back to case-insensitive grouping when embeddings are unavailable. That
    /// is worse, and it is still an answer.
    /// </summary>
    private async Task<List<string>> RankDemandAsync(
        IReadOnlyList<string> requirements,
        IReadOnlySet<string> evidenced,
        int roleCount,
        CancellationToken ct)
    {
        var distinct = requirements.Distinct(StringComparer.OrdinalIgnoreCase).ToList();

        IReadOnlyDictionary<string, float[]>? vectors = null;
        try
        {
            vectors = await _skills.EmbedSkillsAsync(distinct.Concat(evidenced), ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Could not embed skills for demand ranking; grouping lexically.");
        }

        bool Same(string a, string b)
        {
            if (a.Equals(b, StringComparison.OrdinalIgnoreCase)) return true;
            if (vectors is null) return false;
            return vectors.TryGetValue(a, out var va) && vectors.TryGetValue(b, out var vb)
                   && VectorMath.Cosine(va, vb) >= _options.SkillEquivalenceMin;
        }

        // Greedy clustering: the first spelling encountered names the cluster, and the
        // count is how many POSTINGS asked for anything in it.
        var clusters = new List<(string Label, List<string> Members)>();
        foreach (var requirement in distinct)
        {
            var existing = clusters.FirstOrDefault(c => Same(c.Label, requirement));
            if (existing.Members is not null) existing.Members.Add(requirement);
            else clusters.Add((requirement, new List<string> { requirement }));
        }

        var ranked = new List<(string Label, int Count)>();
        foreach (var (label, members) in clusters)
        {
            // Already covered by the CV, however either side spells it.
            if (evidenced.Any(have => members.Any(m => Same(have, m)))) continue;

            var asks = requirements.Count(r => members.Contains(r, StringComparer.OrdinalIgnoreCase));
            ranked.Add((label, asks));
        }

        return ranked
            .OrderByDescending(r => r.Count)
            .ThenBy(r => r.Label, StringComparer.OrdinalIgnoreCase)
            .Select(r => $"{r.Label} — asked for by {r.Count} of {roleCount} open roles")
            .ToList();
    }

    /// <summary>
    /// The candidate's coverage of the given postings, from results already computed.
    ///
    /// This is the ONLY place resume information enters a package, and it is always
    /// tied to a requirement of a real posting — the job-anchor rule in practice.
    /// </summary>
    private static List<RequirementCoverage> Coverage(
        IEnumerable<int> jobIds, IReadOnlyList<JobMatchDto> lastMatches)
    {
        var wanted = jobIds.ToHashSet();
        var coverage = new List<RequirementCoverage>();

        foreach (var match in lastMatches.Where(m => wanted.Contains(m.Job.Id)))
        {
            foreach (var skill in match.Skills)
            {
                coverage.Add(new RequirementCoverage(
                    match.Job.Id, skill.Skill, skill.Status,
                    // The transfer reasoner's evidence, when it found any. A bare
                    // "Transferable" with nothing behind it is not worth showing.
                    string.IsNullOrWhiteSpace(skill.EvidenceSkill) ? null : skill.EvidenceSkill));
            }
        }
        return coverage;
    }
}
