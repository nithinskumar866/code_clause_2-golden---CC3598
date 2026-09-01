using System.Text;
using System.Text.Json;
using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Llm;
using Microsoft.Extensions.Caching.Memory;

namespace JobPortal.Api.Services.Matching;

public interface IJobEvaluationService
{
    /// <summary>
    /// Judges one résumé against one posting with the model doing the reasoning.
    ///
    /// Returns null when no model is configured — the caller falls back to the
    /// deterministic score rather than showing nothing.
    /// </summary>
    Task<JobEvaluationDto?> EvaluateAsync(
        PortalResume resume, PortalJob job, CancellationToken ct = default);
}

/// <summary>
/// Evidence-based evaluation, reasoned by the model rather than computed.
///
/// This is the deliberate counterpart to <see cref="MatchingService"/>. That one
/// computes a fit from four weighted dimensions and can explain every number;
/// this one hands the whole posting and the whole CV to a model and asks it to
/// judge them the way a recruiter would — anchored to the posting's own stated
/// requirements, weighted by WHERE the evidence sits in the résumé, and capped
/// hard when a mandatory requirement is missing entirely.
///
/// Both exist on purpose. The computed score is instant, reproducible and
/// auditable; the reasoned one is slower and less repeatable but explains itself
/// in the language a candidate actually needs — which requirement, what proof,
/// and what to do about it.
///
/// What the model is NOT trusted with: the evidence it cites. Every quote it
/// attributes to the résumé is checked against the document, and anything it
/// could not have read there is dropped. It may judge; it may not invent.
/// </summary>
public class JobEvaluationService : IJobEvaluationService
{
    /// <summary>
    /// Bumped whenever the prompt below changes.
    ///
    /// A stored evaluation is only interpretable against the instructions that
    /// produced it — two scores from different prompts are not comparable, and a
    /// frozen snapshot that does not record which one it came from cannot be
    /// audited later.
    /// </summary>
    public const string PromptVersion = "eval-v7";

    private const int MaxResumeChars = 9000;
    private const int MaxJobChars = 5000;

    /// <summary>
    /// Sampling temperature for a judgement.
    ///
    /// Zero, not the configured 0.2. The score is the model's own opinion, which
    /// means nothing else in the system can steady it — so the one thing that can
    /// is greedy decoding. At 0.2 an 8B model returned 41% and 67% for the same
    /// résumé and posting on consecutive asks, and a recruiter who re-reads an
    /// answer and sees a different number stops believing both.
    /// </summary>
    private const double JudgementTemperature = 0.0;

    /// <summary>
    /// How long a judgement stands for the same résumé, posting and prompt.
    ///
    /// Re-asking is the normal case — the shortlist is re-scored whenever the
    /// conversation is refined — and re-running a 15-second evaluation to get an
    /// answer that is supposed to be identical is pure latency. Bounded by time
    /// rather than kept forever so an edited posting is eventually re-judged.
    /// </summary>
    private static readonly TimeSpan CacheLifetime = TimeSpan.FromHours(6);

    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private readonly IOllamaClient _llm;
    private readonly IResumeEvidenceService _evidence;
    private readonly IMemoryCache _cache;
    private readonly ILogger<JobEvaluationService> _logger;

    public JobEvaluationService(
        IOllamaClient llm,
        IResumeEvidenceService evidence,
        IMemoryCache cache,
        ILogger<JobEvaluationService> logger)
    {
        _llm = llm;
        _evidence = evidence;
        _cache = cache;
        _logger = logger;
    }

    public async Task<JobEvaluationDto?> EvaluateAsync(
        PortalResume resume, PortalJob job, CancellationToken ct = default)
    {
        if (!_llm.ChatAvailable) return null;

        // Everything that can change the verdict is in the key: the two documents,
        // the instructions, and the model. Anything left out would serve a stale
        // judgement as though it were about the current pair.
        // A résumé row is never edited in place — replacing a CV writes a new one —
        // so its id alone pins the document. A posting IS editable, hence its stamp.
        var key = $"eval|{PromptVersion}|{_llm.GetModelName()}|r{resume.Id}|j{job.Id}:{job.UpdatedAt:O}";
        if (_cache.TryGetValue<JobEvaluationDto>(key, out var cached) && cached is not null)
        {
            _logger.LogDebug("Evaluation of job {JobId} served from cache.", job.Id);
            return cached;
        }

        var started = DateTime.UtcNow;

        try
        {
            var reply = await _llm.ChatAsync(
                BuildPrompt(resume, job), jsonMode: true, temperature: JudgementTemperature, ct: ct);

            if (string.IsNullOrWhiteSpace(reply))
            {
                _logger.LogWarning("Evaluation of job {JobId} returned nothing.", job.Id);
                return null;
            }

            var evaluation = Parse(reply, resume, job);
            if (evaluation is null) return null;

            _logger.LogInformation(
                "Evaluated job {JobId} in {Seconds:0.#}s: {Score}% {Category} (model said {Stated}%)",
                job.Id, (DateTime.UtcNow - started).TotalSeconds,
                evaluation.OverallMatch, evaluation.Category, evaluation.ModelMatch);

            _cache.Set(key, evaluation, CacheLifetime);
            return evaluation;
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Evaluation of job {JobId} failed.", job.Id);
            return null;
        }
    }

    // -- the prompt ---------------------------------------------------------

    private static IReadOnlyList<ChatTurn> BuildPrompt(PortalResume resume, PortalJob job)
    {
        const string system = """
            You are a strict, objective AI Resume Auditor. You evaluate ONE résumé
            against ONE job description, in any industry or domain, by extracting
            factual evidence. Return JSON only.

            ── 0. YOU DO NOT SCORE ──
            Never produce a number, a percentage, a grade or a verdict about the
            candidate overall. Your job is evidence extraction, classification and gap
            detection. The score is computed elsewhere, from the rows you return.

            ── 0b. EXPERIENCE: AGGREGATE THE DATES, NEVER JUDGE THEM ──
            You are forbidden from deciding whether the candidate's experience meets
            the posting's minimum. Do not evaluate whether 0.1 is less than 1.0, or
            2.6 less than 2. The backend performs that comparison.
            You MUST still determine the total, into "candidate_years_experience", by
            these steps in order:
              Step 1 — EXPLICIT. If the résumé states a total tenure ("over 2.6 years
                of experience", "3+ years"), extract that exact number.
              Step 2 — CALCULATED. If no total is stated, you MUST NOT return null.
                Scan the employment history, find the start and end dates of each role
                ("12/2023 – 01/2024", "Jun 2021 - Present"), and sum the durations into
                fractional years. One month is 0.08 years; six months is 0.5; eighteen
                months is 1.5. Treat "Present" as today. Do not count overlapping roles
                twice.
              Step 3 — FALLBACK. Return null ONLY when the document contains no
                employment dates and no durations anywhere at all.
            "I could not find a summary line" is not a reason to return null. A résumé
            that lists two roles with dates states its total perfectly well; it just
            does not add it up for you. A null here removes the candidate from a
            comparison the posting requires, which reads as passing a bar nobody
            checked — the opposite of strict.

            ── 1. EVALUATE EACH REQUIREMENT IN A VACUUM ──
            No sentiment bleed. Judge every row on its own evidence, never on your
            impression of the candidate overall. A candidate who is a poor fit for the
            role may still explicitly demonstrate one requirement — mentoring, SQL
            optimisation, a specific tool — and that row is STRONG on its own evidence.
            NEVER mark a requirement MISSING because the candidate failed other
            requirements. Each row is a separate question about a separate sentence.

            ── 2. VERBATIM QUOTES ONLY — NO PHANTOM QUOTES ──
            "resume_quote" must be the exact, word-for-word sentence from the résumé
            that demonstrates THIS requirement. Never paraphrase, summarise or
            synthesise.
            If the skill, an equivalent tool, or the concept is not explicitly in the
            text, output exactly "NONE" — and when you output "NONE" the match_level
            MUST be "MISSING".
            Never reach for an unrelated sentence to fill the slot. Quoting "Prepared
            test strategy" as evidence of SQL or Git is worse than admitting the gap:
            it is a false positive dressed as proof, and every quote is checked against
            the document anyway.

            ── 3. CONCEPTUAL MATCHING WITHIN A FAMILY ──
            Do not penalise a candidate for an equivalent tool or a sub-component of
            the same technical family. Match on capability, not on brand string.
              • Cloud equivalence: AWS, Azure and GCP are functionally equivalent. A
                posting asking for AWS is matched by Azure with Terraform and
                Kubernetes.
              • Conceptual umbrellas: "Networking" is evidenced by its components —
                VNets, subnets, NSGs, firewalls, load balancers, CIDR.
              • Tool families: "Prometheus & Grafana" is evidenced by Grafana alone, or
                by Datadog. Do not demand every brand name in a list.
            Different names for one thing are one thing (React / React.js / ReactJS).

            ── 4. BUT NOT ACROSS DOMAINS ──
            Rule 3 operates INSIDE a technical family. It never crosses between them.
              • Frontend/UI frameworks (React, Angular, Vue) do NOT satisfy design and
                UX tools (Figma, Adobe XD, Sketch).
              • Application programming (Python, Java) does NOT satisfy cloud
                infrastructure or IaC without explicit infrastructure experience.
              • Basic SQL queries do NOT satisfy query optimisation, DBA or data
                engineering requirements.
            Building the thing is not designing the thing. Using a system is not
            operating it. Two monitoring tools are the same family; a monitoring tool
            and a design tool are not.

            ── 5. SENIORITY AND LEADERSHIP CANNOT BE INFERRED ──
            Individual-contributor work does not satisfy requirements for team
            leadership, managing reports, budget ownership or roadmap strategy. A
            senior or lead role needs explicit evidence of leading, mentoring or
            managing people or projects.

            ── 6. SCAN THE WHOLE DOCUMENT ──
            Read all of it before judging anything. Evidence of a skill is often NOT in
            the "Skills" list — it is buried in a project description or a work-history
            bullet. Search every project and every role before declaring anything
            MISSING. A skill demonstrated inside a project is stronger evidence than
            the same skill named in a list, not weaker.

            ── 7. ATOMISE, AND DROP NOTHING ──
            Output exactly one row for EVERY requirement, qualification and
            responsibility the posting states. Not a summary, not a selection, not the
            interesting ones — every one. There is no upper limit; a posting with
            eighteen bullets gets eighteen rows or more.
            Never leave a compound requirement whole: "manage the process and negotiate
            offers" is TWO requirements, and scoring it as one loses the distinction.
            A requirement the candidate has NO evidence for still gets its row, with
            match_level "MISSING" and resume_quote "NONE". Silently dropping it does
            not spare the candidate — the scoring runs on the rows you return, so an
            omitted requirement is one nobody is measured against, and the result is a
            score for a different, easier job than the one advertised.

            ── 7b. DOMAIN ISOLATION EXTENDS TO TITLES AND FUNCTIONS ──
            Rule 4 is about tools. This is about the WORK, and it is the harder one.
            Do not conflate similar-sounding titles or duties when the underlying
            business domain differs.
              • "Project Management", "IT Service Delivery" and "Systems Integration"
                are NOT "Software Product Management".
              • Leading engineers to implement a vendor package for a client is NOT
                owning a software product's commercial lifecycle, market strategy and
                PRDs.
              • Running a delivery schedule is not setting a roadmap. Coordinating
                stakeholders is not defining what to build.
            When the candidate's experience sits in a structurally different domain
            from the one the posting describes, mark the experience row AND the core
            skills that depend on that domain MISSING. Shared vocabulary is not shared
            work — do not award partial credit because two roles use the same nouns.

            ── 8. CLASSIFY EVERY REQUIREMENT ──
              must_have       non-negotiable core tool, technology or hard qualification
                              the posting explicitly mandates
              core_skill      technical ability, domain expertise, language or
                              operational skill
              experience      mandatory years, industry background or seniority level
              responsibility  a daily duty or task of the role being offered
              education       degree, formal academic background or required
                              certification
              nice_to_have    optional plus, preferred secondary skill, bonus
            Be honest about "responsibility": most bullets under "Responsibilities" or
            "What you'll do" describe the job on offer, not a checklist the applicant
            must already have ticked.

            For an "experience" row about YEARS, the verdict describes only whether the
            résumé STATES a duration — STRONG when it does, MISSING when it is silent.
            It never describes whether that duration is enough. See rule 0.

            ── 9. MATCH LEVEL ──
              STRONG   explicitly supported by verbatim evidence in professional work
                       history, leadership experience or executed projects
              WEAK     mentioned only in a passive skills list, an education summary or
                       tangential context, with no demonstrated application
              MISSING  absent from the résumé, OR claimed via an unrelated tool,
                       language or domain

            ── 10. KNOCKOUT INDICATORS ──
            Report each independently. They cap the result outright, so raise one only
            on real evidence — and never on a responsibility or a nice_to_have.
            There is no years indicator. Years are arithmetic and are not yours.

            ── WORKED EXAMPLES ──

            Within-family equivalence — a match, not a gap:
              JD: "Strong experience with AWS cloud infrastructure"
              CV: "Provisioned Azure VNets and AKS clusters using Terraform across three regions"
              {"jd_requirement": "AWS cloud infrastructure", "kind": "core_skill",
               "match_level": "STRONG",
               "resume_quote": "Provisioned Azure VNets and AKS clusters using Terraform across three regions",
               "reasoning": "Azure is functionally equivalent to AWS and the infrastructure work is demonstrated in real projects."}

            Evidence buried in a project, not in the skills list:
              JD: "Containerisation with Docker"
              CV skills list: "Java, Spring Boot, Jenkins"
              CV project bullet: "Packaged the payments service into Docker images for the CI pipeline"
              {"jd_requirement": "Containerisation with Docker", "kind": "core_skill",
               "match_level": "STRONG",
               "resume_quote": "Packaged the payments service into Docker images for the CI pipeline",
               "reasoning": "Docker is absent from the skills list but demonstrated in a project, which is the stronger evidence."}

            Years — extracted, never judged:
              JD: "Minimum 5 years of DevOps experience"
              CV: "DevOps Engineer with 2.6 years of hands-on experience"
              {"jd_requirement": "Minimum 5 years of DevOps experience", "kind": "experience",
               "match_level": "STRONG",
               "resume_quote": "DevOps Engineer with 2.6 years of hands-on experience",
               "reasoning": "The résumé states a total duration. Whether it meets the minimum is not evaluated here."}
              ...and "candidate_years_experience": 2.6

            Years — no total stated, so SUM THE DATES (rule 0b, step 2):
              JD: "1+ year of professional experience"
              CV: "QA Intern, Acme — 12/2023 – 01/2024" and nothing else about tenure
              {"jd_requirement": "1+ year of professional experience", "kind": "experience",
               "match_level": "WEAK",
               "resume_quote": "QA Intern, Acme — 12/2023 – 01/2024",
               "reasoning": "The only employment listed is a one-month internship; the résumé states no total."}
              ...and "candidate_years_experience": 0.08
              NOT null. The dates are right there. Returning null would drop this
              candidate out of the comparison entirely.

            Title conflation — same nouns, different job:
              JD: "5+ years in Software Product Management, owning roadmap and PRDs"
              CV: "Project Manager: led a team of 12 engineers delivering an SAP rollout for a retail client"
              {"jd_requirement": "5+ years in Software Product Management", "kind": "experience",
               "match_level": "MISSING", "resume_quote": "NONE",
               "reasoning": "Delivery project management on a vendor implementation is a structurally different domain from owning a software product's roadmap and commercial lifecycle."}

            A requirement with no evidence at all — the row is still required:
              JD: "Gather market trends and competitive intelligence"
              CV: nothing on the subject anywhere
              {"jd_requirement": "Gather market trends and competitive intelligence", "kind": "responsibility",
               "match_level": "MISSING", "resume_quote": "NONE",
               "reasoning": "No mention of market research, competitor analysis or related work."}

            No sentiment bleed — a weak candidate still earns the row they evidence:
              JD: "Mentoring junior engineers"
              CV: "Mentored two junior QA analysts through their first release"
              {"jd_requirement": "Mentoring junior engineers", "kind": "responsibility",
               "match_level": "STRONG",
               "resume_quote": "Mentored two junior QA analysts through their first release",
               "reasoning": "Explicitly demonstrated, and judged independently of how the candidate scores elsewhere."}

            Cross-domain, application code vs. infrastructure:
              JD: "Hands-on experience with Terraform and Kubernetes for cloud infrastructure"
              CV: "Wrote REST APIs in Python and deployed applications using basic Git commands"
              {"jd_requirement": "Hands-on experience with Terraform and Kubernetes", "kind": "must_have",
               "match_level": "MISSING", "resume_quote": "NONE",
               "reasoning": "Writing REST APIs in Python does not satisfy infrastructure-as-code or container orchestration."}

            Seniority, individual contributor vs. leadership:
              JD: "Lead and mentor a team of 5+ data scientists"
              CV: "Worked as a Data Scientist building machine learning models alongside 4 team members"
              {"jd_requirement": "Lead and mentor a team of 5+ data scientists", "kind": "responsibility",
               "match_level": "MISSING", "resume_quote": "NONE",
               "reasoning": "Working alongside team members as an individual contributor is not evidence of leading, managing or mentoring."}

            A genuine match:
              JD: "Proficiency in SQL for data analysis and reporting"
              CV: "Built complex SQL queries and window functions to extract business metrics from PostgreSQL databases"
              {"jd_requirement": "Proficiency in SQL for data analysis and reporting", "kind": "core_skill",
               "match_level": "STRONG",
               "resume_quote": "Built complex SQL queries and window functions to extract business metrics from PostgreSQL databases",
               "reasoning": "The candidate explicitly demonstrates writing complex SQL for analysis and reporting in their work history."}

            ── 9. WEIGHT THE KINDS TO THIS POSTING ──
            The framework is fixed; the weights are not. Distribute 100 points across
            only the kinds you actually used. Derive the split from what this role IS —
            a missing core framework is decisive for an engineering role and irrelevant
            for a marketing one. Responsibilities never dominate: they describe the job,
            not the candidate. This is a statement about the POSTING, not about the
            candidate, so it is not a score.

            ── OUTPUT ──
            Return ONLY this JSON object. No prose before or after it.
            {
              "requirements_evaluation": [
                {
                  "jd_requirement": "<one atomised requirement, from the JD>",
                  "kind": "must_have" | "core_skill" | "experience" | "responsibility" | "education" | "nice_to_have",
                  "match_level": "STRONG" | "WEAK" | "MISSING",
                  "resume_quote": "<verbatim from the résumé, or 'NONE'>",
                  "reasoning": "<1 concise sentence on why THIS row got THIS verdict>"
                }
              ],
              "candidate_years_experience": <total professional years as a number — stated if the résumé states one, otherwise SUMMED from the employment dates per rule 0b. null only when the document has no dates and no durations at all>,
              "weights": [{"criterion": "must_have" | "core_skill" | "experience" | "responsibility" | "education" | "nice_to_have", "weight": <integer percent>}],
              "knockout_indicators": {
                "missing_must_have_tools": true | false,
                "missing_required_education_or_cert": true | false
              },
              "justification_summary": "<1-2 sentences on factual alignment and the key gaps. State only what the rows above show. Do not credit the candidate with anything whose row says MISSING.>",
              "alternate_role": "<a role this candidate genuinely fits better, or empty>"
            }
            """;

        var facts = new StringBuilder();

        facts.AppendLine("=== JOB DESCRIPTION ===");
        facts.AppendLine($"Title: {job.Title}");
        if (!string.IsNullOrWhiteSpace(job.Company)) facts.AppendLine($"Company: {job.Company}");
        if (!string.IsNullOrWhiteSpace(job.SeniorityLevel)) facts.AppendLine($"Seniority: {job.SeniorityLevel}");
        if (job.MinYearsExperience is { } min) facts.AppendLine($"Minimum years: {min:0.#}");

        var required = TextStructure.ReadLines(job.RequiredSkills);
        if (required.Count > 0) facts.AppendLine($"Required: {string.Join(", ", required)}");

        var preferred = TextStructure.ReadLines(job.PreferredSkills);
        if (preferred.Count > 0) facts.AppendLine($"Preferred (nice to have): {string.Join(", ", preferred)}");

        var qualifications = TextStructure.ReadLines(job.Qualifications);
        if (qualifications.Count > 0)
        {
            facts.AppendLine("Stated qualifications:");
            foreach (var q in qualifications.Take(12)) facts.AppendLine($"- {q}");
        }

        var responsibilities = TextStructure.ReadLines(job.Responsibilities);
        if (responsibilities.Count > 0)
        {
            facts.AppendLine("Responsibilities:");
            foreach (var r in responsibilities.Take(12)) facts.AppendLine($"- {r}");
        }

        // The posting's own words matter: the structured fields above are an
        // extraction, and an extraction can miss a requirement that the prose states.
        facts.AppendLine();
        facts.AppendLine("Full posting text:");
        facts.AppendLine(TextStructure.Clip(job.RawText, MaxJobChars));

        // The whole résumé, headings intact. Section names are the entire basis of
        // the evidence ladder — handing over a parsed skill list instead would throw
        // away exactly the information the model is being asked to weigh.
        facts.AppendLine();
        facts.AppendLine("=== CANDIDATE RÉSUMÉ (verbatim, section headings intact) ===");
        facts.AppendLine(TextStructure.Clip(resume.RawText, MaxResumeChars));

        return new[]
        {
            new ChatTurn("system", system),
            new ChatTurn("user", facts.ToString()),
        };
    }

    // -- parsing and grounding ----------------------------------------------

    private JobEvaluationDto? Parse(string reply, PortalResume resume, PortalJob job)
    {
        var jobId = job.Id;

        // Small models wrap JSON in prose or a code fence often enough that it is
        // worth recovering rather than discarding a whole evaluation over it.
        var start = reply.IndexOf('{');
        var end = reply.LastIndexOf('}');
        if (start < 0 || end <= start) return null;

        var json = reply[start..(end + 1)];

        try
        {
            var raw = JsonSerializer.Deserialize<RawEvaluation>(json, Json);
            if (raw is null) return null;

            var claimed = raw.Rows
                .Where(r => !string.IsNullOrWhiteSpace(r.Jd_Requirement))
                .Select(r => new EvaluationRequirementDto(
                    TextStructure.Collapse(r.Jd_Requirement!),
                    IsNone(r.Resume_Quote) ? "" : TextStructure.Collapse(r.Resume_Quote!),
                    MatchLevels.Normalise(r.Match_Level),
                    RequirementKinds.Normalise(r.Kind),
                    // The section is measured from the document, never taken from the
                    // model: it is the difference between a skill used and a skill
                    // listed, which is exactly the thing worth not guessing at.
                    Where: "",
                    Reasoning: TextStructure.Collapse(r.Reasoning ?? "")))
                .ToList();

            if (claimed.Count == 0)
            {
                _logger.LogWarning("Evaluation of job {JobId} returned no requirements to score.", jobId);
                return null;
            }

            // Grounding runs BEFORE the arithmetic, not after it. The verdicts are
            // what the score is computed from, so a claim the document cannot support
            // has to be corrected while it can still change the number — checking it
            // afterwards would leave a percentage standing on evidence that was
            // subsequently thrown away.
            var grounded = Ground(claimed, resume, job);
            var (requirements, unexamined) = Backfill(grounded, job);

            var proposed = (raw.Weights ?? new())
                .GroupBy(w => RequirementKinds.Normalise(w.Criterion))
                .ToDictionary(g => g.Key, g => g.Sum(w => Math.Max(0, w.Weight)));

            return Compose(
                requirements, unexamined, proposed,
                claimedKnockout: new ClaimedKnockout(
                    raw.Knockout?.Missing_Must_Have_Tools ?? false,
                    raw.Knockout?.Missing_Required_Education_Or_Cert ?? false),
                statedYears: raw.Candidate_Years_Experience,
                resume: resume,
                job: job,
                alternateRole: raw.Alternate_Role,
                justification: raw.Justification_Summary ?? "",
                promptVersion: PromptVersion);
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Evaluation of job {JobId} returned unparseable JSON.", jobId);
            return null;
        }
    }

    // -- the prose, written here rather than by the model --------------------
    //
    // The card's summary and its "why this score" paragraph used to be free text
    // from the model, and it hallucinated into both — crediting a candidate with
    // wireframing and user research that its OWN rows had just marked MISSING.
    // That is the worst possible place for an invention: a recruiter reads the
    // summary and skips the evidence.
    //
    // There is no way to check a free paragraph the way a quote can be checked, so
    // neither is asked for any more. Both sentences below are assembled from the
    // verdicts and the arithmetic, which means they cannot say anything the rows do
    // not. The model's per-requirement `reasoning` survives, because it sits beside
    // its own verdict and quote where a false claim contradicts itself in place.

    /// <summary>The headline: what this candidate has, and what they do not.</summary>
    private static string Summarise(
        IReadOnlyList<EvaluationRequirementDto> requirements, EvaluationKnockoutDto knockout)
    {
        var strong = requirements.Where(r => r.MatchLevel == MatchLevels.Strong).ToList();
        var weak = requirements.Where(r => r.MatchLevel == MatchLevels.Weak).ToList();
        var missing = requirements.Where(r => r.MatchLevel == MatchLevels.Missing).ToList();

        var parts = new List<string>();

        parts.Add(strong.Count > 0
            ? $"Demonstrated in work history: {Names(strong)}."
            : "Nothing the posting asks for is demonstrated in work history.");

        if (weak.Count > 0) parts.Add($"Claimed but not demonstrated: {Names(weak)}.");

        if (missing.Count > 0)
        {
            // Dealbreakers first: which requirements are missing matters far more
            // than how many, and a must-have belongs at the front of the sentence.
            var critical = missing
                .Where(r => r.Kind is RequirementKinds.MustHave or RequirementKinds.Experience
                                   or RequirementKinds.Education)
                .ToList();

            parts.Add(critical.Count > 0
                ? $"Not evidenced anywhere: {Names(critical)}."
                : $"Not evidenced: {Names(missing)}.");
        }

        if (knockout.Fired)
        {
            parts.Add("A mandatory requirement is unmet, so the result is capped regardless " +
                      "of the strengths above.");
        }

        return string.Join(" ", parts);
    }

    /// <summary>The arithmetic, said in words: where the points went and why.</summary>
    private static string Explain(
        IReadOnlyList<EvaluationRequirementDto> requirements,
        Dictionary<string, double> weights,
        EvaluationKnockoutDto knockout,
        int score)
    {
        var parts = new List<string>();

        var byKind = weights
            .OrderByDescending(w => w.Value)
            .Select(w =>
            {
                var inKind = requirements.Where(r => r.Kind == w.Key).ToList();
                var met = inKind.Sum(r => MatchLevels.Credit(r.MatchLevel));
                return $"{RequirementKinds.Label(w.Key).ToLowerInvariant()} " +
                       $"{(int)Math.Round(w.Value)}% of the weighting, " +
                       $"{met:0.#} of {inKind.Count} met";
            })
            .ToList();

        parts.Add($"This posting was weighted as {string.Join("; ", byKind)}.");

        if (knockout.Fired)
        {
            parts.Add($"The result is capped at {KnockoutCeiling}% because {string.Join("; ", knockout.Reasons.Take(2))}.");
        }
        else
        {
            parts.Add($"That totals {score}%.");
        }

        // Said explicitly, because it is the single most misread thing on the card:
        // an absent duty is not the same failure as an absent mandatory tool.
        var missingDuties = requirements
            .Count(r => r.Kind == RequirementKinds.Responsibility && r.MatchLevel == MatchLevels.Missing);
        if (missingDuties > 0)
        {
            parts.Add($"{missingDuties} unevidenced {(missingDuties == 1 ? "item is" : "items are")} " +
                      "a duty of the role rather than a qualification for it, and carry little weight.");
        }

        return string.Join(" ", parts);
    }

    private static string Names(IReadOnlyList<EvaluationRequirementDto> requirements)
    {
        var names = requirements.Select(r => r.Requirement).Take(4).ToList();
        var rest = requirements.Count - names.Count;
        return string.Join(", ", names) + (rest > 0 ? $", and {rest} more" : "");
    }

    /// <summary>
    /// The recommendation, from the score and the knockout.
    ///
    /// Derived rather than asked for, so it can never contradict the number printed
    /// beside it — a model that returns 28% and "Proceed to Interview" is not
    /// offering a second opinion, it is producing a card that argues with itself.
    /// </summary>
    private static string Recommend(int score, EvaluationKnockoutDto knockout) =>
        knockout.Fired ? "Disqualify"
        : score >= 70 ? "Proceed to Interview"
        : score >= 45 ? "Consider Alternate Role"
        : "Disqualify";

    /// <summary>
    /// A dealbreaker the judge raised that its own rows do not account for.
    ///
    /// A plain record rather than the raw JSON shape, so a caller that never
    /// parsed any JSON — the RAG agent, which builds its verdicts from retrieved
    /// evidence — can still contribute one.
    /// </summary>
    public record ClaimedKnockout(bool MissingMustHaveTools, bool MissingEducation);

    /// <summary>
    /// Turns per-requirement verdicts into a scored evaluation.
    ///
    /// The public seam over everything below: weighting with its ceilings, the
    /// knockout derivation, the arithmetic, the bands, the recommendation and the
    /// composed prose. Nothing here calls a model — by the time this runs, every
    /// judgement has already been made.
    ///
    /// It exists so the RAG mode can reuse this arithmetic rather than grow a
    /// second copy of it. Two scorers would drift, and the same candidate would
    /// get one number under "AI evaluation" and a different one under "RAG" for
    /// reasons that had nothing to do with the evidence.
    /// </summary>
    public JobEvaluationDto Compose(
        IReadOnlyList<EvaluationRequirementDto> requirements,
        IReadOnlySet<string> unexamined,
        IReadOnlyDictionary<string, int> proposedWeights,
        ClaimedKnockout claimedKnockout,
        double? statedYears,
        PortalResume resume,
        PortalJob job,
        string? alternateRole,
        string justification,
        string promptVersion)
    {
        var weights = NormaliseWeights(proposedWeights, requirements);
        var knockout = ReadKnockout(claimedKnockout, requirements, unexamined, statedYears, resume, job);
        var computed = Score(requirements, weights, knockout);

        return new JobEvaluationDto(
            job.Id,
            computed,
            Categorise(computed),
            Explain(requirements, weights, knockout, computed),
            Summarise(requirements, knockout),
            weights.Select(w => new EvaluationWeightDto(w.Key, (int)Math.Round(w.Value))).ToList(),
            Group(requirements, MatchLevels.Strong),
            Group(requirements, MatchLevels.Weak),
            requirements
                .Where(r => r.MatchLevel == MatchLevels.Missing)
                .Select(r => new EvaluationGapDto(
                    r.Requirement,
                    string.IsNullOrWhiteSpace(r.Reasoning)
                        ? r.Kind == RequirementKinds.Responsibility
                            ? "not evidenced — a duty of the role rather than a stated qualification"
                            : $"no evidence in the résumé ({RequirementKinds.Label(r.Kind).ToLowerInvariant()})"
                        : r.Reasoning))
                .ToList(),
            Recommend(computed, knockout),
            string.IsNullOrWhiteSpace(alternateRole) ? null : alternateRole,
            promptVersion,
            requirements,
            // The judge is forbidden from producing a number, so there is no second
            // opinion to compare against. Zero means "none offered", and the UI
            // hides the comparison rather than reporting a 0% nobody claimed.
            ModelMatch: 0,
            knockout,
            justification);
    }

    /// <summary>Models write "None", "n/a" and "" for the same thing.</summary>
    private static bool IsNone(string? quote)
    {
        var q = (quote ?? "").Trim().Trim('"', '\'', '.').ToLowerInvariant();
        return q.Length == 0 || q is "none" or "n/a" or "na" or "null" or "not found" or "not stated";
    }

    private static IReadOnlyList<EvaluationMatchDto> Group(
        IEnumerable<EvaluationRequirementDto> requirements, string level) =>
        requirements
            .Where(r => r.MatchLevel == level)
            .Select(r => new EvaluationMatchDto(r.Requirement, r.Quote, r.Where))
            .ToList();

    // -- the arithmetic -----------------------------------------------------

    /// <summary>
    /// The most of the total a posting's RESPONSIBILITIES may be worth.
    ///
    /// Responsibility bullets describe the job on offer, not a checklist the
    /// applicant must already have ticked, and there are usually more of them than
    /// of anything else. Left unbounded they dominate the weighting by sheer count
    /// and every candidate is marked down for not having pre-performed the role —
    /// which is how a recruiter with seven years of full-cycle experience scored 65
    /// against a recruiting job for not mentioning that they negotiate offers.
    /// </summary>
    private const double MaxResponsibilityShare = 15.0;

    /// <summary>A nice-to-have is a bonus. It cannot carry a fifth of the verdict.</summary>
    private const double MaxNiceToHaveShare = 10.0;

    /// <summary>
    /// The weighting, as percentages over the kinds actually present.
    ///
    /// The model proposes; this disposes. Its split is kept wherever it is sane,
    /// because a per-posting weighting is the whole point — but the two ceilings
    /// above are enforced, and a model that returned nothing usable falls back to a
    /// flat split so the score is still defined.
    /// </summary>
    private static Dictionary<string, double> NormaliseWeights(
        IReadOnlyDictionary<string, int> proposed, IReadOnlyList<EvaluationRequirementDto> requirements)
    {
        var present = requirements.Select(r => r.Kind).Distinct().ToList();

        var weights = present.ToDictionary(
            kind => kind,
            kind => proposed.TryGetValue(kind, out var weight) ? (double)Math.Max(0, weight) : 0d);

        // Nothing usable came back — an equal split is at least honest about
        // having no opinion, and keeps the score defined.
        if (weights.Values.Sum() <= 0)
        {
            foreach (var kind in present) weights[kind] = 100.0 / present.Count;
        }

        Rescale(weights, 100);
        Cap(weights, RequirementKinds.Responsibility, MaxResponsibilityShare);
        Cap(weights, RequirementKinds.NiceToHave, MaxNiceToHaveShare);
        return weights;
    }

    private static void Rescale(Dictionary<string, double> weights, double total)
    {
        var sum = weights.Values.Sum();
        if (sum <= 0) return;
        foreach (var key in weights.Keys.ToList()) weights[key] = weights[key] / sum * total;
    }

    /// <summary>
    /// Trims one kind to its ceiling and hands the surplus to the others in
    /// proportion, so the total stays 100 and the model's relative ordering of
    /// everything else survives.
    /// </summary>
    private static void Cap(Dictionary<string, double> weights, string kind, double ceiling)
    {
        if (!weights.TryGetValue(kind, out var value) || value <= ceiling) return;

        var surplus = value - ceiling;
        weights[kind] = ceiling;

        var others = weights.Keys.Where(k => k != kind).ToList();
        if (others.Count == 0)
        {
            // The posting is nothing but responsibilities. Capping would leave the
            // total at 15 and every candidate near zero, so the ceiling does not
            // apply when there is nothing to redistribute to.
            weights[kind] = 100;
            return;
        }

        var othersTotal = others.Sum(k => weights[k]);
        foreach (var k in others)
        {
            weights[k] += othersTotal > 0
                ? surplus * (weights[k] / othersTotal)
                : surplus / others.Count;
        }
    }

    /// <summary>
    /// The percentage, from the model's own verdicts.
    ///
    /// Each kind scores the mean credit of its requirements — STRONG 1.0, WEAK 0.5,
    /// MISSING 0.0 — and contributes its weight. A kind is measured against its own
    /// requirements rather than against the whole list, so a posting with one
    /// must-have and six responsibilities cannot bury the must-have.
    /// </summary>
    private static int Score(
        IReadOnlyList<EvaluationRequirementDto> requirements,
        Dictionary<string, double> weights,
        EvaluationKnockoutDto knockout)
    {
        var total = 0.0;
        foreach (var (kind, weight) in weights)
        {
            var inKind = requirements.Where(r => r.Kind == kind).ToList();
            if (inKind.Count == 0) continue;
            total += weight * inKind.Average(r => MatchLevels.Credit(r.MatchLevel));
        }

        // Away from zero, not .NET's default banker's rounding: 62.5 becoming 62
        // is correct statistically and wrong to everyone reading a percentage.
        var score = (int)Math.Round(Math.Clamp(total, 0, 100), MidpointRounding.AwayFromZero);

        // The cap is applied last and only lowers. A dealbreaker is not something to
        // average against strengths elsewhere — that is the entire meaning of one.
        return knockout.Fired ? Math.Min(score, KnockoutCeiling) : score;
    }

    /// <summary>
    /// The most a candidate with a genuine dealbreaker can score.
    ///
    /// 34, not 35: the bands put Weak Match at 35 and above, so a ceiling of 35
    /// let a knocked-out candidate land in the same band as one who merely scored
    /// poorly. A dealbreaker should read as a dealbreaker.
    /// </summary>
    private const int KnockoutCeiling = 34;

    /// <summary>
    /// The knockout, derived from the VERDICTS rather than from the model's own
    /// summary of them.
    ///
    /// This used to trust two self-reported booleans, and both let real
    /// dealbreakers through. A missing mandatory tool was only caught when the
    /// model had also labelled it <c>must_have</c> — label it <c>core_skill</c>
    /// and a required tool that is nowhere in the CV cost nothing. Worse, an unmet
    /// years requirement was read ONLY from the model's flag, so a requirement it
    /// had itself just marked MISSING on an <c>experience</c> row did not fire the
    /// cap it exists for. Measured: a candidate with no Figma and none of the
    /// required years scored 45%.
    ///
    /// So each indicator is now computed from the requirement rows, and the
    /// model's flags can only ADD to that, never subtract. It is allowed to raise
    /// a dealbreaker its verdicts do not show — it can read a seniority or
    /// domain mismatch that no single row captures — but it cannot wave one away.
    /// </summary>
    private EvaluationKnockoutDto ReadKnockout(
        ClaimedKnockout claimed,
        IReadOnlyList<EvaluationRequirementDto> requirements,
        IReadOnlySet<string> unexamined,
        double? statedYears,
        PortalResume resume,
        PortalJob job)
    {
        List<string> MissingOfKind(string kind) => requirements
            .Where(r => r.Kind == kind && r.MatchLevel == MatchLevels.Missing)
            .Select(r => r.Requirement)
            .ToList();

        var missingMustHaves = MissingOfKind(RequirementKinds.MustHave);
        var missingEducation = MissingOfKind(RequirementKinds.Education);

        // The posting's OWN required-skills list, cross-checked against the missing
        // rows whatever kind the model gave them.
        //
        // Keying the dealbreaker purely on the model's `must_have` label left a hole
        // exactly the width of one misclassification: a tool the posting explicitly
        // requires, absent from the CV, labelled `core_skill`, cost nothing. That is
        // half of how a candidate with no Figma against a "Figma proficiency
        // required" posting reached 45%. The posting already told us which skills are
        // mandatory, so that list decides — not the model's opinion of it.
        var stated = StatedRequiredSkills(job);

        var missingStated = requirements
            .Where(r => r.MatchLevel == MatchLevels.Missing && !missingMustHaves.Contains(r.Requirement))
            // A row nobody evaluated is not evidence of a gap. It scores zero either
            // way, but it may not disqualify: that would cap a candidate over an
            // omission of ours rather than a finding about them.
            .Where(r => !unexamined.Contains(r.Requirement))
            .Where(r => stated.Any(s => NamesTheSameThing(s, r.Requirement)))
            .Select(r => r.Requirement)
            .ToList();

        var noTools = missingMustHaves.Count > 0
                   || missingStated.Count > 0
                   || claimed.MissingMustHaveTools;
        var noEducation = missingEducation.Count > 0 || claimed.MissingEducation;

        var reasons = new List<string>();
        reasons.AddRange(missingMustHaves.Select(r => $"mandatory requirement not evidenced: {r}"));
        reasons.AddRange(missingStated.Select(r => $"the posting lists this as required and it is not evidenced: {r}"));
        reasons.AddRange(missingEducation.Select(r => $"required education or certification absent: {r}"));

        if (noTools && missingMustHaves.Count == 0 && missingStated.Count == 0)
            reasons.Add("a mandatory tool the model judged absent");
        if (noEducation && missingEducation.Count == 0) reasons.Add("a required qualification the model judged absent");

        var (shortOfYears, yearsReason) = CompareYears(statedYears, resume, job);
        if (yearsReason is not null) reasons.Add(yearsReason);

        return new EvaluationKnockoutDto(noTools, shortOfYears, reasons, noEducation);
    }

    /// <summary>The skills the posting itself declares mandatory, normalised.</summary>
    private static List<string> StatedRequiredSkills(PortalJob job) =>
        TextStructure.ReadLines(job.RequiredSkills)
            .Select(TextStructure.Collapse)
            .Where(s => s.Length >= 2)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

    /// <summary>
    /// Adds a row for any required skill the model returned no row for at all.
    ///
    /// The instruction to evaluate every requirement is not self-enforcing: a model
    /// under output pressure summarises, and a dropped requirement is invisible in
    /// the result — the score simply runs over a shorter list, which is a score for
    /// an easier job than the one advertised. The posting's own required-skills list
    /// is the part of that we can check, so it is checked.
    ///
    /// The backfilled rows are returned separately as UNEXAMINED, and they do not
    /// fire a dealbreaker. The distinction matters: "the CV lacks this" is a finding,
    /// "nobody looked" is a failure of ours, and capping a candidate at a
    /// disqualifying score over the second would punish them for our omission. They
    /// score zero on their row — a real penalty — and are labelled so the card can
    /// say what happened.
    /// </summary>
    private (IReadOnlyList<EvaluationRequirementDto> Rows, IReadOnlySet<string> Unexamined) Backfill(
        IReadOnlyList<EvaluationRequirementDto> requirements, PortalJob job)
    {
        var stated = StatedRequiredSkills(job);
        var missing = stated
            .Where(s => !requirements.Any(r => NamesTheSameThing(s, r.Requirement)))
            .ToList();

        if (missing.Count == 0)
        {
            return (requirements, new HashSet<string>(StringComparer.OrdinalIgnoreCase));
        }

        _logger.LogWarning(
            "Job {JobId}: the evaluation returned no row for {Count} required skill(s) — {Skills}. " +
            "Added as unexamined so they still count against the score.",
            job.Id, missing.Count, string.Join(", ", missing));

        var added = missing.Select(s => new EvaluationRequirementDto(
            s, "", MatchLevels.Missing, RequirementKinds.CoreSkill,
            Where: "not evaluated",
            Reasoning: "The posting lists this as required and the evaluation returned no verdict for it."));

        return (
            requirements.Concat(added).ToList(),
            new HashSet<string>(missing, StringComparer.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Whether a requirement row is about a skill the posting explicitly listed.
    ///
    /// Containment either way rather than equality: the posting says "Figma" and the
    /// row reads "Figma proficiency", or the posting says "Figma proficiency
    /// required" and the row reads "Figma". Both are the same requirement, and an
    /// equality test would match neither.
    ///
    /// Guarded on length so a two-letter stated skill cannot swallow every row it
    /// happens to be a substring of.
    /// </summary>
    private static bool NamesTheSameThing(string stated, string requirement)
    {
        var a = stated.ToLowerInvariant();
        var b = TextStructure.Collapse(requirement).ToLowerInvariant();
        if (a.Length < 3 || b.Length < 3) return string.Equals(a, b, StringComparison.Ordinal);
        return a.Contains(b, StringComparison.Ordinal) || b.Contains(a, StringComparison.Ordinal);
    }

    /// <summary>
    /// Whether the candidate is short of the posting's stated minimum — decided
    /// here, in arithmetic, never by the model.
    ///
    /// The model is explicitly forbidden from comparing the two numbers, because it
    /// is bad at it in a way that is expensive: asked whether 2.6 years met a
    /// 2-year minimum, an 8B answered that it did not, and the candidate was capped
    /// at a disqualifying score on a requirement they exceeded. It extracts the
    /// duration the résumé states; the subtraction happens here.
    ///
    /// The model's figure is preferred over the parsed profile's because it read
    /// the sentence in context, and the profile parser is a regex over the same
    /// document. Either may be absent.
    ///
    /// An unknown on either side fires NOTHING. A dealbreaker asserts a fact, and
    /// "the résumé does not say" is not evidence of a shortfall — capping on an
    /// absent number would disqualify every CV that omits a total.
    /// </summary>
    private (bool Short, string? Reason) CompareYears(
        double? statedYears, PortalResume resume, PortalJob job)
    {
        if (job.MinYearsExperience is not { } required || required <= 0) return (false, null);

        var actual = statedYears ?? resume.YearsExperience;
        if (actual is not { } years)
        {
            _logger.LogDebug(
                "Job {JobId} requires {Required:0.#} years but neither the model nor the profile " +
                "read a total from the résumé; no years dealbreaker raised.", job.Id, required);
            return (false, null);
        }

        if (years >= required) return (false, null);

        return (true, $"{years:0.#} years of experience against a required {required:0.#}");
    }

    /// <summary>
    /// The band always follows the number.
    ///
    /// Models routinely return a score and a label that disagree — 32% labelled
    /// "Moderate Match" — and a candidate reading both believes the kinder one.
    /// </summary>
    private static string Categorise(int score) => score switch
    {
        >= 80 => "Strong Match",
        >= 60 => "Moderate Match",
        >= 35 => "Weak Match",
        _ => "Complete Mismatch",
    };

    /// <summary>
    /// Checks the model's QUOTES against the document, and nothing else.
    ///
    /// The division of labour matters, because getting it wrong is what made this
    /// evaluation behave like keyword matching. The model owns the judgement: which
    /// requirements are met, at what depth, and what the whole thing is worth. This
    /// method owns one narrower question — is the sentence it put in quotation marks
    /// actually in the CV? — because a quote is the one part of the output a
    /// candidate takes at face value, and measured on an 8B model roughly every
    /// quote was invented: the right skills, justified with fabricated sentences.
    ///
    /// What it deliberately no longer does is require the REQUIREMENT'S own words to
    /// appear in the résumé before a match is allowed to stand. That check ran a
    /// literal string search over the document, so an evaluation that had correctly
    /// reasoned "this candidate's Angular work evidences component-based frontend
    /// development" was overruled because the phrase itself was not written there —
    /// re-imposing, at the last step, exactly the vocabulary matching the model was
    /// brought in to see past.
    ///
    /// A claim whose quote cannot be found is dropped from the proof lists and NOT
    /// converted into a gap. Promoting it would contradict a score the model has
    /// already justified, and leave a card asserting a requirement is both met and
    /// missing.
    /// </summary>
    private IReadOnlyList<EvaluationRequirementDto> Ground(
        IReadOnlyList<EvaluationRequirementDto> claimed, PortalResume resume, PortalJob job)
    {
        var evidence = _evidence.Build(resume);
        var haystack = Normalise(resume.RawText ?? "");
        var text = resume.RawText ?? "";

        var grounded = new List<EvaluationRequirementDto>(claimed.Count);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var claim in claimed)
        {
            // The same requirement listed twice would be counted twice in its kind's
            // average. Models repeat themselves often enough to guard.
            if (!seen.Add(claim.Requirement)) continue;

            if (claim.MatchLevel == MatchLevels.Missing)
            {
                grounded.Add(claim with { Quote = "" });
                continue;
            }

            var quoted = IsQuoted(haystack, claim.Quote);

            // The requirement's literal appearance is NOT a gate — that check is what
            // made this behave like keyword matching, overruling a correct reading of
            // "this Angular work evidences component-based frontend development"
            // because the phrase itself was not written in the document. It is used
            // only as a second chance and to say where the words actually are.
            var found = evidence.For(claim.Requirement);
            var located = found.Tier >= EvidenceTier.Listed;

            if (quoted)
            {
                grounded.Add(claim with
                {
                    Where = string.IsNullOrWhiteSpace(claim.Where) && located
                        ? string.Join(" and ", found.FoundIn)
                        : claim.Where,
                });
                continue;
            }

            if (located)
            {
                // The requirement is in the document but the sentence offered as proof
                // is not. The claim survives — the words are demonstrably there — but
                // not at full strength: we can confirm the mention and not the depth
                // the model asserted for it.
                _logger.LogDebug(
                    "Job {JobId}: '{Requirement}' is in the CV but its quote was not; capped at WEAK.",
                    job.Id, claim.Requirement);

                grounded.Add(claim with
                {
                    Quote = Excerpt(text, claim.Requirement) ?? "",
                    MatchLevel = MatchLevels.Weak,
                    Where = string.Join(" and ", found.FoundIn),
                });
                continue;
            }

            _logger.LogWarning(
                "Job {JobId}: '{Requirement}' scored {Level} on evidence that is not in the CV — " +
                "neither the quote nor the requirement appears. Recorded as MISSING.",
                job.Id, claim.Requirement, claim.MatchLevel);

            grounded.Add(claim with { Quote = "", MatchLevel = MatchLevels.Missing });
        }

        return grounded;
    }

    /// <summary>
    /// The evaluation's verdicts as skill chips, so a reasoned card carries the same
    /// Have / Transferable / Missing vocabulary as a computed one.
    ///
    /// The two lists mean genuinely different things and the mapping keeps that:
    /// a high-confidence match is <c>Have</c>, a partial one is <c>Transferable</c>
    /// — proven, but not at the depth the posting asked for — and a gap is
    /// <c>Missing</c>. Similarity is zero throughout, deliberately: no cosine was
    /// measured here, and putting a number in that slot would invent a measurement.
    /// </summary>
    public static IReadOnlyList<SkillAssessmentDto> ToSkillAssessments(JobEvaluationDto evaluation)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var chips = new List<SkillAssessmentDto>();

        void Add(string requirement, string status, string? evidenceSkill)
        {
            var name = TextStructure.Collapse(requirement ?? "");
            if (name.Length == 0 || !seen.Add(name)) return;
            chips.Add(new SkillAssessmentDto(name, status, evidenceSkill, 0));
        }

        foreach (var m in evaluation.HighConfidence) Add(m.Requirement, "Have", m.Where);
        foreach (var m in evaluation.Partial) Add(m.Requirement, "Transferable", m.Where);
        foreach (var g in evaluation.Gaps) Add(g.Requirement, "Missing", null);

        return chips;
    }

    /// <summary>
    /// The line of the résumé that actually mentions a requirement.
    ///
    /// This is the proof shown to the candidate, so it comes from the document, not
    /// from the model. Returning null means the word is not there and the caller
    /// must not present it as evidence.
    /// </summary>
    private static string? Excerpt(string resumeText, string requirement)
    {
        var needle = TextStructure.Collapse(requirement);
        if (needle.Length < 2) return null;

        foreach (var line in resumeText.Split('\n'))
        {
            var clean = TextStructure.Collapse(line);
            if (clean.Length < 3) continue;
            if (clean.Contains(needle, StringComparison.OrdinalIgnoreCase))
                return TextStructure.Clip(clean, 240);
        }

        return null;
    }

    /// <summary>
    /// Whether a claimed quote really is in the document.
    ///
    /// Deliberately forgiving about form and strict about substance: whitespace and
    /// punctuation are normalised away, and a long quote passes if a solid run of
    /// it matches, because a model reflowing a bullet is not the failure this
    /// guards against. Inventing a sentence is.
    /// </summary>
    private static bool IsQuoted(string haystack, string evidence)
    {
        var needle = Normalise(evidence);
        if (needle.Length < 3) return false;
        if (haystack.Contains(needle, StringComparison.Ordinal)) return true;

        // Long evidence: accept if most consecutive words of it appear together.
        var words = needle.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length < 4) return false;

        var window = Math.Max(4, words.Length * 2 / 3);
        for (var i = 0; i + window <= words.Length; i++)
        {
            if (haystack.Contains(string.Join(' ', words.Skip(i).Take(window)), StringComparison.Ordinal))
                return true;
        }

        return false;
    }

    private static string Normalise(string text) =>
        new string(text.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) ? c : ' ').ToArray())
            .Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Aggregate(new StringBuilder(), (b, w) => b.Append(w).Append(' '))
            .ToString();

    // -- the shape the model is asked for -----------------------------------

    private sealed class RawEvaluation
    {
        public List<RawRequirement>? Requirements_Evaluation { get; set; }

        /// <summary>
        /// The key this schema used before the auditor prompt renamed it. Read as a
        /// fallback so a cached or replayed reply from the previous version still
        /// parses instead of being discarded as unscoreable.
        /// </summary>
        public List<RawRequirement>? Evidence_Evaluation { get; set; }

        /// <summary>
        /// The total the résumé states. EXTRACTED by the model, compared here — the
        /// model is forbidden from deciding whether it meets the posting's minimum.
        /// </summary>
        public double? Candidate_Years_Experience { get; set; }

        public List<RawWeight>? Weights { get; set; }
        public RawKnockout? Knockout_Indicators { get; set; }
        public RawKnockout? Knockout_Check { get; set; }
        public string? Justification_Summary { get; set; }
        public string? Alternate_Role { get; set; }

        public List<RawRequirement> Rows => Requirements_Evaluation ?? Evidence_Evaluation ?? new();
        public RawKnockout? Knockout => Knockout_Indicators ?? Knockout_Check;
    }

    private sealed class RawRequirement
    {
        public string? Jd_Requirement { get; set; }
        public string? Resume_Quote { get; set; }
        public string? Match_Level { get; set; }
        public string? Kind { get; set; }
        public string? Reasoning { get; set; }
    }

    private sealed class RawWeight
    {
        public string? Criterion { get; set; }
        public int Weight { get; set; }
    }

    private sealed class RawKnockout
    {
        public bool Missing_Must_Have_Tools { get; set; }
        public bool Missing_Required_Education_Or_Cert { get; set; }

        // No years flag, deliberately. Whether a duration meets a minimum is
        // arithmetic, and the model does not do arithmetic here — see CompareYears.
    }
}
