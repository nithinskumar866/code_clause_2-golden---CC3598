using System.Text;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;

namespace JobPortal.Api.Services.Chat;

/// <summary>
/// Everything the model is allowed to know for ONE turn.
///
/// THE JOB-ANCHOR RULE
/// -------------------
/// A package is invalid unless it contains at least one real posting. That single
/// constraint is what keeps the assistant answerable and honest at the same time:
///
///   "what should I study for the data science job"  -> anchored, answerable
///   "what am I lacking"                             -> anchored to the shortlist
///   "how many years of experience do I have"        -> NO anchor, not a job question
///
/// The candidate's resume is never a subject in its own right. It enters only as
/// <see cref="RequirementCoverage"/> — one requirement OF an anchored posting, and
/// whether the CV evidences it. So "tell me about my resume" cannot be *constructed*,
/// which is a far stronger guarantee than instructing a model not to answer it.
///
/// COMPANY FACTS ARE THE DANGEROUS ONES
/// ------------------------------------
/// A company is a real-world entity the model has opinions about. Asked where a firm
/// is headquartered, or what its parental-leave policy is, an 8B model will answer
/// confidently from training data and be plausibly, unverifiably wrong about a real
/// employer. So company facts are lifted from the JD text into
/// <see cref="CompanyFacts"/> and nothing else is available. If the JD does not say
/// it, the honest answer is that the posting does not say.
/// </summary>
public sealed class FactPackage
{
    /// <summary>Postings this turn is about. Never empty — see the anchor rule.</summary>
    public IReadOnlyList<JobFact> Jobs { get; }

    /// <summary>
    /// What the JD says about the employer. Keyed by job id, because two postings
    /// from the same company can carry different text and only the quoted one is
    /// evidence for the question being asked.
    /// </summary>
    public IReadOnlyDictionary<int, CompanyFacts> Companies { get; }

    /// <summary>The ONLY channel through which resume information enters a turn.</summary>
    public IReadOnlyList<RequirementCoverage> Coverage { get; }

    /// <summary>Free-form notes the executor computed (counts, rankings, comparisons).</summary>
    public IReadOnlyDictionary<string, string> Notes { get; }

    public FactPackage(
        IReadOnlyList<JobFact> jobs,
        IReadOnlyDictionary<int, CompanyFacts>? companies = null,
        IReadOnlyList<RequirementCoverage>? coverage = null,
        IReadOnlyDictionary<string, string>? notes = null)
    {
        if (jobs is null || jobs.Count == 0)
        {
            // Enforced here rather than at the call sites: an unanchored package is
            // the one thing that could turn this assistant into a general chatbot,
            // and a constructor is the only place that cannot be forgotten.
            throw new ArgumentException(
                "A fact package must be anchored to at least one real posting.", nameof(jobs));
        }

        Jobs = jobs;
        Companies = companies ?? new Dictionary<int, CompanyFacts>();
        Coverage = coverage ?? Array.Empty<RequirementCoverage>();
        Notes = notes ?? new Dictionary<string, string>();
    }

    public IReadOnlySet<int> AnchoredJobIds => Jobs.Select(j => j.JobId).ToHashSet();

    /// <summary>
    /// The package rendered for the model. This is the ENTIRE world it may answer
    /// from; anything absent here must come back as "the posting does not say".
    /// </summary>
    public string ToPrompt()
    {
        var text = new StringBuilder();
        foreach (var job in Jobs)
        {
            text.AppendLine($"JOB {job.JobId}: {job.Title}");
            text.AppendLine($"  company: {Or(job.Company)}");
            text.AppendLine($"  location: {Or(job.Location)}   work mode: {Or(job.WorkMode)}");
            text.AppendLine($"  employment: {Or(job.EmploymentType)}   level: {Or(job.SeniorityLevel)}");
            if (job.Salary is not null) text.AppendLine($"  salary: {job.Salary}");
            if (job.Experience is not null) text.AppendLine($"  experience: {job.Experience}");
            if (job.Required.Count > 0) text.AppendLine($"  required: {string.Join(", ", job.Required)}");
            if (job.Preferred.Count > 0) text.AppendLine($"  preferred: {string.Join(", ", job.Preferred)}");
            if (!string.IsNullOrWhiteSpace(job.Summary)) text.AppendLine($"  summary: {job.Summary}");

            if (Companies.TryGetValue(job.JobId, out var company))
            {
                foreach (var (heading, body) in company.Sections)
                    text.AppendLine($"  [{heading}] {body}");
            }
            text.AppendLine();
        }

        var byJob = Coverage.GroupBy(c => c.JobId);
        foreach (var group in byJob)
        {
            text.AppendLine($"CANDIDATE vs JOB {group.Key}:");
            foreach (var item in group)
            {
                var evidence = string.IsNullOrWhiteSpace(item.Evidence) ? "" : $" — \"{item.Evidence}\"";
                text.AppendLine($"  {item.Status}: {item.Requirement}{evidence}");
            }
            text.AppendLine();
        }

        foreach (var (key, value) in Notes)
            text.AppendLine($"{key}: {value}");

        return text.ToString().TrimEnd();
    }

    private static string Or(string value) =>
        string.IsNullOrWhiteSpace(value) ? "not stated" : value;
}

/// <summary>One posting, flattened to what a turn can quote.</summary>
public sealed record JobFact(
    int JobId,
    string Title,
    string Company,
    string Location,
    string WorkMode,
    string EmploymentType,
    string SeniorityLevel,
    string? Salary,
    string? Experience,
    IReadOnlyList<string> Required,
    IReadOnlyList<string> Preferred,
    string Summary)
{
    public static JobFact From(PortalJob job) => new(
        job.Id,
        job.Title,
        job.Company,
        job.Location,
        job.WorkMode,
        job.EmploymentType,
        job.SeniorityLevel,
        Money(job),
        Years(job),
        TextStructure.ReadLines(job.RequiredSkills),
        TextStructure.ReadLines(job.PreferredSkills),
        job.Summary);

    private static string? Money(PortalJob job)
    {
        if (job.SalaryMin is null && job.SalaryMax is null) return null;
        var currency = job.SalaryCurrency ?? "";
        if (job.SalaryMin is not null && job.SalaryMax is not null)
            return $"{currency}{job.SalaryMin:0} - {currency}{job.SalaryMax:0}";
        return job.SalaryMin is not null ? $"from {currency}{job.SalaryMin:0}" : $"up to {currency}{job.SalaryMax:0}";
    }

    private static string? Years(PortalJob job)
    {
        if (job.MinYearsExperience is null && job.MaxYearsExperience is null) return null;
        if (job.MinYearsExperience is not null && job.MaxYearsExperience is not null)
            return $"{job.MinYearsExperience:0.#}-{job.MaxYearsExperience:0.#} years";
        return job.MinYearsExperience is not null
            ? $"{job.MinYearsExperience:0.#}+ years"
            : $"up to {job.MaxYearsExperience:0.#} years";
    }
}

/// <summary>
/// What the JD says about the employer — and nothing more.
///
/// Harvested from the document's own sections so a "what's their leave policy"
/// question is answered by quoting the posting, or not at all. The model is never
/// given the company name alone and invited to recall what it knows about it.
/// </summary>
public sealed record CompanyFacts(string Name, IReadOnlyList<(string Heading, string Body)> Sections)
{
    /// <summary>
    /// Section headings that carry employer information rather than role duties.
    ///
    /// Matched as substrings because JDs name these sections a dozen ways — "About
    /// Us", "About the Company", "Who We Are", "Our Culture", "Perks & Benefits".
    /// </summary>
    private static readonly string[] CompanyHeadings =
    {
        "about", "who we are", "company", "culture", "benefit", "perk", "why join",
        "we offer", "our team", "work life", "work-life", "policy", "policies",
        "leave", "compensation", "eeo", "equal opportunity", "diversity",
    };

    private const int MaxSectionChars = 1200;

    /// <summary>
    /// Pulls employer-facing sections out of the JD's raw text.
    ///
    /// Scanned here rather than through <see cref="TextStructure.SplitSections"/>,
    /// which matches headings by exact equality against a supplied list. That works
    /// when you know the headings; here we do not. JDs name these sections a dozen
    /// ways — "About Us", "## About the Company", "Why Join Us?", "Perks &amp;
    /// Benefits" — so the test is "a heading-shaped line CONTAINING a company word",
    /// which generalises instead of enumerating.
    ///
    /// Returns a package with no sections when the JD says nothing about the employer.
    /// That is a useful state, not a failure: it is precisely what lets the assistant
    /// say "the posting doesn't cover that" instead of improvising from what the model
    /// happens to know about a real company.
    /// </summary>
    public static CompanyFacts From(PortalJob job)
    {
        var sections = new List<(string, string)>();
        if (string.IsNullOrWhiteSpace(job.RawText)) return new CompanyFacts(job.Company, sections);

        var lines = job.RawText.Split('\n');
        string? openHeading = null;
        var body = new StringBuilder();

        void Close()
        {
            if (openHeading is null) return;
            var text = TextStructure.Collapse(body.ToString());
            if (text.Length > 0)
            {
                sections.Add((openHeading,
                    text.Length <= MaxSectionChars ? text : text[..MaxSectionChars] + "…"));
            }
            openHeading = null;
            body.Clear();
        }

        foreach (var raw in lines)
        {
            var line = TextStructure.Demarkup(raw);

            if (IsHeadingShaped(line))
            {
                // A new heading always ends the previous section, whether or not the
                // new one is about the company — otherwise "About Us" would swallow
                // the responsibilities that follow it.
                Close();
                var label = line.TrimEnd(':', '-', '–').Trim();
                if (CompanyHeadings.Any(h => label.Contains(h, StringComparison.OrdinalIgnoreCase)))
                    openHeading = label;
                continue;
            }

            if (openHeading is not null) body.AppendLine(line);
        }
        Close();

        return new CompanyFacts(job.Company, sections);
    }

    /// <summary>
    /// Whether a line reads as a section heading rather than prose.
    ///
    /// Short, not a list item, and not a sentence — a trailing full stop is the
    /// clearest signal that a line is prose, and headings that end in ':' are the
    /// clearest signal that one is not.
    /// </summary>
    private static bool IsHeadingShaped(string line)
    {
        if (line.Length is < 3 or > 60) return false;
        if (line.StartsWith('-') || line.StartsWith('•') || line.StartsWith('*')) return false;
        if (line.EndsWith(':')) return true;
        return !line.EndsWith('.') && !line.Contains(". ");
    }

    public bool SaysAnything => Sections.Count > 0;
}

/// <summary>
/// One requirement of an anchored posting, and whether the CV evidences it.
///
/// This record is the only door resume information walks through. There is
/// deliberately no "candidate profile" type in a fact package: the moment one exists,
/// "what are my strongest skills" becomes answerable and the job-anchor rule is gone.
/// </summary>
public sealed record RequirementCoverage(
    int JobId,
    string Requirement,
    /// <summary>Have | Missing | Transferable — the vocabulary the matcher already uses.</summary>
    string Status,
    string? Evidence = null);
