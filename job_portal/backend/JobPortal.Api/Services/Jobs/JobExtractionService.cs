using System.Text.Json;
using System.Text.RegularExpressions;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Llm;

namespace JobPortal.Api.Services.Jobs;

/// <summary>The structured content of a job ad, before it becomes a database row.</summary>
public class ExtractedJob
{
    public string Title { get; set; } = "";
    public string Company { get; set; } = "";
    public string Location { get; set; } = "";
    public string WorkMode { get; set; } = "Unspecified";
    public string EmploymentType { get; set; } = "Unspecified";
    public string SeniorityLevel { get; set; } = "Unspecified";
    public double? MinYearsExperience { get; set; }
    public double? MaxYearsExperience { get; set; }
    public decimal? SalaryMin { get; set; }
    public decimal? SalaryMax { get; set; }
    public string? SalaryCurrency { get; set; }
    public List<string> RequiredSkills { get; set; } = new();
    public List<string> PreferredSkills { get; set; } = new();
    public List<string> Responsibilities { get; set; } = new();
    public List<string> Qualifications { get; set; } = new();
    public string Summary { get; set; } = "";
    public string Mode { get; set; } = "deterministic";
}

public interface IJobExtractionService
{
    Task<ExtractedJob> ExtractAsync(string rawText, string filenameHint, CancellationToken ct = default);

    /// <summary>The text that gets embedded for a job. Public because it is also
    /// how a re-index decides whether anything actually changed.</summary>
    string BuildEmbeddingText(PortalJob job);
}

/// <summary>
/// Turns job-ad text into structured fields.
///
/// Two passes, in this order and never the other way round:
///   1. Deterministic structure reading. Always runs, always produces a usable
///      posting, works offline, and is the same answer every time.
///   2. An optional LLM pass that fills gaps and cleans up phrasing.
///
/// The LLM refines; it does not originate. A field the deterministic pass read
/// straight out of a "Location:" line is not handed to the model to second-guess,
/// because a hallucinated location on a real job posting is worse than a blank
/// one. The model's value is in the places where prose has to be understood:
/// separating must-haves from nice-to-haves, and writing a one-line summary.
/// </summary>
public class JobExtractionService : IJobExtractionService
{
    // Structural vocabulary: the names documents give their own sections. This is
    // not domain knowledge about jobs or skills, and adding a skill name here
    // would be a bug, not a feature.
    private static readonly string[] Headings =
    {
        "Responsibilities", "Key Responsibilities", "What You'll Do", "What You Will Do",
        "The Role", "Role Overview", "About the Role", "Job Description", "Overview", "Summary",
        "Requirements", "Required Skills", "Must Have", "Must Haves", "Essential Skills",
        "Minimum Qualifications", "Basic Qualifications", "What We're Looking For",
        "Qualifications", "Education", "Experience",
        "Preferred", "Preferred Skills", "Preferred Qualifications", "Nice to Have",
        "Nice to Haves", "Good to Have", "Bonus Points", "Desirable",
        "Skills", "Technical Skills", "Tech Stack",
        "Benefits", "Perks", "What We Offer", "About Us", "About the Company",
        "Compensation", "Salary", "Location", "How to Apply",
    };

    private static readonly string[] RequiredHeadings =
    {
        "Requirements", "Required Skills", "Must Have", "Must Haves", "Essential Skills",
        "Minimum Qualifications", "Basic Qualifications", "What We're Looking For",
        "Skills", "Technical Skills", "Tech Stack",
    };

    private static readonly string[] PreferredHeadings =
    {
        "Preferred", "Preferred Skills", "Preferred Qualifications", "Nice to Have",
        "Nice to Haves", "Good to Have", "Bonus Points", "Desirable",
    };

    private static readonly string[] ResponsibilityHeadings =
    {
        "Responsibilities", "Key Responsibilities", "What You'll Do", "What You Will Do",
        "The Role", "Role Overview",
    };

    private static readonly string[] QualificationHeadings = { "Qualifications", "Education", "Experience" };

    // "5+ years", "3 to 5 years", "minimum of 4 years"
    private static readonly Regex YearsRange = new(
        @"(?<min>\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)\s*(?<max>\d{1,2})\s*\+?\s*(?:years|yrs)",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex YearsMinimum = new(
        @"(?<min>\d{1,2})\s*\+?\s*(?:years|yrs)",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    // "$120,000 - $150,000", "INR 12,00,000", "£70k"
    private static readonly Regex SalaryRange = new(
        @"(?<cur>[$£€₹]|USD|EUR|GBP|INR|CAD|AUD)?\s*(?<a>\d[\d,\.]{2,})\s*(?<ka>k)?\s*(?:-|–|to)\s*(?<cur2>[$£€₹]|USD|EUR|GBP|INR|CAD|AUD)?\s*(?<b>\d[\d,\.]{2,})\s*(?<kb>k)?",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private readonly IOllamaClient _llm;
    private readonly ILogger<JobExtractionService> _logger;

    public JobExtractionService(IOllamaClient llm, ILogger<JobExtractionService> logger)
    {
        _llm = llm;
        _logger = logger;
    }

    public async Task<ExtractedJob> ExtractAsync(string rawText, string filenameHint, CancellationToken ct = default)
    {
        var job = ExtractDeterministic(rawText, filenameHint);

        if (_llm.ChatAvailable)
        {
            var refined = await RefineWithLlmAsync(job, rawText, ct);
            if (refined) job.Mode = "llm-assisted";
        }

        // The summary is the one field with no deterministic source worth reading
        // out of the document, so it is composed last from whatever was found.
        if (string.IsNullOrWhiteSpace(job.Summary)) job.Summary = ComposeSummary(job);

        return job;
    }

    // -- pass 1: structure --------------------------------------------------

    private ExtractedJob ExtractDeterministic(string text, string filenameHint)
    {
        var sections = TextStructure.SplitSections(text, Headings);
        var fields = TextStructure.LabelledFields(text);
        var job = new ExtractedJob();

        job.Title = TextStructure.FirstField(fields, "Job Title", "Title", "Position", "Role")
                    ?? TextStructure.FirstMeaningfulLine(text)
                    ?? CleanFilename(filenameHint);

        job.Company = TextStructure.FirstField(fields, "Company", "Organisation", "Organization", "Employer", "Client") ?? "";
        job.Location = TextStructure.FirstField(fields, "Location", "Based in", "Work Location", "Place") ?? "";

        job.WorkMode = DetectWorkMode(text, fields);
        job.EmploymentType = DetectEmploymentType(text, fields);
        // A posting that states its level is not guessed at.
        //
        // Inference reads the title plus the opening of the document, which means a
        // responsibility like "Lead and mentor a team of platform engineers" labels
        // a Senior role as Lead — measured on a posting whose own header said
        // "Seniority: Senior". What the posting declares about itself wins; the
        // detector is the fallback for the postings that declare nothing.
        var statedSeniority = TextStructure.FirstField(
            fields, "Seniority", "Seniority Level", "Level", "Experience Level", "Job Level");

        var declared = string.IsNullOrWhiteSpace(statedSeniority)
            ? "Unspecified"
            : DetectSeniority(statedSeniority, "");

        job.SeniorityLevel = declared != "Unspecified" ? declared : DetectSeniority(job.Title, text);

        (job.MinYearsExperience, job.MaxYearsExperience) = DetectYears(text);
        (job.SalaryMin, job.SalaryMax, job.SalaryCurrency) = DetectSalary(text, fields);

        var requiredBullets = CollectBullets(sections, RequiredHeadings);
        var preferredBullets = CollectBullets(sections, PreferredHeadings);

        var (required, requiredStatements) = AtomiseRequirements(requiredBullets);
        var (preferred, preferredStatements) = AtomiseRequirements(preferredBullets);

        job.RequiredSkills = required;
        job.PreferredSkills = preferred;
        job.Responsibilities = CollectBullets(sections, ResponsibilityHeadings);

        // Prose requirements are still requirements; they are just not skill names.
        // Keeping them under qualifications means nothing the posting said is lost.
        job.Qualifications = CollectBullets(sections, QualificationHeadings)
            .Concat(requiredStatements)
            .Concat(preferredStatements)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(30)
            .ToList();

        // A JD with no recognisable requirements section still has requirements —
        // they are just written in prose. Falling back to the whole document's
        // bullets is worse than nothing only if it produces noise, so it is capped.
        if (job.RequiredSkills.Count == 0 && job.Responsibilities.Count == 0)
        {
            job.RequiredSkills = TextStructure.BulletLines(text).Take(20).ToList();
        }

        // "Preferred" items listed inside the required section would otherwise be
        // scored as must-haves, which is the single most common way a good
        // candidate gets marked down for a nice-to-have.
        job.PreferredSkills = job.PreferredSkills
            .Where(p => !job.RequiredSkills.Contains(p, StringComparer.OrdinalIgnoreCase))
            .ToList();

        if (sections.TryGetValue("Summary", out var summary) ||
            sections.TryGetValue("Overview", out summary) ||
            sections.TryGetValue("About the Role", out summary))
        {
            job.Summary = TextStructure.Clip(TextStructure.Collapse(summary), 400);
        }

        return job;
    }

    /// <summary>
    /// Breaks requirement bullets into one skill per entry, and separates out the
    /// ones that are sentences rather than skill names.
    ///
    /// This matters more than it looks. Job ads bundle requirements
    /// ("Kubernetes, Docker, Helm"; "AWS and Azure"), and a bundled bullet is
    /// matched as a single phrase — so a candidate whose resume lists all three is
    /// told they are MISSING all three, and is advised to go and learn things they
    /// already know. Splitting is what makes the skill verdicts true.
    ///
    /// It is also why prose is separated out: "5+ years of DevOps experience" is a
    /// sentence, and comparing a candidate's skill names against a sentence
    /// produces a low similarity that means nothing at all. The years in it are
    /// already read by <see cref="DetectYears"/>, and the sentence itself is kept
    /// under qualifications.
    ///
    /// Purely structural — separators and word counts. There is no list anywhere of
    /// which words are technologies.
    /// </summary>
    private static (List<string> Skills, List<string> Statements) AtomiseRequirements(List<string> bullets)
    {
        var skills = new List<string>();
        var statements = new List<string>();

        foreach (var bullet in bullets)
        {
            var cleaned = TextStructure.Collapse(bullet).TrimEnd('.', ';', ',');
            if (cleaned.Length < 2) continue;

            // A bullet long enough to be a sentence is treated as one. Six words is
            // comfortably above a bundled skill list ("Kubernetes, Docker, Helm")
            // and below any real requirement sentence.
            //
            // A bullet stating a duration is a sentence too, however short: "5+ years
            // of DevOps/Cloud Engineering experience" is four words past the slash and
            // would otherwise split into "5+ years of DevOps" and "Cloud Engineering
            // experience", two things no resume will ever list as skills. The years in
            // it are already read by DetectYears and scored on their own dimension.
            if (WordCount(cleaned) > 6 || YearsMinimum.IsMatch(cleaned))
            {
                statements.Add(cleaned);
                continue;
            }

            var parts = cleaned
                .Split(new[] { ",", "/", "&", " and ", " or ", "|", ";" }, StringSplitOptions.RemoveEmptyEntries)
                .Select(p => TextStructure.Collapse(p).Trim('.', '-', ' '))
                .Where(p => p.Length is >= 2 and <= 40 && WordCount(p) <= 4 && p.Any(char.IsLetter))
                .ToList();

            // Splitting that yields nothing usable means the separators were part of
            // the name rather than between names, so the original stands.
            skills.AddRange(parts.Count > 0 ? parts : new List<string> { cleaned });
        }

        return (
            skills.Distinct(StringComparer.OrdinalIgnoreCase).Take(30).ToList(),
            statements.Distinct(StringComparer.OrdinalIgnoreCase).Take(15).ToList());
    }

    private static int WordCount(string value) =>
        value.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

    private static List<string> CollectBullets(Dictionary<string, string> sections, string[] headings)
    {
        var items = new List<string>();
        foreach (var heading in headings)
        {
            if (!sections.TryGetValue(heading, out var block)) continue;
            items.AddRange(TextStructure.BulletLines(block));
        }
        return items.Distinct(StringComparer.OrdinalIgnoreCase).Take(30).ToList();
    }

    private static string DetectWorkMode(string text, IReadOnlyDictionary<string, string> fields)
    {
        var probe = ((TextStructure.FirstField(fields, "Work Mode", "Workplace", "Location", "Remote") ?? "")
                     + " " + text[..Math.Min(text.Length, 2500)]).ToLowerInvariant();

        // Order matters: "hybrid (2 days remote)" is hybrid, not remote, so the
        // more specific arrangement is tested first.
        if (probe.Contains("hybrid")) return "Hybrid";
        if (probe.Contains("fully remote") || probe.Contains("100% remote") ||
            probe.Contains("remote-first") || probe.Contains("work from home") ||
            Regex.IsMatch(probe, @"\bremote\b")) return "Remote";
        if (probe.Contains("on-site") || probe.Contains("onsite") || probe.Contains("in office") ||
            probe.Contains("in-office")) return "Onsite";

        return "Unspecified";
    }

    private static string DetectEmploymentType(string text, IReadOnlyDictionary<string, string> fields)
    {
        var probe = ((TextStructure.FirstField(fields, "Employment Type", "Job Type", "Contract Type") ?? "")
                     + " " + text[..Math.Min(text.Length, 2500)]).ToLowerInvariant();

        if (probe.Contains("intern")) return "Internship";
        if (probe.Contains("part-time") || probe.Contains("part time")) return "Part-time";
        if (probe.Contains("contract") || probe.Contains("freelance") || probe.Contains("c2c")) return "Contract";
        if (probe.Contains("temporary") || probe.Contains("temp ")) return "Temporary";
        if (probe.Contains("full-time") || probe.Contains("full time") || probe.Contains("permanent")) return "Full-time";

        return "Unspecified";
    }

    private static string DetectSeniority(string title, string text)
    {
        var probe = (title + " " + text[..Math.Min(text.Length, 1200)]).ToLowerInvariant();

        // Most specific first: "Senior Staff Engineer" is Staff, and a plain
        // "Senior" test placed first would mislabel it.
        if (probe.Contains("principal") || probe.Contains("distinguished")) return "Principal";
        if (probe.Contains("staff engineer") || Regex.IsMatch(probe, @"\bstaff\b")) return "Staff";
        if (probe.Contains("director") || probe.Contains("head of") || probe.Contains("vp ")) return "Leadership";
        if (probe.Contains("manager") || Regex.IsMatch(probe, @"\blead\b")) return "Lead";
        if (Regex.IsMatch(probe, @"\b(senior|sr\.?|snr)\b")) return "Senior";
        if (Regex.IsMatch(probe, @"\b(junior|jr\.?|entry[- ]level|graduate|fresher|trainee|intern)\b")) return "Junior";
        if (Regex.IsMatch(probe, @"\b(mid[- ]level|intermediate)\b")) return "Mid";

        return "Unspecified";
    }

    private static (double?, double?) DetectYears(string text)
    {
        var range = YearsRange.Match(text);
        if (range.Success &&
            double.TryParse(range.Groups["min"].Value, out var lo) &&
            double.TryParse(range.Groups["max"].Value, out var hi) &&
            hi >= lo && hi <= 50)
        {
            return (lo, hi);
        }

        var minimum = YearsMinimum.Match(text);
        if (minimum.Success && double.TryParse(minimum.Groups["min"].Value, out var min) && min <= 50)
        {
            // An open-ended "5+ years" genuinely has no maximum. Inventing one
            // would silently exclude a candidate for being too experienced.
            return (min, null);
        }

        return (null, null);
    }

    private static (decimal?, decimal?, string?) DetectSalary(string text, IReadOnlyDictionary<string, string> fields)
    {
        var probe = TextStructure.FirstField(fields, "Salary", "Compensation", "Pay", "Package", "CTC")
                    ?? FindSalaryLine(text);
        if (probe is null) return (null, null, null);

        var match = SalaryRange.Match(probe);
        if (!match.Success) return (null, null, null);

        var a = ParseAmount(match.Groups["a"].Value, match.Groups["ka"].Success);
        var b = ParseAmount(match.Groups["b"].Value, match.Groups["kb"].Success);
        if (a is null || b is null) return (null, null, null);

        var currency = NormaliseCurrency(
            match.Groups["cur"].Success ? match.Groups["cur"].Value : match.Groups["cur2"].Value);

        return a <= b ? (a, b, currency) : (b, a, currency);
    }

    private static string? FindSalaryLine(string text)
    {
        foreach (var line in text.Split('\n'))
        {
            var lower = line.ToLowerInvariant();
            if (lower.Contains("salary") || lower.Contains("compensation") ||
                lower.Contains("ctc") || lower.Contains("per annum") || lower.Contains("per year"))
            {
                return line;
            }
        }
        return null;
    }

    private static decimal? ParseAmount(string raw, bool thousands)
    {
        // Indian grouping ("12,00,000") and Western grouping ("120,000") both mean
        // the digits; stripping separators avoids having to know which is in use.
        var digits = raw.Replace(",", "").Replace(".", "");
        if (!decimal.TryParse(digits, out var value)) return null;
        if (thousands) value *= 1000;
        return value is > 0 and < 100_000_000 ? value : null;
    }

    private static string? NormaliseCurrency(string symbol) => symbol.ToUpperInvariant() switch
    {
        "$" or "USD" => "USD",
        "£" or "GBP" => "GBP",
        "€" or "EUR" => "EUR",
        "₹" or "INR" => "INR",
        "CAD" => "CAD",
        "AUD" => "AUD",
        _ => null,
    };

    private static string CleanFilename(string filename)
    {
        var name = Path.GetFileNameWithoutExtension(filename ?? "");
        // Uploads arrive as "12_Senior_DotNet_Engineer.pdf" once the Python side
        // has prefixed the row id.
        name = Regex.Replace(name, @"^\d+[_\-]", "");
        name = name.Replace('_', ' ').Replace('-', ' ');
        return TextStructure.Collapse(name);
    }

    private static string ComposeSummary(ExtractedJob job)
    {
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(job.Title)) parts.Add(job.Title);
        if (!string.IsNullOrWhiteSpace(job.Company)) parts.Add("at " + job.Company);
        if (!string.IsNullOrWhiteSpace(job.Location)) parts.Add("in " + job.Location);
        if (job.WorkMode != "Unspecified") parts.Add($"({job.WorkMode})");

        var head = string.Join(" ", parts);
        var skills = job.RequiredSkills.Take(6).ToList();
        return skills.Count > 0 ? $"{head}. Key requirements: {string.Join(", ", skills)}." : head + ".";
    }

    // -- pass 2: LLM refinement --------------------------------------------

    private const string RefinementSystemPrompt = """
        You normalise job postings into JSON. You are given the raw text of one job
        advertisement and the fields a deterministic parser already read from it.

        Rules:
        - Use ONLY information present in the raw text. Never invent a company,
          location, salary or requirement that is not written there.
        - If the text does not state something, return null (or an empty array).
          A null is correct and useful; a guess is not.
        - required_skills: concrete capabilities the posting treats as mandatory.
        - preferred_skills: capabilities it treats as optional, bonus or desirable.
        - Keep each skill short (1-5 words) and as written in the posting.
        - summary: at most 2 sentences describing the role, in plain language.

        Reply with JSON only, matching exactly this shape:
        {"title":string|null,"company":string|null,"location":string|null,
         "work_mode":"Remote"|"Hybrid"|"Onsite"|null,
         "employment_type":string|null,"seniority_level":string|null,
         "required_skills":[string],"preferred_skills":[string],"summary":string|null}
        """;

    private async Task<bool> RefineWithLlmAsync(ExtractedJob job, string rawText, CancellationToken ct)
    {
        var prompt = $"""
            RAW JOB POSTING
            ---------------
            {TextStructure.Clip(rawText, 6000)}

            WHAT THE PARSER ALREADY READ
            ----------------------------
            title: {Show(job.Title)}
            company: {Show(job.Company)}
            location: {Show(job.Location)}
            work_mode: {job.WorkMode}
            employment_type: {job.EmploymentType}
            seniority_level: {job.SeniorityLevel}
            required_skills: {Show(string.Join(" | ", job.RequiredSkills))}
            preferred_skills: {Show(string.Join(" | ", job.PreferredSkills))}
            """;

        var reply = await _llm.ChatAsync(new[]
        {
            new ChatTurn("system", RefinementSystemPrompt),
            new ChatTurn("user", prompt),
        }, jsonMode: true, ct: ct);

        if (string.IsNullOrWhiteSpace(reply)) return false;

        try
        {
            using var doc = JsonDocument.Parse(ExtractJsonObject(reply));
            var root = doc.RootElement;

            // Gap-filling only for the identity fields. If the deterministic pass
            // read a location off a "Location:" line, that came from the document
            // itself and the model has nothing to add but risk.
            job.Title = PreferExisting(job.Title, ReadString(root, "title"));
            job.Company = PreferExisting(job.Company, ReadString(root, "company"));
            job.Location = PreferExisting(job.Location, ReadString(root, "location"));

            if (job.WorkMode == "Unspecified")
                job.WorkMode = ReadString(root, "work_mode") ?? "Unspecified";
            if (job.EmploymentType == "Unspecified")
                job.EmploymentType = ReadString(root, "employment_type") ?? "Unspecified";
            if (job.SeniorityLevel == "Unspecified")
                job.SeniorityLevel = ReadString(root, "seniority_level") ?? "Unspecified";

            // Skills ARE overwritten when the model returns any, because splitting
            // must-have from nice-to-have is prose comprehension, which is the one
            // job the deterministic pass genuinely cannot do well. Every returned
            // skill is still checked against the source text below.
            var required = ReadStrings(root, "required_skills");
            var preferred = ReadStrings(root, "preferred_skills");

            if (required.Count > 0) job.RequiredSkills = Ground(required, rawText, job.RequiredSkills);
            if (preferred.Count > 0) job.PreferredSkills = Ground(preferred, rawText, job.PreferredSkills);

            job.PreferredSkills = job.PreferredSkills
                .Where(p => !job.RequiredSkills.Contains(p, StringComparer.OrdinalIgnoreCase))
                .ToList();

            var summary = ReadString(root, "summary");
            if (!string.IsNullOrWhiteSpace(summary)) job.Summary = TextStructure.Clip(summary, 500);

            return true;
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Job refinement returned unparseable JSON; keeping the deterministic extraction.");
            return false;
        }
    }

    /// <summary>
    /// Keeps only the skills that actually occur in the posting.
    ///
    /// This is the guard that stops the model from adding a requirement nobody
    /// wrote. A fabricated requirement is not a cosmetic error: it becomes a real
    /// gap in a candidate's fit analysis, and they get told to go learn something
    /// the employer never asked for. If nothing survives the check, the
    /// deterministic list stands.
    /// </summary>
    private List<string> Ground(List<string> candidates, string rawText, List<string> fallback)
    {
        var haystack = rawText.ToLowerInvariant();

        var present = candidates
            .Select(TextStructure.Collapse)
            .Where(s => haystack.Contains(s.ToLowerInvariant()))
            .ToList();

        // Held to exactly the standard the deterministic pass holds itself to.
        //
        // The model's list REPLACES the atomised one, so anything it returns that
        // was never atomised goes straight into RequiredSkills — measured on a real
        // posting that meant "Terraform and Ansible" stored as one skill, plus
        // "5+ years of DevOps/Cloud Engineering experience" and "Thorough knowledge
        // of observability best practices" stored as skills at all. Nothing can
        // evidence those, so every candidate was marked as missing them while the
        // technologies named inside them earned no credit.
        //
        // Running the model's output back through the same atomiser splits the
        // compounds and routes the sentences out, rather than duplicating the rules.
        var (atomised, _) = AtomiseRequirements(present);

        var kept = atomised
            .Where(TextStructure.LooksLikeSkillName)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(30)
            .ToList();

        if (kept.Count == 0 && candidates.Count > 0)
        {
            _logger.LogWarning(
                "Every skill the model proposed was absent from the posting text; " +
                "keeping the deterministic list.");
            return fallback;
        }

        return kept;
    }

    private static string PreferExisting(string current, string? proposed) =>
        !string.IsNullOrWhiteSpace(current) ? current : TextStructure.Collapse(proposed ?? "");

    private static string Show(string value) => string.IsNullOrWhiteSpace(value) ? "(not found)" : value;

    private static string? ReadString(JsonElement root, string name) =>
        root.TryGetProperty(name, out var e) && e.ValueKind == JsonValueKind.String &&
        !string.IsNullOrWhiteSpace(e.GetString())
            ? TextStructure.Collapse(e.GetString()!)
            : null;

    private static List<string> ReadStrings(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var e) || e.ValueKind != JsonValueKind.Array)
            return new List<string>();

        return e.EnumerateArray()
            .Where(v => v.ValueKind == JsonValueKind.String)
            .Select(v => v.GetString()!)
            .Where(v => !string.IsNullOrWhiteSpace(v))
            .ToList();
    }

    /// <summary>Pulls the JSON object out of a reply that wrapped it in prose or a
    /// code fence, which happens even with format=json on some models.</summary>
    private static string ExtractJsonObject(string reply)
    {
        var start = reply.IndexOf('{');
        var end = reply.LastIndexOf('}');
        return start >= 0 && end > start ? reply[start..(end + 1)] : reply;
    }

    // -- embedding text -----------------------------------------------------

    /// <summary>
    /// Composes what actually gets embedded.
    ///
    /// Not the raw document: a JD's benefits, equal-opportunity statement and
    /// company boilerplate are half its length and are near-identical across
    /// postings, so embedding them pulls every job toward the same point in the
    /// space and flattens the very distinctions the search depends on. What goes
    /// in is the role and its requirements.
    /// </summary>
    public string BuildEmbeddingText(PortalJob job)
    {
        var parts = new List<string>
        {
            job.Title,
            job.SeniorityLevel != "Unspecified" ? job.SeniorityLevel + " level" : "",
            job.Summary,
            string.Join(", ", TextStructure.ReadLines(job.RequiredSkills)),
            string.Join(", ", TextStructure.ReadLines(job.PreferredSkills)),
            string.Join(". ", TextStructure.ReadLines(job.Responsibilities).Take(10)),
            string.Join(". ", TextStructure.ReadLines(job.Qualifications).Take(6)),
        };

        var text = string.Join("\n", parts.Where(p => !string.IsNullOrWhiteSpace(p)));

        // A posting whose sections were all unreadable would otherwise embed to
        // little more than its title.
        if (text.Length < 120) text += "\n" + TextStructure.Clip(job.RawText, 2000);

        return TextStructure.Clip(text, 4000);
    }
}
