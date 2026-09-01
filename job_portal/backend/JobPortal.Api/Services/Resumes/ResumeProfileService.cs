using System.Text.Json;
using System.Text.RegularExpressions;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Llm;

namespace JobPortal.Api.Services.Resumes;

public class ResumeProfile
{
    public string CandidateName { get; set; } = "";
    public string Email { get; set; } = "";
    public string Phone { get; set; } = "";
    public string Location { get; set; } = "";
    public string CurrentTitle { get; set; } = "";
    public double? YearsExperience { get; set; }
    public List<string> Skills { get; set; } = new();
    public List<string> Titles { get; set; } = new();
    public List<string> Education { get; set; } = new();
    public string Summary { get; set; } = "";
    public string Mode { get; set; } = "deterministic";
}

public interface IResumeProfileService
{
    Task<ResumeProfile> ExtractAsync(string rawText, string filenameHint, CancellationToken ct = default);
    string BuildEmbeddingText(PortalResume resume);
}

/// <summary>
/// Reads a candidate profile out of resume text.
///
/// Same discipline as the job side: deterministic structure first, LLM second and
/// only to fill gaps or read prose. Contact details in particular are NEVER taken
/// from the model — an email address is either literally present in the document
/// or it is wrong, and a hallucinated one on a job application is a real-world
/// failure, not a formatting glitch.
/// </summary>
public class ResumeProfileService : IResumeProfileService
{
    private static readonly string[] Headings =
    {
        "Summary", "Profile", "Professional Summary", "Objective", "About Me", "About",
        "Skills", "Technical Skills", "Core Competencies", "Technologies", "Tech Stack",
        "Expertise", "Areas of Expertise",
        "Experience", "Work Experience", "Professional Experience", "Employment History",
        "Employment", "Career History", "Work History",
        "Projects", "Personal Projects", "Key Projects",
        "Education", "Academic Background", "Qualifications",
        "Certifications", "Certificates", "Licenses",
        "Achievements", "Awards", "Publications", "Interests", "Hobbies",
        "Contact", "Contact Information", "References",
    };

    // "Languages" is deliberately NOT a heading. In a resume it is far more often a
    // sub-label inside the skills block ("Languages: Python, Bash, Go") than a
    // section of its own, and treating it as a heading ended the Skills section
    // early — every skill listed after that line vanished from the profile and was
    // then scored as Missing against jobs that asked for it. Leaving it out costs
    // nothing, because nothing here consumes a spoken-languages section.

    private static readonly string[] SkillHeadings =
    {
        "Skills", "Technical Skills", "Core Competencies", "Technologies", "Tech Stack",
        "Expertise", "Areas of Expertise",
    };

    private static readonly string[] ExperienceHeadings =
    {
        "Experience", "Work Experience", "Professional Experience", "Employment History",
        "Employment", "Career History", "Work History",
    };

    private static readonly string[] EducationHeadings =
    {
        "Education", "Academic Background", "Qualifications",
    };

    private static readonly string[] SummaryHeadings =
    {
        "Summary", "Profile", "Professional Summary", "Objective", "About Me", "About",
    };

    private static readonly Regex EmailPattern = new(
        @"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", RegexOptions.Compiled);

    // Deliberately loose: international formats vary far too much for a strict
    // pattern, and a missed phone number is worse than an over-inclusive one.
    private static readonly Regex PhonePattern = new(
        @"(?<!\d)(\+\d{1,3}[\s\-\.]?)?(\(?\d{2,4}\)?[\s\-\.]?){2,4}\d{2,4}(?!\d)", RegexOptions.Compiled);

    private static readonly Regex YearsClaim = new(
        @"(?<n>\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs)(?:\s+of)?\s+(?:professional\s+|industry\s+|relevant\s+|hands[- ]on\s+)?experience",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    // "Jan 2019 - Present", "2019 – 2023", "03/2020 to 06/2022"
    private static readonly Regex DateRange = new(
        @"(?<from>(?:[A-Za-z]{3,9}\.?\s+)?(?:\d{1,2}[/\-])?(?<fy>(?:19|20)\d{2}))\s*(?:-|–|—|to|until)\s*(?<to>present|current|now|(?:[A-Za-z]{3,9}\.?\s+)?(?:\d{1,2}[/\-])?(?<ty>(?:19|20)\d{2}))",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private readonly IOllamaClient _llm;
    private readonly ILogger<ResumeProfileService> _logger;

    public ResumeProfileService(IOllamaClient llm, ILogger<ResumeProfileService> logger)
    {
        _llm = llm;
        _logger = logger;
    }

    public async Task<ResumeProfile> ExtractAsync(string rawText, string filenameHint, CancellationToken ct = default)
    {
        var profile = ExtractDeterministic(rawText, filenameHint);

        // The deterministic profile is already a complete answer; refinement only
        // improves it. So nothing the model does may fail the ingestion — a resume
        // that parsed must be accepted even if the endpoint is slow, down, or
        // returns something unreadable. Without this guard a single LLM fault takes
        // the upload down with it and the candidate is told their CV is unreadable,
        // which is both wrong and unactionable.
        try
        {
            if (_llm.ChatAvailable && await RefineWithLlmAsync(profile, rawText, ct))
            {
                profile.Mode = "llm-assisted";
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;   // the caller really did go away
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "LLM refinement failed for {File}; keeping the deterministic profile.",
                filenameHint);
        }

        if (string.IsNullOrWhiteSpace(profile.Summary)) profile.Summary = ComposeSummary(profile);
        return profile;
    }

    // -- pass 1 -------------------------------------------------------------

    private ResumeProfile ExtractDeterministic(string text, string filenameHint)
    {
        var sections = TextStructure.SplitSections(text, Headings);
        var fields = TextStructure.LabelledFields(text, scanLines: 40);
        var profile = new ResumeProfile();

        profile.Email = EmailPattern.Match(text).Value;
        profile.Phone = FindPhone(text);
        profile.Location = TextStructure.FirstField(fields, "Location", "Address", "City", "Based in") ?? "";
        profile.CandidateName = FindName(text, fields, filenameHint);
        profile.CurrentTitle = TextStructure.FirstField(fields, "Title", "Designation", "Current Role", "Role") ?? "";

        profile.Skills = CollectSkills(sections);
        profile.Titles = CollectTitles(sections);
        profile.Education = Collect(sections, EducationHeadings, 8);

        if (string.IsNullOrWhiteSpace(profile.CurrentTitle) && profile.Titles.Count > 0)
        {
            // Resumes are reverse-chronological by overwhelming convention, so the
            // first role listed is the current one.
            profile.CurrentTitle = profile.Titles[0];
        }

        profile.YearsExperience = EstimateYears(text, sections);

        foreach (var heading in SummaryHeadings)
        {
            if (sections.TryGetValue(heading, out var block) && block.Trim().Length > 20)
            {
                profile.Summary = TextStructure.Clip(TextStructure.Collapse(block), 400);
                break;
            }
        }

        return profile;
    }

    private static string FindPhone(string text)
    {
        // Only the header block: a phone-shaped run of digits deeper in a resume is
        // far more likely to be a date range, a postcode or a metric.
        var header = string.Join("\n", text.Split('\n').Take(15));
        foreach (Match match in PhonePattern.Matches(header))
        {
            var digits = match.Value.Count(char.IsDigit);
            if (digits is >= 8 and <= 15) return TextStructure.Collapse(match.Value);
        }
        return "";
    }

    private static string FindName(string text, IReadOnlyDictionary<string, string> fields, string filenameHint)
    {
        var labelled = TextStructure.FirstField(fields, "Name", "Candidate Name", "Full Name");
        if (!string.IsNullOrWhiteSpace(labelled)) return labelled;

        // Nearly every resume opens with the person's name on its own line.
        foreach (var line in text.Split('\n').Take(8))
        {
            var trimmed = TextStructure.Collapse(line);
            if (trimmed.Length is < 3 or > 60) continue;
            if (trimmed.Contains('@') || trimmed.Any(char.IsDigit)) continue;

            var words = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (words.Length is < 2 or > 5) continue;
            // A heading like "CURRICULUM VITAE" is all caps but is not a name;
            // requiring every word to start with a capital and the rest to be
            // lower-case rules out both headings and section labels.
            if (!words.All(w => char.IsUpper(w[0]))) continue;

            return trimmed;
        }

        // The filename is the last resort, and often carries it: "Jane_Doe_CV.pdf".
        var name = Path.GetFileNameWithoutExtension(filenameHint ?? "");
        name = Regex.Replace(name, @"(?i)[_\-\s]*(resume|cv|curriculum vitae|final|updated|\d+)[_\-\s]*", " ");
        name = TextStructure.Collapse(name.Replace('_', ' ').Replace('-', ' '));
        return name.Length is >= 3 and <= 60 ? name : "";
    }

    private static List<string> CollectSkills(Dictionary<string, string> sections)
    {
        var skills = new List<string>();

        foreach (var heading in SkillHeadings)
        {
            if (!sections.TryGetValue(heading, out var block)) continue;

            foreach (var line in block.Split('\n'))
            {
                var trimmed = line.Trim();
                if (trimmed.Length < 2) continue;

                // Skills sections are written both ways: "Languages: C#, Python"
                // and a bullet per skill. Splitting on separators handles both,
                // and drops the "Languages:" label when there is one.
                var payload = trimmed.Contains(':') ? trimmed.Split(':', 2)[1] : trimmed;
                skills.AddRange(TextStructure.SplitInlineList(payload));
            }
        }

        return skills
            .Where(s => s.Length is >= 2 and <= 40)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(60)
            .ToList();
    }

    private static List<string> CollectTitles(Dictionary<string, string> sections)
    {
        var titles = new List<string>();

        foreach (var heading in ExperienceHeadings)
        {
            if (!sections.TryGetValue(heading, out var block)) continue;

            foreach (var line in block.Split('\n'))
            {
                var trimmed = TextStructure.Collapse(line);
                if (trimmed.Length is < 4 or > 90) continue;

                // A role header is a short line that is not a bullet and not a
                // sentence. Sentences are what the achievements underneath are
                // made of, and they end in a full stop.
                if (Regex.IsMatch(line, @"^\s*[•●▪‣⁃·\*\-–—]")) continue;
                if (trimmed.EndsWith('.')) continue;

                // Strip the company and dates that usually share the line.
                var head = trimmed.Split(new[] { " at ", " | ", " – ", " — ", ", " }, StringSplitOptions.None)[0];
                head = Regex.Replace(head, @"\s*\(?(19|20)\d{2}.*$", "").Trim();

                if (head.Length is >= 4 and <= 60 && head.Any(char.IsLetter)) titles.Add(head);
            }
        }

        return titles.Distinct(StringComparer.OrdinalIgnoreCase).Take(10).ToList();
    }

    private static List<string> Collect(Dictionary<string, string> sections, string[] headings, int max)
    {
        var items = new List<string>();
        foreach (var heading in headings)
        {
            if (sections.TryGetValue(heading, out var block)) items.AddRange(TextStructure.BulletLines(block));
        }
        return items.Distinct(StringComparer.OrdinalIgnoreCase).Take(max).ToList();
    }

    /// <summary>
    /// Years of experience, preferring what the resume claims and otherwise
    /// measuring what it shows.
    /// </summary>
    private static double? EstimateYears(string text, Dictionary<string, string> sections)
    {
        var claim = YearsClaim.Match(text);
        if (claim.Success && double.TryParse(claim.Groups["n"].Value, out var claimed) && claimed is > 0 and <= 50)
        {
            return claimed;
        }

        // Otherwise, span the employment dates. Union rather than sum: overlapping
        // roles (a promotion, a concurrent contract) are one stretch of time, and
        // adding them would credit someone with two careers run in parallel.
        var block = ExperienceHeadings
            .Where(sections.ContainsKey)
            .Select(h => sections[h])
            .FirstOrDefault() ?? text;

        var spans = new List<(int From, int To)>();
        var thisYear = DateTime.UtcNow.Year;

        foreach (Match match in DateRange.Matches(block))
        {
            if (!int.TryParse(match.Groups["fy"].Value, out var from)) continue;

            var toGroup = match.Groups["ty"];
            var to = toGroup.Success && int.TryParse(toGroup.Value, out var parsed) ? parsed : thisYear;

            if (from < 1960 || to > thisYear + 1 || to < from) continue;
            spans.Add((from, to));
        }

        if (spans.Count == 0) return null;

        spans.Sort((a, b) => a.From.CompareTo(b.From));
        var total = 0;
        var cursor = int.MinValue;

        foreach (var (from, to) in spans)
        {
            var start = Math.Max(from, cursor);
            if (to > start) total += to - start;
            cursor = Math.Max(cursor, to);
        }

        return total is > 0 and <= 50 ? total : null;
    }

    private static string ComposeSummary(ResumeProfile profile)
    {
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(profile.CurrentTitle)) parts.Add(profile.CurrentTitle);
        if (profile.YearsExperience is > 0) parts.Add($"{profile.YearsExperience:0.#} years of experience");
        if (profile.Skills.Count > 0) parts.Add("skilled in " + string.Join(", ", profile.Skills.Take(8)));

        return parts.Count > 0 ? string.Join("; ", parts) + "." : "Profile extracted from the uploaded resume.";
    }

    // -- pass 2 -------------------------------------------------------------

    private const string RefinementSystemPrompt = """
        You normalise resumes into JSON. You are given the raw text of one resume and
        the fields a deterministic parser already read from it.

        Rules:
        - Use ONLY information present in the raw text. Never invent an employer,
          a skill, a qualification or a number of years.
        - If the resume does not state something, return null or an empty array.
        - skills: concrete technologies, tools and capabilities, as written.
        - titles: job titles the person has held, most recent first.
        - years_experience: total professional years, as a number, or null.
        - summary: at most 2 sentences, third person, factual.
        - Do NOT return contact details. They are read from the document directly.

        Reply with JSON only, matching exactly this shape:
        {"current_title":string|null,"location":string|null,"years_experience":number|null,
         "skills":[string],"titles":[string],"education":[string],"summary":string|null}
        """;

    private async Task<bool> RefineWithLlmAsync(ResumeProfile profile, string rawText, CancellationToken ct)
    {
        var prompt = $"""
            RAW RESUME
            ----------
            {TextStructure.Clip(rawText, 6000)}

            WHAT THE PARSER ALREADY READ
            ----------------------------
            current_title: {Show(profile.CurrentTitle)}
            location: {Show(profile.Location)}
            years_experience: {(profile.YearsExperience?.ToString("0.#") ?? "(not found)")}
            skills: {Show(string.Join(" | ", profile.Skills))}
            titles: {Show(string.Join(" | ", profile.Titles))}
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

            if (string.IsNullOrWhiteSpace(profile.CurrentTitle))
                profile.CurrentTitle = ReadString(root, "current_title") ?? "";
            if (string.IsNullOrWhiteSpace(profile.Location))
                profile.Location = ReadString(root, "location") ?? "";

            // A model-stated year count is accepted only when the document itself
            // yielded none, and only within a sane range. Experience drives which
            // roles a candidate is shown; inflating it wastes their time.
            if (profile.YearsExperience is null &&
                root.TryGetProperty("years_experience", out var years) &&
                years.ValueKind == JsonValueKind.Number &&
                years.TryGetDouble(out var value) && value is > 0 and <= 50)
            {
                profile.YearsExperience = value;
            }

            var skills = Ground(ReadStrings(root, "skills"), rawText);
            // Resume skills sections are often a plain comma list, which the
            // deterministic pass reads perfectly well. The model earns an override
            // only when it finds materially more than that pass did, which is the
            // case it exists for: skills mentioned in prose, never listed.
            if (skills.Count > profile.Skills.Count) profile.Skills = skills;

            var titles = Ground(ReadStrings(root, "titles"), rawText);
            if (titles.Count > 0) profile.Titles = titles;

            var education = ReadStrings(root, "education");
            if (education.Count > 0 && profile.Education.Count == 0)
                profile.Education = education.Select(TextStructure.Collapse).Take(8).ToList();

            var summary = ReadString(root, "summary");
            if (!string.IsNullOrWhiteSpace(summary)) profile.Summary = TextStructure.Clip(summary, 500);

            return true;
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Resume refinement returned unparseable JSON; keeping the deterministic profile.");
            return false;
        }
    }

    /// <summary>
    /// Drops anything the model returned that is not in the resume.
    ///
    /// A skill invented here would be shown to the candidate as one of their own
    /// and counted as evidence of fit for a job they cannot do.
    /// </summary>
    private static List<string> Ground(List<string> candidates, string rawText)
    {
        var haystack = rawText.ToLowerInvariant();
        return candidates
            .Select(TextStructure.Collapse)
            .Where(s => s.Length is >= 2 and <= 40)
            .Where(s => haystack.Contains(s.ToLowerInvariant()))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(60)
            .ToList();
    }

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

    private static string ExtractJsonObject(string reply)
    {
        var start = reply.IndexOf('{');
        var end = reply.LastIndexOf('}');
        return start >= 0 && end > start ? reply[start..(end + 1)] : reply;
    }

    // -- embedding text -----------------------------------------------------

    /// <summary>
    /// What gets embedded for a candidate.
    ///
    /// Mirrors <c>JobExtractionService.BuildEmbeddingText</c> on purpose: the two
    /// vectors are compared directly, so they should be built from the same kind
    /// of content. Feeding a whole resume in on one side and a distilled role
    /// summary on the other measures document length as much as fit.
    /// </summary>
    public string BuildEmbeddingText(PortalResume resume)
    {
        var parts = new List<string>
        {
            resume.CurrentTitle,
            string.Join(", ", TextStructure.ReadLines(resume.Titles)),
            resume.YearsExperience is > 0 ? $"{resume.YearsExperience:0.#} years of experience" : "",
            resume.Summary,
            string.Join(", ", TextStructure.ReadLines(resume.Skills)),
            string.Join(". ", TextStructure.ReadLines(resume.Education).Take(4)),
        };

        var text = string.Join("\n", parts.Where(p => !string.IsNullOrWhiteSpace(p)));
        if (text.Length < 120) text += "\n" + TextStructure.Clip(resume.RawText, 2000);

        return TextStructure.Clip(text, 4000);
    }
}
