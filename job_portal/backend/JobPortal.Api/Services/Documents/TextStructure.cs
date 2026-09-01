using System.Text;
using System.Text.RegularExpressions;

namespace JobPortal.Api.Services.Documents;

/// <summary>
/// Generic document-shape utilities: split prose into its headed sections, pull
/// out bullet lines, read labelled fields.
///
/// Everything here keys on DOCUMENT STRUCTURE (headings, bullets, "Label:" lines),
/// never on domain vocabulary. There is no list of known skills, companies or
/// technologies anywhere in this file, and there must not be: a portal that only
/// recognises the skills someone remembered to type into a constant is exactly
/// the keyword ATS this system exists to replace. Section names ARE listed, but a
/// heading is part of a document's grammar rather than its subject matter.
/// </summary>
public static class TextStructure
{
    private static readonly Regex BulletPrefix = new(
        @"^\s*([•●▪‣⁃·\*\-–—\+o]|\d{1,2}[\.\)])\s+",
        RegexOptions.Compiled);

    private static readonly Regex LabelledField = new(
        @"^\s*(?<label>[A-Za-z][A-Za-z /&\-]{2,40})\s*[:\-–]\s*(?<value>.+)$",
        RegexOptions.Compiled);

    private static readonly Regex WhitespaceRun = new(@"\s+", RegexOptions.Compiled);

    /// <summary>
    /// Markdown decoration that is formatting, not content: heading hashes and bold
    /// or underline emphasis.
    ///
    /// Deliberately NOT bullets. A leading '-' or '*' marks a list item, which is a
    /// requirement rather than a header field, and <see cref="BulletPrefix"/> still
    /// skips those.
    /// </summary>
    private static readonly Regex MarkdownNoise = new(@"^\s*#{1,6}\s*|\*\*|__", RegexOptions.Compiled);

    /// <summary>
    /// Strips markdown decoration so a heading can still be read as a labelled field.
    ///
    /// JDs are routinely written as markdown, where the title line is
    /// "# Job Title: HR Generalist". The leading hash stopped that matching the
    /// label pattern, so `FirstField("Job Title")` found nothing and the
    /// `FirstMeaningfulLine` fallback returned the ENTIRE raw line — hash, label and
    /// all — as the job title. Every posting on the board then displayed as
    /// "# Job Title: HR Generalist", and the same failure left Company and Location
    /// empty, which is why every row showed "—" and fell back to the work mode.
    /// </summary>
    public static string Demarkup(string line) => MarkdownNoise.Replace(line ?? "", "").Trim();

    /// <summary>Matches a stated duration — "5+ years", "3-5 years of experience".</summary>
    private static readonly Regex DurationPhrase = new(
        @"\b\d+\s*\+?\s*(-\s*\d+\s*)?(year|yr)s?\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Whether a string is shaped like the NAME of a skill, rather than a sentence
    /// about one.
    ///
    /// Postings routinely yield "Thorough knowledge of employment laws and HR best
    /// practices" or "5+ years of DevOps experience" where a skill list is expected.
    /// Those cost real accuracy: no résumé can evidence a sentence, so every
    /// candidate is scored as missing it, and a candidate who names the actual
    /// technology gets no credit for it.
    ///
    /// Judged on SHAPE — length, word count, and whether it states a duration —
    /// never against a list of known technologies. A maintained vocabulary is what
    /// this codebase exists to avoid, and it would fail on the first tool nobody
    /// remembered to add.
    /// </summary>
    public static bool LooksLikeSkillName(string? value)
    {
        var trimmed = Collapse(value ?? "");
        if (trimmed.Length is < 2 or > 40) return false;
        if (trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length > 4) return false;

        // "5+ years of Java" names a duration, not a capability. The years are read
        // separately and scored on their own dimension.
        if (DurationPhrase.IsMatch(trimmed)) return false;

        return trimmed.Any(char.IsLetter);
    }

    /// <summary>
    /// Splits text into (heading, body) sections. Text before the first heading is
    /// returned under the empty key, because a resume's contact block and a JD's
    /// opening paragraph both live there and both matter.
    /// </summary>
    public static Dictionary<string, string> SplitSections(string text, IReadOnlyCollection<string> headings)
    {
        var sections = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var lines = text.Split('\n');

        var current = "";
        var buffer = new StringBuilder();

        foreach (var line in lines)
        {
            var heading = MatchHeading(line, headings);
            if (heading is not null)
            {
                Flush(sections, current, buffer);
                current = heading;

                // "Skills: C#, SQL" is a heading and its content on one line.
                var inline = line.Split(':', 2);
                if (inline.Length == 2 && inline[1].Trim().Length > 0) buffer.AppendLine(inline[1].Trim());
                continue;
            }
            buffer.AppendLine(line);
        }
        Flush(sections, current, buffer);

        return sections;
    }

    private static void Flush(Dictionary<string, string> sections, string key, StringBuilder buffer)
    {
        var body = buffer.ToString().Trim();
        buffer.Clear();
        if (body.Length == 0) return;

        // A heading repeated later in the document continues the same section
        // rather than replacing it; overwriting would drop the earlier content.
        sections[key] = sections.TryGetValue(key, out var existing)
            ? existing + "\n" + body
            : body;
    }

    /// <summary>
    /// Decides whether a line is a heading. A heading is short, is not a sentence,
    /// and names one of the given sections. All three tests are needed: "Skills"
    /// on its own line is a heading, whereas a bullet reading "Skills gained while
    /// leading the platform team" is not.
    /// </summary>
    private static string? MatchHeading(string line, IReadOnlyCollection<string> headings)
    {
        var trimmed = line.Trim();
        if (trimmed.Length is 0 or > 60) return null;
        if (BulletPrefix.IsMatch(line)) return null;

        var probe = trimmed.TrimEnd(':', '-', '–').Trim();
        var beforeColon = probe.Split(':', 2)[0].Trim();

        foreach (var heading in headings)
        {
            if (probe.Equals(heading, StringComparison.OrdinalIgnoreCase) ||
                beforeColon.Equals(heading, StringComparison.OrdinalIgnoreCase))
            {
                return heading;
            }
        }
        return null;
    }

    /// <summary>Bullet lines in a block, cleaned of their markers. Falls back to
    /// non-empty lines when the block uses no bullets at all, which many JDs
    /// written in prose do.</summary>
    public static List<string> BulletLines(string block, int maxLength = 300)
    {
        if (string.IsNullOrWhiteSpace(block)) return new List<string>();

        var lines = block.Split('\n');
        var bullets = new List<string>();
        var plain = new List<string>();

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (trimmed.Length < 2) continue;

            if (BulletPrefix.IsMatch(line))
            {
                var cleaned = BulletPrefix.Replace(line, "").Trim().TrimEnd(';', '.', ',');
                if (cleaned.Length is >= 2 && cleaned.Length <= maxLength) bullets.Add(cleaned);
            }
            else if (trimmed.Length <= maxLength)
            {
                plain.Add(trimmed.TrimEnd(';', ','));
            }
        }

        return bullets.Count > 0 ? bullets : plain;
    }

    /// <summary>Reads "Label: value" lines into a lookup.</summary>
    public static Dictionary<string, string> LabelledFields(string text, int scanLines = 60)
    {
        var fields = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (var raw in text.Split('\n').Take(scanLines))
        {
            if (BulletPrefix.IsMatch(raw)) continue;

            var line = Demarkup(raw);
            var match = LabelledField.Match(line);
            if (!match.Success) continue;

            var label = Collapse(match.Groups["label"].Value);
            var value = Collapse(match.Groups["value"].Value);
            if (value.Length is 0 or > 200) continue;

            // First occurrence wins: document headers state the real values, and a
            // later mention is usually prose that happens to fit the pattern.
            if (!fields.ContainsKey(label)) fields[label] = value;
        }

        return fields;
    }

    public static string? FirstField(IReadOnlyDictionary<string, string> fields, params string[] labels)
    {
        foreach (var label in labels)
        {
            if (fields.TryGetValue(label, out var value) && value.Length > 0) return value;
        }
        return null;
    }

    /// <summary>Splits a comma/semicolon/pipe/bullet-separated inline list.</summary>
    public static List<string> SplitInlineList(string value, int maxItems = 60)
    {
        return value
            .Split(new[] { ',', ';', '|', '•', '/', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(part => Collapse(part).Trim('.', '-', ' '))
            .Where(part => part.Length is >= 2 and <= 60)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(maxItems)
            .ToList();
    }

    public static string Collapse(string value) => WhitespaceRun.Replace(value ?? "", " ").Trim();

    /// <summary>The first line that could plausibly be a document title.</summary>
    public static string? FirstMeaningfulLine(string text, int maxLength = 120)
    {
        foreach (var line in text.Split('\n').Take(15))
        {
            // Demarkup here too: when this fallback DOES run on a markdown document,
            // returning "# Senior Engineer" as the title is the same bug one step later.
            var trimmed = Collapse(Demarkup(line));
            if (trimmed.Length is < 3 or > 120) continue;
            if (BulletPrefix.IsMatch(line)) continue;
            // A contact line is not a title.
            if (trimmed.Contains('@') || trimmed.Count(char.IsDigit) > 6) continue;
            return trimmed.Length <= maxLength ? trimmed : trimmed[..maxLength];
        }
        return null;
    }

    public static string JoinLines(IEnumerable<string> values) =>
        string.Join("\n", values.Select(Collapse).Where(v => v.Length > 0).Distinct(StringComparer.OrdinalIgnoreCase));

    public static List<string> ReadLines(string? value) =>
        string.IsNullOrWhiteSpace(value)
            ? new List<string>()
            : value.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                   .Select(v => v.Trim())
                   .Where(v => v.Length > 0)
                   .ToList();

    /// <summary>Truncates on a word boundary, for prompt budgets.</summary>
    public static string Clip(string text, int maxChars)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= maxChars) return text ?? "";

        var cut = text[..maxChars];
        var lastBreak = cut.LastIndexOfAny(new[] { '\n', '.', ' ' });
        return (lastBreak > maxChars / 2 ? cut[..lastBreak] : cut).TrimEnd() + "\n[...]";
    }
}
