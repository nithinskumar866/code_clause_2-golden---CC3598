using System.Text;
using System.Text.RegularExpressions;
using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Llm;

namespace JobPortal.Api.Services.Applications;

/// <summary>A letter and which path produced it.</summary>
public record CoverLetter(string Text, string Mode)
{
    public const string Llm = "llm";
    public const string Deterministic = "deterministic";
}

public interface ICoverLetterService
{
    Task<CoverLetter> WriteAsync(
        PortalResume resume, PortalJob job, JobMatchDto match, CancellationToken ct = default);

    /// <summary>
    /// The composed letter, with no model call — instant.
    ///
    /// Exposed so a review panel can open immediately on real letters and upgrade
    /// them on request, rather than making the candidate wait tens of seconds per
    /// role before they can see anything at all.
    /// </summary>
    CoverLetter Compose(PortalResume resume, PortalJob job, JobMatchDto match);
}

/// <summary>
/// Writes the letter that goes out with an application.
///
/// This is the most dangerous text the portal produces. Everything else the
/// model writes is a description of a result the candidate can see for
/// themselves; this goes to an employer, under the candidate's name, and a
/// sentence claiming a skill they do not have is a lie they did not tell. So the
/// rule the rest of the system already holds — <b>the model never supplies a
/// fact</b> — is enforced here twice over:
///
///  1. <b>By construction.</b> The model is given only the candidate's real
///     evidenced skills, their real title and years, and the posting's real
///     requirements. It is never given the resume text to embroider, and never
///     asked for contact details — those are appended from the parsed profile,
///     because an email is either literally in the document or it is wrong.
///  2. <b>By verification.</b> Any letter mentioning a skill the candidate was
///     scored as <i>Missing</i> is discarded and the deterministic letter is used
///     instead. That check is deliberately strict: it rejects even an honest
///     "keen to learn Kubernetes", because distinguishing that from "experienced
///     in Kubernetes" is prose comprehension, and being wrong costs the candidate
///     an interview they cannot get back. It applies only to genuinely
///     skill-shaped requirements — see <see cref="IsSkillShaped"/>.
///  3. <b>By exclusion.</b> Contact details are never taken from the model; an
///     invented address routes an employer's reply into a void.
///
/// What is deliberately NOT done here is check the letter against a list of known
/// technologies. That was tried and removed: it rejected letters for naming the
/// posting's own requirements, fired on every single letter, and reintroduced the
/// maintained keyword vocabulary this codebase exists to avoid.
///
/// With no model configured the deterministic letter is what ships, and it is a
/// real letter rather than a placeholder — the portal's standing rule is that
/// every feature works without an LLM.
/// </summary>
public class CoverLetterService : ICoverLetterService
{
    private const int MaxLetterChars = 1600;

    private readonly IOllamaClient _llm;
    private readonly ILogger<CoverLetterService> _logger;

    public CoverLetterService(IOllamaClient llm, ILogger<CoverLetterService> logger)
    {
        _llm = llm;
        _logger = logger;
    }

    public async Task<CoverLetter> WriteAsync(
        PortalResume resume, PortalJob job, JobMatchDto match, CancellationToken ct = default)
    {
        var evidenced = Evidenced(match);
        var missing = match.Skills
            .Where(s => string.Equals(s.Status, "Missing", StringComparison.OrdinalIgnoreCase))
            .Select(s => s.Skill)
            .Where(IsSkillShaped)
            .ToList();

        var deterministic = Compose(resume, job, evidenced, match);

        if (!_llm.ChatAvailable) return new CoverLetter(deterministic, CoverLetter.Deterministic);

        try
        {
            var written = await _llm.ChatAsync(Prompt(resume, job, evidenced, match), false, ct: ct);
            if (string.IsNullOrWhiteSpace(written))
                return new CoverLetter(deterministic, CoverLetter.Deterministic);

            var body = TextStructure.Clip(written.Trim(), MaxLetterChars);

            var overclaimed = missing.FirstOrDefault(skill => Mentions(body, skill));
            if (overclaimed is not null)
            {
                _logger.LogWarning(
                    "Cover letter for job {JobId} mentioned {Skill}, which this candidate does not " +
                    "evidence; using the deterministic letter instead.", job.Id, overclaimed);
                return new CoverLetter(deterministic, CoverLetter.Deterministic);
            }

            if (ContainsContactDetail(body))
            {
                _logger.LogWarning(
                    "Cover letter for job {JobId} contained a contact detail; using the deterministic " +
                    "letter instead. Contact details are appended from the parsed resume, never written " +
                    "by the model.", job.Id);
                return new CoverLetter(deterministic, CoverLetter.Deterministic);
            }

            return new CoverLetter(WithSignature(body, resume), CoverLetter.Llm);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Cover letter generation failed for job {JobId}; composing it instead.", job.Id);
            return new CoverLetter(deterministic, CoverLetter.Deterministic);
        }
    }

    public CoverLetter Compose(PortalResume resume, PortalJob job, JobMatchDto match) =>
        new(Compose(resume, job, Evidenced(match), match), CoverLetter.Deterministic);

    /// <summary>
    /// Whether a "required skill" is actually a skill, rather than a whole
    /// responsibility sentence the extractor put into the skills array.
    ///
    /// Postings routinely yield entries like "Thorough knowledge of employment laws
    /// and HR best practices" or "Optimize database queries". No résumé can ever
    /// evidence a sentence, so they are permanently scored <i>Missing</i> — and then
    /// the letter, which naturally paraphrases what the role asks for, trips the
    /// over-claim guard and is thrown away. Measured on this board, that discarded
    /// every model-written letter and made the feature look hard-coded.
    ///
    /// Judged on SHAPE, never against a vocabulary list: a skill is short and a few
    /// words at most. A maintained list of real skill names is exactly what this
    /// codebase forbids, and it would fail on the first technology nobody thought
    /// to add.
    /// </summary>
    private static bool IsSkillShaped(string value)
    {
        var trimmed = TextStructure.Collapse(value ?? "");
        if (trimmed.Length is < 2 or > 40) return false;
        return trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length <= 4;
    }

    /// <summary>
    /// The skills this candidate can actually stand behind for this posting,
    /// named as the candidate's own resume names them.
    ///
    /// A "Have" verdict says the posting's phrasing was matched by something on
    /// the resume; the letter uses the resume's word, because that is the one the
    /// candidate can defend in an interview.
    /// </summary>
    private static List<string> Evidenced(JobMatchDto match) => match.Skills
        .Where(s => !string.Equals(s.Status, "Missing", StringComparison.OrdinalIgnoreCase))
        .Select(s => string.IsNullOrWhiteSpace(s.EvidenceSkill) ? s.Skill : s.EvidenceSkill!)
        .Where(s => !string.IsNullOrWhiteSpace(s))
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .Take(8)
        .ToList();

    private IReadOnlyList<ChatTurn> Prompt(
        PortalResume resume, PortalJob job, IReadOnlyList<string> evidenced, JobMatchDto match)
    {
        var system =
            "You write short, plain job application letters. Rules you must not break:\n" +
            "- Use ONLY the facts given below. Invent nothing.\n" +
            "- Mention ONLY the skills in EVIDENCED SKILLS. Never mention any other technology, " +
            "tool or skill, not even to say the candidate is willing to learn it.\n" +
            "- Do not state an employer, university, date, salary, email or phone number.\n" +
            "- Three short paragraphs, under 200 words. No subject line, no signature, no placeholders " +
            "like [Your Name]. Plain prose, no markdown.";

        var facts = new StringBuilder();
        facts.AppendLine($"ROLE: {job.Title}");
        if (!string.IsNullOrWhiteSpace(job.Company)) facts.AppendLine($"COMPANY: {job.Company}");
        if (!string.IsNullOrWhiteSpace(resume.CurrentTitle)) facts.AppendLine($"CANDIDATE'S CURRENT TITLE: {resume.CurrentTitle}");
        if (resume.YearsExperience is > 0) facts.AppendLine($"YEARS OF EXPERIENCE: {resume.YearsExperience:0.#}");
        facts.AppendLine($"EVIDENCED SKILLS: {string.Join(", ", evidenced)}");
        // The role's OTHER requirements are deliberately withheld.
        //
        // Listing them and then forbidding the model from mentioning them is an
        // instruction small models do not reliably follow — measured here, it named
        // TypeScript straight out of the list it had just been told to avoid, and
        // the letter was discarded. It does not need them: the letter is about what
        // this candidate brings, and the requirements they actually meet are already
        // in EVIDENCED SKILLS. Removing the temptation is more reliable than
        // repeating the prohibition.
        if (match.Strengths.Count > 0)
            facts.AppendLine($"ASSESSED STRENGTHS: {string.Join("; ", match.Strengths.Take(3))}");

        return new[]
        {
            new ChatTurn("system", system),
            new ChatTurn("user", facts.ToString()),
        };
    }

    /// <summary>
    /// The letter when there is no model — and the letter that wins whenever the
    /// model oversteps. Composed from the same facts, so it says less but never
    /// says anything untrue.
    /// </summary>
    private static string Compose(
        PortalResume resume, PortalJob job, IReadOnlyList<string> evidenced, JobMatchDto match)
    {
        var letter = new StringBuilder();

        letter.AppendLine(string.IsNullOrWhiteSpace(job.Company)
            ? "Dear Hiring Team,"
            : $"Dear Hiring Team at {job.Company},");
        letter.AppendLine();

        var opening = new StringBuilder($"I would like to apply for the {job.Title} role");
        if (!string.IsNullOrWhiteSpace(job.Company)) opening.Append($" at {job.Company}");
        opening.Append('.');

        if (!string.IsNullOrWhiteSpace(resume.CurrentTitle))
        {
            opening.Append($" I am currently working as a {resume.CurrentTitle}");
            opening.Append(resume.YearsExperience is > 0
                ? $", with around {resume.YearsExperience:0.#} years of experience."
                : ".");
        }
        letter.AppendLine(opening.ToString());
        letter.AppendLine();

        if (evidenced.Count > 0)
        {
            letter.AppendLine(
                $"My background covers {Join(evidenced)}, which is where this role's requirements " +
                "and my experience overlap most directly.");
        }
        else
        {
            // Saying nothing is better than manufacturing an overlap that the
            // scoring did not find.
            letter.AppendLine(
                "I am applying on the strength of my broader experience rather than a direct " +
                "overlap with the listed requirements.");
        }

        if (match.Strengths.Count > 0) letter.AppendLine(match.Strengths[0]);
        letter.AppendLine();

        letter.AppendLine(
            "I would welcome the chance to talk through how that experience applies here. " +
            "My full CV is attached.");

        return WithSignature(letter.ToString().TrimEnd(), resume);
    }

    /// <summary>
    /// Signs off with contact details taken from the parsed document only.
    ///
    /// Never from the model: an email address is either literally in the resume or
    /// it is wrong, and a plausible-looking wrong one is worse than none.
    /// </summary>
    private static string WithSignature(string body, PortalResume resume)
    {
        var contact = new[] { resume.Email, resume.Phone }
            .Where(part => !string.IsNullOrWhiteSpace(part))
            .ToList();

        var signature = new StringBuilder();
        signature.AppendLine();
        signature.AppendLine();
        signature.AppendLine("Kind regards,");
        signature.Append(string.IsNullOrWhiteSpace(resume.CandidateName) ? "" : resume.CandidateName);
        if (contact.Count > 0) signature.Append($"{(string.IsNullOrWhiteSpace(resume.CandidateName) ? "" : "\n")}{string.Join(" · ", contact)}");

        return (body.TrimEnd() + signature).Trim();
    }

    /// <summary>
    /// Whether the model put a contact detail in the letter.
    ///
    /// It is told not to, and the real details are appended afterwards from the
    /// parsed document. A letter carrying an address the model produced is a letter
    /// that may route an employer's reply into a void, so it is discarded rather
    /// than edited.
    ///
    /// The digit threshold is 9 deliberately: it catches phone numbers and leaves
    /// date ranges alone. "2019 - 2023" is eight digits and is ordinary content in
    /// a cover letter; rejecting it would quietly downgrade every letter that
    /// mentions when someone held a job.
    /// </summary>
    private static bool ContainsContactDetail(string text)
    {
        if (EmailPattern.IsMatch(text)) return true;

        return DigitRunPattern.Matches(text)
            .Any(match => match.Value.Count(char.IsDigit) >= 9);
    }

    private static readonly System.Text.RegularExpressions.Regex EmailPattern =
        new(@"[\w.+-]+@[\w-]+\.[\w.-]+", System.Text.RegularExpressions.RegexOptions.Compiled);

    private static readonly System.Text.RegularExpressions.Regex DigitRunPattern =
        new(@"\+?\d[\d\s().\-]{5,}\d", System.Text.RegularExpressions.RegexOptions.Compiled);

    /// <summary>
    /// Whether the letter refers to a skill at all.
    ///
    /// Bounded on both sides so "Go" does not match "going" and "R" does not match
    /// every word containing it — a false positive here silently downgrades a
    /// perfectly good letter, and single-letter skill names are real.
    /// </summary>
    private static bool Mentions(string text, string skill)
    {
        var needle = TextStructure.Collapse(skill);
        if (needle.Length < 2) return false;

        var haystack = text;
        var index = haystack.IndexOf(needle, StringComparison.OrdinalIgnoreCase);

        while (index >= 0)
        {
            var before = index == 0 || !char.IsLetterOrDigit(haystack[index - 1]);
            var afterAt = index + needle.Length;
            var after = afterAt >= haystack.Length || !char.IsLetterOrDigit(haystack[afterAt]);
            if (before && after) return true;

            index = haystack.IndexOf(needle, index + 1, StringComparison.OrdinalIgnoreCase);
        }

        return false;
    }

    private static string Join(IReadOnlyList<string> values) => values.Count switch
    {
        0 => "",
        1 => values[0],
        2 => $"{values[0]} and {values[1]}",
        _ => $"{string.Join(", ", values.Take(values.Count - 1))} and {values[^1]}",
    };
}
