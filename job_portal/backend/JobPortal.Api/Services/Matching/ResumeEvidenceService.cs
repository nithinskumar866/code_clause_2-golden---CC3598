using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;

namespace JobPortal.Api.Services.Matching;

/// <summary>
/// How well a claim is backed up, in the order a recruiter would trust it.
///
/// The order is the whole point: saying you want HR work and having done HR work
/// are not the same claim, and a system that scores them alike is stuffable by
/// anyone who writes the right sentence.
/// </summary>
public enum EvidenceTier
{
    /// <summary>Nowhere in the document.</summary>
    None = 0,

    /// <summary>
    /// Only in a summary, objective or interests line — "I prefer HR roles".
    /// A statement of intent, not of capability.
    /// </summary>
    Aspirational = 1,

    /// <summary>Listed under Skills. Claimed, but nothing shows it in use.</summary>
    Listed = 2,

    /// <summary>Used in a project. Something was actually built with it.</summary>
    Applied = 3,

    /// <summary>Studied or certified. Formally grounded.</summary>
    Credentialed = 4,

    /// <summary>Used in employment or an internship. The strongest evidence there is.</summary>
    Professional = 5,
}

/// <summary>What the document proves about one requirement.</summary>
public record SkillEvidence(
    string Skill,
    EvidenceTier Tier,
    double Weight,
    IReadOnlyList<string> FoundIn,
    string Explanation)
{
    /// <summary>Claimed somewhere, however weakly.</summary>
    public bool IsEvidenced => Tier >= EvidenceTier.Listed;
}

/// <summary>A résumé's sections, and what they prove.</summary>
public class ResumeEvidence
{
    private readonly IReadOnlyDictionary<EvidenceTier, string> _byTier;

    internal ResumeEvidence(IReadOnlyDictionary<EvidenceTier, string> byTier) => _byTier = byTier;

    /// <summary>
    /// Where a requirement is actually backed up.
    ///
    /// Deliberately a whole-word search over the section text rather than a match
    /// against a parsed skill list: a résumé that never writes "Kubernetes" under
    /// Skills but describes migrating a cluster in a job entry has stronger
    /// evidence than one that lists the word and nothing else.
    /// </summary>
    public SkillEvidence For(string skill)
    {
        var needle = TextStructure.Collapse(skill ?? "");
        if (needle.Length < 2)
            return new SkillEvidence(skill ?? "", EvidenceTier.None, 0, Array.Empty<string>(), "not stated");

        var found = new List<EvidenceTier>();
        foreach (var (tier, text) in _byTier)
        {
            if (ContainsWord(text, needle)) found.Add(tier);
        }

        if (found.Count == 0)
            return new SkillEvidence(skill!, EvidenceTier.None, 0, Array.Empty<string>(),
                "no mention anywhere in the CV");

        var best = found.Max();

        // Base is the strongest place it appears; breadth adds a little on top, so
        // "listed AND built AND used at work" outranks "used at work" alone. Capped
        // at 1 so corroboration can never manufacture more than full confidence.
        var weight = Math.Min(1.0, BaseWeight(best) + 0.05 * (found.Count - 1));

        var names = found.OrderByDescending(t => t).Select(Describe).ToList();
        return new SkillEvidence(skill!, best, weight, names, Explain(best, names));
    }

    /// <summary>
    /// The share of a role's requirements this candidate can actually back up.
    ///
    /// This is what stops a stated preference from carrying a match: a résumé that
    /// says "interested in HR" and evidences none of an HR posting's requirements
    /// scores near zero here, however close the two documents look as vectors.
    /// </summary>
    public double CoverageOf(IEnumerable<string> requirements)
    {
        var wanted = requirements
            .Select(r => TextStructure.Collapse(r ?? ""))
            .Where(r => r.Length >= 2)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        // Nothing measurable was asked for. Neutral rather than full marks: a
        // posting whose requirements the extractor could not read must not
        // out-rank one whose requirements were read properly and found wanting.
        // Returning 1.0 here inverted the whole board — every badly-parsed role
        // sailed past every clean one.
        if (wanted.Count == 0) return 0.5;

        return wanted.Average(Score);
    }

    /// <summary>
    /// What one stated requirement is worth, whether it is a skill name or a
    /// whole sentence.
    ///
    /// Sentences are routine in extracted postings — "Thorough knowledge of
    /// employment laws and HR best practices" — and no résumé will ever contain
    /// one verbatim. Discarding them would leave such a posting unmeasurable;
    /// matching them whole would score every candidate zero. So a sentence is
    /// judged by the strongest evidence behind any of its distinctive words,
    /// which is what a person skimming for relevance actually does.
    /// </summary>
    private double Score(string requirement)
    {
        var direct = For(requirement);
        if (direct.Tier != EvidenceTier.None) return direct.Weight;

        var words = requirement
            .Split(new[] { ' ', ',', '/', '(', ')', '-', ':', ';', '.' }, StringSplitOptions.RemoveEmptyEntries)
            .Where(w => w.Length >= 4 && !Filler.Contains(w))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(8)
            .ToList();

        if (words.Count == 0) return direct.Weight;

        // The strongest single hit, not the average: a requirement is met by
        // evidence of the thing it is about, and averaging over its grammar would
        // punish a long sentence for containing ordinary words.
        return words.Max(w => For(w).Weight);
    }

    /// <summary>
    /// Words that carry no evidence either way. Not a domain vocabulary — these
    /// are the connective tissue of a requirement sentence in any field, and the
    /// list would be identical for law, nursing or catering.
    /// </summary>
    private static readonly HashSet<string> Filler = new(StringComparer.OrdinalIgnoreCase)
    {
        "with", "and", "the", "for", "from", "into", "over", "under", "years",
        "year", "experience", "experienced", "strong", "excellent", "good", "solid",
        "knowledge", "understanding", "ability", "able", "skills", "skill",
        "proficiency", "proficient", "familiar", "familiarity", "working", "work",
        "including", "such", "etc", "plus", "must", "have", "having", "role",
        "similar", "related", "relevant", "team", "teams", "using", "used", "use",
        "best", "practices", "practice", "thorough", "demonstrated", "proven",
        "track", "record", "responsible", "responsibility", "responsibilities",
    };

    private static double BaseWeight(EvidenceTier tier) => tier switch
    {
        EvidenceTier.Professional => 1.00,
        EvidenceTier.Credentialed => 0.75,
        EvidenceTier.Applied      => 0.60,
        EvidenceTier.Listed       => 0.30,
        // Almost nothing. Wanting a job is not a qualification for it, and this
        // number is the difference between a job board and a keyword matcher.
        EvidenceTier.Aspirational => 0.08,
        _ => 0.0,
    };

    private static string Describe(EvidenceTier tier) => tier switch
    {
        EvidenceTier.Professional => "work experience",
        EvidenceTier.Credentialed => "education or certification",
        EvidenceTier.Applied      => "projects",
        EvidenceTier.Listed       => "the skills list",
        EvidenceTier.Aspirational => "a summary or interests line",
        _ => "nowhere",
    };

    private static string Explain(EvidenceTier best, IReadOnlyList<string> where) => best switch
    {
        EvidenceTier.Professional => $"used professionally — found in {string.Join(", ", where)}",
        EvidenceTier.Credentialed => $"formally studied — found in {string.Join(", ", where)}",
        EvidenceTier.Applied      => $"applied in projects — found in {string.Join(", ", where)}",
        EvidenceTier.Listed       => "listed as a skill, but nothing shows it in use",
        EvidenceTier.Aspirational => "only mentioned as an interest or goal, with nothing behind it",
        _ => "no mention anywhere in the CV",
    };

    /// <summary>
    /// Whole-word containment, so "Go" does not match "going" and "R" does not
    /// match every word on the page. Single-letter skill names are real, which is
    /// why this is bounded rather than a plain IndexOf.
    /// </summary>
    private static bool ContainsWord(string haystack, string needle)
    {
        var i = haystack.IndexOf(needle, StringComparison.OrdinalIgnoreCase);
        while (i >= 0)
        {
            var before = i == 0 || !char.IsLetterOrDigit(haystack[i - 1]);
            var afterAt = i + needle.Length;
            var after = afterAt >= haystack.Length || !char.IsLetterOrDigit(haystack[afterAt]);
            if (before && after) return true;
            i = haystack.IndexOf(needle, i + 1, StringComparison.OrdinalIgnoreCase);
        }
        return false;
    }
}

public interface IResumeEvidenceService
{
    ResumeEvidence Build(PortalResume resume);
}

/// <summary>
/// Reads a résumé as a body of evidence rather than a bag of words.
///
/// The problem this exists to solve: a single vector over the whole document
/// cannot tell a stated preference from a demonstrated capability. Write "I would
/// like to move into HR" and the document drifts toward HR postings, which then
/// surface with a confident percentage and nothing behind it. Retrieval stays
/// generous — recall is its job — and this decides what the match is actually
/// worth.
///
/// Entirely deterministic and instant: where a word appears is a property of the
/// document, not a judgement. Only genuine world knowledge — whether AWS
/// experience transfers to an Azure role — needs a model, and that happens once
/// per search rather than once per skill.
/// </summary>
public class ResumeEvidenceService : IResumeEvidenceService
{
    // Section headings, grouped by what finding a word there actually proves.
    private static readonly (EvidenceTier Tier, string[] Headings)[] TierHeadings =
    {
        (EvidenceTier.Professional, new[]
        {
            "Experience", "Work Experience", "Professional Experience", "Employment History",
            "Employment", "Career History", "Work History", "Internship", "Internships",
        }),
        (EvidenceTier.Credentialed, new[]
        {
            "Education", "Academic Background", "Qualifications",
            "Certifications", "Certificates", "Licenses",
        }),
        (EvidenceTier.Applied, new[]
        {
            "Projects", "Personal Projects", "Key Projects", "Achievements", "Publications",
        }),
        (EvidenceTier.Listed, new[]
        {
            "Skills", "Technical Skills", "Core Competencies", "Technologies", "Tech Stack",
            "Expertise", "Areas of Expertise",
        }),
        (EvidenceTier.Aspirational, new[]
        {
            "Summary", "Profile", "Professional Summary", "Objective", "About Me", "About",
            "Interests", "Hobbies",
        }),
    };

    private static readonly string[] AllHeadings =
        TierHeadings.SelectMany(t => t.Headings).Distinct().ToArray();

    public ResumeEvidence Build(PortalResume resume)
    {
        var sections = TextStructure.SplitSections(resume.RawText ?? "", AllHeadings);

        var byTier = new Dictionary<EvidenceTier, string>();
        foreach (var (tier, headings) in TierHeadings)
        {
            var text = string.Join("\n", headings
                .Where(sections.ContainsKey)
                .Select(h => sections[h]));

            if (!string.IsNullOrWhiteSpace(text)) byTier[tier] = text;
        }

        // A résumé with no recognisable headings at all — a single prose block, or
        // a layout the splitter could not read — would otherwise evidence nothing
        // and score every role at zero. Treating the whole document as Listed is
        // the honest reading: the words are there, but nothing shows where.
        if (byTier.Count == 0 && !string.IsNullOrWhiteSpace(resume.RawText))
        {
            byTier[EvidenceTier.Listed] = resume.RawText;
        }

        return new ResumeEvidence(byTier);
    }
}
