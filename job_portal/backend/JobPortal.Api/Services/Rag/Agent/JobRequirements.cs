using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;

namespace JobPortal.Api.Services.Rag.Agent;

/// <summary>One thing a posting asks for, before anyone has judged it.</summary>
public record JobRequirement(string Text, string Kind);

/// <summary>
/// Breaks a posting into requirements — WITHOUT a model.
///
/// The extraction already happened: `JobExtractionService` parsed this posting
/// into required skills, preferred skills, responsibilities and qualifications
/// when it was uploaded. Asking a model to re-derive that list per evaluation
/// would be paying ten seconds to rediscover something already in the database,
/// and doing it less reliably — the atomisation failures earlier in this project
/// were all cases of a model merging or dropping requirements it had been asked
/// to enumerate.
///
/// So the kinds come from WHICH FIELD a line was extracted into, which is a fact
/// about the posting rather than an opinion about it:
///
///   RequiredSkills   -> core_skill    (and the posting's own required list is
///                                      separately cross-checked for dealbreakers)
///   PreferredSkills  -> nice_to_have
///   Responsibilities -> responsibility
///   Qualifications   -> education, or experience when it states a duration
///   MinYearsExperience -> experience
/// </summary>
public static class JobRequirements
{
    /// <summary>
    /// The most requirements one evaluation will consider.
    ///
    /// A posting with forty bullets would otherwise produce forty retrievals and a
    /// prompt nobody can read. The cap keeps the highest-signal fields — skills
    /// and experience come before responsibilities, which are the job's duties
    /// rather than the applicant's qualifications.
    /// </summary>
    public const int MaxRequirements = 14;

    public static IReadOnlyList<JobRequirement> For(PortalJob job)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var requirements = new List<JobRequirement>();

        void Add(string text, string kind)
        {
            var clean = TextStructure.Collapse(text ?? "");
            if (clean.Length is < 3 or > 300) return;
            if (!seen.Add(clean)) return;
            requirements.Add(new JobRequirement(clean, kind));
        }

        // Years first: it is the one requirement the backend can settle by
        // arithmetic, and it must never be crowded out by the cap.
        if (job.MinYearsExperience is { } years and > 0)
        {
            Add($"{years:0.#}+ years of relevant professional experience", RequirementKinds.Experience);
        }

        foreach (var skill in TextStructure.ReadLines(job.RequiredSkills))
        {
            Add(skill, RequirementKinds.CoreSkill);
        }

        foreach (var qualification in TextStructure.ReadLines(job.Qualifications).Take(6))
        {
            Add(qualification, MentionsDuration(qualification)
                ? RequirementKinds.Experience
                : RequirementKinds.Education);
        }

        foreach (var responsibility in TextStructure.ReadLines(job.Responsibilities).Take(8))
        {
            Add(responsibility, RequirementKinds.Responsibility);
        }

        foreach (var preferred in TextStructure.ReadLines(job.PreferredSkills).Take(4))
        {
            Add(preferred, RequirementKinds.NiceToHave);
        }

        // A posting with nothing structured still has to be evaluable.
        if (requirements.Count == 0 && !string.IsNullOrWhiteSpace(job.Title))
        {
            Add($"Experience working as a {job.Title}", RequirementKinds.Experience);
        }

        return requirements.Take(MaxRequirements).ToList();
    }

    private static bool MentionsDuration(string text) =>
        text.Contains("year", StringComparison.OrdinalIgnoreCase) ||
        text.Contains("experience", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// The weighting handed to the scorer.
    ///
    /// Fixed rather than model-chosen, because nothing here asks a model for an
    /// opinion about the posting. The scorer renormalises across the kinds that
    /// are actually present and applies its own ceilings, so these are relative
    /// weights rather than percentages that must sum to anything.
    /// </summary>
    public static IReadOnlyDictionary<string, int> Weights() => new Dictionary<string, int>
    {
        [RequirementKinds.MustHave] = 35,
        [RequirementKinds.CoreSkill] = 30,
        [RequirementKinds.Experience] = 15,
        [RequirementKinds.Responsibility] = 10,
        [RequirementKinds.Education] = 5,
        [RequirementKinds.NiceToHave] = 5,
    };
}
