using JobPortal.Api.Contracts;
using JobPortal.Api.Data;

namespace JobPortal.Api.Services.Matching;

/// <summary>
/// Whether a posting satisfies the conversation's stated constraints.
///
/// Lifted out of <see cref="MatchingService"/> so that browsing without a resume
/// obeys exactly the same rules as matching with one. Two copies of this would
/// drift, and the day they did, "remote only" would mean one thing when the
/// candidate had uploaded a CV and another when they had not.
/// </summary>
public static class JobFilterRules
{
    /// <summary>
    /// Deterministic, and deliberately generous about silence: a job that does not
    /// state its work mode is NOT excluded by a "remote" filter. Excluding it
    /// would treat an unstated fact as a stated contradiction, and most postings
    /// leave most of these fields unstated.
    /// </summary>
    public static bool Passes(PortalJob job, JobFilters filters)
    {
        if (filters.WorkMode is { Length: > 0 } mode &&
            !job.WorkMode.Equals("Unspecified", StringComparison.OrdinalIgnoreCase) &&
            !job.WorkMode.Equals(mode, StringComparison.OrdinalIgnoreCase))
        {
            // "Remote" asked for, "Hybrid" offered: a hybrid role is a partial
            // answer to a remote request, so it stays. Onsite is not.
            var remoteAskedHybridOffered =
                mode.Equals("Remote", StringComparison.OrdinalIgnoreCase) &&
                job.WorkMode.Equals("Hybrid", StringComparison.OrdinalIgnoreCase);

            if (!remoteAskedHybridOffered) return false;
        }

        if (filters.EmploymentType is { Length: > 0 } employment &&
            !job.EmploymentType.Equals("Unspecified", StringComparison.OrdinalIgnoreCase) &&
            !job.EmploymentType.Equals(employment, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (filters.SeniorityLevel is { Length: > 0 } seniority &&
            !job.SeniorityLevel.Equals("Unspecified", StringComparison.OrdinalIgnoreCase) &&
            !job.SeniorityLevel.Equals(seniority, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (filters.Location is { Length: > 0 } location &&
            job.Location.Length > 0 &&
            !job.Location.Contains(location, StringComparison.OrdinalIgnoreCase) &&
            !location.Contains(job.Location, StringComparison.OrdinalIgnoreCase))
        {
            // A remote job satisfies any location request by definition.
            if (!job.WorkMode.Equals("Remote", StringComparison.OrdinalIgnoreCase)) return false;
        }

        // Compared against the TOP of the range: a candidate asking for at least
        // 100k is answered by a 90k-120k posting, and comparing against the floor
        // would hide it.
        if (filters.MinSalary is { } minSalary && job.SalaryMax is { } jobMax && jobMax < minSalary)
        {
            return false;
        }

        if (filters.MaxYearsRequired is { } maxYears &&
            job.MinYearsExperience is { } jobMin && jobMin > maxYears)
        {
            return false;
        }

        if (filters.Keywords is { Count: > 0 } keywords)
        {
            // Every keyword must appear somewhere in the posting. These come from
            // the candidate naming something explicitly ("must involve Kubernetes"),
            // which is a constraint rather than a preference.
            var haystack = $"{job.Title} {job.Company} {job.RequiredSkills} {job.PreferredSkills} " +
                           $"{job.Summary} {job.Responsibilities}".ToLowerInvariant();

            if (!keywords.All(k => haystack.Contains(k.ToLowerInvariant()))) return false;
        }

        return true;
    }
}
