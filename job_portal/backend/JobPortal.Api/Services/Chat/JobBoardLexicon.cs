using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Services.Chat;

/// <summary>
/// What the board itself says its words mean: the locations, companies and skill
/// names that actually occur in the postings.
///
/// This exists so message understanding never needs a maintained keyword list.
/// Whether "Pune" is a place or "Go" is a technology is not answerable in the
/// abstract, only against this corpus — and a portal that only recognises the
/// cities and tools someone thought to type into a constant will silently fail on
/// every posting outside that list.
/// </summary>
public class JobBoardLexicon
{
    public IReadOnlySet<string> Locations { get; private init; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    public IReadOnlySet<string> Companies { get; private init; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    public IReadOnlySet<string> Skills { get; private init; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    public static async Task<JobBoardLexicon> BuildAsync(PortalDbContext db, CancellationToken ct = default)
    {
        var rows = await db.Jobs
            .AsNoTracking()
            .Where(j => j.IsPublished)
            .Select(j => new { j.Location, j.Company, j.RequiredSkills, j.PreferredSkills })
            .ToListAsync(ct);

        var locations = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var companies = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var skills = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var row in rows)
        {
            // A location field is often "Bengaluru, India" or "London / Remote";
            // the parts are what a candidate actually types.
            foreach (var part in SplitLocation(row.Location))
            {
                if (part.Length is >= 3 and <= 40) locations.Add(part);
            }

            var company = TextStructure.Collapse(row.Company);
            if (company.Length is >= 2 and <= 60) companies.Add(company);

            foreach (var skill in TextStructure.ReadLines(row.RequiredSkills)
                                               .Concat(TextStructure.ReadLines(row.PreferredSkills)))
            {
                // Requirements are frequently whole sentences ("Experience building
                // distributed systems at scale"). Those are not terms a candidate
                // will name in a filter, and admitting them would make almost any
                // message look like it contained a keyword constraint.
                var trimmed = TextStructure.Collapse(skill);
                if (trimmed.Length is >= 2 and <= 30 && trimmed.Count(char.IsWhiteSpace) <= 2)
                {
                    skills.Add(trimmed);
                }
            }
        }

        return new JobBoardLexicon
        {
            Locations = locations,
            Companies = companies,
            Skills = skills,
        };
    }

    private static IEnumerable<string> SplitLocation(string location)
    {
        if (string.IsNullOrWhiteSpace(location)) yield break;

        yield return TextStructure.Collapse(location);

        foreach (var part in location.Split(new[] { ',', '/', '|', ';', '-' },
                                            StringSplitOptions.RemoveEmptyEntries))
        {
            var trimmed = TextStructure.Collapse(part);
            if (trimmed.Length > 0) yield return trimmed;
        }
    }
}
