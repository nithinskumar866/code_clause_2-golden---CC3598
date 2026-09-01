using System.Collections.Concurrent;
using JobPortal.Api.Contracts;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Vectors;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Matching;

/// <summary>
/// Process-wide cache of skill-name vectors.
///
/// Skill names repeat relentlessly across a job board: fifty postings will name
/// the same dozen technologies. Embedding them per request would be the dominant
/// cost of every search. Keyed by model as well as text, because the same word
/// embedded by two models gives two incomparable vectors.
/// </summary>
public class SkillVectorCache
{
    private readonly ConcurrentDictionary<(string Model, string Text), float[]> _cache = new();

    /// <summary>Bounded so a long-running server cannot grow it without limit;
    /// far above the number of distinct skills any real board contains.</summary>
    private const int MaxEntries = 50_000;

    public bool TryGet(string model, string text, out float[] vector) =>
        _cache.TryGetValue((model, Key(text)), out vector!);

    public void Set(string model, string text, float[] vector)
    {
        if (_cache.Count >= MaxEntries) _cache.Clear();
        _cache[(model, Key(text))] = vector;
    }

    private static string Key(string text) => text.Trim().ToLowerInvariant();
}

public interface ISkillSemanticsService
{
    /// <summary>Embeds every distinct skill name in one round trip and returns a
    /// lookup keyed by the original text.</summary>
    Task<IReadOnlyDictionary<string, float[]>> EmbedSkillsAsync(
        IEnumerable<string> skills, CancellationToken ct = default);

    /// <summary>
    /// Judges each required skill against what the candidate has: Have,
    /// Transferable, or Missing.
    /// </summary>
    IReadOnlyList<SkillAssessmentDto> Assess(
        IReadOnlyList<string> required,
        IReadOnlyList<string> candidateSkills,
        IReadOnlyDictionary<string, float[]> vectors);
}

public class SkillSemanticsService : ISkillSemanticsService
{
    private readonly IEmbeddingService _embeddings;
    private readonly SkillVectorCache _cache;
    private readonly MatchingOptions _options;

    public SkillSemanticsService(
        IEmbeddingService embeddings,
        SkillVectorCache cache,
        IOptions<MatchingOptions> options)
    {
        _embeddings = embeddings;
        _cache = cache;
        _options = options.Value;
    }

    public async Task<IReadOnlyDictionary<string, float[]>> EmbedSkillsAsync(
        IEnumerable<string> skills, CancellationToken ct = default)
    {
        var model = _embeddings.PreferredModelId;
        var result = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);

        var missing = new List<string>();
        foreach (var skill in skills.Select(s => s.Trim()).Where(s => s.Length > 1).Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (_cache.TryGet(model, skill, out var cached)) result[skill] = cached;
            else missing.Add(skill);
        }

        if (missing.Count > 0)
        {
            var batch = await _embeddings.EmbedAsync(missing, ct);
            for (var i = 0; i < missing.Count && i < batch.Vectors.Count; i++)
            {
                var vector = VectorMath.Normalize(batch.Vectors[i]);
                result[missing[i]] = vector;
                // Cache under the model the batch actually used, which may be the
                // fallback rather than the preferred one.
                _cache.Set(batch.ModelId, missing[i], vector);
            }
        }

        return result;
    }

    public IReadOnlyList<SkillAssessmentDto> Assess(
        IReadOnlyList<string> required,
        IReadOnlyList<string> candidateSkills,
        IReadOnlyDictionary<string, float[]> vectors)
    {
        var assessments = new List<SkillAssessmentDto>(required.Count);

        foreach (var requirement in required)
        {
            // A textual match needs no vector and no threshold, so this verdict is
            // right even under the deterministic fallback embedder. Normalised
            // rather than exact because "GitHub Actions" and "Github actions" are
            // the same skill, and cosine gives punctuation and casing variants a
            // surprisingly wide spread.
            var exact = candidateSkills.FirstOrDefault(s => Normalize(s) == Normalize(requirement));
            if (exact is not null)
            {
                assessments.Add(new SkillAssessmentDto(requirement, "Have", exact, 1.0));
                continue;
            }

            var best = BestMatch(requirement, candidateSkills, vectors);

            var status = best.Similarity switch
            {
                var s when s >= _options.SkillEquivalenceMin => "Have",
                var s when s >= _options.SkillRelatedMin => "Transferable",
                _ => "Missing",
            };

            assessments.Add(new SkillAssessmentDto(
                requirement,
                status,
                // Naming the evidence only when there is some: a "Missing" verdict
                // paired with a weakly-similar skill reads as a claim, and the
                // candidate would reasonably ask why it did not count.
                status == "Missing" ? null : best.Skill,
                Math.Round(best.Similarity, 3)));
        }

        return assessments;
    }

    /// <summary>
    /// Case, spacing and punctuation carry no meaning in a skill name. Purely
    /// structural: no aliases, no abbreviation table, nothing domain-specific.
    /// </summary>
    private static string Normalize(string skill)
    {
        Span<char> buffer = stackalloc char[skill.Length];
        var length = 0;
        foreach (var c in skill)
        {
            if (char.IsLetterOrDigit(c)) buffer[length++] = char.ToLowerInvariant(c);
        }
        return new string(buffer[..length]);
    }

    private (string? Skill, double Similarity) BestMatch(
        string requirement,
        IReadOnlyList<string> candidateSkills,
        IReadOnlyDictionary<string, float[]> vectors)
    {
        if (!vectors.TryGetValue(requirement, out var requirementVector))
            return (null, 0);

        string? bestSkill = null;
        var bestScore = 0.0;

        foreach (var skill in candidateSkills)
        {
            if (!vectors.TryGetValue(skill, out var skillVector)) continue;

            var score = VectorMath.Dot(requirementVector, skillVector);
            if (score > bestScore)
            {
                bestScore = score;
                bestSkill = skill;
            }
        }

        return (bestSkill, bestScore);
    }
}
