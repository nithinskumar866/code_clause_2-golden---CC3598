using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Llm;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Services.Matching;

public record SkillTransfer(string Requirement, string CandidateSkill, string Reason);

public interface ISkillTransferService
{
    /// <summary>
    /// For requirements the candidate does not have, decides which are covered by
    /// something adjacent they DO have. Returns only genuine transfers.
    /// </summary>
    Task<IReadOnlyList<SkillTransfer>> FindTransfersAsync(
        int jobId,
        string jobTitle,
        IReadOnlyList<string> missingRequirements,
        IReadOnlyList<string> candidateSkills,
        CancellationToken ct = default);
}

/// <summary>
/// Decides skill transferability.
///
/// WHY THIS IS NOT DONE WITH EMBEDDINGS. Measured over hand-labelled skill pairs
/// on this deployment's models, cosine over skill NAMES separates "the same skill
/// written differently" from everything else, but does NOT separate "adjacent and
/// genuinely transferable" from "unrelated":
///
///   nomic-embed-text, bare      equivalent 0.703 | adjacent 0.379 | unrelated 0.383
///   nomic, clustering: prefix   equivalent 0.892 | adjacent 0.693 | unrelated 0.682
///   bge-m3, bare                equivalent 0.823 | adjacent 0.494 | unrelated 0.411
///
/// The adjacent and unrelated distributions sit on top of each other. Any
/// threshold picked in that overlap produces confident falsehoods in both
/// directions: telling a candidate their Figma experience transfers to AWS, or
/// telling them their AWS does not transfer to Azure. Both are worse than saying
/// nothing, because the whole point of the recruiter note is that the candidate
/// acts on it in an interview.
///
/// So the split follows the house rule — deterministic algorithms where the
/// answer must be reproducible, the LLM only where genuine reasoning is required.
/// Equivalence stays algorithmic (<see cref="SkillSemanticsService"/>); this is
/// the reasoning part.
///
/// The model is still not allowed to invent anything. It chooses from a closed
/// list of skills the candidate actually listed, against a closed list of
/// requirements the posting actually stated, and every returned pair is checked
/// back against both lists before it is used.
/// </summary>
public class SkillTransferService : ISkillTransferService
{
    private const string SystemPrompt = """
        You judge whether a candidate's existing skills transfer to a job requirement
        they do not have.

        You are given a requirement list and the candidate's skill list. For each
        requirement, decide whether ONE of the candidate's listed skills makes them
        substantially able to pick it up quickly.

        Rules:
        - candidate_skill MUST be copied exactly from the candidate's skill list.
          Never name a skill that is not on it.
        - requirement MUST be copied exactly from the requirement list.
        - Only include a requirement when the transfer is real and specific: same
          category of tool solving the same problem (one cloud platform to another,
          one container orchestrator to another, one UI framework to another).
        - Do NOT include a requirement just because both are "technology". If nothing
          the candidate has genuinely transfers, leave it out. An empty list is a
          correct and useful answer.
        - reason: one short clause the candidate could say in an interview.

        Reply with JSON only:
        {"transfers":[{"requirement":string,"candidate_skill":string,"reason":string}]}
        """;

    // Prompt version - bump this when the SystemPrompt changes to invalidate cache
    private const string PromptVersion = "v1";

    private readonly IOllamaClient _llm;
    private readonly PortalDbContext _db;
    private readonly ILogger<SkillTransferService> _logger;

    public SkillTransferService(IOllamaClient llm, PortalDbContext db, ILogger<SkillTransferService> logger)
    {
        _llm = llm;
        _db = db;
        _logger = logger;
    }

    public async Task<IReadOnlyList<SkillTransfer>> FindTransfersAsync(
        int jobId,
        string jobTitle,
        IReadOnlyList<string> missingRequirements,
        IReadOnlyList<string> candidateSkills,
        CancellationToken ct = default)
    {
        if (!_llm.ChatAvailable || missingRequirements.Count == 0 || candidateSkills.Count == 0)
        {
            // No model, no transfer verdicts. The assessment stays Have/Missing,
            // which is exactly what the deterministic path can support honestly.
            return Array.Empty<SkillTransfer>();
        }

        var skillSetHash = ComputeSkillSetHash(candidateSkills);
        var modelVersion = _llm.GetModelName() ?? "unknown";

        // Check cache for each missing requirement
        var cachedVerdicts = await _db.SkillVerdictCache
            .Where(v => v.JobId == jobId && v.CandidateSkillSetHash == skillSetHash && v.PromptVersion == PromptVersion)
            .ToDictionaryAsync(v => v.RequiredSkill, v => v, ct);

        var results = new List<SkillTransfer>();
        var uncachedRequirements = new List<string>();

        foreach (var requirement in missingRequirements)
        {
            if (cachedVerdicts.TryGetValue(requirement, out var cached))
            {
                // Cache hit - use cached verdict
                if (cached.Verdict == "Transferable" && cached.EvidenceSkill is not null)
                {
                    results.Add(new SkillTransfer(
                        requirement,
                        cached.EvidenceSkill,
                        TextStructure.Clip(cached.EvidenceSkill, 160))); // Reason not stored, use skill name as placeholder
                }
                // For "Have" or "Missing" verdicts, we don't add to transfers list
                _logger.LogDebug("Cache hit for skill verdict: job={JobId}, requirement={Requirement}, verdict={Verdict}", 
                    jobId, requirement, cached.Verdict);
            }
            else
            {
                uncachedRequirements.Add(requirement);
            }
        }

        // If all requirements were cached, return early
        if (uncachedRequirements.Count == 0)
        {
            return results;
        }

        // Query LLM for uncached requirements
        var prompt = $"""
            Role: {jobTitle}

            REQUIREMENTS THE CANDIDATE DOES NOT HAVE
            {string.Join("\n", uncachedRequirements.Select(r => "- " + r))}

            THE CANDIDATE'S SKILLS
            {string.Join("\n", candidateSkills.Select(s => "- " + s))}
            """;

        var reply = await _llm.ChatAsync(new[]
        {
            new ChatTurn("system", SystemPrompt),
            new ChatTurn("user", prompt),
        }, jsonMode: true, ct: ct); // Low temperature is configured in OllamaOptions

        if (string.IsNullOrWhiteSpace(reply)) return results;

        try
        {
            using var doc = JsonDocument.Parse(ExtractJsonObject(reply));
            if (!doc.RootElement.TryGetProperty("transfers", out var array) ||
                array.ValueKind != JsonValueKind.Array)
            {
                return results;
            }

            var requirements = missingRequirements.ToHashSet(StringComparer.OrdinalIgnoreCase);
            var skills = candidateSkills.ToHashSet(StringComparer.OrdinalIgnoreCase);

            var newCacheEntries = new List<PortalSkillVerdictCache>();

            foreach (var element in array.EnumerateArray())
            {
                var requirement = Read(element, "requirement");
                var skill = Read(element, "candidate_skill");
                var reason = Read(element, "reason");

                if (requirement is null || skill is null) continue;

                // The grounding check. A transfer naming a skill the candidate never
                // listed is a fabricated qualification, and it would be shown to them
                // as advice to lean on it in an interview.
                if (!requirements.Contains(requirement) || !skills.Contains(skill))
                {
                    _logger.LogWarning(
                        "Discarded an ungrounded transfer: '{Requirement}' <- '{Skill}'", requirement, skill);
                    continue;
                }

                var canonicalRequirement = requirements.First(r => r.Equals(requirement, StringComparison.OrdinalIgnoreCase));
                var canonicalSkill = skills.First(s => s.Equals(skill, StringComparison.OrdinalIgnoreCase));

                results.Add(new SkillTransfer(
                    canonicalRequirement,
                    canonicalSkill,
                    TextStructure.Clip(reason ?? "", 160)));

                // Cache the verdict
                newCacheEntries.Add(new PortalSkillVerdictCache
                {
                    JobId = jobId,
                    CandidateSkillSetHash = skillSetHash,
                    RequiredSkill = canonicalRequirement,
                    Verdict = "Transferable",
                    EvidenceSkill = canonicalSkill,
                    Confidence = 0.8, // Default confidence for LLM-derived transfers
                    ModelVersion = modelVersion,
                    PromptVersion = PromptVersion,
                });
            }

            // Cache "Missing" verdicts for requirements not returned by LLM
            var returnedRequirements = newCacheEntries.Select(e => e.RequiredSkill).ToHashSet(StringComparer.OrdinalIgnoreCase);
            foreach (var req in uncachedRequirements)
            {
                if (!returnedRequirements.Contains(req))
                {
                    newCacheEntries.Add(new PortalSkillVerdictCache
                    {
                        JobId = jobId,
                        CandidateSkillSetHash = skillSetHash,
                        RequiredSkill = req,
                        Verdict = "Missing",
                        EvidenceSkill = null,
                        Confidence = 0.9,
                        ModelVersion = modelVersion,
                        PromptVersion = PromptVersion,
                    });
                }
            }

            // Save cache entries
            if (newCacheEntries.Count > 0)
            {
                _db.SkillVerdictCache.AddRange(newCacheEntries);
                await _db.SaveChangesAsync(ct);
            }

            // One verdict per requirement; a model that lists the same requirement
            // twice should not produce two contradictory rows.
            return results
                .GroupBy(t => t.Requirement, StringComparer.OrdinalIgnoreCase)
                .Select(g => g.First())
                .ToList();
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Skill-transfer reply was not parseable JSON; reporting no transfers.");
            return results;
        }
    }

    private static string ComputeSkillSetHash(IReadOnlyList<string> skills)
    {
        var sorted = skills.OrderBy(s => s, StringComparer.OrdinalIgnoreCase).ToList();
        var joined = string.Join("|", sorted);
        using var sha256 = SHA256.Create();
        var hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(joined));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string? Read(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String &&
        !string.IsNullOrWhiteSpace(value.GetString())
            ? TextStructure.Collapse(value.GetString()!)
            : null;

    private static string ExtractJsonObject(string reply)
    {
        var start = reply.IndexOf('{');
        var end = reply.LastIndexOf('}');
        return start >= 0 && end > start ? reply[start..(end + 1)] : reply;
    }
}
