using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Mapping;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Vectors;
using Microsoft.EntityFrameworkCore;

namespace JobPortal.Api.Services.Jobs;

public interface IJobSuggestionService
{
    /// <summary>
    /// Roles to show someone who has not uploaded a CV.
    ///
    /// <paramref name="query"/> empty means "suggest something" — the newest
    /// postings that satisfy the filters. With a query, the board is searched
    /// semantically against the words themselves.
    /// </summary>
    Task<JobSuggestionResultDto> SuggestAsync(
        string? query, JobFilters filters, int top = 5, CancellationToken ct = default);
}

/// <summary>
/// Finding roles with no resume to match against.
///
/// This exists because the assistant used to answer every question from someone
/// without a CV with the same sentence asking for one. A job board is browsable
/// by definition — "show me remote Python roles" is answerable from the postings
/// alone — and refusing to answer it until a document is uploaded turns the
/// front door into a gate.
///
/// What it deliberately does NOT do is score. A fit percentage is a claim about
/// a person, and with no resume there is no person to make it about; showing 0%
/// or a similarity dressed up as a fit would be inventing one. So these come back
/// as postings, ranked by relevance to what was asked, and the scoring only
/// appears once there is a CV to score.
/// </summary>
public class JobSuggestionService : IJobSuggestionService
{
    private readonly PortalDbContext _db;
    private readonly IEmbeddingService _embeddings;
    private readonly IVectorStore _vectors;
    private readonly ILogger<JobSuggestionService> _logger;

    public JobSuggestionService(
        PortalDbContext db,
        IEmbeddingService embeddings,
        IVectorStore vectors,
        ILogger<JobSuggestionService> logger)
    {
        _db = db;
        _embeddings = embeddings;
        _vectors = vectors;
        _logger = logger;
    }

    public async Task<JobSuggestionResultDto> SuggestAsync(
        string? query, JobFilters filters, int top = 5, CancellationToken ct = default)
    {
        top = Math.Clamp(top, 1, 25);

        var published = await _db.Jobs
            .AsNoTracking()
            .Where(j => j.IsPublished)
            .ToListAsync(ct);

        var eligible = published.Where(job => JobFilterRules.Passes(job, filters)).ToList();
        if (eligible.Count == 0)
        {
            return new JobSuggestionResultDto(
                Array.Empty<JobSummaryDto>(), 0, filters, false, _embeddings.PreferredModelId, false);
        }

        var trimmed = (query ?? "").Trim();
        var ranked = trimmed.Length >= 2
            ? await RankByRelevanceAsync(trimmed, eligible, ct)
            : null;

        // Newest first is the honest default when nothing was asked for. Any other
        // order would be presenting an arbitrary sequence as relevance.
        var ordered = ranked ?? eligible.OrderByDescending(j => j.CreatedAt).ToList();

        var indexed = (await _vectors.GetIndexedHashesAsync(_embeddings.PreferredModelId, ct)).Keys.ToHashSet();

        return new JobSuggestionResultDto(
            ordered.Take(top).Select(job => job.ToSummaryDto(indexed.Contains(job.Id))).ToList(),
            eligible.Count,
            filters,
            ranked is not null && _embeddings.SemanticAvailable,
            _embeddings.PreferredModelId,
            ranked is not null);
    }

    /// <summary>
    /// Orders the eligible postings by how close they are to what was typed.
    ///
    /// Returns null when the query could not be embedded, so the caller can fall
    /// back to recency and say so, rather than silently presenting an unranked
    /// list as if it were relevance-ordered.
    /// </summary>
    private async Task<List<PortalJob>?> RankByRelevanceAsync(
        string query, List<PortalJob> eligible, CancellationToken ct)
    {
        try
        {
            var batch = await _embeddings.EmbedAsync(TextStructure.Clip(query, 2000), ct);
            if (batch.Vectors.Count == 0) return null;

            // Restricted to the postings that already passed the filters, so the
            // constraints the candidate stated are never overruled by similarity.
            var allowed = eligible.Select(j => j.Id).ToHashSet();

            var hits = await _vectors.SearchAsync(
                batch.Vectors[0], _embeddings.PreferredModelId, allowed.Count,
                minScore: 0, restrictTo: allowed, ct: ct);

            if (hits.Count == 0) return null;

            var byId = eligible.ToDictionary(j => j.Id);
            var ordered = hits
                .Where(h => byId.ContainsKey(h.JobId))
                .Select(h => byId[h.JobId])
                .ToList();

            // Anything unindexed cannot be ranked, but it is still a real posting
            // that passed the filters — appended rather than dropped.
            ordered.AddRange(eligible
                .Where(j => ordered.All(o => o.Id != j.Id))
                .OrderByDescending(j => j.CreatedAt));

            return ordered;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not rank suggestions for a query; falling back to recency.");
            return null;
        }
    }
}
