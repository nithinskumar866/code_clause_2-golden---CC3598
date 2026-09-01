using System.Text;
using System.Text.Json;
using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Llm;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Rag.Data;
using JobPortal.Api.Services.Rag.Retrieval;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Rag.Agent;

/// <summary>
/// Judges one résumé against one posting by RETRIEVING the evidence for each
/// requirement instead of reading the whole document.
///
/// The loop, per posting:
///
///   1. Break the posting into requirements — from its own structured fields, no
///      model involved (see <see cref="JobRequirements"/>).
///   2. For each requirement, retrieve the passages of THIS résumé that bear on
///      it. Milliseconds, no model.
///   3. If a requirement retrieved nothing, widen the query and try once more.
///      This is the agentic step, and it is the difference between "the résumé
///      does not contain this phrase" and "the résumé does not evidence this
///      requirement".
///   4. Hand the model every requirement with its retrieved passages, and ask for
///      a verdict on each. One call per posting, not one per requirement.
///   5. Score the verdicts with the arithmetic the AI-evaluation mode already
///      uses, via <see cref="JobEvaluationService.Compose"/>.
///
/// What this buys over pasting the CV into a prompt:
///
///   NO TRUNCATION. The old evaluator clips a résumé at nine thousand characters
///   and never says so. Here the document's length is irrelevant — only the
///   passages that matched are sent.
///
///   NO INVENTED QUOTES. The model chooses a quote from the passages it was
///   given, so a fabricated one is not merely detectable, it is unavailable.
/// </summary>
public class RagEvaluationAgent
{
    /// <summary>Bumped when the prompt below changes; part of the cache key.</summary>
    public const string PromptVersion = "rag-v1";

    private const double JudgementTemperature = 0.0;
    private static readonly TimeSpan CacheLifetime = TimeSpan.FromHours(6);
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private readonly IRagRetriever _retriever;
    private readonly IOllamaClient _llm;
    private readonly JobEvaluationService _scoring;
    private readonly IMemoryCache _cache;
    private readonly RagOptions _options;
    private readonly ILogger<RagEvaluationAgent> _logger;

    public RagEvaluationAgent(
        IRagRetriever retriever,
        IOllamaClient llm,
        JobEvaluationService scoring,
        IMemoryCache cache,
        IOptions<RagOptions> options,
        ILogger<RagEvaluationAgent> logger)
    {
        _retriever = retriever;
        _llm = llm;
        _scoring = scoring;
        _cache = cache;
        _options = options.Value;
        _logger = logger;
    }

    public async Task<JobEvaluationDto?> EvaluateAsync(
        PortalResume resume, PortalJob job, CancellationToken ct = default)
    {
        if (!_llm.ChatAvailable) return null;

        var key = $"rag|{PromptVersion}|{_llm.GetModelName()}|r{resume.Id}|j{job.Id}:{job.UpdatedAt:O}";
        if (_cache.TryGetValue<JobEvaluationDto>(key, out var cached) && cached is not null) return cached;

        var requirements = JobRequirements.For(job);
        if (requirements.Count == 0) return null;

        var started = DateTime.UtcNow;

        // STAGE 2 retrieval — one targeted search per requirement, with a retry.
        var evidence = new List<(JobRequirement Requirement, IReadOnlyList<RetrievedChunk> Chunks, bool Widened)>();
        var retries = 0;

        foreach (var requirement in requirements)
        {
            ct.ThrowIfCancellationRequested();

            var chunks = await _retriever.ForRequirementAsync(
                requirement.Text, RagParentTypes.Resume, resume.Id, ct);

            var widened = false;
            if (chunks.Count == 0 && _options.ExpandOnEmpty)
            {
                // Nothing cleared the floor. Before recording this as absent, ask
                // the question a different way — the phrasing of a posting and the
                // phrasing of a résumé rarely coincide, and one missed retrieval
                // becomes a MISSING verdict on a requirement the candidate meets.
                chunks = await _retriever.ForRequirementAsync(
                    Widen(requirement, job), RagParentTypes.Resume, resume.Id, ct);

                widened = true;
                retries++;
            }

            evidence.Add((requirement, chunks, widened));
        }

        var verdicts = await JudgeAsync(resume, job, evidence, ct);
        if (verdicts is null) return null;

        var evaluation = _scoring.Compose(
            verdicts,
            unexamined: new HashSet<string>(StringComparer.OrdinalIgnoreCase),
            proposedWeights: JobRequirements.Weights(),
            claimedKnockout: new JobEvaluationService.ClaimedKnockout(false, false),
            statedYears: resume.YearsExperience,
            resume: resume,
            job: job,
            alternateRole: null,
            justification: "",
            promptVersion: PromptVersion);

        _logger.LogInformation(
            "RAG evaluated job {JobId} in {Seconds:0.#}s: {Score}% {Category} " +
            "({Requirements} requirements, {Retries} widened).",
            job.Id, (DateTime.UtcNow - started).TotalSeconds,
            evaluation.OverallMatch, evaluation.Category, requirements.Count, retries);

        _cache.Set(key, evaluation, CacheLifetime);
        return evaluation;
    }

    /// <summary>
    /// The second attempt's query.
    ///
    /// Not a synonym list — there is no maintained vocabulary here and there is not
    /// going to be one. It widens by CONTEXT: the requirement said with the role it
    /// belongs to, which pulls the query toward the same region of the embedding
    /// space as the work rather than the words.
    /// </summary>
    private static string Widen(JobRequirement requirement, PortalJob job)
    {
        var role = string.IsNullOrWhiteSpace(job.Title) ? "this role" : job.Title;
        return $"{requirement.Text}. Experience relevant to a {role}, described in any wording.";
    }

    // -- the judge ----------------------------------------------------------

    private const string System = """
        You judge whether a candidate's RETRIEVED RÉSUMÉ PASSAGES satisfy each
        requirement of a job posting. Return JSON only.

        You are given, for every requirement, the passages of this candidate's
        résumé that a search returned for it. Those passages are the ONLY thing you
        know about the candidate. There is no other document to recall.

        ── VERDICTS ──
          STRONG   the passages show the requirement demonstrated in work history,
                   a professional internship, or a delivered project
          WEAK     the passages only name it — a skills list, a summary line, a
                   qualification in passing — without showing it applied
          MISSING  the passages do not support it, or support only a different
                   thing that happens to use similar words

        ── QUOTE FROM WHAT YOU WERE GIVEN ──
        "quote" must be copied from the passages shown for that requirement. You
        cannot quote anything else, because you have not been shown anything else.
        If no passage supports the requirement, the quote is "" and the verdict is
        MISSING.

        ── NO PASSAGES IS NOT AUTOMATICALLY MISSING ──
        A requirement with an empty passage list means the search found nothing
        above its threshold. That is strong evidence of absence and MISSING is
        usually right — but say so on that basis, not by pretending to have looked
        at a document you were not given.

        ── JUDGE EACH ROW ON ITS OWN ──
        No sentiment bleed. A candidate who is a poor fit overall may still fully
        evidence one requirement, and that row is STRONG on its own passages.
        Never mark a requirement MISSING because other requirements failed.

        ── MEANING, NOT SPELLING — WITHIN A DOMAIN ──
        Different names for one technology are one technology. A specific service
        evidences the general capability. But this never crosses domains: a
        frontend framework does not satisfy a design tool, and application code
        does not satisfy cloud infrastructure.

        ── DO NOT SCORE ──
        Never produce a number, a percentage or an overall verdict. Row verdicts
        only. The score is computed elsewhere from what you return.

        Return ONLY:
        {"verdicts":[{"index":<the requirement's number>,"match_level":"STRONG"|"WEAK"|"MISSING","quote":"<copied from that requirement's passages, or empty>","reasoning":"<one sentence>"}]}
        """;

    private async Task<IReadOnlyList<EvaluationRequirementDto>?> JudgeAsync(
        PortalResume resume,
        PortalJob job,
        IReadOnlyList<(JobRequirement Requirement, IReadOnlyList<RetrievedChunk> Chunks, bool Widened)> evidence,
        CancellationToken ct)
    {
        var prompt = new StringBuilder();
        prompt.AppendLine($"POSTING: {job.Title}");
        if (!string.IsNullOrWhiteSpace(job.SeniorityLevel)) prompt.AppendLine($"Seniority: {job.SeniorityLevel}");
        prompt.AppendLine();

        for (var i = 0; i < evidence.Count; i++)
        {
            var (requirement, chunks, widened) = evidence[i];

            prompt.AppendLine($"--- REQUIREMENT {i + 1} [{RequirementKinds.Label(requirement.Kind)}] ---");
            prompt.AppendLine(requirement.Text);

            if (chunks.Count == 0)
            {
                prompt.AppendLine(widened
                    ? "PASSAGES: none. Two searches, including a widened one, returned nothing above the threshold."
                    : "PASSAGES: none returned above the threshold.");
            }
            else
            {
                prompt.AppendLine("PASSAGES from this candidate's résumé:");
                foreach (var chunk in chunks)
                {
                    prompt.AppendLine($"  [{chunk.Section}] {TextStructure.Clip(chunk.Text, 600)}");
                }
            }

            prompt.AppendLine();
        }

        string? reply;
        try
        {
            reply = await _llm.ChatAsync(
                new[] { new ChatTurn("system", System), new ChatTurn("user", prompt.ToString()) },
                jsonMode: true, temperature: JudgementTemperature, ct: ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "RAG judgement of job {JobId} failed.", job.Id);
            return null;
        }

        if (string.IsNullOrWhiteSpace(reply)) return null;

        return ReadVerdicts(reply, evidence, job.Id);
    }

    /// <summary>
    /// Turns the model's verdicts into scored rows.
    ///
    /// Every row is built from the REQUIREMENT LIST, not from the reply: a
    /// requirement the model forgot to mention still gets a row, because a
    /// requirement nobody judged must not silently leave the denominator. That was
    /// a real failure in the other evaluator — a shorter list is a score for an
    /// easier job than the one advertised.
    /// </summary>
    private IReadOnlyList<EvaluationRequirementDto>? ReadVerdicts(
        string reply,
        IReadOnlyList<(JobRequirement Requirement, IReadOnlyList<RetrievedChunk> Chunks, bool Widened)> evidence,
        int jobId)
    {
        var start = reply.IndexOf('{');
        var end = reply.LastIndexOf('}');
        if (start < 0 || end <= start) return null;

        Dictionary<int, RawVerdict> byIndex;
        try
        {
            var parsed = JsonSerializer.Deserialize<RawReply>(reply[start..(end + 1)], Json);
            byIndex = (parsed?.Verdicts ?? new())
                .Where(v => v.Index >= 1 && v.Index <= evidence.Count)
                .GroupBy(v => v.Index)
                .ToDictionary(g => g.Key, g => g.Last());
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "RAG judgement of job {JobId} returned unparseable JSON.", jobId);
            return null;
        }

        if (byIndex.Count == 0) return null;

        var rows = new List<EvaluationRequirementDto>(evidence.Count);

        for (var i = 0; i < evidence.Count; i++)
        {
            var (requirement, chunks, _) = evidence[i];
            byIndex.TryGetValue(i + 1, out var verdict);

            var level = MatchLevels.Normalise(verdict?.Match_Level);

            // A verdict the model skipped is MISSING, not absent: the requirement
            // was asked about, and no answer is not the same as no requirement.
            if (verdict is null) level = MatchLevels.Missing;

            // Nothing was retrieved, so nothing can be evidenced. The model
            // claiming otherwise is claiming to have read a document it was never
            // shown.
            if (chunks.Count == 0) level = MatchLevels.Missing;

            // A verdict with no quote is not demoted. The model was shown a passage
            // that cleared the retrieval floor and judged on it; omitting the quote
            // is a formatting lapse, not weaker evidence. Verify falls back to that
            // passage, so the card still shows something the candidate wrote.
            var (quote, section) = Verify(verdict?.Quote, chunks);

            rows.Add(new EvaluationRequirementDto(
                requirement.Text,
                level == MatchLevels.Missing ? "" : quote,
                level,
                requirement.Kind,
                // The section of the passage the QUOTE came from — not of whichever
                // chunk happened to rank first, and empty when there is no quote.
                // "User research — MISSING — Summary" reads as though something in
                // the summary was weighed, when nothing was found at all.
                Where: level == MatchLevels.Missing ? "" : section,
                Reasoning: TextStructure.Collapse(verdict?.Reasoning ?? "")));
        }

        return rows;
    }

    /// <summary>
    /// Whether the quote really came from the passages supplied.
    ///
    /// Cheaper and stricter than the document-wide check the other evaluator
    /// needs: the candidate set is four passages, so a quote either appears in one
    /// of them or was invented. Whitespace and punctuation are normalised away,
    /// because a model reflowing a bullet is not the failure this guards against.
    /// </summary>
    private static (string Quote, string Section) Verify(
        string? quote, IReadOnlyList<RetrievedChunk> chunks)
    {
        if (chunks.Count == 0) return ("", "");

        var candidate = TextStructure.Collapse(quote ?? "");

        if (candidate.Length >= 3)
        {
            var needle = Normalise(candidate);

            foreach (var chunk in chunks)
            {
                if (!Normalise(chunk.Text).Contains(needle, StringComparison.Ordinal)) continue;

                // A one-word quote is verbatim and useless. "Figma" tells a
                // candidate nothing about WHY it counted as weak evidence, whereas
                // the line it sits on — a skills list — says exactly that. So a
                // very short quote is widened to its passage, which is equally
                // verbatim and actually informative.
                var shown = candidate.Length < MinimumUsefulQuote
                    ? TextStructure.Clip(chunk.Text, 240)
                    : candidate;

                return (shown, chunk.Section);
            }
        }

        // Not found, or nothing offered. Fall back to the strongest passage
        // retrieved — the candidate did write that, and it is what the search
        // thought was relevant.
        return (TextStructure.Clip(chunks[0].Text, 240), chunks[0].Section);
    }

    /// <summary>
    /// Below this, a quote is a term rather than evidence.
    ///
    /// Twenty-five characters is roughly "a few words": enough to exclude a bare
    /// skill name lifted out of a comma-separated list, short enough to keep a
    /// genuinely terse line like "Led the platform team".
    /// </summary>
    private const int MinimumUsefulQuote = 25;

    private static string Normalise(string text) =>
        new string(text.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) ? c : ' ').ToArray())
            .Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Aggregate(new StringBuilder(), (b, w) => b.Append(w).Append(' '))
            .ToString();

    private sealed class RawReply
    {
        public List<RawVerdict>? Verdicts { get; set; }
    }

    private sealed class RawVerdict
    {
        public int Index { get; set; }
        public string? Match_Level { get; set; }
        public string? Quote { get; set; }
        public string? Reasoning { get; set; }
    }
}
