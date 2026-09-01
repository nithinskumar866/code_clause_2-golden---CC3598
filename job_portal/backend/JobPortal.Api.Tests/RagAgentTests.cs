using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Rag;
using JobPortal.Api.Services.Rag.Agent;
using JobPortal.Api.Services.Rag.Data;
using JobPortal.Api.Services.Rag.Retrieval;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging.Abstractions;

namespace JobPortal.Api.Tests;

/// <summary>
/// A retriever that returns whatever the test scripted, per requirement.
///
/// The agent is the piece worth testing — it turns passages into verdicts — and
/// it should not need Qdrant and Postgres running to do that.
/// </summary>
internal class FakeRetriever : IRagRetriever
{
    private readonly Dictionary<string, IReadOnlyList<RetrievedChunk>> _byRequirement;

    /// <summary>Requirements asked for, in order — including widened retries.</summary>
    public List<string> Queries { get; } = new();

    public FakeRetriever(Dictionary<string, IReadOnlyList<RetrievedChunk>> byRequirement) =>
        _byRequirement = byRequirement;

    public Task<IReadOnlyList<DocumentMatch>> ShortlistAsync(
        string query, string parentType, IReadOnlySet<int>? restrictTo = null,
        int? take = null, CancellationToken ct = default) =>
        Task.FromResult<IReadOnlyList<DocumentMatch>>(Array.Empty<DocumentMatch>());

    public Task<IReadOnlyList<RetrievedChunk>> ForRequirementAsync(
        string requirement, string parentType, int parentId, CancellationToken ct = default)
    {
        Queries.Add(requirement);

        // Match on prefix so a widened query ("X. Experience relevant to…") can be
        // scripted to return either the same passages or nothing.
        var hit = _byRequirement.FirstOrDefault(kv => requirement.StartsWith(kv.Key, StringComparison.OrdinalIgnoreCase));

        return Task.FromResult(hit.Value ?? (IReadOnlyList<RetrievedChunk>)Array.Empty<RetrievedChunk>());
    }
}

/// <summary>
/// The agent: retrieve per requirement, judge the passages, score the verdicts.
///
/// The property that separates this from the other evaluator is that the model
/// is only ever shown the retrieved passages. It cannot quote a document it was
/// not given, and a requirement whose search came back empty cannot be talked
/// into being evidenced.
/// </summary>
public class RagAgentTests
{
    private static PortalResume Candidate() => new()
    {
        Id = 1,
        RawText = "irrelevant — the agent never reads this, only retrieved passages",
        YearsExperience = 5,
    };

    private static PortalJob Posting() => new()
    {
        Id = 10,
        Title = "Frontend Engineer",
        RequiredSkills = "React\nTypeScript",
        UpdatedAt = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc),
    };

    private static RetrievedChunk Chunk(string text, string section, double score = 0.8) =>
        new(ChunkId: Math.Abs(text.GetHashCode()), ParentId: 1, Section: section, Text: text, Score: score);

    private static RagEvaluationAgent Agent(
        FakeRetriever retriever, string? reply, RagOptions? options = null)
    {
        var cache = new MemoryCache(new MemoryCacheOptions());
        var scoring = new JobEvaluationService(
            new ScriptedLlm(reply), new ResumeEvidenceService(), cache,
            NullLogger<JobEvaluationService>.Instance);

        return new RagEvaluationAgent(
            retriever,
            new ScriptedLlm(reply),
            scoring,
            cache,
            Microsoft.Extensions.Options.Options.Create(options ?? new RagOptions()),
            NullLogger<RagEvaluationAgent>.Instance);
    }

    private static string Verdicts(params string[] rows) =>
        $$"""{"verdicts":[{{string.Join(",", rows)}}]}""";

    private static string Verdict(int index, string level, string quote, string reasoning = "because") =>
        $$"""{"index":{{index}},"match_level":"{{level}}","quote":"{{quote}}","reasoning":"{{reasoning}}"}""";

    // -- the passages are the only evidence ---------------------------------

    /// <summary>
    /// A requirement whose search returned nothing cannot be evidenced, whatever
    /// the model says. It was shown no passages, so a STRONG verdict on it is a
    /// claim about a document it never saw.
    /// </summary>
    [Fact]
    public async Task A_requirement_with_no_retrieved_passages_is_MISSING_whatever_the_model_says()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React over two years.", "Experience") },
            // TypeScript retrieves nothing, and the widened retry finds nothing either.
        });

        var evaluation = await Agent(retriever, Verdicts(
            Verdict(1, "STRONG", "Built the checkout in React over two years."),
            Verdict(2, "STRONG", "TypeScript everywhere")))
            .EvaluateAsync(Candidate(), Posting());

        Assert.NotNull(evaluation);

        var typescript = evaluation!.Requirements!.Single(r => r.Requirement == "TypeScript");
        Assert.Equal(MatchLevels.Missing, typescript.MatchLevel);
        Assert.Equal("", typescript.Quote);

        // And the section is blank, not the section of some unrelated chunk.
        Assert.Equal("", typescript.Where);
    }

    [Fact]
    public async Task An_empty_retrieval_is_retried_with_a_widened_query()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React.", "Experience") },
        });

        await Agent(retriever, Verdicts(Verdict(1, "STRONG", "Built the checkout in React.")))
            .EvaluateAsync(Candidate(), Posting());

        // TypeScript found nothing, so it was asked a second time with the role
        // attached — the difference between "this phrase is absent" and "this
        // requirement is unevidenced".
        Assert.Contains(retriever.Queries, q =>
            q.StartsWith("TypeScript") && q.Contains("Frontend Engineer"));
    }

    [Fact]
    public async Task Widening_can_be_switched_off()
    {
        var retriever = new FakeRetriever(new());

        await Agent(retriever, Verdicts(Verdict(1, "MISSING", "")), new RagOptions { ExpandOnEmpty = false })
            .EvaluateAsync(Candidate(), Posting());

        Assert.DoesNotContain(retriever.Queries, q => q.Contains("Experience relevant to"));
    }

    // -- quotes --------------------------------------------------------------

    [Fact]
    public async Task An_invented_quote_falls_back_to_a_passage_the_candidate_wrote()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React over two years.", "Experience") },
        });

        var evaluation = await Agent(retriever, Verdicts(
            Verdict(1, "STRONG", "Led the React platform team of twelve engineers")))
            .EvaluateAsync(Candidate(), Posting());

        var react = evaluation!.Requirements!.Single(r => r.Requirement == "React");

        // The sentence it offered is not in the passages, so it cannot be shown as
        // a quote. What IS shown is the passage that was retrieved.
        Assert.DoesNotContain("twelve engineers", react.Quote);
        Assert.Contains("Built the checkout in React", react.Quote);
    }

    /// <summary>
    /// A one-word quote is verbatim and useless: "React" does not explain why the
    /// evidence was weak, whereas the skills line it sits on says exactly that.
    /// </summary>
    [Fact]
    public async Task A_bare_term_is_widened_to_the_line_it_came_from()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("React, Redux, TypeScript, Vite, Jest, Storybook", "Skills") },
        });

        var evaluation = await Agent(retriever, Verdicts(Verdict(1, "WEAK", "React")))
            .EvaluateAsync(Candidate(), Posting());

        var react = evaluation!.Requirements!.Single(r => r.Requirement == "React");

        Assert.Contains("Redux", react.Quote);
        Assert.Equal("Skills", react.Where);
    }

    [Fact]
    public async Task The_section_reported_is_the_one_the_quote_came_from()
    {
        var retriever = new FakeRetriever(new()
        {
            // The skills line ranks first; the quote comes from the second passage.
            ["React"] = new[]
            {
                Chunk("React, Redux, TypeScript", "Skills", 0.9),
                Chunk("Rebuilt the checkout flow in React, cutting drop-off by a fifth.", "Experience", 0.8),
            },
        });

        var evaluation = await Agent(retriever, Verdicts(
            Verdict(1, "STRONG", "Rebuilt the checkout flow in React, cutting drop-off by a fifth.")))
            .EvaluateAsync(Candidate(), Posting());

        var react = evaluation!.Requirements!.Single(r => r.Requirement == "React");
        Assert.Equal("Experience", react.Where);
    }

    /// <summary>
    /// Omitting the quote is a formatting lapse, not weaker evidence: the model
    /// was shown a passage that cleared the retrieval floor and judged on it. The
    /// verdict stands and the card falls back to that passage, so it still shows
    /// something the candidate actually wrote.
    /// </summary>
    [Fact]
    public async Task A_verdict_with_no_quote_keeps_its_level_and_shows_the_passage()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React.", "Experience") },
        });

        var evaluation = await Agent(retriever, Verdicts(Verdict(1, "STRONG", "")))
            .EvaluateAsync(Candidate(), Posting());

        var react = evaluation!.Requirements!.Single(r => r.Requirement == "React");

        Assert.Equal(MatchLevels.Strong, react.MatchLevel);
        Assert.Contains("Built the checkout in React", react.Quote);
        Assert.Equal("Experience", react.Where);
    }

    // -- completeness --------------------------------------------------------

    /// <summary>
    /// A requirement the model forgot still gets a row. A shorter list is a score
    /// for an easier job than the one advertised.
    /// </summary>
    [Fact]
    public async Task A_requirement_the_model_skipped_still_gets_a_row()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React.", "Experience") },
            ["TypeScript"] = new[] { Chunk("TypeScript across the whole codebase.", "Experience") },
        });

        // Only the first requirement is answered.
        var evaluation = await Agent(retriever, Verdicts(
            Verdict(1, "STRONG", "Built the checkout in React.")))
            .EvaluateAsync(Candidate(), Posting());

        Assert.Equal(2, evaluation!.Requirements!.Count);
        Assert.Equal(MatchLevels.Missing,
            evaluation.Requirements!.Single(r => r.Requirement == "TypeScript").MatchLevel);
    }

    [Fact]
    public async Task A_verdict_for_a_requirement_that_does_not_exist_is_ignored()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React.", "Experience") },
        });

        var evaluation = await Agent(retriever, Verdicts(
            Verdict(1, "STRONG", "Built the checkout in React."),
            Verdict(99, "STRONG", "invented requirement")))
            .EvaluateAsync(Candidate(), Posting());

        Assert.Equal(2, evaluation!.Requirements!.Count);
    }

    [Fact]
    public async Task Unusable_output_returns_nothing_rather_than_a_score()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React.", "Experience") },
        });

        Assert.Null(await Agent(retriever, "not json").EvaluateAsync(Candidate(), Posting()));
        Assert.Null(await Agent(retriever, null).EvaluateAsync(Candidate(), Posting()));
        Assert.Null(await Agent(retriever, """{"verdicts":[]}""").EvaluateAsync(Candidate(), Posting()));
    }

    [Fact]
    public async Task The_same_pair_is_not_re_judged()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React.", "Experience") },
        });

        var agent = Agent(retriever, Verdicts(Verdict(1, "STRONG", "Built the checkout in React.")));
        var resume = Candidate();
        var job = Posting();

        var first = await agent.EvaluateAsync(resume, job);
        var queriesAfterFirst = retriever.Queries.Count;
        var second = await agent.EvaluateAsync(resume, job);

        Assert.Same(first, second);
        Assert.Equal(queriesAfterFirst, retriever.Queries.Count);
    }

    [Fact]
    public async Task The_score_comes_from_the_shared_arithmetic()
    {
        var retriever = new FakeRetriever(new()
        {
            ["React"] = new[] { Chunk("Built the checkout in React over two years.", "Experience") },
            ["TypeScript"] = new[] { Chunk("TypeScript across the whole codebase.", "Experience") },
        });

        var evaluation = await Agent(retriever, Verdicts(
            Verdict(1, "STRONG", "Built the checkout in React over two years."),
            Verdict(2, "STRONG", "TypeScript across the whole codebase.")))
            .EvaluateAsync(Candidate(), Posting());

        // Both core skills fully evidenced, no dealbreaker: the same arithmetic the
        // AI-evaluation mode uses, reached through JobEvaluationService.Compose.
        Assert.Equal(100, evaluation!.OverallMatch);
        Assert.Equal("Strong Match", evaluation.Category);
        Assert.Equal(RagEvaluationAgent.PromptVersion, evaluation.PromptVersion);
    }
}
