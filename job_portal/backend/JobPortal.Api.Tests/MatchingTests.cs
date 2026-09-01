using JobPortal.Api.Contracts;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Chat;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Vectors;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Tests;

public class VectorMathTests
{
    [Fact]
    public void Bytes_round_trip_exactly()
    {
        var original = new[] { 0.5f, -0.25f, 1f, 0f, 0.125f };
        var restored = VectorMath.FromBytes(VectorMath.ToBytes(original));

        Assert.Equal(original, restored);
    }

    [Fact]
    public void Normalising_yields_unit_length()
    {
        var normalised = VectorMath.Normalize(new[] { 3f, 4f });
        Assert.Equal(1.0, Math.Sqrt(normalised.Sum(v => v * (double)v)), 5);
    }

    /// <summary>An all-zero vector has no direction. Scaling it would divide by
    /// zero; leaving it gives a similarity of 0 against everything, which is the
    /// correct reading of "no signal".</summary>
    [Fact]
    public void A_zero_vector_survives_normalisation()
    {
        var zero = VectorMath.Normalize(new[] { 0f, 0f, 0f });
        Assert.All(zero, v => Assert.Equal(0f, v));
        Assert.Equal(0, VectorMath.Dot(zero, new[] { 1f, 0f, 0f }));
    }
}

public class HashingEmbeddingTests
{
    /// <summary>
    /// Vectors are persisted, so the same text must embed identically in a later
    /// process. string.GetHashCode is randomised per process and would make
    /// yesterday's stored vectors incomparable with today's queries.
    /// </summary>
    [Fact]
    public void The_same_text_embeds_identically_across_instances()
    {
        var a = new HashingEmbeddingService(256).Embed("Senior DevOps Engineer, Kubernetes and Terraform");
        var b = new HashingEmbeddingService(256).Embed("Senior DevOps Engineer, Kubernetes and Terraform");

        Assert.Equal(a, b);
    }

    [Fact]
    public void Related_text_scores_higher_than_unrelated_text()
    {
        var service = new HashingEmbeddingService(512);

        var job = service.Embed("DevOps engineer working with Kubernetes, Terraform and AWS");
        var close = service.Embed("Kubernetes and Terraform on AWS, DevOps automation");
        var far = service.Embed("Pastry chef specialising in laminated dough and viennoiserie");

        Assert.True(VectorMath.Dot(job, close) > VectorMath.Dot(job, far));
    }

    /// <summary>Trigrams are what let the deterministic embedder see through
    /// suffixes; without them "Postgres" and "PostgreSQL" share nothing.</summary>
    [Fact]
    public void Spelling_variants_are_closer_than_unrelated_words()
    {
        var service = new HashingEmbeddingService(512);

        var a = service.Embed("PostgreSQL database administration");
        var b = service.Embed("Postgres database administration");
        var c = service.Embed("Woodworking joinery techniques");

        Assert.True(VectorMath.Dot(a, b) > VectorMath.Dot(a, c));
    }

    [Fact]
    public void Empty_text_yields_a_zero_vector_rather_than_throwing()
    {
        var vector = new HashingEmbeddingService(128).Embed("   ");
        Assert.All(vector, v => Assert.Equal(0f, v));
    }
}

public class SkillSemanticsTests
{
    private static SkillSemanticsService Service(double equivalence = 0.80, double related = 0.68) =>
        // Fully qualified: `JobPortal.Api.Options` (imported above) shadows the
        // Microsoft.Extensions.Options.Options static class.
        new(null!, new SkillVectorCache(), Microsoft.Extensions.Options.Options.Create(new MatchingOptions
        {
            SkillEquivalenceMin = equivalence,
            SkillRelatedMin = related,
        }));

    /// <summary>An exact textual match needs no vector and no threshold, so the
    /// verdict stays right even under the deterministic embedder.</summary>
    [Fact]
    public void An_exact_match_is_Have_without_consulting_any_vector()
    {
        var assessments = Service().Assess(
            required: new[] { "Kubernetes" },
            candidateSkills: new[] { "kubernetes" },
            vectors: new Dictionary<string, float[]>());

        Assert.Equal("Have", assessments[0].Status);
        Assert.Equal(1.0, assessments[0].Similarity);
    }

    [Fact]
    public void Adjacent_skills_are_Transferable_and_name_their_evidence()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase)
        {
            // 0.70 cosine: adjacent, below the 0.80 equivalence floor.
            ["Azure"] = new[] { 1f, 0f },
            ["AWS"] = new[] { 0.7f, 0.714f },
        };

        var assessments = Service().Assess(new[] { "Azure" }, new[] { "AWS" }, vectors);

        Assert.Equal("Transferable", assessments[0].Status);
        Assert.Equal("AWS", assessments[0].EvidenceSkill);
    }

    /// <summary>
    /// A "Missing" verdict must not name a near-miss skill. Showing one reads as a
    /// claim, and the candidate would reasonably ask why it did not count.
    /// </summary>
    [Fact]
    public void A_missing_skill_names_no_evidence()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["Rust"] = new[] { 1f, 0f },
            ["Excel"] = new[] { 0.1f, 0.995f },
        };

        var assessments = Service().Assess(new[] { "Rust" }, new[] { "Excel" }, vectors);

        Assert.Equal("Missing", assessments[0].Status);
        Assert.Null(assessments[0].EvidenceSkill);
    }
}

public class ChatQueryUnderstandingTests
{
    private static JobBoardLexicon Lexicon() => new();

    private readonly ChatQueryUnderstanding _understanding = new();

    [Fact]
    public void A_question_about_a_previous_answer_is_an_explain_not_a_search()
    {
        var parsed = _understanding.Parse("why is the second one only a 60% fit?", new JobFilters(), Lexicon());

        Assert.Equal(ChatIntent.Explain, parsed.Intent);
        Assert.Equal(2, parsed.ReferencedRank);
    }

    [Fact]
    public void A_bare_constraint_is_a_refinement_even_with_no_search_verb()
    {
        var parsed = _understanding.Parse("remote only please", new JobFilters(), Lexicon());

        Assert.Equal(ChatIntent.Refine, parsed.Intent);
        Assert.Equal("Remote", parsed.Filters.WorkMode);
    }

    /// <summary>Filters accumulate across turns, because that is what a person
    /// means when they add a second constraint to the first.</summary>
    [Fact]
    public void Filters_are_additive_across_the_conversation()
    {
        var first = _understanding.Parse("remote roles", new JobFilters(), Lexicon());
        var second = _understanding.Parse("senior level", first.Filters, Lexicon());

        Assert.Equal("Remote", second.Filters.WorkMode);
        Assert.Equal("Senior", second.Filters.SeniorityLevel);
    }

    [Fact]
    public void Restating_a_dimension_replaces_it()
    {
        var first = _understanding.Parse("remote roles", new JobFilters(), Lexicon());
        var second = _understanding.Parse("actually hybrid is fine", first.Filters, Lexicon());

        Assert.Equal("Hybrid", second.Filters.WorkMode);
    }

    [Fact]
    public void Asking_to_start_over_clears_every_filter()
    {
        var narrowed = new JobFilters(WorkMode: "Remote", SeniorityLevel: "Senior");
        var parsed = _understanding.Parse("show me everything", narrowed, Lexicon());

        Assert.True(parsed.ResetFilters);
        Assert.True(parsed.Filters.IsEmpty);
    }

    [Theory]
    [InlineData("paying over $100k", 100000)]
    [InlineData("at least 120,000", 120000)]
    [InlineData("above 90k", 90000)]
    public void Salary_floors_are_read_from_natural_phrasing(string message, decimal expected)
    {
        var parsed = _understanding.Parse(message, new JobFilters(), Lexicon());
        Assert.Equal(expected, parsed.Filters.MinSalary);
    }

    /// <summary>"over 3 years" is not a salary. Treating it as one would silently
    /// remove every job without a stated pay range.</summary>
    [Fact]
    public void A_small_number_is_not_mistaken_for_a_salary()
    {
        var parsed = _understanding.Parse("roles needing over 3 years", new JobFilters(), Lexicon());
        Assert.Null(parsed.Filters.MinSalary);
    }

    /// <summary>
    /// A keyword constraint can only come from a term the board actually uses. An
    /// empty lexicon means nothing in an ordinary sentence can become a hard
    /// filter that silently deletes results.
    /// </summary>
    [Fact]
    public void No_keyword_filter_is_invented_from_ordinary_words()
    {
        var parsed = _understanding.Parse("find me something good near home", new JobFilters(), Lexicon());
        Assert.Null(parsed.Filters.Keywords);
    }
}
