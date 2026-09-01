using JobPortal.Api.Data;
using JobPortal.Api.Services.Chat;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace JobPortal.Api.Tests;

/// <summary>
/// Composing an answer, and refusing to ship one that left the facts behind.
///
/// The composer is the only place a model's prose reaches the candidate, so the tests
/// that matter are the ones where the model misbehaves.
/// </summary>
public class AnswerComposerTests
{
    private const string Fallback = "The posting asks for Python, Pandas and SQL.";

    private static PortalJob Job(string raw = "") => new()
    {
        Id = 1,
        Title = "Data Scientist",
        Company = "Acme Corp",
        Location = "Chennai",
        WorkMode = "Remote",
        MinYearsExperience = 3,
        MaxYearsExperience = 5,
        SalaryMin = 1200000,
        SalaryCurrency = "₹",
        RequiredSkills = "Python\nPandas\nSQL",
        RawText = raw,
    };

    private static FactPackage Package(PortalJob? job = null)
    {
        var j = job ?? Job();
        return new FactPackage(
            new[] { JobFact.From(j) },
            new Dictionary<int, CompanyFacts> { [j.Id] = CompanyFacts.From(j) });
    }

    private static AnswerComposer Composer(string? reply) => new(
        reply is null ? new OfflineLlm() : new ScriptedLlm(reply),
        NullLogger<AnswerComposer>.Instance);

    [Fact]
    public async Task With_no_model_the_deterministic_reply_ships()
    {
        var result = await Composer(null).ComposeAsync("is it remote?", Package(), Fallback);
        Assert.False(result.Generated);
        Assert.Equal(Fallback, result.Text);
    }

    [Fact]
    public async Task A_grounded_answer_is_kept()
    {
        var result = await Composer("Yes — it's remote, based out of Chennai, and asks for 3-5 years.")
            .ComposeAsync("is it remote?", Package(), Fallback);

        Assert.True(result.Generated);
        Assert.Contains("remote", result.Text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task An_invented_figure_is_overruled()
    {
        // The failure the whole design exists to prevent: a number quoted at a
        // candidate that no posting ever stated.
        var result = await Composer("It pays about 4500000 and needs 12 years of experience.")
            .ComposeAsync("what does it pay?", Package(), Fallback);

        Assert.False(result.Generated);
        Assert.Equal(Fallback, result.Text);
    }

    [Fact]
    public async Task Figures_that_come_from_the_posting_are_allowed()
    {
        var result = await Composer("The range starts at ₹1200000 and it wants 3-5 years.")
            .ComposeAsync("what does it pay?", Package(), Fallback);
        Assert.True(result.Generated);
    }

    [Fact]
    public async Task A_long_company_answer_is_overruled_when_the_jd_is_silent()
    {
        // "Acme Corp" is a real-looking name; a model will happily describe it.
        var invented = string.Join(" ", Enumerable.Repeat(
            "Acme is a well-established firm known for excellent work-life balance and generous leave.", 6));

        var result = await Composer(invented).ComposeAsync(
            "what's their leave policy?", Package(), Fallback, isCompanyQuestion: true);

        Assert.False(result.Generated);
    }

    [Fact]
    public async Task A_company_answer_is_kept_when_the_jd_does_describe_the_employer()
    {
        var job = Job("""
            ## About the Company
            Acme Corp builds pricing software and offers a four-day week in August.
            """);

        var result = await Composer("They build pricing software and offer a four-day week in August.")
            .ComposeAsync("tell me about them", Package(job), Fallback, isCompanyQuestion: true);

        Assert.True(result.Generated);
    }

    [Fact]
    public async Task An_empty_reply_falls_back()
    {
        var result = await Composer("   ").ComposeAsync("is it remote?", Package(), Fallback);
        Assert.False(result.Generated);
    }

    // --- Deterministic fallbacks are real answers, not apologies ------------------
    [Fact]
    public void DescribeJobs_states_the_postings_actual_values()
    {
        var text = AnswerComposer.DescribeJobs(Package());
        Assert.Contains("Data Scientist", text);
        Assert.Contains("Chennai", text);
        Assert.Contains("Python, Pandas, SQL", text);
    }

    [Fact]
    public void DescribeCompany_says_so_when_the_posting_is_silent()
    {
        var text = AnswerComposer.DescribeCompany(Package());
        Assert.Contains("doesn't say", text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DescribeCompany_quotes_the_posting_when_it_does_speak()
    {
        var job = Job("""
            ## Benefits
            26 days of annual leave.
            """);
        var text = AnswerComposer.DescribeCompany(Package(job));
        Assert.Contains("26 days", text);
    }

    [Fact]
    public void DescribeGaps_prefers_the_computed_ranking()
    {
        var package = new FactPackage(
            new[] { JobFact.From(Job()) },
            notes: new Dictionary<string, string> { ["TO CLOSE"] = "Kubernetes, Spark" });

        Assert.Contains("Kubernetes, Spark", AnswerComposer.DescribeGaps(package));
    }
}
