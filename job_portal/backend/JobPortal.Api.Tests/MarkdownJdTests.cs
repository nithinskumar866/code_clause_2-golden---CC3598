using JobPortal.Api.Services.Documents;
using Xunit;

namespace JobPortal.Api.Tests;

/// <summary>
/// Reading a JD written as markdown.
///
/// Regression cover for the defect that put "# Job Title: HR Generalist" into every
/// row of the match table: a leading hash stopped the line matching the labelled-field
/// pattern, so the title fell through to the raw-first-line fallback and Company and
/// Location came back empty.
/// </summary>
public class MarkdownJdTests
{
    private const string MarkdownJd = """
        # Job Title: HR Generalist
        ## Company: Acme Corp
        **Location:** Chennai
        Employment Type: Full-time

        ## Requirements
        - 3-5 years of experience in an HR Generalist or similar role
        - Thorough knowledge of employment laws
        """;

    [Fact]
    public void Heading_markers_do_not_hide_a_labelled_field()
    {
        var fields = TextStructure.LabelledFields(MarkdownJd);

        Assert.Equal("HR Generalist", TextStructure.FirstField(fields, "Job Title", "Title"));
        Assert.Equal("Acme Corp", TextStructure.FirstField(fields, "Company"));
        Assert.Equal("Chennai", TextStructure.FirstField(fields, "Location"));
    }

    [Fact]
    public void A_plain_labelled_field_still_parses()
    {
        var fields = TextStructure.LabelledFields("Job Title: Backend Engineer\nCompany: Globex");
        Assert.Equal("Backend Engineer", TextStructure.FirstField(fields, "Job Title"));
        Assert.Equal("Globex", TextStructure.FirstField(fields, "Company"));
    }

    [Fact]
    public void Bullets_are_still_not_treated_as_header_fields()
    {
        // "- Thorough knowledge of employment laws" must not become a field named
        // "Thorough knowledge of employment laws" — bullets are requirements.
        var fields = TextStructure.LabelledFields(MarkdownJd);
        Assert.DoesNotContain(fields.Keys, k => k.Contains("Thorough", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void The_title_fallback_does_not_return_markdown_decoration()
    {
        var title = TextStructure.FirstMeaningfulLine("# Senior Platform Engineer\n\nAbout us...");
        Assert.Equal("Senior Platform Engineer", title);
    }

    [Theory]
    [InlineData("# Job Title: X", "Job Title: X")]
    [InlineData("### Company: Y", "Company: Y")]
    [InlineData("**Location:** Z", "Location: Z")]
    [InlineData("Plain: value", "Plain: value")]
    public void Demarkup_strips_decoration_and_leaves_content(string input, string expected)
    {
        Assert.Equal(expected, TextStructure.Demarkup(input));
    }
}
