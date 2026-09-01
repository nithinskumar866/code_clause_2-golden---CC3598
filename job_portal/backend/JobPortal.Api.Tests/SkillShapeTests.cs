using JobPortal.Api.Services.Documents;

namespace JobPortal.Api.Tests;

/// <summary>
/// What may be stored as a required SKILL.
///
/// This is the single rule that decides whether a posting's requirements are
/// matchable at all. A sentence or a duration stored as a skill can never be
/// evidenced by any résumé, so every candidate is scored as missing it — and the
/// real technology named inside it earns nobody any credit. Measured on the
/// repository's own postings, that was most of the requirement list.
/// </summary>
public class SkillShapeTests
{
    [Theory]
    [InlineData("Kubernetes")]
    [InlineData("Docker")]
    [InlineData("SQL Server")]
    [InlineData("Amazon Web Services")]
    [InlineData("CI/CD")]
    [InlineData("Go")]
    [InlineData("C#")]
    [InlineData(".NET")]
    public void Real_skill_names_are_kept(string skill) =>
        Assert.True(TextStructure.LooksLikeSkillName(skill), $"'{skill}' should be a skill name");

    [Theory]
    [InlineData("Thorough knowledge of employment laws and HR best practices")]
    [InlineData("Optimize database queries and improve service latency")]
    [InlineData("Lead and mentor a team of data scientists and ML engineers")]
    [InlineData("Excellent communication and stakeholder management skills")]
    [InlineData("Strong understanding of CI/CD pipelines and release automation")]
    public void Responsibility_sentences_are_rejected(string sentence) =>
        Assert.False(TextStructure.LooksLikeSkillName(sentence), $"'{sentence}' is a sentence, not a skill");

    /// <summary>
    /// Durations are read separately and scored on their own dimension. Stored as
    /// a skill, "5+ years of DevOps experience" is a requirement no CV can satisfy
    /// by naming a technology.
    /// </summary>
    [Theory]
    [InlineData("5+ years of DevOps experience")]
    [InlineData("3-5 years experience")]
    [InlineData("7+ years")]
    [InlineData("2 yrs Java")]
    public void Duration_phrases_are_rejected(string duration) =>
        Assert.False(TextStructure.LooksLikeSkillName(duration), $"'{duration}' states a duration");

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("a")]
    [InlineData("---")]
    [InlineData("123")]
    public void Junk_is_rejected(string junk) =>
        Assert.False(TextStructure.LooksLikeSkillName(junk));

    /// <summary>
    /// The markdown bug that titled every posting "# Job Title: HR Generalist" and
    /// left company and location empty, because the label was never seen as a label.
    /// </summary>
    [Theory]
    [InlineData("# Job Title: Senior Platform Engineer", "Job Title: Senior Platform Engineer")]
    [InlineData("## Requirements", "Requirements")]
    [InlineData("**Company:** Northwind", "Company: Northwind")]
    [InlineData("Plain heading", "Plain heading")]
    public void Markdown_decoration_is_stripped(string raw, string expected) =>
        Assert.Equal(expected, TextStructure.Demarkup(raw));
}
