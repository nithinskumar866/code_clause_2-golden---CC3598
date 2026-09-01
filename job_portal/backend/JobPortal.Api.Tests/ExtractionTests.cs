using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Jobs;
using JobPortal.Api.Services.Llm;
using JobPortal.Api.Services.Resumes;
using Microsoft.Extensions.Logging.Abstractions;

namespace JobPortal.Api.Tests;

public class JobExtractionTests
{
    private static JobExtractionService Service() =>
        new(new OfflineLlm(), NullLogger<JobExtractionService>.Instance);

    private const string Posting = """
        Senior DevOps Engineer
        Company: Northwind Systems
        Location: Austin, TX (Hybrid/Remote)
        Employment Type: Full-time

        Requirements
        - 5+ years of DevOps/Cloud Engineering experience.
        - AWS and Azure
        - Kubernetes, Docker, Helm
        - Terraform and Ansible
        - Python / Bash scripting

        Nice to Have
        - Production Kubernetes experience
        - SRE practices
        """;

    /// <summary>
    /// The defect this guards against: a bundled bullet was compared as one
    /// phrase, so a candidate listing all three of Kubernetes, Docker and Helm was
    /// told they were MISSING all three and advised to go and learn them.
    /// </summary>
    [Fact]
    public async Task Bundled_requirements_are_split_into_one_skill_each()
    {
        var job = await Service().ExtractAsync(Posting, "jd.txt");

        Assert.Contains("Kubernetes", job.RequiredSkills);
        Assert.Contains("Docker", job.RequiredSkills);
        Assert.Contains("Helm", job.RequiredSkills);
        Assert.Contains("AWS", job.RequiredSkills);
        Assert.Contains("Azure", job.RequiredSkills);
        Assert.Contains("Terraform", job.RequiredSkills);
        Assert.Contains("Ansible", job.RequiredSkills);

        Assert.DoesNotContain("Kubernetes, Docker, Helm", job.RequiredSkills);
        Assert.DoesNotContain("AWS and Azure", job.RequiredSkills);
    }

    /// <summary>
    /// A duration is not a skill. Left in, it split into "5+ years of DevOps" and
    /// "Cloud Engineering experience" — two entries no resume will ever list, both
    /// scored as gaps against every candidate.
    /// </summary>
    [Fact]
    public async Task Experience_statements_are_not_treated_as_skills()
    {
        var job = await Service().ExtractAsync(Posting, "jd.txt");

        Assert.DoesNotContain(job.RequiredSkills, s => s.Contains("years", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(job.RequiredSkills, s => s.Equals("Cloud Engineering experience", StringComparison.OrdinalIgnoreCase));

        // Not discarded — kept where it belongs, and its number read separately.
        Assert.Contains(job.Qualifications, q => q.Contains("5+ years", StringComparison.OrdinalIgnoreCase));
        Assert.Equal(5, job.MinYearsExperience);
    }

    [Fact]
    public async Task Labelled_fields_are_read_from_the_header()
    {
        var job = await Service().ExtractAsync(Posting, "jd.txt");

        Assert.Equal("Northwind Systems", job.Company);
        Assert.Equal("Austin, TX (Hybrid/Remote)", job.Location);
        Assert.Equal("Full-time", job.EmploymentType);
        Assert.Equal("Senior", job.SeniorityLevel);
    }

    /// <summary>"Hybrid/Remote" is a hybrid role. Testing the looser "remote"
    /// pattern first would classify every hybrid posting as fully remote.</summary>
    [Fact]
    public async Task Hybrid_wins_over_remote_when_both_words_appear()
    {
        var job = await Service().ExtractAsync(Posting, "jd.txt");
        Assert.Equal("Hybrid", job.WorkMode);
    }

    [Fact]
    public async Task Preferred_skills_are_kept_separate_from_required()
    {
        var job = await Service().ExtractAsync(Posting, "jd.txt");

        Assert.Contains(job.PreferredSkills, s => s.Contains("SRE", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(job.RequiredSkills, s => s.Contains("SRE", StringComparison.OrdinalIgnoreCase));
    }

    [Theory]
    [InlineData("3 to 5 years of experience", 3.0, 5.0)]
    [InlineData("Minimum 7 years required", 7.0, null)]
    [InlineData("No duration mentioned at all", null, null)]
    public async Task Years_are_read_as_a_range_or_an_open_minimum(string text, double? min, double? max)
    {
        var job = await Service().ExtractAsync($"Engineer\n\nRequirements\n- {text}", "jd.txt");

        Assert.Equal(min, job.MinYearsExperience);
        // "7+ years" genuinely has no ceiling; inventing one would exclude
        // candidates for being too experienced.
        Assert.Equal(max, job.MaxYearsExperience);
    }
}

public class ResumeProfileTests
{
    private static ResumeProfileService Service() =>
        new(new OfflineLlm(), NullLogger<ResumeProfileService>.Instance);

    private const string Resume = """
        Priya Raman
        priya.raman@example.com | +1 512 555 0134

        Summary
        Cloud infrastructure engineer with 6 years of professional experience.

        Skills
        Cloud: AWS, Terraform
        Containers: Docker, Kubernetes
        Languages: Python, Bash, Go
        Other: Linux Administration, Git

        Experience
        Senior DevOps Engineer
        Northwind Systems | 2021 - Present
        - Migrated 40 services onto Kubernetes.

        Cloud Engineer
        Bluepoint Labs | 2019 - 2021
        - Automated provisioning with Ansible.

        Education
        B.E. Computer Science, 2019
        """;

    /// <summary>
    /// The defect this guards against: "Languages:" was treated as a section
    /// heading, which ended the Skills section early and silently dropped every
    /// skill listed after it. Those skills were then scored as Missing against
    /// jobs that asked for them.
    /// </summary>
    [Fact]
    public async Task Sub_labels_inside_the_skills_block_do_not_end_the_section()
    {
        var profile = await Service().ExtractAsync(Resume, "resume.txt");

        Assert.Contains("Python", profile.Skills);
        Assert.Contains("Bash", profile.Skills);
        Assert.Contains("Go", profile.Skills);
        Assert.Contains("Linux Administration", profile.Skills);
        Assert.Contains("Git", profile.Skills);

        // The ones before the sub-label must still be there too.
        Assert.Contains("AWS", profile.Skills);
        Assert.Contains("Kubernetes", profile.Skills);
    }

    [Fact]
    public async Task Contact_details_are_read_verbatim()
    {
        var profile = await Service().ExtractAsync(Resume, "resume.txt");

        Assert.Equal("Priya Raman", profile.CandidateName);
        Assert.Equal("priya.raman@example.com", profile.Email);
        Assert.Contains("512", profile.Phone);
    }

    [Fact]
    public async Task A_stated_year_count_is_preferred_over_inferring_one()
    {
        var profile = await Service().ExtractAsync(Resume, "resume.txt");
        Assert.Equal(6, profile.YearsExperience);
    }

    /// <summary>
    /// Overlapping roles are one stretch of time. Summing them would credit a
    /// candidate with two careers run in parallel, and inflated experience decides
    /// which roles they are shown.
    /// </summary>
    [Fact]
    public async Task Overlapping_employment_dates_are_unioned_rather_than_summed()
    {
        var resume = """
            Jane Doe

            Experience
            Staff Engineer
            Acme | 2018 - 2024
            - Led platform work.

            Advisor (concurrent)
            Beta Corp | 2020 - 2023
            - Advised on architecture.
            """;

        var profile = await Service().ExtractAsync(resume, "resume.txt");

        // 2018-2024 is six years. Summing both spans would give nine.
        Assert.Equal(6, profile.YearsExperience);
    }

    [Fact]
    public async Task The_most_recent_title_becomes_the_current_one()
    {
        var profile = await Service().ExtractAsync(Resume, "resume.txt");
        Assert.Equal("Senior DevOps Engineer", profile.CurrentTitle);
    }
}

public class TextStructureTests
{
    [Fact]
    public void A_bullet_that_mentions_a_heading_word_is_not_a_heading()
    {
        var text = """
            Skills
            - C#
            - Skills gained while leading the platform team
            """;

        var sections = TextStructure.SplitSections(text, new[] { "Skills" });

        Assert.True(sections.ContainsKey("Skills"));
        Assert.Contains("leading the platform team", sections["Skills"]);
    }

    [Fact]
    public void A_repeated_heading_appends_rather_than_replacing()
    {
        var text = """
            Skills
            - C#

            Experience
            - Built things

            Skills
            - SQL
            """;

        var sections = TextStructure.SplitSections(text, new[] { "Skills", "Experience" });

        Assert.Contains("C#", sections["Skills"]);
        Assert.Contains("SQL", sections["Skills"]);
    }

    [Fact]
    public void Bullets_fall_back_to_plain_lines_when_the_block_uses_none()
    {
        var lines = TextStructure.BulletLines("Owns the CI pipeline\nRuns the on-call rota");
        Assert.Equal(2, lines.Count);
    }

    [Fact]
    public void Clipping_breaks_on_a_word_boundary()
    {
        var clipped = TextStructure.Clip("the quick brown fox jumps over the lazy dog", 20);

        Assert.True(clipped.Length < 45);
        Assert.EndsWith("[...]", clipped);
        Assert.DoesNotContain("jumpe", clipped);
    }
}
