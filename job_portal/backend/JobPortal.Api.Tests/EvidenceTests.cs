using JobPortal.Api.Data;
using JobPortal.Api.Services.Matching;

namespace JobPortal.Api.Tests;

/// <summary>
/// The keyword-stuffing problem, in one file.
///
/// A résumé is not a bag of words: where a term appears decides what it proves.
/// "I would like to move into HR" and "HR Generalist, 2019-2024" are the same
/// token to a vector and completely different claims to a person, and the whole
/// point of this layer is that the system agrees with the person.
/// </summary>
public class ResumeEvidenceTests
{
    private static readonly ResumeEvidenceService Service = new();

    /// <summary>A developer who fancies a career change and has done nothing about it.</summary>
    private static PortalResume AspiringHrCandidate() => new()
    {
        Id = 1,
        RawText = """
            Objective
            I am a backend developer and I prefer HR related jobs and recruitment roles.

            Skills
            Java, Spring Boot, PostgreSQL, Docker

            Experience
            Backend Developer, Acme Ltd, 2019-2024
            Built and maintained Spring Boot services backed by PostgreSQL.
            Containerised the deployment pipeline with Docker.

            Education
            BSc Computer Science
            """,
    };

    [Fact]
    public void A_preference_with_nothing_behind_it_is_barely_evidence()
    {
        var evidence = Service.Build(AspiringHrCandidate());
        var hr = evidence.For("HR");

        Assert.Equal(EvidenceTier.Aspirational, hr.Tier);
        Assert.True(hr.Weight < 0.15, $"a stated preference scored {hr.Weight:0.00}");
        Assert.Contains("interest", hr.Explanation);
    }

    [Fact]
    public void Work_experience_is_the_strongest_evidence()
    {
        var evidence = Service.Build(AspiringHrCandidate());
        var spring = evidence.For("Spring Boot");

        Assert.Equal(EvidenceTier.Professional, spring.Tier);
        Assert.True(spring.Weight >= 0.95, $"professional use scored only {spring.Weight:0.00}");
    }

    /// <summary>The ladder the product asked for, in order.</summary>
    [Fact]
    public void Corroboration_across_sections_outranks_a_bare_listing()
    {
        var listedOnly = Service.Build(new PortalResume
        {
            RawText = "Skills\nKubernetes\n\nExperience\nSupport Analyst, 2020-2024\nHandled tickets.",
        }).For("Kubernetes");

        var alsoUsed = Service.Build(new PortalResume
        {
            RawText = "Skills\nKubernetes\n\nExperience\nPlatform Engineer, 2020-2024\n" +
                      "Ran Kubernetes clusters in production.",
        }).For("Kubernetes");

        Assert.Equal(EvidenceTier.Listed, listedOnly.Tier);
        Assert.Equal(EvidenceTier.Professional, alsoUsed.Tier);
        Assert.True(alsoUsed.Weight > listedOnly.Weight);
    }

    [Fact]
    public void Breadth_of_corroboration_adds_on_top_of_depth()
    {
        var deepOnly = Service.Build(new PortalResume
        {
            RawText = "Experience\nEngineer, 2020-2024\nWorked with Terraform daily.",
        }).For("Terraform");

        var deepAndBroad = Service.Build(new PortalResume
        {
            RawText = "Skills\nTerraform\n\nProjects\nBuilt a Terraform module library.\n\n" +
                      "Experience\nEngineer, 2020-2024\nWorked with Terraform daily.",
        }).For("Terraform");

        Assert.Equal(EvidenceTier.Professional, deepOnly.Tier);
        Assert.Equal(EvidenceTier.Professional, deepAndBroad.Tier);
        Assert.True(deepAndBroad.Weight >= deepOnly.Weight);
        Assert.True(deepAndBroad.FoundIn.Count > deepOnly.FoundIn.Count);
    }

    [Fact]
    public void A_skill_named_nowhere_is_no_evidence()
    {
        var evidence = Service.Build(AspiringHrCandidate());
        var missing = evidence.For("Kubernetes");

        Assert.Equal(EvidenceTier.None, missing.Tier);
        Assert.Equal(0, missing.Weight);
        Assert.False(missing.IsEvidenced);
    }

    /// <summary>
    /// The number the fit score is gated by. An HR posting asking for HR things
    /// must come back near zero for this candidate, however much the two documents
    /// look alike as vectors.
    /// </summary>
    [Fact]
    public void An_unevidenced_role_gets_near_zero_coverage()
    {
        var evidence = Service.Build(AspiringHrCandidate());

        var hrRole = evidence.CoverageOf(new[] { "HR", "Recruitment", "Onboarding", "Payroll" });
        var backendRole = evidence.CoverageOf(new[] { "Java", "Spring Boot", "PostgreSQL", "Docker" });

        Assert.True(hrRole < 0.1, $"an unevidenced HR role covered {hrRole:0.00}");
        Assert.True(backendRole > 0.8, $"a fully evidenced backend role only covered {backendRole:0.00}");
        Assert.True(backendRole > hrRole * 5);
    }

    /// <summary>
    /// A skill name inside a longer word is not a mention — "Go" must not be
    /// evidenced by "going", or every résumé would prove every Go role.
    /// </summary>
    [Fact]
    public void Matching_respects_word_boundaries()
    {
        var evidence = Service.Build(new PortalResume
        {
            RawText = "Experience\nEngineer, 2020-2024\nI was going to learn more about algorithms.",
        });

        Assert.Equal(EvidenceTier.None, evidence.For("Go").Tier);
    }

    /// <summary>
    /// A CV with no headings the splitter recognises must not evidence nothing —
    /// that would score every role at the floor and look like a broken matcher.
    /// </summary>
    [Fact]
    public void A_resume_with_no_recognisable_sections_still_counts_as_listed()
    {
        var evidence = Service.Build(new PortalResume
        {
            RawText = "Jane Doe. Ten years building Python services and maintaining Postgres.",
        });

        Assert.Equal(EvidenceTier.Listed, evidence.For("Python").Tier);
        Assert.Equal(EvidenceTier.None, evidence.For("Kubernetes").Tier);
    }
}
