using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Mapping;
using JobPortal.Api.Services.Applications;
using JobPortal.Api.Services.Llm;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;

namespace JobPortal.Api.Tests;

public class CoverLetterTests
{
    private static PortalResume Candidate() => new()
    {
        Id = 1,
        Filename = "sanjay.docx",
        CandidateName = "Sanjay Singh",
        Email = "sanjay.singh@example.com",
        Phone = "+91-9876543030",
        CurrentTitle = "System Administrator",
        YearsExperience = 4,
        Skills = "Docker\nSQL Server Configuration\nDatabase Migrations",
    };

    private static PortalJob Posting() => new()
    {
        Id = 7,
        Title = "DevOps Engineer",
        Company = "Northwind Systems",
        RequiredSkills = "Kubernetes\nDocker\nTerraform",
    };

    /// <summary>Docker is evidenced; Kubernetes and Terraform are not.</summary>
    private static JobMatchDto Match(PortalJob job) => new(
        job.ToSummaryDto(true),
        46.4, "Worth a look", 50, 33, 40, 60,
        new[]
        {
            new SkillAssessmentDto("Docker", "Have", "Docker", 1.0),
            new SkillAssessmentDto("Kubernetes", "Missing", null, 0.2),
            new SkillAssessmentDto("Terraform", "Missing", null, 0.1),
        },
        new[] { "Four years of hands-on infrastructure administration." },
        new[] { "Kubernetes", "Terraform" },
        "Worth a look for an infrastructure-leaning DevOps role.");

    private static CoverLetterService Service(IOllamaClient llm) =>
        new(llm, NullLogger<CoverLetterService>.Instance);

    [Fact]
    public async Task Without_a_model_it_still_writes_a_real_letter()
    {
        var job = Posting();
        var letter = await Service(new OfflineLlm()).WriteAsync(Candidate(), job, Match(job));

        Assert.Equal(CoverLetter.Deterministic, letter.Mode);
        Assert.Contains("DevOps Engineer", letter.Text);
        Assert.Contains("Northwind Systems", letter.Text);
        Assert.Contains("Sanjay Singh", letter.Text);
        // No placeholders ever reach an employer.
        Assert.DoesNotContain("[", letter.Text);
    }

    /// <summary>
    /// The whole point of the feature's safety story: the letter may name what the
    /// candidate can evidence, and nothing else.
    /// </summary>
    [Fact]
    public async Task The_composed_letter_claims_only_evidenced_skills()
    {
        var job = Posting();
        var letter = await Service(new OfflineLlm()).WriteAsync(Candidate(), job, Match(job));

        Assert.Contains("Docker", letter.Text);
        Assert.DoesNotContain("Kubernetes", letter.Text);
        Assert.DoesNotContain("Terraform", letter.Text);
    }

    /// <summary>
    /// An address the model produced can route an employer's reply into a void.
    /// The letter is discarded, and the real details are appended from the parsed
    /// document — an email is either literally in the resume or it is wrong.
    /// </summary>
    [Fact]
    public async Task An_invented_email_address_throws_the_letter_away()
    {
        var job = Posting();
        var invented = "I am a great fit. Reach me at hire.me@totally-real.example any time.";
        var letter = await Service(new ScriptedLlm(invented)).WriteAsync(Candidate(), job, Match(job));

        Assert.Equal(CoverLetter.Deterministic, letter.Mode);
        Assert.DoesNotContain("hire.me@totally-real.example", letter.Text);
        Assert.Contains("sanjay.singh@example.com", letter.Text);
    }

    [Fact]
    public async Task An_invented_phone_number_throws_the_letter_away()
    {
        var job = Posting();
        var invented = "My Docker work speaks for itself. Call me on +1 415 555 0199.";
        var letter = await Service(new ScriptedLlm(invented)).WriteAsync(Candidate(), job, Match(job));

        Assert.Equal(CoverLetter.Deterministic, letter.Mode);
        Assert.DoesNotContain("415 555 0199", letter.Text);
        Assert.Contains("+91-9876543030", letter.Text);
    }

    /// <summary>
    /// Dates are ordinary content in a cover letter. Treating them as phone numbers
    /// would silently downgrade every letter that says when someone held a job.
    /// </summary>
    [Fact]
    public async Task A_date_range_is_not_mistaken_for_a_phone_number()
    {
        var job = Posting();
        var dated = "From 2019 - 2023 I ran Docker estates for a mid-sized platform team.";
        var letter = await Service(new ScriptedLlm(dated)).WriteAsync(Candidate(), job, Match(job));

        Assert.Equal(CoverLetter.Llm, letter.Mode);
    }

    /// <summary>
    /// A letter claiming a skill the candidate was scored as Missing is a lie sent
    /// under their name. It is discarded rather than trimmed, because editing prose
    /// to remove a claim is exactly the comprehension we refuse to rely on.
    /// </summary>
    [Fact]
    public async Task A_letter_claiming_a_missing_skill_is_thrown_away()
    {
        var job = Posting();
        var overclaiming = "I have deep production experience with Kubernetes and Docker at scale.";
        var letter = await Service(new ScriptedLlm(overclaiming)).WriteAsync(Candidate(), job, Match(job));

        Assert.Equal(CoverLetter.Deterministic, letter.Mode);
        Assert.DoesNotContain("Kubernetes", letter.Text);
    }

    /// <summary>Even an honest one, because "keen to learn X" and "experienced in
    /// X" are not separable without reading the prose.</summary>
    [Fact]
    public async Task Willingness_to_learn_a_missing_skill_is_also_refused()
    {
        var job = Posting();
        var honest = "My Docker background is strong and I am keen to learn Terraform.";
        var letter = await Service(new ScriptedLlm(honest)).WriteAsync(Candidate(), job, Match(job));

        Assert.Equal(CoverLetter.Deterministic, letter.Mode);
    }

    [Fact]
    public async Task A_clean_letter_from_the_model_is_kept()
    {
        var job = Posting();
        var clean = "I would like to apply for the DevOps Engineer role. My Docker experience is directly relevant.";
        var letter = await Service(new ScriptedLlm(clean)).WriteAsync(Candidate(), job, Match(job));

        Assert.Equal(CoverLetter.Llm, letter.Mode);
        Assert.Contains("Docker", letter.Text);
        Assert.Contains("Sanjay Singh", letter.Text);
    }

    /// <summary>
    /// A skill name inside a longer word is not a mention. Without the boundary
    /// check, a "Go" requirement would reject every letter containing "going" and
    /// silently downgrade good letters forever.
    /// </summary>
    [Fact]
    public async Task A_skill_name_inside_another_word_is_not_a_claim()
    {
        var job = new PortalJob { Id = 9, Title = "Backend Engineer", RequiredSkills = "Go" };
        var match = new JobMatchDto(
            job.ToSummaryDto(true), 50, "Worth a look", 50, 50, 50, 50,
            new[]
            {
                new SkillAssessmentDto("Docker", "Have", "Docker", 1.0),
                new SkillAssessmentDto("Go", "Missing", null, 0.1),
            },
            Array.Empty<string>(), new[] { "Go" }, "");

        var reply = "I am going to be a strong fit, and my Docker work shows it.";
        var letter = await Service(new ScriptedLlm(reply)).WriteAsync(Candidate(), job, match);

        Assert.Equal(CoverLetter.Llm, letter.Mode);
    }

    [Fact]
    public async Task With_no_overlap_it_says_so_rather_than_inventing_one()
    {
        var job = Posting();
        var match = new JobMatchDto(
            job.ToSummaryDto(true), 12, "Weak", 12, 0, 10, 20,
            new[] { new SkillAssessmentDto("Kubernetes", "Missing", null, 0.1) },
            Array.Empty<string>(), new[] { "Kubernetes" }, "");

        var letter = await Service(new OfflineLlm()).WriteAsync(Candidate(), job, match);

        Assert.Contains("broader experience", letter.Text);
        Assert.DoesNotContain("Kubernetes", letter.Text);
    }
}

/// <summary>
/// The duplicate guard, exercised against the real DDL rather than a mock.
///
/// A service-level "have they applied already?" check cannot be the guarantee:
/// two clicks race, and a candidate who re-uploads the same CV gets a new
/// portal_resumes row and applies again in good faith. The unique index on
/// (JobId, ResumeContentHash) is what actually holds, so that is what is tested.
/// </summary>
public class ApplicationSchemaTests : IAsyncLifetime
{
    private string _dbPath = "";
    private PortalDbContext _db = null!;

    public async Task InitializeAsync()
    {
        // A real file rather than :memory: — the initializer opens and closes the
        // connection, and an in-memory database is destroyed when it closes.
        _dbPath = Path.Combine(Path.GetTempPath(), $"portal-apps-{Guid.NewGuid():N}.db");

        var options = new DbContextOptionsBuilder<PortalDbContext>()
            .UseSqlite($"Data Source={_dbPath}")
            .Options;

        _db = new PortalDbContext(options);
        await PortalSchemaInitializer.InitializeAsync(_db, NullLogger.Instance);

        _db.Jobs.Add(new PortalJob { Id = 1, Title = "DevOps Engineer" });
        _db.Jobs.Add(new PortalJob { Id = 2, Title = "Platform Engineer" });
        _db.Resumes.Add(new PortalResume { Id = 1, Filename = "a.docx", ContentHash = "hash-a" });
        // The SAME document, uploaded again under a different name.
        _db.Resumes.Add(new PortalResume { Id = 2, Filename = "a-copy.docx", ContentHash = "hash-a" });
        await _db.SaveChangesAsync();
    }

    public async Task DisposeAsync()
    {
        await _db.DisposeAsync();
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        if (File.Exists(_dbPath)) File.Delete(_dbPath);
    }

    private static PortalApplication Application(int jobId, int resumeId, string hash) => new()
    {
        JobId = jobId,
        ResumeId = resumeId,
        ResumeContentHash = hash,
        Status = ApplicationStatus.Submitted,
        FitScore = 46.4,
        FitBand = "Worth a look",
    };

    [Fact]
    public async Task An_application_is_stored()
    {
        _db.Applications.Add(Application(1, 1, "hash-a"));
        await _db.SaveChangesAsync();

        Assert.Equal(1, await _db.Applications.CountAsync());
    }

    [Fact]
    public async Task The_same_cv_cannot_apply_to_one_posting_twice()
    {
        _db.Applications.Add(Application(1, 1, "hash-a"));
        await _db.SaveChangesAsync();

        _db.Applications.Add(Application(1, 1, "hash-a"));
        await Assert.ThrowsAsync<DbUpdateException>(() => _db.SaveChangesAsync());
    }

    /// <summary>The point of keying on the document rather than the row.</summary>
    [Fact]
    public async Task Re_uploading_the_cv_does_not_permit_a_second_application()
    {
        _db.Applications.Add(Application(1, 1, "hash-a"));
        await _db.SaveChangesAsync();

        // Different portal_resumes row, same document.
        _db.Applications.Add(Application(1, 2, "hash-a"));
        await Assert.ThrowsAsync<DbUpdateException>(() => _db.SaveChangesAsync());
    }

    [Fact]
    public async Task The_same_cv_may_apply_to_different_postings()
    {
        _db.Applications.Add(Application(1, 1, "hash-a"));
        _db.Applications.Add(Application(2, 1, "hash-a"));
        await _db.SaveChangesAsync();

        Assert.Equal(2, await _db.Applications.CountAsync());
    }

    [Fact]
    public async Task Deleting_a_posting_takes_its_applications_with_it()
    {
        _db.Applications.Add(Application(1, 1, "hash-a"));
        await _db.SaveChangesAsync();

        _db.Jobs.Remove(await _db.Jobs.FirstAsync(j => j.Id == 1));
        await _db.SaveChangesAsync();

        Assert.Equal(0, await _db.Applications.CountAsync());
    }

    /// <summary>The recruiter platform's tables must stay untouched by all of this.</summary>
    [Fact]
    public async Task Only_portal_tables_are_created()
    {
        var connection = (Microsoft.Data.Sqlite.SqliteConnection)_db.Database.GetDbConnection();
        await connection.OpenAsync();

        await using var command = connection.CreateCommand();
        command.CommandText = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'";

        var tables = new List<string>();
        await using (var reader = await command.ExecuteReaderAsync())
        {
            while (await reader.ReadAsync()) tables.Add(reader.GetString(0));
        }
        await connection.CloseAsync();

        Assert.NotEmpty(tables);
        Assert.All(tables, table => Assert.StartsWith("portal_", table));
    }
}
