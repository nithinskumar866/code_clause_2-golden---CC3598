using JobPortal.Api.Data;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Screening;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Tests;

/// <summary>
/// The pre-application gate.
///
/// Two properties matter more than the wording. First, a question must come from
/// the POSTING — a gate that invents its own requirements rejects people for
/// failing a test nobody set, and that is worse than no gate. Second, it has to
/// work with the model switched off, or it cannot be depended on to gate anything.
/// </summary>
public class ApplicationScreeningTests
{
    private static PortalJob SpringBootPosting() => new()
    {
        Id = 5,
        Title = "Backend Developer",
        Company = "Acme",
        Location = "Berlin",
        WorkMode = "Onsite",
        EmploymentType = "Full-time",
        SeniorityLevel = "Mid",
        MinYearsExperience = 3,
        RequiredSkills = "Spring Boot\nJava\nPostgreSQL",
        RawText = "Backend Developer in Berlin. Onsite, full-time. You will build services "
                + "with Spring Boot and Java against PostgreSQL. 3+ years required.",
        UpdatedAt = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc),
    };

    private static PortalJob RemoteDataSciencePosting() => new()
    {
        Id = 6,
        Title = "Data Scientist",
        WorkMode = "Remote",
        EmploymentType = "Contract",
        RequiredSkills = "Python\nmachine learning",
        RawText = "Remote Data Scientist, contract. Python and machine learning.",
        UpdatedAt = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc),
    };

    /// <summary>Two contacts, as configured in appsettings.</summary>
    private static IOptions<ScreeningOptions> Contacts() => Microsoft.Extensions.Options.Options.Create(new ScreeningOptions
    {
        Contacts =
        {
            new ScreeningContactOption
            {
                WorkMode = "Remote", Name = "John Nacks",
                Url = "https://in.linkedin.com/in/john-nacks-990195228", Handles = "remote roles",
            },
            new ScreeningContactOption
            {
                WorkMode = "Onsite", Name = "Abinaya Jagdish",
                Url = "https://in.linkedin.com/in/abinaya-jagdish-466822210",
                Handles = "onsite and hybrid roles",
            },
        },
    });

    private static ApplicationScreeningService Service(
        string? reply, IOptions<ScreeningOptions>? contacts = null) =>
        new(new ScriptedLlm(reply), new MemoryCache(new MemoryCacheOptions()),
            contacts ?? Contacts(), NullLogger<ApplicationScreeningService>.Instance);

    private static ApplicationScreeningService Offline(IOptions<ScreeningOptions>? contacts = null) =>
        new(new OfflineLlm(), new MemoryCache(new MemoryCacheOptions()),
            contacts ?? Contacts(), NullLogger<ApplicationScreeningService>.Instance);

    // -- with no model at all ------------------------------------------------

    [Fact]
    public async Task The_gate_works_with_the_model_switched_off()
    {
        var set = await Offline().BuildAsync(SpringBootPosting());

        Assert.False(set.Generated);
        Assert.Equal(4, set.Questions.Count);

        // Distinct topics: four questions about four skills is a quiz, and it leaves
        // location and work mode unasked.
        Assert.Equal(4, set.Questions.Select(q => q.Topic).Distinct().Count());
    }

    [Fact]
    public async Task A_located_posting_asks_about_relocation()
    {
        var set = await Offline().BuildAsync(SpringBootPosting());
        var location = set.Questions.Single(q => q.Topic == ScreeningTopics.Location);

        Assert.Contains("Berlin", location.Question);
        Assert.Contains("relocate", location.Question, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task A_remote_posting_asks_about_remote_instead_of_relocation()
    {
        var set = await Offline().BuildAsync(RemoteDataSciencePosting());

        Assert.DoesNotContain(set.Questions, q => q.Topic == ScreeningTopics.Location);
        Assert.Contains(set.Questions, q =>
            q.Topic == ScreeningTopics.WorkMode &&
            q.Question.Contains("remote", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task A_posting_with_almost_nothing_still_produces_a_gate()
    {
        var bare = new PortalJob { Id = 9, Title = "Analyst", UpdatedAt = DateTime.UtcNow };

        var set = await Offline().BuildAsync(bare);

        Assert.NotEmpty(set.Questions);
        Assert.Contains("Analyst", set.Questions[0].Question);
    }

    // -- with a model --------------------------------------------------------

    [Fact]
    public async Task Generated_questions_are_used_when_they_trace_to_the_posting()
    {
        var set = await Service("""
            {"questions":[
              {"question":"This role is based in Berlin. Are you able to work there?","topic":"location","basis":"Berlin"},
              {"question":"Do you have hands-on experience with Java?","topic":"skill","basis":"Java"},
              {"question":"Do you have at least 3 years of backend experience?","topic":"experience","basis":"3+ years required"},
              {"question":"This is a full-time onsite role. Does that suit you?","topic":"work_mode","basis":"Onsite"}
            ]}
            """).BuildAsync(SpringBootPosting());

        Assert.True(set.Generated);
        Assert.Equal(4, set.Questions.Count);

        // The Spring Boot rule: ask about the language under the framework.
        Assert.Contains(set.Questions, q => q.Question.Contains("Java"));
        Assert.Equal(new[] { 1, 2, 3, 4 }, set.Questions.Select(q => q.Id));
    }

    /// <summary>
    /// The property that makes this safe to put in front of an applicant.
    ///
    /// A question about a requirement the employer never set would block people over
    /// a test nobody wrote. It is dropped, and a real question takes its place.
    /// </summary>
    [Fact]
    public async Task A_question_about_something_the_posting_never_mentions_is_dropped()
    {
        var set = await Service("""
            {"questions":[
              {"question":"Do you have hands-on experience with Java?","topic":"skill","basis":"Java"},
              {"question":"Do you hold an active AWS Solutions Architect certification?","topic":"skill","basis":"AWS"},
              {"question":"Are you willing to travel to Singapore each quarter?","topic":"location","basis":"Singapore"}
            ]}
            """).BuildAsync(SpringBootPosting());

        Assert.DoesNotContain(set.Questions, q => q.Question.Contains("AWS"));
        Assert.DoesNotContain(set.Questions, q => q.Question.Contains("Singapore"));
        Assert.Contains(set.Questions, q => q.Question.Contains("Java"));

        // Topped up from the deterministic set rather than returning a short gate.
        Assert.Equal(4, set.Questions.Count);
    }

    [Fact]
    public async Task Unusable_output_falls_back_rather_than_asking_nothing()
    {
        foreach (var reply in new[] { "not json at all", """{"questions":[]}""", null })
        {
            var set = await Service(reply).BuildAsync(SpringBootPosting());
            Assert.False(set.Generated);
            Assert.Equal(4, set.Questions.Count);
        }
    }

    // -- who to ask ----------------------------------------------------------

    /// <summary>
    /// The posting already says which contact matters, so the applicant should not
    /// have to work it out. "Remote queries go to John" is noise on an onsite job.
    /// </summary>
    [Fact]
    public async Task An_onsite_posting_leads_with_the_onsite_contact()
    {
        var set = await Offline().BuildAsync(SpringBootPosting());

        Assert.Equal(2, set.Contacts.Count);
        Assert.Equal("Abinaya Jagdish", set.Contacts[0].Name);
        Assert.True(set.Contacts[0].Relevant);

        // The other stays available — someone may want to ask about something else.
        Assert.False(set.Contacts[1].Relevant);
    }

    [Fact]
    public async Task A_remote_posting_leads_with_the_remote_contact()
    {
        var set = await Offline().BuildAsync(RemoteDataSciencePosting());

        Assert.Equal("John Nacks", set.Contacts[0].Name);
        Assert.True(set.Contacts[0].Relevant);
    }

    /// <summary>Hybrid means being in the office some days, so its questions are the
    /// onsite ones — which days, how far, is the commute workable.</summary>
    [Fact]
    public async Task A_hybrid_posting_routes_to_the_onsite_contact()
    {
        var job = SpringBootPosting();
        job.WorkMode = "Hybrid";

        var set = await Offline().BuildAsync(job);

        Assert.Equal("Abinaya Jagdish", set.Contacts[0].Name);
        Assert.True(set.Contacts[0].Relevant);
    }

    /// <summary>Nothing to route on: both offered, neither pushed forward.</summary>
    [Fact]
    public async Task An_unstated_work_mode_marks_neither_contact_relevant()
    {
        var job = SpringBootPosting();
        job.WorkMode = "Unspecified";

        var set = await Offline().BuildAsync(job);

        Assert.Equal(2, set.Contacts.Count);
        Assert.All(set.Contacts, c => Assert.False(c.Relevant));
    }

    /// <summary>
    /// A config file is not a trusted input just because it lives on the server. A
    /// typo putting a javascript: URL here would otherwise become a link the app
    /// renders and someone clicks.
    /// </summary>
    [Fact]
    public async Task A_contact_whose_url_is_not_http_is_dropped()
    {
        var dodgy = Microsoft.Extensions.Options.Options.Create(new ScreeningOptions
        {
            Contacts =
            {
                new ScreeningContactOption { WorkMode = "Remote", Name = "Real", Url = "https://example.com/x" },
                new ScreeningContactOption { WorkMode = "Onsite", Name = "Bad", Url = "javascript:alert(1)" },
                new ScreeningContactOption { WorkMode = "Onsite", Name = "Empty", Url = "" },
            },
        });

        var set = await Offline(dodgy).BuildAsync(RemoteDataSciencePosting());

        var only = Assert.Single(set.Contacts);
        Assert.Equal("Real", only.Name);
    }

    [Fact]
    public async Task No_configured_contacts_is_not_an_error()
    {
        var set = await Offline(Microsoft.Extensions.Options.Options.Create(new ScreeningOptions())).BuildAsync(SpringBootPosting());

        Assert.Empty(set.Contacts);
        Assert.Equal(4, set.Questions.Count);
    }

    // -- the gate ------------------------------------------------------------

    [Fact]
    public async Task All_yes_passes()
    {
        var service = Offline();
        var job = SpringBootPosting();
        var set = await service.BuildAsync(job);

        var result = await service.EvaluateAsync(
            job, set.Questions.Select(q => new ScreeningAnswer(q.Id, true)).ToList());

        Assert.True(result.Passed);
        Assert.Empty(result.Declined);
    }

    /// <summary>
    /// A "no" is recorded and reported, and does not block.
    ///
    /// Answering is the gate; the answers are not. Someone who reads "Berlin,
    /// onsite", says they will not relocate and applies anyway has made a decision
    /// that is theirs — and the recruiter now knows something no CV would have said.
    /// </summary>
    [Fact]
    public async Task A_no_is_recorded_and_reported_but_does_not_block()
    {
        var service = Offline();
        var job = SpringBootPosting();
        var set = await service.BuildAsync(job);

        var answers = set.Questions
            .Select(q => new ScreeningAnswer(q.Id, q.Topic != ScreeningTopics.Location))
            .ToList();

        var result = await service.EvaluateAsync(job, answers);

        Assert.True(result.Passed);

        var declined = Assert.Single(result.Declined);
        Assert.Equal(ScreeningTopics.Location, declined.Topic);
        Assert.Contains("shared with the employer", result.Message);
    }

    [Fact]
    public async Task Every_answer_no_still_passes()
    {
        var service = Offline();
        var job = SpringBootPosting();
        var set = await service.BuildAsync(job);

        var result = await service.EvaluateAsync(
            job, set.Questions.Select(q => new ScreeningAnswer(q.Id, false)).ToList());

        Assert.True(result.Passed);
        Assert.Equal(set.Questions.Count, result.Declined.Count);
    }

    [Fact]
    public async Task An_unanswered_question_blocks_without_being_a_refusal()
    {
        var service = Offline();
        var job = SpringBootPosting();
        var set = await service.BuildAsync(job);

        var result = await service.EvaluateAsync(
            job, set.Questions.Take(2).Select(q => new ScreeningAnswer(q.Id, true)).ToList());

        Assert.False(result.Passed);
        Assert.Empty(result.Declined);
        Assert.Equal(2, result.Unanswered.Count);
    }

    /// <summary>
    /// Answers are matched against the questions the service produces, never against
    /// the client's account of them — otherwise passing is a matter of inventing
    /// four ids and answering them all yes.
    /// </summary>
    [Fact]
    public async Task Answers_to_questions_that_were_never_asked_do_not_pass_the_gate()
    {
        var service = Offline();
        var job = SpringBootPosting();

        var result = await service.EvaluateAsync(job, new[]
        {
            new ScreeningAnswer(99, true), new ScreeningAnswer(98, true),
            new ScreeningAnswer(97, true), new ScreeningAnswer(96, true),
        });

        Assert.False(result.Passed);
        Assert.Equal(4, result.Unanswered.Count);
    }

    [Fact]
    public async Task No_answers_at_all_blocks()
    {
        var service = Offline();
        var result = await service.EvaluateAsync(SpringBootPosting(), Array.Empty<ScreeningAnswer>());

        Assert.False(result.Passed);
        Assert.Contains("Answer the screening questions", result.Message);
    }
}
