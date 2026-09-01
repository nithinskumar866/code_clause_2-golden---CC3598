using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Matching;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging.Abstractions;

namespace JobPortal.Api.Tests;

/// <summary>
/// The reasoned scorer: who decides what, and why the split is drawn where it is.
///
/// The model judges — it atomises the posting, classifies each requirement, finds
/// the quote, and rules STRONG / WEAK / MISSING. Everything after that is
/// arithmetic done here, because a small model is competent at "is this proven in
/// work history?" and unreliable at "what is all of that worth out of a hundred".
///
/// Two failure modes are what these tests exist to hold shut. Let the document
/// veto the judgement and the evaluation collapses into keyword matching, which is
/// what it replaced. Let a posting's RESPONSIBILITIES score like mandatory skills
/// and every candidate is marked down for not having pre-performed the job — which
/// is exactly how a recruiter with seven years of full-cycle experience scored 65%
/// against a recruiting role.
/// </summary>
public class ReasonedEvaluationTests
{
    /// <summary>An Angular developer. Note the words that are NOT in this document:
    /// "React", "component-based", "frontend framework".</summary>
    private static PortalResume AngularDeveloper() => new()
    {
        Id = 7,
        RawText = """
            Summary
            Frontend engineer with 3+ years of experience.

            Skills
            Angular, TypeScript, RxJS, NgRx, HTML5, SCSS, Git

            Experience
            Angular Developer, Digit Insurance, 2021-2024
            Optimized UI/UX by incorporating Figma designs into responsive Angular components.
            Streamlined UX based on user feedback and Figma prototypes.

            Education
            BE Computer Science
            """,
    };

    /// <summary>
    /// A posting with NO structured required-skills list.
    ///
    /// The default for tests about one mechanism, because a posting that declares
    /// required skills also triggers the backfill and the required-skill
    /// dealbreaker — both correct, both irrelevant to whatever else is being
    /// measured, and both loud enough to drown it.
    /// </summary>
    private static PortalJob UxDesignerPosting() => new()
    {
        Id = 42,
        Title = "UX/UI Designer",
        RawText = "We are hiring a UX/UI Designer. You will create wireframes, storyboards, "
                + "user flows and site maps, and conduct user research. Figma proficiency required.",
        RequiredSkills = "",
        UpdatedAt = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc),
    };

    /// <summary>The same posting, declaring three required skills.</summary>
    private static PortalJob PostingRequiringSkills()
    {
        var job = UxDesignerPosting();
        job.RequiredSkills = "Figma\nwireframes\nuser research";
        return job;
    }

    private static JobEvaluationService Evaluator(string? reply) =>
        new(new ScriptedLlm(reply),
            new ResumeEvidenceService(),
            new MemoryCache(new MemoryCacheOptions()),
            NullLogger<JobEvaluationService>.Instance);

    private const string NoKnockout =
        """{"missing_must_have_tools": false, "missing_required_education_or_cert": false}""";

    /// <summary>A posting with a hard years minimum, for the arithmetic tests.</summary>
    private static PortalJob PostingRequiring(double years)
    {
        var job = UxDesignerPosting();
        job.MinYearsExperience = years;
        return job;
    }

    /// <summary>Builds the model's JSON so a test states only what it is about.</summary>
    private static string Reply(
        string requirements,
        string weights = """[{"criterion": "core_skill", "weight": 100}]""",
        string knockout = NoKnockout,
        string years = "null") => $$"""
        {
          "requirements_evaluation": [{{requirements}}],
          "candidate_years_experience": {{years}},
          "weights": {{weights}},
          "knockout_indicators": {{knockout}},
          "justification_summary": "the candidate is a wireframing and user research expert",
          "alternate_role": "Frontend Engineer"
        }
        """;

    private static string Req(string requirement, string quote, string level, string kind) =>
        $$"""{"jd_requirement": "{{requirement}}", "resume_quote": "{{quote}}", "match_level": "{{level}}", "kind": "{{kind}}", "reasoning": "as judged"}""";

    // -- the score is counted, not stated -----------------------------------

    [Fact]
    public async Task The_score_is_computed_from_the_verdicts_not_taken_from_the_model()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "Optimized UI/UX by incorporating Figma designs into responsive Angular components", "STRONG", "core_skill"),
                Req("Figma", "Streamlined UX based on user feedback and Figma prototypes", "STRONG", "core_skill"),
                Req("TypeScript", "Angular, TypeScript, RxJS, NgRx, HTML5, SCSS, Git", "WEAK", "core_skill"),
                // Deliberately NOT one of the posting's RequiredSkills — this test is
                // about the arithmetic, and a required skill would cap it instead.
                Req("storyboards", "NONE", "MISSING", "core_skill"))))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);

        // One kind at 100%: (1.0 + 1.0 + 0.5 + 0.0) / 4 = 62.5 -> 63.
        Assert.Equal(63, evaluation!.OverallMatch);
        Assert.Equal("Moderate Match", evaluation.Category);

        // The auditor prompt forbids the model from producing any number, so there
        // is nothing to compare against and the card must not imply a 0%.
        Assert.Equal(0, evaluation.ModelMatch);
    }

    /// <summary>
    /// The hallucination the summary used to carry.
    ///
    /// The model was asked for a free paragraph and wrote that a candidate was a
    /// "wireframing and user research expert" — crediting them with the two things
    /// its own rows had marked MISSING. A recruiter reads the summary and skips the
    /// evidence, so that is the worst place on the card for an invention. Neither
    /// the summary nor the explanation is written by the model any more.
    /// </summary>
    [Fact]
    public async Task The_summary_cannot_credit_the_candidate_with_a_missing_requirement()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"),
                Req("wireframes", "None", "MISSING", "core_skill"),
                Req("user research", "None", "MISSING", "core_skill"))))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);

        // The model's sentence claimed exactly this. It is not what gets rendered.
        Assert.DoesNotContain("expert", evaluation!.ExecutiveSummary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("wireframes", evaluation.ExecutiveSummary);
        Assert.Contains("Not evidenced", evaluation.ExecutiveSummary);

        // The composed sentences are assembled from the rows, so they cannot assert
        // anything the rows do not.
        Assert.Contains("Angular", evaluation.ExecutiveSummary);
        Assert.Contains("%", evaluation.Reasoning);
    }

    /// <summary>
    /// The regression this whole revision is about.
    ///
    /// A posting's duties are not a checklist the applicant must already have
    /// ticked. Scored like mandatory skills they dominate by sheer count, and a
    /// candidate who plainly can do the job is marked down for not having listed
    /// every part of it.
    /// </summary>
    [Fact]
    public async Task Missing_responsibilities_cannot_sink_a_candidate_who_has_the_skills()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "Optimized UI/UX by incorporating Figma designs into responsive Angular components", "STRONG", "core_skill"),
                Req("Figma", "Streamlined UX based on user feedback and Figma prototypes", "STRONG", "core_skill"),
                Req("3+ years experience", "Frontend engineer with 3+ years of experience", "STRONG", "experience"),
                Req("negotiate offers", "None", "MISSING", "responsibility"),
                Req("run design workshops", "None", "MISSING", "responsibility"),
                Req("write status reports", "None", "MISSING", "responsibility")),
            // The model tries to give the duties most of the weight. It may not.
            weights: """[{"criterion": "core_skill", "weight": 25}, {"criterion": "experience", "weight": 15}, {"criterion": "responsibility", "weight": 60}]"""))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);

        // Responsibilities are capped at 15% of the total and score zero; the other
        // 85% is fully evidenced. Half the requirements are missing and the
        // candidate still — correctly — clears a strong band.
        Assert.Equal(85, evaluation!.OverallMatch);

        var responsibility = evaluation.Weights.Single(w => w.Criterion == RequirementKinds.Responsibility);
        Assert.Equal(15, responsibility.Weight);
    }

    [Fact]
    public async Task A_kind_is_scored_against_its_own_requirements_so_one_must_have_cannot_be_buried()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Figma", "None", "MISSING", "must_have"),
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"),
                Req("TypeScript", "Angular, TypeScript, RxJS", "STRONG", "core_skill"),
                Req("SCSS", "HTML5, SCSS, Git", "STRONG", "core_skill")),
            weights: """[{"criterion": "must_have", "weight": 40}, {"criterion": "core_skill", "weight": 60}]"""))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);

        // The three core skills do not dilute the one must-have: it is measured
        // against itself, scores zero, and takes its whole 40 points with it. The
        // knockout then caps what is left.
        Assert.True(evaluation!.Knockout!.MissingMandatorySkills);
        Assert.Equal(34, evaluation.OverallMatch);
    }

    // -- knockouts ----------------------------------------------------------

    /// <summary>
    /// The bug that produced a 45%, in its final form.
    ///
    /// The posting lists Figma under required skills and the CV does not evidence
    /// it. The model labels the row <c>core_skill</c> rather than <c>must_have</c>
    /// and reports no dealbreaker at all — and it still caps, because the POSTING
    /// said the skill was required and that is not the model's opinion to hold.
    /// Keying the cap on the model's label left a hole exactly the width of one
    /// misclassification.
    /// </summary>
    [Fact]
    public async Task A_skill_the_posting_requires_caps_the_score_whatever_kind_the_model_gave_it()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"),
                Req("TypeScript", "Angular, TypeScript, RxJS", "STRONG", "core_skill"),
                Req("SCSS", "HTML5, SCSS, Git", "STRONG", "core_skill"),
                Req("Figma proficiency", "NONE", "MISSING", "core_skill")),
            weights: """[{"criterion": "core_skill", "weight": 100}]""",
            knockout: NoKnockout))
            .EvaluateAsync(AngularDeveloper(), PostingRequiringSkills());

        Assert.NotNull(evaluation);
        Assert.True(evaluation!.Knockout!.MissingMandatorySkills);
        Assert.Equal(34, evaluation.OverallMatch);
        Assert.Equal("Complete Mismatch", evaluation.Category);
        Assert.Equal("Disqualify", evaluation.Decision);
        Assert.Contains("the posting lists this as required", string.Join(" ", evaluation.Knockout.Reasons));
    }

    [Fact]
    public async Task A_missing_education_row_is_its_own_dealbreaker()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"),
                Req("PMP certification", "None", "MISSING", "education")),
            weights: """[{"criterion": "core_skill", "weight": 60}, {"criterion": "education", "weight": 40}]"""))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);
        Assert.True(evaluation!.Knockout!.MissingRequiredEducation);
        Assert.Equal(34, evaluation.OverallMatch);
    }

    // -- dropped rows -------------------------------------------------------

    /// <summary>
    /// A requirement the model never returned a row for is not a requirement the
    /// candidate passes. The posting lists three required skills; the model
    /// evaluated one, and the score must reflect all three.
    /// </summary>
    [Fact]
    public async Task Required_skills_the_model_skipped_are_added_and_scored()
    {
        // The posting requires Figma, wireframes and user research. Only Figma gets
        // a row.
        var evaluation = await Evaluator(Reply(
            Req("Figma", "Streamlined UX based on user feedback and Figma prototypes", "STRONG", "core_skill")))
            .EvaluateAsync(AngularDeveloper(), PostingRequiringSkills());

        Assert.NotNull(evaluation);

        var rows = evaluation!.Requirements!;
        Assert.Equal(3, rows.Count);
        Assert.Contains(rows, r => r.Requirement == "wireframes" && r.MatchLevel == MatchLevels.Missing);
        Assert.Contains(rows, r => r.Requirement == "user research" && r.MatchLevel == MatchLevels.Missing);

        // One of three evidenced. Without the backfill this scored a clean 100%,
        // because the list it averaged over had a single row on it.
        Assert.Equal(33, evaluation.OverallMatch);
    }

    /// <summary>
    /// A row nobody evaluated is our omission, not a finding about the candidate.
    /// It costs them the marks; it must not disqualify them.
    /// </summary>
    [Fact]
    public async Task An_unexamined_requirement_scores_zero_but_raises_no_dealbreaker()
    {
        var evaluation = await Evaluator(Reply(
            Req("Figma", "Streamlined UX based on user feedback and Figma prototypes", "STRONG", "core_skill")))
            .EvaluateAsync(AngularDeveloper(), PostingRequiringSkills());

        Assert.NotNull(evaluation);
        Assert.False(evaluation!.Knockout!.Fired);

        // Labelled, so the card can say what happened rather than presenting it as a
        // judgement that was made.
        var backfilled = evaluation.Requirements!.Single(r => r.Requirement == "wireframes");
        Assert.Equal("not evaluated", backfilled.Where);
    }

    // -- years are arithmetic, and arithmetic is not the model's ------------

    /// <summary>
    /// The bug this rule exists for: asked whether 2.6 years met a 2-year minimum,
    /// an 8B said it did not, and a candidate who EXCEEDED the requirement was
    /// capped at a disqualifying score. The model now only extracts the duration.
    /// </summary>
    [Fact]
    public async Task Two_point_six_years_meets_a_two_year_minimum()
    {
        var evaluation = await Evaluator(Reply(
            Req("2+ years of experience", "Frontend engineer with 3+ years of experience", "STRONG", "experience"),
            weights: """[{"criterion": "experience", "weight": 100}]""",
            years: "2.6"))
            .EvaluateAsync(AngularDeveloper(), PostingRequiring(2));

        Assert.NotNull(evaluation);
        Assert.False(evaluation!.Knockout!.MissingYearsOfExperience);
        Assert.Equal(100, evaluation.OverallMatch);
    }

    [Fact]
    public async Task A_genuine_years_shortfall_still_caps_the_score()
    {
        var evaluation = await Evaluator(Reply(
            Req("8+ years of experience", "Frontend engineer with 3+ years of experience", "STRONG", "experience"),
            weights: """[{"criterion": "experience", "weight": 100}]""",
            years: "3"))
            .EvaluateAsync(AngularDeveloper(), PostingRequiring(8));

        Assert.NotNull(evaluation);
        Assert.True(evaluation!.Knockout!.MissingYearsOfExperience);
        Assert.Equal(34, evaluation.OverallMatch);
        Assert.Contains("3 years of experience against a required 8", string.Join(" ", evaluation.Knockout.Reasons));
    }

    /// <summary>
    /// A dealbreaker asserts a fact. "The résumé does not say" is not evidence of a
    /// shortfall, and capping on an absent number would disqualify every CV that
    /// omits a total.
    /// </summary>
    [Fact]
    public async Task An_unstated_duration_raises_no_years_dealbreaker()
    {
        var resume = AngularDeveloper();
        resume.YearsExperience = null;

        var evaluation = await Evaluator(Reply(
            Req("8+ years of experience", "NONE", "MISSING", "experience"),
            weights: """[{"criterion": "experience", "weight": 100}]""",
            years: "null"))
            .EvaluateAsync(resume, PostingRequiring(8));

        Assert.NotNull(evaluation);
        Assert.False(evaluation!.Knockout!.MissingYearsOfExperience);

        // The row still scores zero for its kind — the requirement is unevidenced —
        // but that is a low score, not a disqualification.
        Assert.Equal(0, evaluation.OverallMatch);
    }

    /// <summary>The parsed profile stands in when the model extracts nothing.</summary>
    [Fact]
    public async Task The_parsed_profile_supplies_the_duration_the_model_omitted()
    {
        var resume = AngularDeveloper();
        resume.YearsExperience = 3;

        var evaluation = await Evaluator(Reply(
            Req("10+ years of experience", "Frontend engineer with 3+ years of experience", "STRONG", "experience"),
            weights: """[{"criterion": "experience", "weight": 100}]""",
            years: "null"))
            .EvaluateAsync(resume, PostingRequiring(10));

        Assert.NotNull(evaluation);
        Assert.True(evaluation!.Knockout!.MissingYearsOfExperience);
    }

    /// <summary>
    /// A MISSING experience row is no longer a dealbreaker by itself.
    ///
    /// It was, and it had to stop: under the new rules that row means "the résumé
    /// does not state a duration", not "the candidate is short". Only the
    /// subtraction decides.
    /// </summary>
    [Fact]
    public async Task A_missing_experience_row_alone_no_longer_disqualifies()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"),
                Req("industry background in insurance", "NONE", "MISSING", "experience")),
            weights: """[{"criterion": "core_skill", "weight": 50}, {"criterion": "experience", "weight": 50}]"""))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);
        Assert.False(evaluation!.Knockout!.Fired);
        Assert.Equal(50, evaluation.OverallMatch);
    }

    /// <summary>
    /// The model may ADD a dealbreaker its rows do not show, and may not remove one.
    ///
    /// The asymmetry is deliberate. A seniority or domain mismatch can be real
    /// without any single row capturing it, so the flag is worth honouring; but a
    /// model that quietly clears a knockout its own verdicts demand is how an
    /// unqualified candidate reaches a shortlist.
    /// </summary>
    [Fact]
    public async Task A_model_raised_flag_caps_the_score_and_says_why()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"),
                Req("Figma", "Streamlined UX based on user feedback and Figma prototypes", "STRONG", "core_skill")),
            knockout: """{"missing_must_have_tools": true, "missing_required_education_or_cert": false}"""))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);
        Assert.Equal(34, evaluation!.OverallMatch);
        Assert.Contains("a mandatory tool the model judged absent", string.Join(" ", evaluation.Knockout!.Reasons));
    }

    [Fact]
    public async Task A_missing_responsibility_is_never_a_dealbreaker()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"),
                Req("Figma", "Streamlined UX based on user feedback and Figma prototypes", "STRONG", "core_skill"),
                Req("run design workshops", "None", "MISSING", "responsibility"),
                Req("nice: Adobe XD", "None", "MISSING", "nice_to_have"))))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);
        Assert.False(evaluation!.Knockout!.Fired);
        Assert.True(evaluation.OverallMatch > 34);
    }

    // -- grounding ----------------------------------------------------------

    /// <summary>
    /// The other regression: a correct reading across vocabulary must survive.
    ///
    /// The requirement's own words appear nowhere in the CV, which is precisely the
    /// case the old grounding pass discarded — turning a right answer into a
    /// fabricated gap and re-imposing keyword matching at the last step.
    /// </summary>
    [Fact]
    public async Task A_verdict_survives_when_the_requirements_words_are_absent_but_the_quote_is_real()
    {
        var evaluation = await Evaluator(Reply(
            Req("component-based frontend development",
                "Optimized UI/UX by incorporating Figma designs into responsive Angular components",
                "STRONG", "core_skill")))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);
        var kept = Assert.Single(evaluation!.Requirements!);
        Assert.Equal(MatchLevels.Strong, kept.MatchLevel);
        Assert.Equal(100, evaluation.OverallMatch);
    }

    [Fact]
    public async Task An_invented_quote_cannot_earn_credit()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                // Not one of the posting's RequiredSkills, so the correction shows up
                // in the score rather than being masked by a dealbreaker cap.
                Req("usability workshops", "Ran moderated usability sessions with twelve participants", "STRONG", "core_skill"),
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"))))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);

        // Neither the sentence nor the requirement is in the document. The verdict
        // is corrected to MISSING BEFORE the arithmetic runs, so the score reflects
        // what the CV supports rather than what the model asserted.
        var research = evaluation!.Requirements!.Single(r => r.Requirement == "usability workshops");
        Assert.Equal(MatchLevels.Missing, research.MatchLevel);
        Assert.Equal("", research.Quote);
        Assert.Equal(50, evaluation.OverallMatch);
    }

    [Fact]
    public async Task A_real_requirement_with_an_unverifiable_quote_is_capped_at_weak()
    {
        // "Git" is in the CV's skills list. The quote offered for it is not in the
        // document at all, so the mention stands and the asserted depth does not.
        var evaluation = await Evaluator(Reply(
            Req("Git", "Led the migration of forty repositories to a trunk-based workflow", "STRONG", "core_skill")))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);
        var git = Assert.Single(evaluation!.Requirements!);
        Assert.Equal(MatchLevels.Weak, git.MatchLevel);
        Assert.Equal(50, evaluation.OverallMatch);
    }

    // -- robustness ---------------------------------------------------------

    [Fact]
    public async Task A_verdict_is_not_recomputed_for_the_same_pair()
    {
        var evaluator = Evaluator(Reply(Req("Angular", "responsive Angular components", "STRONG", "core_skill")));

        var resume = AngularDeveloper();
        var job = UxDesignerPosting();

        var first = await evaluator.EvaluateAsync(resume, job);
        var second = await evaluator.EvaluateAsync(resume, job);

        Assert.NotNull(first);
        Assert.Same(first, second);
    }

    [Fact]
    public async Task Output_with_no_requirements_falls_back_rather_than_inventing_a_score()
    {
        Assert.Null(await Evaluator("I think this candidate is pretty good honestly")
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting()));

        Assert.Null(await Evaluator(null)
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting()));

        // Well-formed JSON with nothing to score is just as unusable as no JSON:
        // an empty requirement list would otherwise compute a confident 0%.
        Assert.Null(await Evaluator(Reply("")).EvaluateAsync(AngularDeveloper(), UxDesignerPosting()));
    }

    [Fact]
    public async Task Weights_the_model_omitted_fall_back_to_an_even_split()
    {
        var evaluation = await Evaluator(Reply(
            string.Join(",",
                Req("Angular", "responsive Angular components", "STRONG", "core_skill"),
                Req("3+ years", "Frontend engineer with 3+ years of experience", "STRONG", "experience")),
            weights: "[]"))
            .EvaluateAsync(AngularDeveloper(), UxDesignerPosting());

        Assert.NotNull(evaluation);
        Assert.Equal(100, evaluation!.Weights.Sum(w => w.Weight));
        Assert.Equal(100, evaluation.OverallMatch);
    }

    [Fact]
    public void Verdicts_become_skill_chips_without_inventing_a_similarity()
    {
        var evaluation = new JobEvaluationDto(
            42, 50, "Weak Match", "", "",
            Array.Empty<EvaluationWeightDto>(),
            new[] { new EvaluationMatchDto("Figma", "…Figma prototypes", "work experience") },
            new[] { new EvaluationMatchDto("Git", "Git", "skills list") },
            new[] { new EvaluationGapDto("wireframes", "absent") },
            "Consider Alternate Role", null, "eval-v4");

        var chips = JobEvaluationService.ToSkillAssessments(evaluation);

        Assert.Equal("Have", chips.Single(c => c.Skill == "Figma").Status);
        Assert.Equal("Transferable", chips.Single(c => c.Skill == "Git").Status);
        Assert.Equal("Missing", chips.Single(c => c.Skill == "wireframes").Status);

        // No cosine was measured anywhere in this path. Putting a number in that
        // slot would be a measurement nobody took.
        Assert.All(chips, c => Assert.Equal(0, c.Similarity));
    }

    [Theory]
    [InlineData("reasoned", ScoringModes.Reasoned)]
    [InlineData("Reasoned", ScoringModes.Reasoned)]
    [InlineData("computed", ScoringModes.Computed)]
    [InlineData("nonsense", ScoringModes.Computed)]
    [InlineData(null, ScoringModes.Computed)]
    public void An_unrecognised_mode_falls_back_to_the_arithmetic(string? input, string expected) =>
        Assert.Equal(expected, ScoringModes.Normalise(input));

    [Theory]
    [InlineData("must_have", RequirementKinds.MustHave)]
    [InlineData("Nice To Have", RequirementKinds.NiceToHave)]
    [InlineData("nice-to-have", RequirementKinds.NiceToHave)]
    [InlineData("something the model made up", RequirementKinds.CoreSkill)]
    public void An_unrecognised_kind_lands_in_the_middle_of_the_range(string input, string expected) =>
        Assert.Equal(expected, RequirementKinds.Normalise(input));
}
