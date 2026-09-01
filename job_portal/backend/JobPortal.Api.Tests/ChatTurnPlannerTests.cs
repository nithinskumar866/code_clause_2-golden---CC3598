using JobPortal.Api.Contracts;
using JobPortal.Api.Services.Chat;
using JobPortal.Api.Services.Llm;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace JobPortal.Api.Tests;

/// <summary>
/// Turn planning.
///
/// The planner is where an LLM is allowed to decide what a follow-up MEANS. These
/// tests are almost entirely about the boundary of that permission: it may point at
/// postings and name filters, and everything it points at must turn out to be real.
/// </summary>
public class ChatTurnPlannerTests
{
    private static readonly IReadOnlyList<(int, string, string)> OnScreen = new[]
    {
        (7, "HR Generalist", "Acme"),
        (9, "Data Scientist", "Globex"),
    };

    private static readonly IReadOnlySet<int> OnBoard = new HashSet<int> { 7, 9, 42 };

    private static ChatTurnPlanner Planner(string? reply) => new(
        reply is null ? new OfflineLlm() : new ScriptedLlm(reply),
        new ChatQueryUnderstanding(),
        NullLogger<ChatTurnPlanner>.Instance);

    private static Task<TurnPlan> Plan(
        string? reply, string message, JobFilters? filters = null, int? focus = null) =>
        Planner(reply).PlanAsync(
            message, filters ?? new JobFilters(), new JobBoardLexicon(),
            OnScreen, OnBoard, Array.Empty<(string, string)>(), focus);

    // --- Fallback -------------------------------------------------------------
    [Fact]
    public async Task With_no_model_the_deterministic_parser_decides()
    {
        var plan = await Plan(null, "show me remote roles");
        Assert.True(plan.FromFallback);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("")]
    [InlineData("{\"action\":\"teleport\"}")]
    [InlineData("{\"nonsense\":true}")]
    public async Task Unusable_planner_output_falls_back(string reply)
    {
        var plan = await Plan(reply, "why is this a good fit?");
        Assert.True(plan.FromFallback);
    }

    [Fact]
    public async Task Json_wrapped_in_prose_is_still_read()
    {
        // Verified on the live pod: llama3.1:8b prefaces its JSON and fences it.
        var plan = await Plan(
            "Sure! Here you go:\n```json\n{\"action\":\"gap_plan\",\"job_ids\":[9]}\n```",
            "what should I study for the data science job?");

        Assert.False(plan.FromFallback);
        Assert.Equal(TurnAction.GapPlan, plan.Action);
        Assert.Equal(new[] { 9 }, plan.JobIds);
    }

    // --- The model may not invent things --------------------------------------
    [Fact]
    public async Task A_job_id_that_does_not_exist_is_discarded()
    {
        var plan = await Plan(
            "{\"action\":\"fact_about\",\"job_ids\":[999]}", "what does it pay?");

        // 999 is dropped; with nothing left to anchor to, the turn asks rather than guesses.
        Assert.Equal(TurnAction.Clarify, plan.Action);
        Assert.Empty(plan.JobIds);
    }

    [Fact]
    public async Task A_posting_that_is_off_screen_but_real_is_allowed()
    {
        // "the data science job" may have scrolled out of the visible five.
        var plan = await Plan("{\"action\":\"gap_plan\",\"job_ids\":[42]}", "what do I need for that role?");
        Assert.Equal(TurnAction.GapPlan, plan.Action);
        Assert.Equal(new[] { 42 }, plan.JobIds);
    }

    [Fact]
    public async Task An_invented_location_is_never_applied()
    {
        // The lexicon here knows no locations, so "Atlantis" resolves to nothing.
        // Applying it would return an empty board the candidate cannot explain.
        var plan = await Plan(
            "{\"action\":\"refine\",\"filters\":{\"location\":\"Atlantis\"}}", "anything in Atlantis?");

        Assert.Null(plan.Filters.Location);
    }

    [Fact]
    public async Task Filter_values_are_normalised_to_the_boards_vocabulary()
    {
        var plan = await Plan(
            "{\"action\":\"refine\",\"filters\":{\"workMode\":\"remote\",\"employmentType\":\"CONTRACT\"}}",
            "remote contract only");

        Assert.Equal("Remote", plan.Filters.WorkMode);
        Assert.Equal("Contract", plan.Filters.EmploymentType);
    }

    [Fact]
    public async Task A_work_mode_outside_the_vocabulary_is_dropped()
    {
        var plan = await Plan(
            "{\"action\":\"refine\",\"filters\":{\"workMode\":\"remote-first-ish\"}}", "remote-ish?");
        Assert.Null(plan.Filters.WorkMode);
    }

    [Fact]
    public async Task Existing_filters_survive_a_partial_update()
    {
        var current = new JobFilters(WorkMode: "Remote");
        var plan = await Plan(
            "{\"action\":\"refine\",\"filters\":{\"seniority\":\"Senior\"}}", "senior ones", current);

        Assert.Equal("Remote", plan.Filters.WorkMode);
        Assert.Equal("Senior", plan.Filters.SeniorityLevel);
    }

    // --- Anchoring ------------------------------------------------------------
    [Fact]
    public async Task A_posting_question_with_no_posting_becomes_a_clarify()
    {
        var plan = await Plan("{\"action\":\"fact_about\"}", "is it remote?");

        Assert.Equal(TurnAction.Clarify, plan.Action);
        Assert.False(string.IsNullOrWhiteSpace(plan.ClarifyPrompt));
    }

    [Fact]
    public async Task The_focus_job_anchors_a_bare_follow_up()
    {
        // "is it remote?" after discussing job 7 is about job 7.
        var plan = await Plan("{\"action\":\"fact_about\"}", "is it remote?", focus: 7);

        Assert.Equal(TurnAction.FactAbout, plan.Action);
        Assert.Equal(new[] { 7 }, plan.JobIds);
    }

    [Fact]
    public async Task Actions_that_need_no_posting_are_not_forced_to_clarify()
    {
        var plan = await Plan("{\"action\":\"skill_up\"}", "what should I learn next?");
        Assert.Equal(TurnAction.SkillUpAcrossBoard, plan.Action);
        Assert.Empty(plan.JobIds);
    }

    // --- Routing the cases the regex cascade got wrong -------------------------
    [Fact]
    public async Task Why_is_this_a_good_fit_routes_to_explain_not_advise()
    {
        // The exact collision in the old parser: "why" hits ExplainPattern and
        // "good fit" hits AdvisePattern, so declaration order decided the answer.
        var plan = await Plan("{\"action\":\"explain\",\"job_ids\":[7]}", "why is this a good fit?");
        Assert.Equal(TurnAction.Explain, plan.Action);
    }

    [Fact]
    public async Task A_company_question_gets_its_own_action()
    {
        var plan = await Plan(
            "{\"action\":\"company_fact\",\"job_ids\":[7]}", "what's their leave policy?");
        Assert.Equal(TurnAction.CompanyFact, plan.Action);
    }

    [Fact]
    public async Task Compare_carries_every_posting_named()
    {
        var plan = await Plan("{\"action\":\"compare\",\"job_ids\":[7,9]}", "compare the first two");
        Assert.Equal(new[] { 7, 9 }, plan.JobIds);
    }

    [Fact]
    public async Task The_question_is_kept_for_the_composer()
    {
        var plan = await Plan(
            "{\"action\":\"fact_about\",\"job_ids\":[7],\"question\":\"Does the HR Generalist role allow remote work?\"}",
            "is it remote?");

        Assert.Contains("HR Generalist", plan.Question);
    }

    [Fact]
    public async Task A_missing_question_falls_back_to_the_raw_message()
    {
        var plan = await Plan("{\"action\":\"fact_about\",\"job_ids\":[7]}", "is it remote?");
        Assert.Equal("is it remote?", plan.Question);
    }
    // --- Wrapper objects ------------------------------------------------------
    [Fact]
    public async Task A_plan_nested_in_a_response_envelope_is_still_read()
    {
        // qwen2.5:14b's actual habit, captured live: the payload is right, it just
        // arrives one level down. Rejecting it would waste a good plan.
        var plan = await Plan(
            "{\"response\":{\"action\":\"gap_plan\",\"job_ids\":[9],\"status\":\"success\"}}",
            "what should I study for the data science job?");

        Assert.False(plan.FromFallback);
        Assert.Equal(TurnAction.GapPlan, plan.Action);
        Assert.Equal(new[] { 9 }, plan.JobIds);
    }

    [Fact]
    public async Task A_top_level_plan_is_not_mistaken_for_an_envelope()
    {
        // "filters" is an object property, but the root has its own action, so
        // unwrapping must stop before it descends into the filters.
        var plan = await Plan(
            "{\"action\":\"refine\",\"filters\":{\"workMode\":\"Remote\"}}", "remote only");

        Assert.Equal(TurnAction.Refine, plan.Action);
        Assert.Equal("Remote", plan.Filters.WorkMode);
    }

    [Fact]
    public async Task An_envelope_with_no_usable_plan_still_falls_back()
    {
        var plan = await Plan("{\"response\":{\"status\":\"success\"}}", "is it remote?");
        Assert.True(plan.FromFallback);
    }

}
