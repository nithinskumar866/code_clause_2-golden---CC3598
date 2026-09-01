using JobPortal.Api.Contracts;
using JobPortal.Api.Services.Chat;
using Xunit;

namespace JobPortal.Api.Tests;

/// <summary>
/// Follow-up chips.
///
/// A chip is a promise: tap it and you get an answer. So the tests are mostly about
/// what is NOT offered — a chip that leads nowhere is worse than no chip, because the
/// assistant then declines something it just volunteered.
/// </summary>
public class ChatFollowUpTests
{
    private static JobMatchDto Match(int id, string title, double fit = 60, params string[] gaps) =>
        new(new JobSummaryDto(id, title, "Acme", "Chennai", "Remote", "Full-time", "Mid",
                null, null, null, null, null,
                Array.Empty<string>(), Array.Empty<string>(), "", null, true, DateTime.UtcNow),
            fit, "Moderate fit", 60, 50, 70, 80,
            Array.Empty<SkillAssessmentDto>(), Array.Empty<string>(), gaps, "note");

    [Fact]
    public void Nothing_on_screen_still_offers_the_board_wide_question()
    {
        // "What should I learn?" is answerable from the postings alone, so it is the
        // one chip that never depends on a shortlist.
        var chips = ChatFollowUps.For(Array.Empty<JobMatchDto>(), null);
        Assert.Contains(chips, c => c.Message.Contains("skill up", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Compare_is_not_offered_for_a_single_result()
    {
        // "Compare these" against one role is a dead end.
        var chips = ChatFollowUps.For(new[] { Match(1, "Data Scientist") }, null);
        Assert.DoesNotContain(chips, c => c.Label.Contains("Compare"));
    }

    [Fact]
    public void Compare_names_the_two_roles_it_would_compare()
    {
        var chips = ChatFollowUps.For(
            new[] { Match(1, "Data Scientist"), Match(2, "ML Engineer") }, null);

        var compare = Assert.Single(chips, c => c.Label.Contains("Compare"));
        Assert.Contains("Data Scientist", compare.Message);
        Assert.Contains("ML Engineer", compare.Message);
    }

    [Fact]
    public void The_turn_just_taken_is_not_offered_again()
    {
        var shortlist = new[] { Match(1, "Data Scientist"), Match(2, "ML Engineer") };
        var chips = ChatFollowUps.For(shortlist, shortlist[0], TurnAction.Compare);
        Assert.DoesNotContain(chips, c => c.Label.Contains("Compare"));
    }

    [Fact]
    public void A_gap_chip_is_only_offered_when_the_role_has_gaps()
    {
        var covered = Match(1, "Data Scientist");
        Assert.DoesNotContain(
            ChatFollowUps.For(new[] { covered }, covered),
            c => c.Label.StartsWith("What do I need"));

        var lacking = Match(2, "ML Engineer", 50, "Kubernetes");
        Assert.Contains(
            ChatFollowUps.For(new[] { lacking }, lacking),
            c => c.Label.StartsWith("What do I need"));
    }

    [Fact]
    public void Chip_messages_name_the_role_so_they_survive_scrolling()
    {
        var focus = Match(1, "Data Scientist", 72, "Spark");
        var chips = ChatFollowUps.For(new[] { focus }, focus);

        // The label may be short, but every message has to stand alone — by the time
        // it is sent it is the user's turn in the transcript.
        Assert.All(chips.Where(c => c.Label != "What should I learn?"
                                    && c.Label != "What am I missing?"),
            c => Assert.Contains("Data Scientist", c.Message));
    }

    [Fact]
    public void Long_titles_are_truncated_in_the_label_but_not_the_message()
    {
        var focus = Match(1, "Senior Principal Machine Learning Platform Engineer", 50, "Go");
        var chips = ChatFollowUps.For(new[] { focus }, focus);
        var chip = Assert.Single(chips, c => c.Label.StartsWith("What do I need"));

        Assert.True(chip.Label.Length <= 46, $"Label too long for one line: {chip.Label}");
        Assert.Contains("Senior Principal Machine Learning Platform Engineer", chip.Message);
    }

    [Fact]
    public void At_most_four_are_offered()
    {
        var shortlist = Enumerable.Range(1, 6).Select(i => Match(i, $"Role {i}", 50, "Gap")).ToList();
        Assert.True(ChatFollowUps.For(shortlist, shortlist[0]).Count <= 4);
    }

    // --- The ask-back ---------------------------------------------------------
    [Fact]
    public void Clarify_offers_the_real_postings_on_screen()
    {
        var chips = ChatFollowUps.Clarify(
            new[] { (7, "HR Generalist", "Acme"), (9, "Data Scientist", "Globex") },
            "is it remote?");

        Assert.Equal(2, chips.Count);
        Assert.Contains(chips, c => c.Label == "HR Generalist");
        // The original question is re-asked against a named role, so the next turn
        // resolves rather than needing a third exchange.
        Assert.Contains(chips, c => c.Message.Contains("is it remote") && c.Message.Contains("HR Generalist"));
    }

    [Fact]
    public void Clarify_with_nothing_on_screen_offers_nothing()
    {
        Assert.Empty(ChatFollowUps.Clarify(Array.Empty<(int, string, string)>(), "is it remote?"));
    }
}
