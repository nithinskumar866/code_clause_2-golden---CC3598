using JobPortal.Api.Contracts;

namespace JobPortal.Api.Services.Chat;

/// <summary>
/// What to offer the candidate next.
///
/// Deterministic on purpose, and not because the model could not phrase these more
/// nicely. A chip is a PROMISE: tap it and you get an answer. Every one here is built
/// from the shortlist or the board that exists right now, so a chip can never offer
/// something the assistant would then have to decline — which is what would happen
/// the first time a model invented a plausible-sounding follow-up.
///
/// They also carry the discovery burden. Nobody guesses unprompted that "what should
/// I learn across all these roles" is answerable, so the capabilities advertise
/// themselves rather than waiting to be found.
/// </summary>
public static class ChatFollowUps
{
    private const int Max = 4;

    /// <summary>
    /// Follow-ups for a turn that produced or discussed results.
    /// </summary>
    /// <param name="shortlist">Roles on screen, best first.</param>
    /// <param name="focusJob">The posting just discussed, if any.</param>
    /// <param name="lastAction">What the turn did — so the chips move the conversation
    /// on rather than offering what was just answered.</param>
    public static IReadOnlyList<FollowUpDto> For(
        IReadOnlyList<JobMatchDto> shortlist,
        JobMatchDto? focusJob,
        TurnAction? lastAction = null)
    {
        var chips = new List<FollowUpDto>();

        // About the role in hand. Titled explicitly rather than "this one", because a
        // chip is read out of context once the transcript scrolls.
        if (focusJob is not null)
        {
            var title = focusJob.Job.Title;

            if (lastAction != TurnAction.GapPlan && focusJob.Gaps.Count > 0)
                chips.Add(new FollowUpDto($"What do I need for {Short(title)}?",
                    $"What skills do I need to get the {title} role?"));

            if (lastAction != TurnAction.Explain)
                chips.Add(new FollowUpDto("Why this score?",
                    $"Why is my fit for the {title} role {focusJob.FitScore:0.#}%?"));

            if (lastAction != TurnAction.CompanyFact)
                chips.Add(new FollowUpDto("About the company",
                    $"What does the {title} posting say about the company?"));
        }

        // Across the shortlist. Only offered when there is genuinely more than one
        // role to compare — "compare these" against a single result is a dead end.
        if (shortlist.Count >= 2 && lastAction != TurnAction.Compare)
        {
            var first = shortlist[0].Job.Title;
            var second = shortlist[1].Job.Title;
            chips.Add(new FollowUpDto("Compare the top two",
                $"Compare the {first} and {second} roles for me."));
        }

        if (shortlist.Count > 0 && lastAction != TurnAction.GapsAcrossShortlist)
            chips.Add(new FollowUpDto("What am I missing?",
                "What am I lacking across these roles?"));

        if (lastAction != TurnAction.SkillUpAcrossBoard)
            chips.Add(new FollowUpDto("What should I learn?",
                "Based on the open roles, what should I skill up on next?"));

        return chips.Take(Max).ToList();
    }

    /// <summary>
    /// The ask-back: which posting did they mean?
    ///
    /// Offered as chips rather than a typed answer because the whole reason we are
    /// here is that a reference did not resolve, and asking someone to re-type a role
    /// name invites the same miss again. Every option is a real posting on screen.
    /// </summary>
    public static IReadOnlyList<FollowUpDto> Clarify(
        IReadOnlyList<(int Id, string Title, string Company)> visible, string question)
    {
        return visible
            .Take(Max)
            .Select(job => new FollowUpDto(
                Short(job.Title),
                // The original question re-asked against a named role, so the next
                // turn resolves cleanly instead of needing a third exchange.
                $"{question.TrimEnd('?', '.', ' ')} — for the {job.Title} role?"))
            .ToList();
    }

    /// <summary>Openers for a candidate with no CV yet.</summary>
    public static IReadOnlyList<FollowUpDto> Browsing() => new[]
    {
        new FollowUpDto("Show remote roles", "Show me the remote roles."),
        new FollowUpDto("What's on the board?", "What roles are open right now?"),
        new FollowUpDto("What should I learn?", "Based on the open roles, what should I skill up on next?"),
    };

    /// <summary>
    /// Chip labels have to fit on one line next to three others.
    ///
    /// 24, not 28: the longest label wraps this in "What do I need for …?", and the
    /// budget that matters is the whole chip rather than the title inside it.
    /// </summary>
    private static string Short(string title) =>
        title.Length <= 24 ? title : title[..23].TrimEnd() + "…";
}
