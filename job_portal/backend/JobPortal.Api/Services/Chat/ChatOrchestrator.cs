using System.Text;
using System.Text.Json;
using JobPortal.Api.Contracts;
using JobPortal.Api.Data;
using JobPortal.Api.Mapping;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Services.Jobs;
using JobPortal.Api.Services.Llm;
using JobPortal.Api.Services.Matching;
using JobPortal.Api.Services.Resumes;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Chat;

public interface IChatOrchestrator
{
    Task<ChatSessionDto> GetOrCreateSessionAsync(string sessionId, CancellationToken ct = default);

    /// <param name="scoringMode">
    /// The scorer this turn should use — see <see cref="ScoringModes"/>. Null keeps
    /// whatever the session was already set to, so a caller that does not know the
    /// setting exists cannot silently reset it.
    /// </param>
    Task HandleResumeAsync(
        string sessionId, int resumeId, IChatSink sink,
        string? scoringMode = null, CancellationToken ct = default);

    /// <inheritdoc cref="HandleResumeAsync"/>
    Task HandleMessageAsync(
        string sessionId, string message, IChatSink sink,
        string? scoringMode = null, CancellationToken ct = default);

    /// <summary>
    /// Forgets the conversation, and only the conversation.
    ///
    /// Messages, narrowed filters and the stored last result all go; the CV stays
    /// attached. "Start a new chat" is a request to drop the context, not to
    /// become a stranger — making someone re-upload their resume to ask a fresh
    /// question would be a worse product and a slower one.
    /// </summary>
    Task<ChatSessionDto> ResetSessionAsync(string sessionId, CancellationToken ct = default);
}

/// <summary>
/// Runs one turn of the conversation.
///
/// The rule that shapes everything here: THE MODEL NEVER SUPPLIES A FACT. Every
/// job title, fit percentage, skill verdict and salary the candidate sees comes
/// out of deterministic retrieval and scoring. The model's entire job is to say
/// those findings in natural language. That is why the structured result is sent
/// to the UI separately from the prose, and why the markdown table is composed
/// here rather than requested from the model.
/// </summary>
public class ChatOrchestrator : IChatOrchestrator
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private readonly PortalDbContext _db;
    private readonly IResumeService _resumes;
    private readonly IMatchingService _matching;
    private readonly IReasonedMatchingService _reasoned;
    private readonly Rag.IRagMatchingService _rag;
    private readonly IServiceScopeFactory _scopes;
    private readonly IJobSuggestionService _suggestions;
    private readonly IOllamaClient _llm;
    private readonly ChatQueryUnderstanding _understanding;
    private readonly ChatTurnPlanner _planner;
    private readonly ChatCapabilities _capabilities;
    private readonly AnswerComposer _composer;
    private readonly MatchingOptions _options;
    private readonly ILogger<ChatOrchestrator> _logger;

    public ChatOrchestrator(
        PortalDbContext db,
        IResumeService resumes,
        IMatchingService matching,
        IReasonedMatchingService reasoned,
        Rag.IRagMatchingService rag,
        IServiceScopeFactory scopes,
        IJobSuggestionService suggestions,
        IOllamaClient llm,
        ChatQueryUnderstanding understanding,
        ChatTurnPlanner planner,
        ChatCapabilities capabilities,
        AnswerComposer composer,
        IOptions<MatchingOptions> options,
        ILogger<ChatOrchestrator> logger)
    {
        _planner = planner;
        _capabilities = capabilities;
        _composer = composer;
        _db = db;
        _resumes = resumes;
        _matching = matching;
        _reasoned = reasoned;
        _rag = rag;
        _scopes = scopes;
        _suggestions = suggestions;
        _llm = llm;
        _understanding = understanding;
        _options = options.Value;
        _logger = logger;
    }

    // -- session ------------------------------------------------------------

    public async Task<ChatSessionDto> GetOrCreateSessionAsync(string sessionId, CancellationToken ct = default)
    {
        var session = await LoadSessionAsync(sessionId, ct);
        var resume = session.ResumeId is { } id ? await _resumes.GetAsync(id, ct) : null;

        var messages = await _db.ChatMessages
            .AsNoTracking()
            .Where(m => m.SessionId == sessionId)
            .OrderBy(m => m.Id)
            .ToListAsync(ct);

        return new ChatSessionDto(
            sessionId,
            resume?.ToDto(),
            ReadFilters(session),
            messages.Select(m => m.ToDto()).ToList());
    }

    private async Task<PortalChatSession> LoadSessionAsync(string sessionId, CancellationToken ct)
    {
        var session = await _db.ChatSessions.FirstOrDefaultAsync(s => s.Id == sessionId, ct);
        if (session is not null) return session;

        session = new PortalChatSession { Id = sessionId };
        _db.ChatSessions.Add(session);
        await _db.SaveChangesAsync(ct);
        return session;
    }

    private static JobFilters ReadFilters(PortalChatSession session)
    {
        try
        {
            return JsonSerializer.Deserialize<JobFilters>(session.FiltersJson, Json) ?? new JobFilters();
        }
        catch (JsonException)
        {
            // A filter set that cannot be read is a filter set that should not be
            // silently applied. Starting clean is visible; guessing is not.
            return new JobFilters();
        }
    }

    private static List<JobMatchDto> ReadLastMatches(PortalChatSession session)
    {
        try
        {
            return JsonSerializer.Deserialize<List<JobMatchDto>>(session.LastMatchesJson, Json) ?? new();
        }
        catch (JsonException)
        {
            return new List<JobMatchDto>();
        }
    }

    private async Task SaveTurnAsync(
        PortalChatSession session, string userMessage, string assistantMessage,
        JobFilters filters, IReadOnlyList<JobMatchDto>? matches, CancellationToken ct)
    {
        if (!string.IsNullOrWhiteSpace(userMessage))
        {
            _db.ChatMessages.Add(new PortalChatMessage
            { SessionId = session.Id, Role = "user", Content = userMessage });
        }
        if (!string.IsNullOrWhiteSpace(assistantMessage))
        {
            _db.ChatMessages.Add(new PortalChatMessage
            { SessionId = session.Id, Role = "assistant", Content = assistantMessage });
        }

        session.FiltersJson = JsonSerializer.Serialize(filters, Json);
        if (matches is not null) session.LastMatchesJson = JsonSerializer.Serialize(matches, Json);
        session.UpdatedAt = DateTime.UtcNow;

        await _db.SaveChangesAsync(ct);
    }

    /// <summary>
    /// Records the scorer the client asked for on the session.
    ///
    /// Not saved here: the entity is tracked, and every path that gets far enough to
    /// matter saves it. A turn refused by the guardrails never reaches a scorer, so
    /// losing the setting on that path costs nothing — the client sends it again
    /// with the next question.
    /// </summary>
    private void ApplyScoringMode(PortalChatSession session, string? scoringMode)
    {
        if (scoringMode is null) return;

        var mode = ScoringModes.Normalise(scoringMode);
        if (session.ScoringMode == mode) return;

        _logger.LogInformation("Session {SessionId} switched to {Mode} scoring.", session.Id, mode);
        session.ScoringMode = mode;
    }

    // -- resume upload ------------------------------------------------------

    public async Task HandleResumeAsync(
        string sessionId, int resumeId, IChatSink sink,
        string? scoringMode = null, CancellationToken ct = default)
    {
        var session = await LoadSessionAsync(sessionId, ct);
        var resume = await _resumes.GetAsync(resumeId, ct);

        if (resume is null)
        {
            await sink.ErrorAsync("That resume could not be found. Try uploading it again.", ct);
            return;
        }

        session.ResumeId = resume.Id;
        ApplyScoringMode(session, scoringMode);
        await _db.SaveChangesAsync(ct);

        // The narration reports what the pipeline actually found, at the moment it
        // found it. A fixed script on a timer would be theatre, and it would keep
        // saying "5 years of experience" for a resume that stated none.
        await sink.ThoughtAsync("parse", $"Reading {resume.Filename}...", ct);

        var facts = new List<string>();
        if (!string.IsNullOrWhiteSpace(resume.CurrentTitle)) facts.Add($"you are a {resume.CurrentTitle}");
        if (resume.YearsExperience is > 0) facts.Add($"about {resume.YearsExperience:0.#} years of experience");

        var skills = Documents.TextStructure.ReadLines(resume.Skills);
        if (skills.Count > 0) facts.Add($"{skills.Count} skills including {string.Join(", ", skills.Take(4))}");

        await sink.ThoughtAsync("profile", facts.Count > 0
            ? "I can see " + string.Join("; ", facts) + "."
            : "I read the document but could not pull out a clear profile. I will match on the full text.", ct);

        await sink.ActionAsync(new UiActionDto("resume_ready", new { resumeId = resume.Id }), ct);

        await RunSearchAsync(session, resume, ReadFilters(session), userMessage: "", sink, ct);
    }

    public async Task<ChatSessionDto> ResetSessionAsync(string sessionId, CancellationToken ct = default)
    {
        var session = await LoadSessionAsync(sessionId, ct);

        var messages = await _db.ChatMessages.Where(m => m.SessionId == session.Id).ToListAsync(ct);
        if (messages.Count > 0) _db.ChatMessages.RemoveRange(messages);

        // The filters and the stored result are conversation state too: leaving
        // them would make a "new" chat quietly inherit "remote only" from the last
        // one, and let an explain-follow-up answer about a ranking that is no
        // longer on screen.
        session.FiltersJson = "{}";
        session.LastMatchesJson = "[]";

        // The CV goes with it. Keeping it seemed kinder — nobody wants to re-upload
        // to ask a fresh question — but it meant a new chat still answered as the
        // last person, and the only way out was a Replace button on a document the
        // candidate thought they had cleared. Starting over means starting over.
        session.ResumeId = null;

        session.UpdatedAt = DateTime.UtcNow;

        await _db.SaveChangesAsync(ct);

        return await GetOrCreateSessionAsync(sessionId, ct);
    }

    // -- message ------------------------------------------------------------

    public async Task HandleMessageAsync(
        string sessionId, string message, IChatSink sink,
        string? scoringMode = null, CancellationToken ct = default)
    {
        var session = await LoadSessionAsync(sessionId, ct);
        ApplyScoringMode(session, scoringMode);
        var resume = session.ResumeId is { } id ? await _resumes.GetAsync(id, ct) : null;

        var lexicon = await JobBoardLexicon.BuildAsync(_db, ct);

        // Refuse before doing any work. Injection, questions about other candidates
        // and infrastructure fishing never reach a model or the database.
        var gate = ChatGuardrails.InspectInput(message);
        if (!gate.Allowed)
        {
            _logger.LogInformation("Refused a message ({Reason}) for session {SessionId}",
                gate.Reason, sessionId);
            await StreamPlainAsync(gate.Message, sink, ct);
            await sink.CompleteAsync(gate.Message, ct);
            await SaveTurnAsync(session, message, gate.Message, ReadFilters(session), null, ct);
            return;
        }

        // The intelligent path: understand the turn in context, execute deterministically,
        // then write the answer from what was found. Handles follow-ups the five-intent
        // regex cascade could not, and returns false whenever it cannot — so the
        // original routing below remains the floor rather than being replaced by it.
        if (await TryIntelligentTurnAsync(session, message, lexicon, sink, ct)) return;

        var parsed = _understanding.Parse(message, ReadFilters(session), lexicon);

        if (resume is null)
        {
            // The board is browsable without a CV. Answering "show me remote Python
            // roles" with a request for a document turns the front door into a gate,
            // and the postings needed to answer it are already indexed.
            await BrowseAsync(session, parsed.Filters, message, sink, ct);
            return;
        }

        switch (parsed.Intent)
        {
            case ChatIntent.Explain:
                await ExplainAsync(session, parsed, message, sink, ct);
                break;

            case ChatIntent.Advise:
                await AdviseAsync(session, resume, message, sink, ct);
                break;

            case ChatIntent.Chitchat when parsed.Filters.IsEmpty:
                await ChitchatAsync(session, message, sink, ct);
                break;

            default:
                if (parsed.ResetFilters)
                {
                    await sink.ActionAsync(new UiActionDto("update_filters", new JobFilters()), ct);
                }
                await RunSearchAsync(session, resume, parsed.Filters, message, sink, ct);
                break;
        }
    }

    /// <summary>
    /// The planner → capability → composer path.
    ///
    /// Returns true when it answered. Returns FALSE for anything it does not own —
    /// searching, refining, browsing without a CV — so those keep running through the
    /// paths that already work. This is deliberately additive: the turn types that were
    /// fine are untouched, and the ones that were broken get the new pipeline.
    ///
    /// Every exit is safe. No model, an unroutable message, no posting to anchor to,
    /// or an answer that failed grounding all end with the deterministic reply.
    /// </summary>
    private async Task<bool> TryIntelligentTurnAsync(
        PortalChatSession session, string message, JobBoardLexicon lexicon,
        IChatSink sink, CancellationToken ct)
    {
        if (!_llm.ChatAvailable) return false;

        var lastMatches = ReadLastMatches(session);
        var visible = lastMatches
            .Select(m => (m.Job.Id, m.Job.Title, m.Job.Company))
            .ToList();

        var knownJobIds = await _db.Jobs.AsNoTracking()
            .Where(j => j.IsPublished)
            .Select(j => j.Id)
            .ToListAsync(ct);

        var history = await _db.ChatMessages.AsNoTracking()
            .Where(m => m.SessionId == session.Id)
            .OrderByDescending(m => m.Id)
            .Take(6)
            .Select(m => new { m.Role, m.Content })
            .ToListAsync(ct);

        await sink.ThoughtAsync("understanding", "Working out what you're asking", ct);

        var plan = await _planner.PlanAsync(
            message, ReadFilters(session), lexicon, visible, knownJobIds.ToHashSet(),
            history.AsEnumerable().Reverse().Select(m => (m.Role, m.Content)).ToList(),
            session.FocusJobId, ct);

        // Search, refine and browse already work well and own state this method does
        // not manage. Handing them over would be change for its own sake.
        if (plan.Action is TurnAction.Search or TurnAction.Refine or TurnAction.Chitchat) return false;

        if (plan.Action == TurnAction.Clarify)
        {
            var ask = plan.ClarifyPrompt ?? "Which role did you mean?";
            await StreamPlainAsync(ask, sink, ct);
            await sink.CompleteAsync(ask, ct);
            // The options are chips, not prose. Re-typing a role name is what failed
            // to resolve in the first place.
            await sink.FollowUpsAsync(ChatFollowUps.Clarify(visible, plan.Question), ct);
            await SaveTurnAsync(session, message, ask, ReadFilters(session), null, ct);
            return true;
        }

        await sink.ThoughtAsync("gathering", "Reading the postings that answer it", ct);

        var package = plan.Action switch
        {
            TurnAction.FactAbout => await _capabilities.FactAboutAsync(plan.JobIds, lastMatches, ct),
            TurnAction.CompanyFact => await _capabilities.CompanyFactAsync(plan.JobIds, ct),
            TurnAction.Compare => await _capabilities.CompareAsync(plan.JobIds, lastMatches, ct),
            TurnAction.GapPlan => await _capabilities.GapPlanAsync(plan.JobIds, lastMatches, ct),
            TurnAction.GapsAcrossShortlist => await _capabilities.GapsAcrossShortlistAsync(lastMatches, ct),
            TurnAction.SkillUpAcrossBoard => await _capabilities.SkillUpAcrossBoardAsync(lastMatches, ct),
            TurnAction.Explain => await _capabilities.ExplainAsync(plan.JobIds, lastMatches, ct),
            TurnAction.OpenQuestion => await _capabilities.OpenQuestionAsync(plan.JobIds, lastMatches, ct),
            _ => null,
        };

        // Nothing to anchor to. Falling through would answer from the old paths, which
        // for these actions means answering a different question — so say so instead.
        if (package is null)
        {
            const string nothing =
                "I don't have a posting to answer that against. Upload your CV or ask about " +
                "one of the roles on the board and I can be specific.";
            await StreamPlainAsync(nothing, sink, ct);
            await sink.CompleteAsync(nothing, ct);
            await SaveTurnAsync(session, message, nothing, ReadFilters(session), null, ct);
            return true;
        }

        var fallback = plan.Action switch
        {
            TurnAction.CompanyFact => AnswerComposer.DescribeCompany(package),
            TurnAction.GapPlan or TurnAction.GapsAcrossShortlist or TurnAction.SkillUpAcrossBoard
                => AnswerComposer.DescribeGaps(package),
            TurnAction.Explain or TurnAction.Compare => AnswerComposer.DescribeFit(package),
            _ => AnswerComposer.DescribeJobs(package),
        };

        await sink.ThoughtAsync("answering", "Putting it in plain terms", ct);

        var composed = await _composer.ComposeAsync(
            plan.Question, package, fallback,
            isCompanyQuestion: plan.Action == TurnAction.CompanyFact
                               || ChatGuardrails.LooksLikeCompanyQuestion(message),
            ct);

        await StreamPlainAsync(composed.Text, sink, ct);
        await sink.CompleteAsync(composed.Text, ct);

        // Remember what this turn was about, so the NEXT bare follow-up — "and is that
        // remote?" — has a subject. This is the single line that makes a conversation
        // out of a sequence of unrelated questions.
        if (plan.JobIds.Count > 0) session.FocusJobId = plan.JobIds[0];

        var focused = plan.JobIds.Count > 0
            ? lastMatches.FirstOrDefault(m => m.Job.Id == plan.JobIds[0])
            : null;
        await sink.FollowUpsAsync(ChatFollowUps.For(lastMatches, focused, plan.Action), ct);
        await SaveTurnAsync(session, message, composed.Text, ReadFilters(session), null, ct);

        _logger.LogInformation(
            "Chat turn [{Action}] session={SessionId} jobs={Jobs} generated={Generated} fallbackPlanner={Fallback}",
            plan.Action, session.Id, plan.JobIds.Count, composed.Generated, plan.FromFallback);

        return true;
    }

    // -- browsing, with no resume -------------------------------------------

    /// <summary>
    /// Answers from the postings alone.
    ///
    /// Every number that normally appears — fit, skill verdicts, gaps — is absent
    /// here, because all of them are claims about a candidate and there is no
    /// candidate yet. What the reply does instead is state plainly what uploading
    /// a CV would add, so the ask is a reason rather than a gate.
    /// </summary>
    private async Task BrowseAsync(
        PortalChatSession session, JobFilters filters, string userMessage,
        IChatSink sink, CancellationToken ct)
    {
        await sink.ThoughtAsync("browse",
            filters.IsEmpty
                ? "No CV yet, so I am searching the board on your words alone..."
                : $"No CV yet — searching the board for {Describe(filters)}...", ct);

        JobSuggestionResultDto suggestions;
        try
        {
            suggestions = await _suggestions.SuggestAsync(userMessage, filters, 5, ct);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Browsing failed for session {SessionId}", session.Id);
            await sink.ErrorAsync("Something went wrong while searching. Try again in a moment.", ct);
            return;
        }

        if (!suggestions.Ranked && suggestions.Jobs.Count > 0)
        {
            // Said out loud rather than presented as relevance: without a usable
            // query these are simply the newest postings.
            await sink.ThoughtAsync("browse", "Showing the newest postings rather than a ranking.", ct);
        }

        await sink.ThoughtAsync("rank",
            suggestions.TotalMatching == 0
                ? "Nothing on the board matches that."
                : $"{suggestions.TotalMatching} role(s) match. Showing {suggestions.Jobs.Count}.", ct);

        var deterministic = suggestions.Jobs.Count == 0
            ? "Nothing on the board matches that yet. Try relaxing one of the constraints, or drop " +
              "your CV in and I will find the closest roles for you."
            : $"Here {(suggestions.Jobs.Count == 1 ? "is" : "are")} {suggestions.Jobs.Count} " +
              $"role{(suggestions.Jobs.Count == 1 ? "" : "s")} from the board. " +
              "Drop your CV in and I will score you against each one, show what you are missing, " +
              "and let you apply.";

        string reply;
        if (suggestions.Jobs.Count == 0)
        {
            reply = deterministic;
            await StreamPlainAsync(reply, sink, ct);
        }
        else
        {
            // The postings were chosen by retrieval; the model only introduces them.
            // It is given the ones already selected and cannot reach past that list.
            var facts = new StringBuilder();
            facts.AppendLine($"The candidate asked: \"{userMessage}\"");
            facts.AppendLine($"Roles found on the board ({suggestions.TotalMatching} match in total, showing these):");
            foreach (var job in suggestions.Jobs)
            {
                var skills = job.RequiredSkills.Where(IsSkillShaped).Take(4).ToList();
                facts.AppendLine(
                    $"- {job.Title} ({Or(job.Company, "company not stated")}), " +
                    $"{Or(job.Location, "location not stated")}, {job.WorkMode}" +
                    (skills.Count > 0 ? $" — asks for {string.Join(", ", skills)}" : ""));
            }

            reply = await StreamGroundedAsync(
                systemPrompt:
                    "You introduce job-board results to someone who has not uploaded a CV yet. The roles " +
                    "below were already found; your job is to present them.\n\n" +
                    "Rules:\n" +
                    "- Use ONLY these roles. Never invent a role, company or requirement.\n" +
                    "- Do NOT give a fit percentage or say how well they match — no CV has been read, " +
                    "so you know nothing about this person.\n" +
                    "- Two or three sentences. Say what the roles have in common, then invite them to " +
                    "add their CV to be scored against them.\n" +
                    "- Plain prose, no lists, no headings. A card for each role is shown below yours.",
                userPrompt: facts.ToString(),
                fallback: deterministic,
                sink: sink,
                ct: ct);
        }
        await sink.SuggestionsAsync(suggestions, ct);
        await sink.ActionAsync(new UiActionDto("update_filters", filters), ct);
        await SaveTurnAsync(session, userMessage, reply, filters, null, ct);
    }

    /// <summary>Renders a filter set as the phrase the narration uses.</summary>
    private static string Describe(JobFilters filters)
    {
        var parts = new List<string>();
        if (filters.WorkMode is { Length: > 0 } mode) parts.Add(mode.ToLowerInvariant());
        if (filters.SeniorityLevel is { Length: > 0 } level) parts.Add($"{level.ToLowerInvariant()} roles");

        // "Remote" arrives as both a work mode and a location from the query
        // parser, which would otherwise narrate as "remote, in Remote".
        if (filters.Location is { Length: > 0 } location &&
            !location.Equals(filters.WorkMode, StringComparison.OrdinalIgnoreCase))
        {
            parts.Add($"in {location}");
        }
        if (filters.EmploymentType is { Length: > 0 } type) parts.Add(type.ToLowerInvariant());
        if (filters.MinSalary is { } salary) parts.Add($"paying at least {salary:N0}");
        if (filters.Keywords is { Count: > 0 } keywords) parts.Add($"involving {string.Join(" and ", keywords)}");
        return parts.Count > 0 ? string.Join(", ", parts) : "anything";
    }

    // -- the search turn ----------------------------------------------------

    private async Task RunSearchAsync(
        PortalChatSession session, PortalResume resume, JobFilters filters,
        string userMessage, IChatSink sink, CancellationToken ct)
    {
        await sink.ThoughtAsync("search", DescribeSearch(resume, filters), ct);

        var mode = ScoringModes.Normalise(session.ScoringMode);

        // Both progressive modes stream their cards as each posting is judged, so
        // neither may send the whole list again afterwards.
        var progressive = mode is ScoringModes.Reasoned or ScoringModes.Rag;

        ReasonedShortlist? shortlist = null;
        Rag.RagShortlist? ragShortlist = null;

        MatchResultDto result;
        try
        {
            // Every emission carries the same batch id, so the client updates one
            // set of cards rather than stacking a new answer per role. The last one
            // is kept because it is also what the prose is written from.
            MatchResultDto? latest = null;
            Task Emit(MatchResultDto partial, CancellationToken token)
            {
                latest = partial;
                return sink.MatchesAsync(partial, token);
            }

            switch (mode)
            {
                case ScoringModes.Rag when await _rag.IsAvailableAsync(ct):
                    await sink.ThoughtAsync("retrieve",
                        "Searching the chunked index for the passages of your CV that bear on each " +
                        "posting's requirements, then judging those passages one posting at a time.", ct);

                    ragShortlist = await _rag.StartAsync(
                        resume, filters, Guid.NewGuid().ToString("N"), Emit, ct);

                    result = latest ?? EmptyResult(filters, ScoringModes.Rag);
                    break;

                case ScoringModes.Rag:
                    // Unavailable, and said so rather than quietly scored by a
                    // different engine. A number presented as RAG that was not
                    // produced by it is worse than an honest refusal.
                    await sink.ThoughtAsync("degraded",
                        "The RAG index is not reachable, so this answer is the computed score instead.", ct);
                    result = await _matching.MatchAsync(resume, filters, ct: ct);
                    progressive = false;
                    break;

                case ScoringModes.Reasoned:
                    await sink.ThoughtAsync("reason",
                        "Reading each posting against your CV requirement by requirement. " +
                        "The closest few are judged first — the rest keep arriving after this reply.", ct);

                    shortlist = await _reasoned.StartAsync(
                        resume, filters, Guid.NewGuid().ToString("N"), Emit, ct);

                    result = latest ?? EmptyResult(filters, ScoringModes.Reasoned);
                    break;

                default:
                    result = await _matching.MatchAsync(resume, filters, ct: ct);
                    break;
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex, "Matching failed for session {SessionId}", session.Id);
            await sink.ErrorAsync("Something went wrong while searching. Try again in a moment.", ct);
            return;
        }

        if (!result.SemanticMatching)
        {
            // Said out loud, because it changes what the results mean. Under the
            // fallback embedder a candidate who wrote "React" will not surface for
            // a posting asking for a "modern frontend framework", and that is a
            // capability difference the person deserves to know about.
            await sink.ThoughtAsync("degraded",
                "The embedding model is unavailable, so I am matching on wording rather than meaning. " +
                "Results will be narrower than usual.", ct);
        }

        await sink.ThoughtAsync("rank",
            result.TotalCandidates == 0
                ? "No roles cleared the relevance threshold."
                : $"{result.TotalCandidates} role(s) matched. Scoring the closest {result.Matches.Count} " +
                  "on skills, seniority and experience...", ct);

        // Under reasoned scoring the cards were already pushed, one per judgement.
        // Sending the same list again would make the last role appear twice.
        if (!progressive) await sink.MatchesAsync(result, ct);

        await sink.ActionAsync(new UiActionDto("update_filters", filters), ct);
        if (result.Matches.Count > 0)
        {
            await sink.ActionAsync(new UiActionDto("show_jobs",
                new { jobIds = result.Matches.Select(m => m.Job.Id).ToList() }), ct);
        }

        var reply = await ComposeSearchReplyAsync(resume, result, filters, sink, ct);
        await SaveTurnAsync(session, userMessage, reply, filters, result.Matches, ct);

        // The rest of the shortlist, after the turn is over.
        //
        // Deliberately not awaited: judging seven more roles is another minute or
        // two, and holding the turn open for it would leave the composer disabled
        // and the typing indicator spinning while the candidate has three scored
        // roles already on screen and something to ask about them.
        if (shortlist is { HasMore: true }) ContinueReasoningInBackground(session.Id, shortlist, sink);
        if (ragShortlist is { HasMore: true }) ContinueRagInBackground(session.Id, ragShortlist, sink);
    }

    /// <summary>
    /// Finishes judging a shortlist after its turn has ended.
    ///
    /// It runs in a NEW dependency-injection scope, which is the whole reason this
    /// is a method rather than three lines inline: the request's <see
    /// cref="PortalDbContext"/> is disposed the moment the hub invocation returns,
    /// and a background task holding it would fail on its first query. The SignalR
    /// client proxy behind the sink has no such lifetime — it stays valid as long as
    /// the browser is connected — so results can still be delivered.
    ///
    /// Nothing awaits this and nothing can fail because of it. A background pass that
    /// dies takes its remaining roles with it and leaves the ones already judged
    /// exactly as they are.
    /// </summary>
    private void ContinueReasoningInBackground(string sessionId, ReasonedShortlist shortlist, IChatSink sink)
    {
        _ = Task.Run(async () =>
        {
            using var scope = _scopes.CreateScope();
            var reasoned = scope.ServiceProvider.GetRequiredService<IReasonedMatchingService>();
            var logger = scope.ServiceProvider.GetRequiredService<ILogger<ChatOrchestrator>>();

            try
            {
                await reasoned.ContinueAsync(
                    shortlist,
                    (partial, token) => sink.MatchesAsync(partial, token),
                    CancellationToken.None);
            }
            catch (OperationCanceledException)
            {
                // The browser went away mid-pass. Nothing to report to.
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex,
                    "Background evaluation for session {SessionId} stopped early; {Remaining} role(s) unjudged.",
                    sessionId, shortlist.Remaining.Count);
            }
        });
    }

    /// <summary>
    /// Finishes judging a RAG shortlist after its turn has ended.
    ///
    /// Same shape and same reasoning as <see cref="ContinueReasoningInBackground"/>:
    /// a fresh DI scope, because the request's DbContexts are disposed the moment
    /// the hub invocation returns, and nothing awaits it because judging the rest
    /// is another minute the candidate should not spend watching a spinner.
    /// </summary>
    private void ContinueRagInBackground(string sessionId, Rag.RagShortlist shortlist, IChatSink sink)
    {
        _ = Task.Run(async () =>
        {
            using var scope = _scopes.CreateScope();
            var rag = scope.ServiceProvider.GetRequiredService<Rag.IRagMatchingService>();
            var logger = scope.ServiceProvider.GetRequiredService<ILogger<ChatOrchestrator>>();

            try
            {
                await rag.ContinueAsync(
                    shortlist, (partial, token) => sink.MatchesAsync(partial, token), CancellationToken.None);
            }
            catch (OperationCanceledException) { /* the browser went away */ }
            catch (Exception ex)
            {
                logger.LogWarning(ex,
                    "Background RAG evaluation for session {SessionId} stopped early; {Remaining} posting(s) unjudged.",
                    sessionId, shortlist.RemainingJobIds.Count);
            }
        });
    }

    /// <summary>A shortlist that found nothing, in the mode that looked.</summary>
    private static MatchResultDto EmptyResult(JobFilters filters, string scoringMode) =>
        new(Array.Empty<JobMatchDto>(), 0, filters, false, "", scoringMode);

    private static string DescribeSearch(PortalResume resume, JobFilters filters)
    {
        var subject = string.IsNullOrWhiteSpace(resume.CurrentTitle)
            ? "roles that fit your profile"
            : $"roles close to {resume.CurrentTitle}";

        var constraints = DescribeFilters(filters);
        return constraints.Count == 0
            ? $"Searching the board for {subject}..."
            : $"Searching for {subject}, {string.Join(", ", constraints)}...";
    }

    private static List<string> DescribeFilters(JobFilters filters)
    {
        var parts = new List<string>();
        if (filters.WorkMode is { Length: > 0 }) parts.Add(filters.WorkMode.ToLowerInvariant());
        if (filters.Location is { Length: > 0 }) parts.Add("in " + filters.Location);
        if (filters.EmploymentType is { Length: > 0 }) parts.Add(filters.EmploymentType.ToLowerInvariant());
        if (filters.SeniorityLevel is { Length: > 0 }) parts.Add(filters.SeniorityLevel.ToLowerInvariant() + " level");
        if (filters.MinSalary is { } salary) parts.Add($"paying at least {salary:N0}");
        if (filters.Keywords is { Count: > 0 }) parts.Add("involving " + string.Join(" and ", filters.Keywords));
        return parts;
    }

    /// <summary>
    /// Builds the visible reply: a natural-language opening (streamed from the
    /// model when one is available, deterministic otherwise) followed by a
    /// comparison table and per-role recruiter notes that are ALWAYS composed
    /// here, from the scored result.
    /// </summary>
    private async Task<string> ComposeSearchReplyAsync(
        PortalResume resume, MatchResultDto result, JobFilters filters, IChatSink sink, CancellationToken ct)
    {
        var assembled = new StringBuilder();

        if (result.Matches.Count == 0)
        {
            var empty = ComposeEmptyReply(filters);
            await StreamPlainAsync(empty, sink, ct);
            await sink.CompleteAsync(empty, ct);
            return empty;
        }

        var opening = await StreamOpeningAsync(resume, result, sink, ct);
        assembled.Append(opening);

        var body = new StringBuilder();
        body.AppendLine();
        body.AppendLine();
        body.AppendLine(ComposeTable(result.Matches));
        body.AppendLine();

        foreach (var match in result.Matches)
        {
            body.AppendLine($"**{match.Job.Title}** — {match.FitScore:0.#}% ({match.FitBand})  ");
            body.AppendLine(match.RecruiterNote);
            body.AppendLine();
        }

        // Conversational filtering: too many results is its own problem, and
        // dumping fifty roles on someone is how a useful match gets lost. The
        // question is asked only when narrowing would actually help, and it asks
        // about a dimension the board can genuinely distinguish.
        if (result.TotalCandidates > _options.ConversationalFilterThreshold)
        {
            var question = ComposeNarrowingQuestion(result, filters);
            if (question is not null)
            {
                body.AppendLine($"_I found {result.TotalCandidates} matches in total._ {question}");
                body.AppendLine();
            }
        }

        var rest = body.ToString();
        await StreamPlainAsync(rest, sink, ct);
        assembled.Append(rest);

        var full = assembled.ToString();
        await sink.CompleteAsync(full, ct);
        return full;
    }

    private static string ComposeEmptyReply(JobFilters filters)
    {
        var constraints = DescribeFilters(filters);
        return constraints.Count == 0
            ? "I could not find any roles on the board that match your profile closely enough to be worth " +
              "your time. That usually means the board is thin in your area rather than anything about " +
              "your resume — check back as new roles are posted."
            : $"Nothing on the board matches once I apply: {string.Join(", ", constraints)}. " +
              "Say **show me everything** to drop those constraints and see the closest roles overall.";
    }

    private static string ComposeTable(IReadOnlyList<JobMatchDto> matches)
    {
        var table = new StringBuilder();
        table.AppendLine("| Role | Company | Location | Fit | Missing |");
        table.AppendLine("| --- | --- | --- | --- | --- |");

        foreach (var match in matches)
        {
            var gaps = match.Gaps.Count > 0 ? string.Join(", ", match.Gaps.Take(3)) : "—";
            table.AppendLine(
                $"| {Cell(match.Job.Title)} " +
                $"| {Cell(Or(match.Job.Company, "—"))} " +
                $"| {Cell(Or(match.Job.Location, match.Job.WorkMode))} " +
                $"| {match.FitScore:0.#}% " +
                $"| {Cell(gaps)} |");
        }

        return table.ToString();
    }

    // A pipe inside a cell ends the cell early and shifts every column after it.
    private static string Cell(string value) => value.Replace("|", "\\|").Replace("\n", " ");

    private static string Or(string value, string fallback) =>
        string.IsNullOrWhiteSpace(value) ? fallback : value;

    /// <summary>
    /// Picks a narrowing question that the board can actually answer.
    ///
    /// Asking "remote or hybrid?" when every posting is remote wastes a turn and
    /// makes the bot look like it is not reading its own results, so a dimension
    /// only qualifies if the current results genuinely differ along it.
    /// </summary>
    private static string? ComposeNarrowingQuestion(MatchResultDto result, JobFilters filters)
    {
        var jobs = result.Matches.Select(m => m.Job).ToList();

        if (filters.WorkMode is null)
        {
            var modes = jobs.Select(j => j.WorkMode)
                .Where(m => !m.Equals("Unspecified", StringComparison.OrdinalIgnoreCase))
                .Distinct(StringComparer.OrdinalIgnoreCase).ToList();

            if (modes.Count > 1)
                return "Are you looking for remote work only, or is hybrid okay?";
        }

        if (filters.SeniorityLevel is null)
        {
            var levels = jobs.Select(j => j.SeniorityLevel)
                .Where(l => !l.Equals("Unspecified", StringComparison.OrdinalIgnoreCase))
                .Distinct(StringComparer.OrdinalIgnoreCase).ToList();

            if (levels.Count > 1)
                return $"These span {string.Join(" and ", levels.Take(3))} levels — which are you targeting?";
        }

        if (filters.Location is null)
        {
            var locations = jobs.Select(j => j.Location)
                .Where(l => !string.IsNullOrWhiteSpace(l))
                .Distinct(StringComparer.OrdinalIgnoreCase).ToList();

            if (locations.Count > 1)
                return $"They are spread across {string.Join(", ", locations.Take(3))} — anywhere in particular?";
        }

        // Nothing left that would meaningfully narrow the list. Saying nothing is
        // better than asking a question whose answer changes nothing.
        return null;
    }

    // -- advise -------------------------------------------------------------

    /// <summary>
    /// "What suits me?" and "what should I learn?"
    ///
    /// Both are answered from the stored result, for the same reason
    /// <see cref="ExplainAsync"/> is: the question is about roles the candidate is
    /// looking at, and re-running retrieval would answer about a different ranking.
    ///
    /// The advice itself is not invented either. The roles come from the stored
    /// matches in their scored order; the skills to learn are the requirements that
    /// actually came back <i>Missing</i>, counted across those roles so the one
    /// blocking the most opportunities is named first. The model turns that ordered
    /// evidence into a paragraph — it does not decide what is in it.
    /// </summary>
    private async Task AdviseAsync(
        PortalChatSession session, PortalResume resume, string userMessage,
        IChatSink sink, CancellationToken ct)
    {
        var matches = ReadLastMatches(session);

        // Nothing scored yet: run the search first, so the advice has something to
        // be about. Answering "what suits me" with a guess would be the exact
        // failure this whole design exists to prevent.
        if (matches.Count == 0)
        {
            await sink.ThoughtAsync("advise", "No scored results yet — searching first...", ct);
            await RunSearchAsync(session, resume, ReadFilters(session), userMessage, sink, ct);
            return;
        }

        await sink.ThoughtAsync("advise",
            $"Reading your last {matches.Count} result(s) rather than searching again...", ct);

        // How many of the candidate's matches each missing requirement blocks.
        // A skill that is missing from four roles is worth more than one missing
        // from a single outlier, and that ordering is arithmetic, not judgement.
        var gapWeight = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var match in matches)
        {
            foreach (var gap in match.Gaps.Where(IsSkillShaped).Distinct(StringComparer.OrdinalIgnoreCase))
            {
                gapWeight[gap] = gapWeight.GetValueOrDefault(gap) + 1;
            }
        }

        var ranked = gapWeight.OrderByDescending(g => g.Value).ThenBy(g => g.Key).Take(6).ToList();

        await sink.ThoughtAsync("advise", ranked.Count == 0
            ? "Nothing is consistently missing across these roles."
            : $"{ranked.Count} requirement(s) recur across them; ordering by how many roles each blocks.", ct);

        var facts = new StringBuilder();
        facts.AppendLine($"Candidate: {Or(resume.CurrentTitle, "role not stated")}, " +
                         $"{(resume.YearsExperience is { } y ? $"{y:0.#} years" : "years not stated")}.");
        facts.AppendLine("Their strongest matches, best first:");
        foreach (var match in matches.Take(5))
        {
            facts.AppendLine($"- {match.Job.Title} ({Or(match.Job.Company, "company not stated")}) " +
                             $"— {match.FitScore:0.#}% {match.FitBand}");
        }
        facts.AppendLine();
        facts.AppendLine(ranked.Count == 0
            ? "No requirement is missing across these roles."
            : "Requirements they do NOT evidence, with how many of the above roles each blocks:");
        foreach (var (skill, count) in ranked)
        {
            facts.AppendLine($"- {skill} — blocks {count} of {matches.Count}");
        }

        var deterministic = ComposeAdvice(matches, ranked);

        var reply = await StreamGroundedAsync(
            systemPrompt:
                "You are advising a job candidate about their own scored results. You are given the " +
                "complete list; it is already computed.\n\n" +
                "Rules:\n" +
                "- Use ONLY these facts. Never invent a role, company, percentage or skill.\n" +
                "- Name the roles that suit them best and say why, using the fit percentages given.\n" +
                "- Then say which skills to learn first, in the order given — that order is how many " +
                "of their matches each one blocks, so lead with the highest.\n" +
                "- Be direct and encouraging without overpromising. Under 150 words. Plain prose, " +
                "short paragraphs, no headings, no bullet lists.",
            userPrompt: facts.ToString(),
            fallback: deterministic,
            sink: sink,
            ct: ct);

        await SaveTurnAsync(session, userMessage, reply, ReadFilters(session), null, ct);
    }

    /// <summary>The same advice, composed, for when no model is configured.</summary>
    private static string ComposeAdvice(
        IReadOnlyList<JobMatchDto> matches, IReadOnlyList<KeyValuePair<string, int>> gaps)
    {
        var text = new StringBuilder();
        var top = matches[0];

        text.Append($"Your strongest match is **{top.Job.Title}**");
        if (!string.IsNullOrWhiteSpace(top.Job.Company)) text.Append($" at {top.Job.Company}");
        text.Append($" at **{top.FitScore:0.#}%** ({top.FitBand}).");

        if (matches.Count > 1)
        {
            text.Append($" {matches[1].Job.Title} follows at {matches[1].FitScore:0.#}%.");
        }
        text.AppendLine();
        text.AppendLine();

        if (gaps.Count == 0)
        {
            text.Append("Nothing is consistently missing across these roles — your gaps are " +
                        "role-specific rather than a pattern.");
            return text.ToString();
        }

        text.Append("The requirement blocking the most of these roles is " +
                    $"**{gaps[0].Key}** ({gaps[0].Value} of {matches.Count}). ");

        if (gaps.Count > 1)
        {
            text.Append("After that: " +
                        string.Join(", ", gaps.Skip(1).Take(3).Select(g => $"{g.Key} ({g.Value})")) + ".");
        }

        return text.ToString();
    }

    /// <summary>
    /// Whether a gap is a real skill name rather than a whole responsibility
    /// sentence the extractor put into the requirements. Judged on shape — telling
    /// someone to go and learn "Thorough knowledge of employment laws and HR best
    /// practices" is not advice.
    /// </summary>
    private static bool IsSkillShaped(string value)
    {
        var trimmed = TextStructure.Collapse(value ?? "");
        if (trimmed.Length is < 2 or > 40) return false;
        return trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length <= 4;
    }

    // -- explain ------------------------------------------------------------

    /// <summary>
    /// Answers a question about a result already given, FROM that stored result.
    ///
    /// Re-running retrieval here was the tempting shortcut and it is the wrong
    /// one: scores would be recomputed while the candidate is looking at the old
    /// ones, and the explanation would end up defending a number that is no
    /// longer on their screen.
    /// </summary>
    private async Task ExplainAsync(
        PortalChatSession session, ParsedMessage parsed, string message, IChatSink sink, CancellationToken ct)
    {
        var matches = ReadLastMatches(session);
        if (matches.Count == 0)
        {
            const string nothingToExplain =
                "I have not shown you any roles yet, so there is nothing to explain. " +
                "Upload your resume and I will find some.";
            await StreamPlainAsync(nothingToExplain, sink, ct);
            await SaveTurnAsync(session, message, nothingToExplain, ReadFilters(session), null, ct);
            return;
        }

        // "the last one" is only resolvable against a specific list.
        var rank = parsed.ReferencedRank == -1 ? matches.Count : parsed.ReferencedRank ?? 1;
        var match = matches[Math.Clamp(rank, 1, matches.Count) - 1];

        await sink.ThoughtAsync("recall", $"Looking back at {match.Job.Title}...", ct);

        var grounded = ComposeExplanation(match);
        var reply = await StreamGroundedAsync(
            systemPrompt: ExplainSystemPrompt,
            userPrompt: $"""
                The candidate asked: "{message}"

                Here is the scoring for the role they are asking about. Explain it to them in
                plain language. Use only these facts.

                {grounded}
                """,
            fallback: grounded,
            sink, ct);

        await SaveTurnAsync(session, message, reply, ReadFilters(session), null, ct);
    }

    private static string ComposeExplanation(JobMatchDto match)
    {
        var text = new StringBuilder();
        text.AppendLine($"Role: {match.Job.Title}" +
                        (string.IsNullOrWhiteSpace(match.Job.Company) ? "" : $" at {match.Job.Company}"));
        text.AppendLine($"Overall fit: {match.FitScore:0.#}% ({match.FitBand})");
        text.AppendLine($"Breakdown - semantic similarity {match.SemanticScore:0.#}%, " +
                        $"required-skill coverage {match.SkillScore:0.#}%, " +
                        $"title alignment {match.TitleScore:0.#}%, " +
                        $"experience {match.ExperienceScore:0.#}%");
        text.AppendLine();

        foreach (var skill in match.Skills)
        {
            var verdict = skill.Status switch
            {
                "Have" => "covered" + (skill.EvidenceSkill is { } s ? $" (your {s})" : ""),
                "Transferable" => $"not listed, but close to your {skill.EvidenceSkill}",
                _ => "not evidenced on your resume",
            };
            text.AppendLine($"- {skill.Skill}: {verdict}");
        }

        text.AppendLine();
        text.AppendLine(match.RecruiterNote);
        return text.ToString();
    }

    private const string ExplainSystemPrompt = """
        You are a job-search assistant explaining a match score to the candidate it
        belongs to. You are given the complete scoring for one role.

        Rules:
        - Use ONLY the facts given. Never invent a skill, a percentage, a company,
          a salary or a requirement.
        - Do not recompute or re-estimate the score. Explain the one you were given.
        - Be direct and encouraging, and never more than 150 words.
        - Speak to the candidate as "you".
        """;

    // -- chitchat -----------------------------------------------------------

    private async Task ChitchatAsync(
        PortalChatSession session, string message, IChatSink sink, CancellationToken ct)
    {
        const string fallback =
            "I match your resume against the roles on this board. Ask me to find jobs, narrow by " +
            "location, work mode or salary, or ask why any particular role scored the way it did.";

        var reply = await StreamGroundedAsync(
            systemPrompt: """
                You are the assistant for a job board. You can: match an uploaded resume against open
                roles, narrow results by work mode, location, seniority, employment type and salary,
                and explain why a role scored as it did.

                Rules:
                - Never claim a specific job, company or opening exists. You have not searched.
                - If asked something outside job matching, say briefly that it is not what you do.
                - At most 60 words.
                """,
            userPrompt: message,
            fallback,
            sink, ct);

        await SaveTurnAsync(session, message, reply, ReadFilters(session), null, ct);
    }

    // -- streaming ----------------------------------------------------------

    /// <summary>
    /// Streams the opening line of a search reply.
    ///
    /// The model is handed the findings and asked to introduce them. It is not
    /// asked what was found: every number in the prompt is already fixed, and the
    /// table that follows is built from the same data regardless of what comes
    /// back here.
    /// </summary>
    private async Task<string> StreamOpeningAsync(
        PortalResume resume, MatchResultDto result, IChatSink sink, CancellationToken ct)
    {
        var top = result.Matches[0];

        var deterministic =
            $"I found **{result.TotalCandidates}** role(s) that fit your profile. " +
            $"The closest is **{top.Job.Title}**" +
            (string.IsNullOrWhiteSpace(top.Job.Company) ? "" : $" at {top.Job.Company}") +
            $" at **{top.FitScore:0.#}%** ({top.FitBand}).";

        var facts = new StringBuilder();
        facts.AppendLine($"Candidate: {Or(resume.CurrentTitle, "role not stated")}, " +
                         $"{(resume.YearsExperience is { } y ? $"{y:0.#} years" : "years not stated")}.");
        facts.AppendLine($"Total matches: {result.TotalCandidates}. Showing: {result.Matches.Count}.");
        foreach (var match in result.Matches)
        {
            facts.AppendLine($"- {match.Job.Title} ({Or(match.Job.Company, "company not stated")}), " +
                             $"{match.FitScore:0.#}% {match.FitBand}, " +
                             $"gaps: {(match.Gaps.Count > 0 ? string.Join(", ", match.Gaps) : "none")}");
        }

        return await StreamGroundedAsync(
            systemPrompt: """
                You introduce job-search results to the candidate they belong to. You are given the
                complete result set, already scored.

                Rules:
                - Use ONLY these facts. Never invent a role, company, percentage or requirement.
                - Two sentences at most. A detailed table follows yours, so do not list the roles.
                - Lead with the strongest match and its fit percentage.
                - Speak to the candidate as "you". No preamble, no greeting.
                """,
            userPrompt: facts.ToString(),
            fallback: deterministic,
            sink, ct);
    }

    /// <summary>
    /// Streams a model reply, falling back to the given deterministic text if the
    /// model is absent or produces nothing.
    ///
    /// Nothing is emitted until the first real token arrives, so a failure shows
    /// the fallback cleanly rather than a half-sentence followed by a different
    /// half-sentence.
    /// </summary>
    private async Task<string> StreamGroundedAsync(
        string systemPrompt, string userPrompt, string fallback, IChatSink sink, CancellationToken ct)
    {
        if (!_llm.ChatAvailable)
        {
            await StreamPlainAsync(fallback, sink, ct);
            return fallback;
        }

        var buffer = new StringBuilder();
        try
        {
            var turns = new[]
            {
                new ChatTurn("system", systemPrompt),
                new ChatTurn("user", userPrompt),
            };

            await foreach (var token in _llm.StreamChatAsync(turns, ct))
            {
                buffer.Append(token);
                await sink.TokenAsync(token, ct);
            }
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Streaming reply failed part-way through.");
        }

        if (buffer.Length > 0) return buffer.ToString();

        await StreamPlainAsync(fallback, sink, ct);
        return fallback;
    }

    /// <summary>
    /// Sends fixed text through the same token channel the model uses.
    ///
    /// Chunked by word so the deterministic path animates like the model path.
    /// This is presentation, and it is worth it: a UI that renders one path
    /// smoothly and the other as an instant wall of text makes the fallback look
    /// broken when it is working exactly as designed.
    /// </summary>
    private static async Task StreamPlainAsync(string text, IChatSink sink, CancellationToken ct)
    {
        if (string.IsNullOrEmpty(text)) return;

        var parts = text.Split(' ');
        for (var i = 0; i < parts.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            await sink.TokenAsync(i == parts.Length - 1 ? parts[i] : parts[i] + " ", ct);
        }
    }
}
