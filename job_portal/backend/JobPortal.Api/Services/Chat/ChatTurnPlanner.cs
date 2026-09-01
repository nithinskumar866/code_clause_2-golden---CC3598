using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using JobPortal.Api.Contracts;
using JobPortal.Api.Services.Llm;

namespace JobPortal.Api.Services.Chat;

/// <summary>What the turn is asking for. One entry per executor.</summary>
public enum TurnAction
{
    /// <summary>Find or re-find roles for the uploaded CV.</summary>
    Search,
    /// <summary>Narrow, widen or undo the current filters.</summary>
    Refine,
    /// <summary>A property of one posting: pay, location, requirements.</summary>
    FactAbout,
    /// <summary>What the JD says about the employer — never what the model knows.</summary>
    CompanyFact,
    /// <summary>Side by side on fit, gaps and requirements.</summary>
    Compare,
    /// <summary>What to learn for ONE posting.</summary>
    GapPlan,
    /// <summary>What is missing across the roles currently on screen.</summary>
    GapsAcrossShortlist,
    /// <summary>What to learn next, ranked by demand across the whole board.</summary>
    SkillUpAcrossBoard,
    /// <summary>Why a score or a verdict is what it is. Answered from stored facts.</summary>
    Explain,
    /// <summary>Anything else, answered from an anchored package or not at all.</summary>
    OpenQuestion,
    /// <summary>No anchor resolved. Ask the candidate which role they mean.</summary>
    Clarify,
    /// <summary>Greetings and portal questions.</summary>
    Chitchat,
}

/// <summary>
/// A validated plan for one turn.
///
/// Every job id here has been checked against postings that actually exist, and every
/// filter value against <see cref="JobBoardLexicon"/>. A planner output that referred
/// to something unreal has already been discarded by the time this record is built.
/// </summary>
public sealed record TurnPlan(
    TurnAction Action,
    IReadOnlyList<int> JobIds,
    JobFilters Filters,
    bool ResetFilters,
    bool UndoFilters,
    /// <summary>The candidate's question, normalised. What COMPOSE answers.</summary>
    string Question,
    /// <summary>Filled when Action is Clarify.</summary>
    string? ClarifyPrompt = null,
    /// <summary>True when the deterministic parser produced this, not the model.</summary>
    bool FromFallback = false);

/// <summary>
/// Decides what a message is asking for.
///
/// WHY A MODEL AND NOT MORE REGEX
/// ------------------------------
/// The deterministic router (<see cref="ChatQueryUnderstanding"/>) is a keyword
/// cascade over five intents, and follow-ups are exactly where that shape fails.
/// "Why is this a good fit?" matches the explain pattern AND the advise pattern, so
/// declaration order decides the answer rather than meaning. Worse, the parser never
/// saw the conversation — it received the message and the current filters and nothing
/// else — so "what about the second one?" and "and in Chennai?" had nothing to
/// resolve against, and anything unrecognised fell through to a fresh search.
///
/// WHAT THE MODEL IS STILL NOT TRUSTED WITH
/// ----------------------------------------
/// It picks an action and points at things. It does not get to invent what those
/// things are. Every id it returns must be a posting that exists; every filter value
/// must be one the board actually contains. Anything else is dropped, and if nothing
/// survives, the turn becomes <see cref="TurnAction.Clarify"/> rather than a guess.
/// A hallucinated CONSTRAINT silently changes the question before the search runs,
/// which is the failure the deterministic parser was written to avoid — the model
/// does not get to reintroduce it.
///
/// The regex parser remains as the fallback for every failure path: no model, bad
/// JSON, a timeout, or an action nobody recognises. Behaviour then is exactly today's.
/// </summary>
public class ChatTurnPlanner
{
    private static readonly Regex JsonBlock = new(@"\{.*\}", RegexOptions.Compiled | RegexOptions.Singleline);

    private readonly IOllamaClient _llm;
    private readonly ChatQueryUnderstanding _fallback;
    private readonly ILogger<ChatTurnPlanner> _logger;

    public ChatTurnPlanner(IOllamaClient llm, ChatQueryUnderstanding fallback, ILogger<ChatTurnPlanner> logger)
    {
        _llm = llm;
        _fallback = fallback;
        _logger = logger;
    }

    private const string System = """
        You route a job-seeker's message to ONE action. You do not answer it.

        Actions:
          search               find roles for their CV
          refine               change filters (remote, salary, location, seniority, type)
          fact_about           a property of one posting: pay, location, hours, requirements
          company_fact         about the employer: culture, benefits, policies, where they are
          compare              two or more postings side by side
          gap_plan             what to learn for ONE named posting
          gaps_shortlist       what they lack across the roles on screen
          skill_up             what to learn next, across all open roles
          explain              why a score or verdict is what it is
          open_question        a real question about the roles that fits nothing above
          clarify              you cannot tell which posting they mean
          chitchat             greeting or a question about this site

        Rules:
        - job_ids MUST come from the CONTEXT below. Never invent an id.
        - Resolve references: "it", "that one", "the second" -> the id from the context.
        - Only set filters the message actually states. Never add one to be helpful.
        - If the message names a role that is NOT in the context, use clarify.
        - question: rewrite their message as a standalone question, keeping their words.

        Reply with JSON only:
        {"action":"...","job_ids":[1],"filters":{"workMode":"Remote|Hybrid|Onsite",
         "location":"...","employmentType":"Full-time|Part-time|Contract|Internship",
         "seniority":"Junior|Mid|Senior|Lead|Staff|Principal","minSalary":100000},
         "reset_filters":false,"undo_filters":false,"question":"...","clarify":"..."}
        Omit any field you are not setting.
        """;

    /// <summary>
    /// Plans a turn, falling back to the deterministic parser on any failure.
    /// </summary>
    /// <param name="visibleJobs">What is on screen — the ids "the second one" can mean.</param>
    /// <param name="knownJobIds">Every posting on the board. A named role may be off screen.</param>
    public async Task<TurnPlan> PlanAsync(
        string message,
        JobFilters current,
        JobBoardLexicon lexicon,
        IReadOnlyList<(int Id, string Title, string Company)> visibleJobs,
        IReadOnlySet<int> knownJobIds,
        IReadOnlyList<(string Role, string Content)> history,
        int? focusJobId,
        CancellationToken ct = default)
    {
        var deterministic = Fallback(message, current, lexicon, visibleJobs, focusJobId);

        if (!_llm.ChatAvailable) return deterministic;

        try
        {
            var reply = await _llm.ChatAsync(new[]
            {
                new ChatTurn("system", System),
                new ChatTurn("user", Context(message, current, visibleJobs, history, focusJobId)),
            }, jsonMode: true, ct: ct);

            var plan = Parse(reply, message, current, lexicon, visibleJobs, knownJobIds, focusJobId);
            if (plan is not null) return plan;

            _logger.LogInformation("Planner output unusable; using the deterministic parser.");
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Planner call failed; using the deterministic parser.");
        }

        return deterministic;
    }

    private static string Context(
        string message, JobFilters current,
        IReadOnlyList<(int Id, string Title, string Company)> visibleJobs,
        IReadOnlyList<(string Role, string Content)> history,
        int? focusJobId)
    {
        var text = new StringBuilder();

        if (history.Count > 0)
        {
            text.AppendLine("CONVERSATION SO FAR:");
            // Only the tail: a long transcript pushes the actual question out of a
            // small model's attention, and follow-ups depend on the recent turns.
            foreach (var (role, content) in history.TakeLast(6))
                text.AppendLine($"  {role}: {Clip(content, 300)}");
            text.AppendLine();
        }

        text.AppendLine("ROLES ON SCREEN (the only ids you may use):");
        if (visibleJobs.Count == 0) text.AppendLine("  (none yet)");
        foreach (var (id, title, company) in visibleJobs)
            text.AppendLine($"  id {id}: {title}{(string.IsNullOrWhiteSpace(company) ? "" : $" at {company}")}");

        if (focusJobId is not null) text.AppendLine($"\nCURRENTLY DISCUSSING: id {focusJobId}");

        var filters = Describe(current);
        text.AppendLine($"\nACTIVE FILTERS: {(filters.Length == 0 ? "none" : filters)}");
        text.AppendLine($"\nMESSAGE: {message}");
        return text.ToString();
    }

    private static string Describe(JobFilters f)
    {
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(f.WorkMode)) parts.Add(f.WorkMode);
        if (!string.IsNullOrWhiteSpace(f.Location)) parts.Add($"in {f.Location}");
        if (!string.IsNullOrWhiteSpace(f.EmploymentType)) parts.Add(f.EmploymentType);
        if (!string.IsNullOrWhiteSpace(f.SeniorityLevel)) parts.Add(f.SeniorityLevel);
        if (f.MinSalary is not null) parts.Add($"min salary {f.MinSalary:0}");
        return string.Join(", ", parts);
    }

    private static string Clip(string text, int limit) =>
        text.Length <= limit ? text : text[..limit] + "…";

    // --- Parsing and validation ---------------------------------------------------
    // --- Tolerant JSON reading ----------------------------------------------------
    // A strongly-typed Deserialize is the wrong tool here. With web defaults it binds
    // camelCase only, so the "job_ids" this prompt asks for silently arrived as an
    // empty list and every anchored action collapsed into a clarify. Small models
    // also drift between snake_case and camelCase between calls. So each field is
    // read by name with its plausible spellings, and anything unreadable is simply
    // absent — the same posture as the rest of this class.

    /// <summary>
    /// Descends through a wrapper object when the model nested its answer inside one.
    ///
    /// qwen2.5:14b, asked for {"action":...}, routinely replies
    /// {"response":{"action":"gap_plan","job_ids":[9],"status":"success"}}. The payload
    /// is correct; it just arrived one level down. Rejecting that would send a
    /// perfectly good plan to the regex fallback and make the model look worse than it
    /// is — the same class of tolerance as reading JSON out of prose and fences.
    ///
    /// Only unwraps when the outer object has NO action of its own and exactly one
    /// object-valued property, so a real plan is never mistaken for a wrapper.
    /// </summary>
    private static JsonElement Unwrap(JsonElement root)
    {
        for (var depth = 0; depth < 2; depth++)
        {
            if (Prop(root, "action", "Action") is not null) return root;

            var objects = root.EnumerateObject()
                .Where(p => p.Value.ValueKind == JsonValueKind.Object)
                .ToList();
            if (objects.Count != 1) return root;

            root = objects[0].Value;
        }
        return root;
    }

    private static JsonElement? Prop(JsonElement parent, params string[] names)
    {
        foreach (var name in names)
        {
            if (parent.TryGetProperty(name, out var value) && value.ValueKind != JsonValueKind.Null)
                return value;
        }
        return null;
    }

    private static string? Str(JsonElement parent, params string[] names) =>
        Prop(parent, names) is { ValueKind: JsonValueKind.String } e ? e.GetString() : null;

    private static bool Flag(JsonElement parent, params string[] names) =>
        Prop(parent, names) is { ValueKind: JsonValueKind.True };

    private static decimal? Money(JsonElement parent, params string[] names) =>
        Prop(parent, names) is { ValueKind: JsonValueKind.Number } e && e.TryGetDecimal(out var d) ? d : null;

    private static List<int> Ints(JsonElement parent, params string[] names)
    {
        var found = new List<int>();
        if (Prop(parent, names) is not { ValueKind: JsonValueKind.Array } array) return found;
        foreach (var item in array.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.Number && item.TryGetInt32(out var id)) found.Add(id);
            else if (item.ValueKind == JsonValueKind.String && int.TryParse(item.GetString(), out var parsed))
                found.Add(parsed);   // an 8B model quoting its numbers is not a reason to fail
        }
        return found;
    }

    private sealed record RawFilters(
        string? WorkMode, string? Location, string? EmploymentType,
        string? Seniority, decimal? MinSalary);

    // Closed vocabularies. The board stores these as free text extracted from prose,
    // so they are normalised against the set the UI and the filters actually use —
    // "wfh" or "remote-first" must become "Remote" or be dropped, never passed through.
    private static readonly string[] WorkModes = { "Remote", "Hybrid", "Onsite" };
    private static readonly string[] EmploymentTypes = { "Full-time", "Part-time", "Contract", "Internship" };
    private static readonly string[] Seniorities = { "Junior", "Mid", "Senior", "Lead", "Staff", "Principal" };

    private static string? Closest(string? value, IReadOnlyCollection<string> vocabulary) =>
        string.IsNullOrWhiteSpace(value)
            ? null
            : vocabulary.FirstOrDefault(v => v.Equals(value.Trim(), StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Turns the model's JSON into a plan, or null if it cannot be trusted.
    ///
    /// Null is a normal outcome, not an error: the caller has a deterministic plan
    /// already and loses nothing by using it.
    /// </summary>
    private TurnPlan? Parse(
        string? reply, string message, JobFilters current, JobBoardLexicon lexicon,
        IReadOnlyList<(int Id, string Title, string Company)> visibleJobs,
        IReadOnlySet<int> knownJobIds, int? focusJobId)
    {
        if (string.IsNullOrWhiteSpace(reply)) return null;

        // Small models fence their JSON or preface it with a sentence. Verified on the
        // pod: llama3.1:8b replied "Here is the JSON array:\n```\n[...]".
        var match = JsonBlock.Match(reply);
        if (!match.Success) return null;

        JsonDocument document;
        try { document = JsonDocument.Parse(match.Value); }
        catch (JsonException) { return null; }

        using var _ = document;
        var root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object) return null;
        root = Unwrap(root);

        var action = Map(Str(root, "action", "Action") ?? "");
        if (action is null) return null;

        var rawFilters = Prop(root, "filters", "Filters") is { ValueKind: JsonValueKind.Object } f
            ? new RawFilters(
                Str(f, "workMode", "work_mode", "mode"),
                Str(f, "location"),
                Str(f, "employmentType", "employment_type", "jobType", "job_type"),
                Str(f, "seniority", "seniorityLevel", "seniority_level"),
                Money(f, "minSalary", "min_salary", "salaryMin"))
            : null;

        // Ids: on-screen first, then anywhere on the board — a candidate can name a
        // role that scrolled off. Anything else the model produced is discarded.
        var visible = visibleJobs.Select(j => j.Id).ToHashSet();
        var ids = Ints(root, "job_ids", "jobIds", "jobids", "ids")
            .Where(id => visible.Contains(id) || knownJobIds.Contains(id))
            .Distinct()
            .ToList();

        var question = Str(root, "question");
        var clarify = Str(root, "clarify", "clarify_prompt", "clarifyPrompt");

        if (ids.Count == 0 && focusJobId is not null && NeedsAnchor(action.Value))
            ids.Add(focusJobId.Value);

        // An action about a posting, with no posting resolved, is the case the
        // clarify branch exists for. Answering it anyway is how a follow-up ends up
        // describing a role the candidate never asked about.
        if (NeedsAnchor(action.Value) && ids.Count == 0)
        {
            return new TurnPlan(
                TurnAction.Clarify, Array.Empty<int>(), current, false, false,
                Normalise(question, message),
                ClarifyPrompt: clarify ?? "Which role did you mean?");
        }

        return new TurnPlan(
            action.Value, ids, Validate(rawFilters, current, lexicon),
            Flag(root, "reset_filters", "resetFilters"),
            Flag(root, "undo_filters", "undoFilters"),
            Normalise(question, message), clarify);
    }

    /// <summary>Actions that are meaningless without a posting to be about.</summary>
    private static bool NeedsAnchor(TurnAction action) => action is
        TurnAction.FactAbout or TurnAction.CompanyFact or TurnAction.Compare or
        TurnAction.GapPlan or TurnAction.OpenQuestion;

    private static string Normalise(string? question, string message) =>
        string.IsNullOrWhiteSpace(question) ? message : question.Trim();

    private static TurnAction? Map(string action) => action.Trim().ToLowerInvariant() switch
    {
        "search" => TurnAction.Search,
        "refine" => TurnAction.Refine,
        "fact_about" => TurnAction.FactAbout,
        "company_fact" => TurnAction.CompanyFact,
        "compare" => TurnAction.Compare,
        "gap_plan" => TurnAction.GapPlan,
        "gaps_shortlist" => TurnAction.GapsAcrossShortlist,
        "skill_up" => TurnAction.SkillUpAcrossBoard,
        "explain" => TurnAction.Explain,
        "open_question" => TurnAction.OpenQuestion,
        "clarify" => TurnAction.Clarify,
        "chitchat" => TurnAction.Chitchat,
        _ => null,
    };

    /// <summary>
    /// Keeps only filter values the board can actually satisfy.
    ///
    /// This is the line the model may not cross. A location or job type it invented
    /// would silently narrow the search to nothing and the candidate would be shown an
    /// empty board with no indication why — the exact damage the deterministic parser
    /// was built to prevent.
    /// </summary>
    private static JobFilters Validate(RawFilters? raw, JobFilters current, JobBoardLexicon lexicon)
    {
        if (raw is null) return current;

        // A location the board has never heard of is discarded rather than applied.
        // Applying it would return an empty result the candidate cannot explain.
        var location = string.IsNullOrWhiteSpace(raw.Location)
            ? null
            : lexicon.Locations.FirstOrDefault(l => l.Equals(raw.Location.Trim(), StringComparison.OrdinalIgnoreCase))
              ?? lexicon.Locations.FirstOrDefault(l => l.Contains(raw.Location.Trim(), StringComparison.OrdinalIgnoreCase));

        return current with
        {
            WorkMode = Closest(raw.WorkMode, WorkModes) ?? current.WorkMode,
            Location = location ?? current.Location,
            EmploymentType = Closest(raw.EmploymentType, EmploymentTypes) ?? current.EmploymentType,
            SeniorityLevel = Closest(raw.Seniority, Seniorities) ?? current.SeniorityLevel,
            MinSalary = raw.MinSalary ?? current.MinSalary,
        };
    }

    /// <summary>
    /// The deterministic reading — today's behaviour, unchanged.
    ///
    /// Also the answer whenever the model is absent, slow or wrong, which is why the
    /// old parser is kept rather than deleted.
    /// </summary>
    private TurnPlan Fallback(
        string message, JobFilters current, JobBoardLexicon lexicon,
        IReadOnlyList<(int Id, string Title, string Company)> visibleJobs, int? focusJobId)
    {
        var parsed = _fallback.Parse(message, current, lexicon);

        var ids = new List<int>();
        if (parsed.ReferencedRank is { } rank && rank >= 1 && rank <= visibleJobs.Count)
            ids.Add(visibleJobs[rank - 1].Id);
        else if (focusJobId is not null)
            ids.Add(focusJobId.Value);

        var action = parsed.Intent switch
        {
            ChatIntent.Search => TurnAction.Search,
            ChatIntent.Refine => TurnAction.Refine,
            ChatIntent.Explain => TurnAction.Explain,
            ChatIntent.Advise => TurnAction.GapsAcrossShortlist,
            _ => TurnAction.Chitchat,
        };

        return new TurnPlan(
            action, ids, parsed.Filters, parsed.ResetFilters, UndoFilters: false,
            message, ClarifyPrompt: null, FromFallback: true);
    }
}
