using System.Text;
using System.Text.Json;
using JobPortal.Api.Data;
using JobPortal.Api.Services.Documents;
using JobPortal.Api.Options;
using JobPortal.Api.Services.Llm;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;

namespace JobPortal.Api.Services.Screening;

// ─────────────────────────────────────────────────────────────────────────────
//  PRE-APPLICATION SCREENING
//
//  Deliberately self-contained: the contract, the prompt, the generation, the
//  validation and the gate all live in this one file, and nothing outside it
//  changes. It is not wired into the existing apply flow — the intent is that it
//  is dropped into an application and connected there.
//
//  What it does: before someone applies, the posting is turned into FOUR yes/no
//  questions drawn from what that posting actually asks for — its location, its
//  work mode, its mandatory skills, its domain. The applicant must answer all
//  four affirmatively to proceed.
//
//  Why it is worth doing at all: a fit score tells a recruiter how well a CV
//  matches. It cannot tell them whether the person would actually relocate, or
//  wants a contract, or considers themselves a data scientist. Those are facts
//  only the applicant has, and asking is cheaper and more accurate than inferring.
//
//  The rule that shapes it: A QUESTION MUST COME FROM THE POSTING. The model
//  chooses the wording; it does not choose the subject. Every question generated
//  here is checked back against the job record, and one that asks about something
//  the posting never mentioned is discarded and replaced with a deterministic one.
//  A screening gate that invents its own requirements is not screening — it is
//  rejecting people for failing a test nobody set.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>What a screening question is about. Drives ordering and the fallbacks.</summary>
public static class ScreeningTopics
{
    /// <summary>Willingness to work at, or move to, the stated location.</summary>
    public const string Location = "location";
    /// <summary>Remote / hybrid / onsite, and whether that suits them.</summary>
    public const string WorkMode = "work_mode";
    /// <summary>A mandatory skill or technology the posting names.</summary>
    public const string Skill = "skill";
    /// <summary>Domain, seniority or years — whether this is their field.</summary>
    public const string Experience = "experience";
    /// <summary>Employment type: full-time, contract, internship.</summary>
    public const string EmploymentType = "employment_type";

    public static readonly string[] All =
        { Location, WorkMode, Skill, Experience, EmploymentType };

    public static string Normalise(string? value)
    {
        var v = (value ?? "").Trim().ToLowerInvariant().Replace(' ', '_').Replace('-', '_');
        return All.Contains(v) ? v : Skill;
    }
}

/// <summary>
/// One question, and the posting text it came from.
///
/// <paramref name="BasedOn"/> is not decoration. It is the applicant's answer to
/// "why are you asking me this?", and it is the thing this service checks the
/// model against — a question whose basis is not in the job record does not ship.
/// </summary>
public record ScreeningQuestion(
    int Id,
    string Question,
    string Topic,
    string BasedOn);

/// <summary>
/// Someone to talk to about this kind of role.
///
/// <paramref name="Relevant"/> marks the one whose remit matches THIS posting.
/// Both are offered — an applicant may want to ask about something else — but the
/// relevant one leads, because "remote queries go to John" is noise on an onsite
/// job in Chennai and the posting already told us which it is.
/// </summary>
public record ScreeningContact(
    string Name,
    string Url,
    string Handles,
    bool Relevant);

public record ScreeningQuestionSet(
    int JobId,
    string JobTitle,
    IReadOnlyList<ScreeningQuestion> Questions,
    /// <summary>True when a model wrote these, false when the deterministic set stood in.</summary>
    bool Generated,
    /// <summary>Who to ask, most relevant to this posting first. May be empty.</summary>
    IReadOnlyList<ScreeningContact> Contacts);

/// <summary>One answer. <paramref name="Yes"/> false is a decline, not a skip.</summary>
public record ScreeningAnswer(int QuestionId, bool Yes);

/// <summary>
/// The verdict.
///
/// <paramref name="Passed"/> means EVERY question was answered — not that every
/// answer was yes. A "no" is a valid, informative answer and does not block: the
/// questions exist to make someone state a position on the role's terms, and a
/// candidate who reads "Berlin, onsite", says no to relocating and applies anyway
/// has made a decision that is theirs to make.
///
/// <paramref name="Declined"/> carries those noes so they can travel with the
/// application. That is the value the gate actually produces — a recruiter
/// learning "would not relocate" up front is worth far more than a candidate
/// silently filtered out.
/// </summary>
public record ScreeningResult(
    bool Passed,
    IReadOnlyList<ScreeningQuestion> Declined,
    IReadOnlyList<int> Unanswered,
    string Message);

public interface IApplicationScreeningService
{
    /// <summary>The four questions for one posting. Never returns fewer than one.</summary>
    Task<ScreeningQuestionSet> BuildAsync(PortalJob job, CancellationToken ct = default);

    /// <summary>
    /// Whether these answers clear the gate. Re-derives the questions rather than
    /// trusting the client's copy of them, so a caller cannot pass by submitting
    /// answers to questions that were never asked.
    /// </summary>
    Task<ScreeningResult> EvaluateAsync(
        PortalJob job, IReadOnlyList<ScreeningAnswer> answers, CancellationToken ct = default);
}

public class ApplicationScreeningService : IApplicationScreeningService
{
    /// <summary>Bump when the prompt changes — it is part of the cache key.</summary>
    public const string PromptVersion = "screen-v4";

    /// <summary>
    /// Four, as specified. Not three and not six.
    ///
    /// The number is a product decision, not a technical one: a gate long enough to
    /// feel like a form is a gate people abandon, and every question past the fourth
    /// buys less than it costs in applications never sent.
    /// </summary>
    public const int QuestionCount = 4;

    private const int MaxJobChars = 3500;

    /// <summary>
    /// Questions depend only on the POSTING, never on the applicant, so one set
    /// serves everyone who opens that job until it is edited.
    /// </summary>
    private static readonly TimeSpan CacheLifetime = TimeSpan.FromHours(12);

    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private readonly IOllamaClient _llm;
    private readonly IMemoryCache _cache;
    private readonly ScreeningOptions _screening;
    private readonly ILogger<ApplicationScreeningService> _logger;

    public ApplicationScreeningService(
        IOllamaClient llm,
        IMemoryCache cache,
        IOptions<ScreeningOptions> screening,
        ILogger<ApplicationScreeningService> logger)
    {
        _llm = llm;
        _cache = cache;
        _screening = screening.Value;
        _logger = logger;
    }

    // -- the prompt ---------------------------------------------------------

    private const string System = """
        You write pre-application screening questions for ONE job posting.

        You are given the posting. Write exactly 4 questions the applicant answers
        YES or NO, covering the things a CV cannot tell anyone — willingness,
        preference and self-declared domain.

        WHAT TO ASK ABOUT, in this order of priority, using only what the posting states:
          1. Location — if the posting names a place, ask whether they are based there
             or willing to relocate. If it is remote, ask whether remote suits them.
          2. Work mode or employment type — remote/hybrid/onsite, full-time/contract —
             whenever the posting states one.
          3. Mandatory skills — ask about the MOST important named technology or
             qualification. For a framework, ask about the language underneath it:
             a Spring Boot role asks "Do you have hands-on experience with Java?".
          4. Domain or seniority — whether they work in this field, or have the years
             the posting asks for.

        RULES
        - EVERY question must trace to something written in the posting. Never ask
          about a skill, a place or a condition the posting does not mention.
        - ONE thing per question. Never join two skills with "and" or "/". "Do you
          have experience with Excel and SQL?" is forbidden — someone who knows one
          and not the other cannot answer it honestly, and their yes means nothing.
          Pick the single most important one.
        - Never ask twice about the same subject. If one question mentions Excel, no
          other question may mention Excel.
        - Answerable YES or NO. Not "how many years", not "describe your", and never
          a menu: "Do you prefer remote, hybrid, or onsite work?" has no no-answer.
          Ask "This role is remote. Does that suit you?" instead.
        - Second person, plain, under 20 words. No preamble, no numbering.
        - "basis" is the exact phrase from the posting that prompted the question.
        - Do not ask for personal information: age, nationality, marital status,
          health, salary history. Ask about the job, not about the person.

        Return ONLY this JSON:
        {"questions":[{"question":"...","topic":"location|work_mode|skill|experience|employment_type","basis":"<phrase from the posting>"}]}
        """;

    private static IReadOnlyList<ChatTurn> Prompt(PortalJob job)
    {
        var facts = new StringBuilder();
        facts.AppendLine($"Title: {job.Title}");
        if (Has(job.Company)) facts.AppendLine($"Company: {job.Company}");
        if (Has(job.Location)) facts.AppendLine($"Location: {job.Location}");
        if (IsStated(job.WorkMode)) facts.AppendLine($"Work mode: {job.WorkMode}");
        if (IsStated(job.EmploymentType)) facts.AppendLine($"Employment type: {job.EmploymentType}");
        if (IsStated(job.SeniorityLevel)) facts.AppendLine($"Seniority: {job.SeniorityLevel}");
        if (job.MinYearsExperience is { } min) facts.AppendLine($"Minimum years: {min:0.#}");

        var required = TextStructure.ReadLines(job.RequiredSkills);
        if (required.Count > 0) facts.AppendLine($"Required skills: {string.Join(", ", required)}");

        var preferred = TextStructure.ReadLines(job.PreferredSkills);
        if (preferred.Count > 0) facts.AppendLine($"Preferred skills: {string.Join(", ", preferred)}");

        facts.AppendLine();
        facts.AppendLine("Posting text:");
        facts.AppendLine(TextStructure.Clip(job.RawText, MaxJobChars));

        return new[] { new ChatTurn("system", System), new ChatTurn("user", facts.ToString()) };
    }

    // -- generation ---------------------------------------------------------

    public async Task<ScreeningQuestionSet> BuildAsync(PortalJob job, CancellationToken ct = default)
    {
        var key = $"screen|{PromptVersion}|{_llm.GetModelName()}|j{job.Id}:{job.UpdatedAt:O}";
        if (_cache.TryGetValue<ScreeningQuestionSet>(key, out var cached) && cached is not null) return cached;

        var deterministic = Deterministic(job);

        var set = await GenerateAsync(job, deterministic, ct) ?? new ScreeningQuestionSet(
            job.Id, job.Title, deterministic, Generated: false, Contacts: Array.Empty<ScreeningContact>());

        set = set with { Contacts = ContactsFor(job) };

        _cache.Set(key, set, CacheLifetime);
        return set;
    }

    private async Task<ScreeningQuestionSet?> GenerateAsync(
        PortalJob job, IReadOnlyList<ScreeningQuestion> fallback, CancellationToken ct)
    {
        if (!_llm.ChatAvailable) return null;

        string? reply;
        try
        {
            // Temperature 0: two applicants opening the same posting must be asked
            // the same questions. A gate whose wording drifts between people is not
            // one gate.
            reply = await _llm.ChatAsync(Prompt(job), jsonMode: true, temperature: 0, ct: ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Screening generation failed for job {JobId}.", job.Id);
            return null;
        }

        if (string.IsNullOrWhiteSpace(reply)) return null;

        var start = reply.IndexOf('{');
        var end = reply.LastIndexOf('}');
        if (start < 0 || end <= start) return null;

        var raw = ReadQuestions(reply[start..(end + 1)], job.Id);
        if (raw.Count == 0) return null;

        var kept = new List<ScreeningQuestion>();
        var seenTopics = new HashSet<string>();

        foreach (var q in raw)
        {
            if (kept.Count == QuestionCount) break;

            var text = TextStructure.Collapse(q.Question ?? "");
            if (text.Length < 8) continue;
            if (OffersAMenu(text)) continue;

            var topic = ScreeningTopics.Normalise(q.Topic);
            var basis = TextStructure.Collapse(q.Basis ?? "");

            // The check that makes this safe to put in front of an applicant: the
            // subject has to exist in the posting. A model that asked about AWS on a
            // posting that never mentions it would be blocking people over a
            // requirement the employer never set.
            if (!Grounded(job, text, basis))
            {
                _logger.LogWarning(
                    "Job {JobId}: dropped screening question '{Question}' — nothing in the posting supports it.",
                    job.Id, text);
                continue;
            }

            // Prefer one per topic: four questions about four required skills is a
            // quiz, and it leaves location and work mode unasked. A second pass below
            // relaxes this when the posting cannot fill four topics.
            if (!seenTopics.Add(topic)) continue;
            if (kept.Any(k => k.Question.Equals(text, StringComparison.OrdinalIgnoreCase))) continue;

            kept.Add(new ScreeningQuestion(kept.Count + 1, text, topic, basis));
        }

        if (kept.Count == 0) return null;

        // Second pass: the model's remaining questions, with skill repeats allowed.
        foreach (var q in raw)
        {
            if (kept.Count == QuestionCount) break;

            var text = TextStructure.Collapse(q.Question ?? "");
            if (OffersAMenu(text)) continue;
            if (text.Length < 8) continue;
            if (Repeats(kept, ScreeningTopics.Normalise(q.Topic))) continue;
            if (kept.Any(k => k.Question.Equals(text, StringComparison.OrdinalIgnoreCase))) continue;
            if (!Grounded(job, text, TextStructure.Collapse(q.Basis ?? ""))) continue;

            kept.Add(new ScreeningQuestion(
                kept.Count + 1, text, ScreeningTopics.Normalise(q.Topic), TextStructure.Collapse(q.Basis ?? "")));
        }

        // Still short: top up from the deterministic set rather than asking fewer.
        // The count is part of the contract. Matched on SUBJECT, not on sentence —
        // a filler asking about Excel must not follow a generated question that
        // already mentioned Excel.
        foreach (var filler in fallback)
        {
            if (kept.Count == QuestionCount) break;
            if (Repeats(kept, filler.Topic)) continue;
            if (AlreadyCovers(kept, filler.BasedOn)) continue;
            kept.Add(filler with { Id = kept.Count + 1 });
        }

        return new ScreeningQuestionSet(
            job.Id, job.Title, Renumber(kept), Generated: true, Contacts: Array.Empty<ScreeningContact>());
    }

    /// <summary>
    /// Reads the questions array, tolerating the shapes models actually return.
    ///
    /// Asked for objects, qwen returns a mix: some entries are objects, others are
    /// bare strings — the question with no topic or basis attached. Deserialising
    /// the whole array into a typed list throws on the first string and discards
    /// three perfectly good questions with it, which is how a live posting fell back
    /// to the deterministic set while the model was working fine.
    ///
    /// So each element is read on its own terms. A bare string keeps its text and
    /// loses its metadata, which the grounding check can still work without.
    /// </summary>
    private List<RawQuestion> ReadQuestions(string json, int jobId)
    {
        var questions = new List<RawQuestion>();

        try
        {
            using var doc = JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty("questions", out var array) ||
                array.ValueKind != JsonValueKind.Array)
            {
                return questions;
            }

            foreach (var element in array.EnumerateArray())
            {
                switch (element.ValueKind)
                {
                    case JsonValueKind.String:
                        questions.Add(new RawQuestion { Question = element.GetString() });
                        break;

                    case JsonValueKind.Object:
                        questions.Add(new RawQuestion
                        {
                            Question = Text(element, "question") ?? Text(element, "text"),
                            Topic = Text(element, "topic"),
                            Basis = Text(element, "basis") ?? Text(element, "based_on"),
                        });
                        break;
                }
            }
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Screening for job {JobId} returned unparseable JSON.", jobId);
        }

        return questions;

        static string? Text(JsonElement element, string name) =>
            element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
                ? value.GetString()
                : null;
    }

    /// <summary>
    /// Whether the posting actually contains what a question is about.
    ///
    /// Forgiving on purpose — the model rephrases, and demanding an exact echo would
    /// reject good questions. It asks only that SOMETHING connects: the stated basis
    /// appears in the posting, or the question names a skill, place or mode the
    /// posting names. What it stops is the invented requirement.
    /// </summary>
    private static bool Grounded(PortalJob job, string question, string basis)
    {
        var haystack = Normalise(
            $"{job.Title} {job.Location} {job.WorkMode} {job.EmploymentType} {job.SeniorityLevel} " +
            $"{job.RequiredSkills} {job.PreferredSkills} {job.Responsibilities} {job.Qualifications} {job.RawText}");

        if (basis.Length >= 4 && haystack.Contains(Normalise(basis), StringComparison.Ordinal)) return true;

        // No usable basis: fall back to asking whether the question mentions any
        // token the posting owns — a skill name, the location, the work mode.
        var owned = TextStructure.ReadLines(job.RequiredSkills)
            .Concat(TextStructure.ReadLines(job.PreferredSkills))
            .Append(job.Location)
            .Append(job.WorkMode)
            .Append(job.EmploymentType)
            .Append(job.SeniorityLevel)
            .Select(TextStructure.Collapse)
            .Where(v => v.Length >= 3 && !IsPlaceholder(v))
            .Select(Normalise);

        var asked = Normalise(question);
        return owned.Any(token => asked.Contains(token, StringComparison.Ordinal));
    }

    // -- the deterministic set ----------------------------------------------

    /// <summary>
    /// The questions this posting yields with no model at all.
    ///
    /// Not a degraded mode to be embarrassed about: these are drawn straight from
    /// the structured fields, so they are always accurate, just blunter. The gate
    /// works when the model is down, which is the only reason it can be depended on
    /// as a gate.
    /// </summary>
    private static IReadOnlyList<ScreeningQuestion> Deterministic(PortalJob job)
    {
        var questions = new List<ScreeningQuestion>();

        var remote = job.WorkMode.Contains("remote", StringComparison.OrdinalIgnoreCase);

        if (remote)
        {
            Add(ScreeningTopics.WorkMode,
                "This role is remote. Are you set up to work remotely?", job.WorkMode);
        }
        else if (Has(job.Location))
        {
            Add(ScreeningTopics.Location,
                $"This role is based in {job.Location}. Are you based there, or willing to relocate?",
                job.Location);
        }

        if (!remote && IsStated(job.WorkMode))
        {
            Add(ScreeningTopics.WorkMode,
                $"The role is {job.WorkMode.ToLowerInvariant()}. Does that suit you?", job.WorkMode);
        }

        var skills = TextStructure.ReadLines(job.RequiredSkills)
            .Concat(TextStructure.ReadLines(job.PreferredSkills))
            .Select(TextStructure.Collapse)
            .Where(s => s.Length is >= 2 and <= 40 && TextStructure.LooksLikeSkillName(s))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (skills.Count > 0)
        {
            Add(ScreeningTopics.Skill, $"Do you have hands-on experience with {skills[0]}?", skills[0]);
        }

        if (job.MinYearsExperience is { } years and > 0)
        {
            Add(ScreeningTopics.Experience,
                $"Do you have at least {years:0.#} years of relevant professional experience?",
                $"{years:0.#}+ years");
        }
        else if (Has(job.Title))
        {
            Add(ScreeningTopics.Experience,
                $"Have you worked in a {job.Title} role or something close to it?", job.Title);
        }

        if (IsStated(job.EmploymentType))
        {
            Add(ScreeningTopics.EmploymentType,
                $"This is a {job.EmploymentType.ToLowerInvariant()} position. Are you looking for that?",
                job.EmploymentType);
        }

        // Top up with further skills when the posting was too sparse to fill four
        // distinct topics.
        //
        // Most postings state almost nothing structured — no location, work mode
        // "Unspecified", no employment type — and the one-question-per-topic rule
        // then produced a two-question gate on a real posting. Topic variety is a
        // preference, not a constraint: four is the contract.
        foreach (var skill in skills.Skip(1))
        {
            if (questions.Count >= QuestionCount) break;
            if (AlreadyCovers(questions, skill)) continue;
            questions.Add(new ScreeningQuestion(
                0, $"Do you have hands-on experience with {skill}?", ScreeningTopics.Skill, skill));
        }

        // A posting with almost nothing structured still has to produce a gate.
        if (questions.Count == 0)
        {
            Add(ScreeningTopics.Experience,
                $"Are you confident you meet the requirements for the {job.Title} role?", job.Title);
        }

        return Renumber(questions.Take(QuestionCount).ToList());

        void Add(string topic, string text, string basis)
        {
            if (questions.Count >= QuestionCount) return;
            if (questions.Any(q => q.Topic == topic)) return;
            questions.Add(new ScreeningQuestion(questions.Count + 1, text, topic, TextStructure.Collapse(basis)));
        }
    }

    // -- who to ask ---------------------------------------------------------

    /// <summary>
    /// The contacts for this posting, the one whose remit matches it first.
    ///
    /// A hybrid role routes to the ONSITE contact: hybrid means being in the office
    /// some days, so the questions it raises — which days, how far, is the commute
    /// workable — are the onsite ones. Anything unrecognised falls back to whatever
    /// is configured as <c>Any</c>, or simply lists everyone.
    /// </summary>
    private IReadOnlyList<ScreeningContact> ContactsFor(PortalJob job)
    {
        var configured = _screening.Contacts ?? new List<ScreeningContactOption>();
        if (configured.Count == 0) return Array.Empty<ScreeningContact>();

        var mode = job.WorkMode ?? "";
        var wanted =
            mode.Contains("remote", StringComparison.OrdinalIgnoreCase) ? "remote" :
            mode.Contains("hybrid", StringComparison.OrdinalIgnoreCase) ? "onsite" :
            mode.Contains("onsite", StringComparison.OrdinalIgnoreCase) ||
            mode.Contains("on-site", StringComparison.OrdinalIgnoreCase) ? "onsite" :
            "";

        return configured
            .Where(c => IsSafeUrl(c.Url) && !string.IsNullOrWhiteSpace(c.Name))
            .Select(c => new ScreeningContact(
                c.Name.Trim(),
                c.Url.Trim(),
                string.IsNullOrWhiteSpace(c.Handles) ? $"{c.WorkMode} roles" : c.Handles.Trim(),
                Relevant: wanted.Length > 0 &&
                          c.WorkMode.Contains(wanted, StringComparison.OrdinalIgnoreCase)))
            .OrderByDescending(c => c.Relevant)
            .ToList();
    }

    /// <summary>
    /// Whether a configured URL is safe to render as a link.
    ///
    /// http and https only. A config file is not a trusted input just because it
    /// lives on the server — a typo or a bad edit putting <c>javascript:</c> in
    /// here would otherwise become a link the app renders and someone clicks.
    /// </summary>
    private static bool IsSafeUrl(string? url) =>
        Uri.TryCreate((url ?? "").Trim(), UriKind.Absolute, out var parsed) &&
        (parsed.Scheme == Uri.UriSchemeHttp || parsed.Scheme == Uri.UriSchemeHttps);

    // -- the gate -----------------------------------------------------------

    public async Task<ScreeningResult> EvaluateAsync(
        PortalJob job, IReadOnlyList<ScreeningAnswer> answers, CancellationToken ct = default)
    {
        var set = await BuildAsync(job, ct);

        // Answers are matched against the questions THIS service produces, not
        // against whatever the client sends back. Otherwise passing the gate is a
        // matter of inventing four ids and answering them all yes.
        var byId = answers
            .GroupBy(a => a.QuestionId)
            .ToDictionary(g => g.Key, g => g.Last().Yes);

        var unanswered = set.Questions
            .Where(q => !byId.ContainsKey(q.Id))
            .Select(q => q.Id)
            .ToList();

        var declined = set.Questions
            .Where(q => byId.TryGetValue(q.Id, out var yes) && !yes)
            .ToList();

        if (unanswered.Count > 0)
        {
            return new ScreeningResult(false, declined, unanswered,
                unanswered.Count == set.Questions.Count
                    ? "Answer the screening questions to apply."
                    : $"{unanswered.Count} question(s) still unanswered.");
        }

        // ANSWERING is the gate. The answers themselves are not.
        //
        // A "no" used to block, and that was the wrong instrument. The questions
        // exist to make someone read the terms of the role and state a position on
        // them — a candidate who knows the job is in Berlin and applies anyway has
        // made an informed choice, and refusing them is the platform substituting
        // its judgement for theirs on a fact only they hold.
        //
        // The declines are still reported, because they are the useful part: a
        // recruiter seeing "would not relocate" attached to an application learns
        // something no CV would have told them. Recorded, not enforced.
        return new ScreeningResult(true, declined, unanswered,
            declined.Count == 0
                ? "Screening complete. You can apply."
                : $"Screening complete. You answered no to {declined.Count} question(s) — " +
                  "these are shared with the employer alongside your application.");
    }

    // -- helpers ------------------------------------------------------------

    /// <summary>
    /// Whether adding this topic again would repeat a question already asked.
    ///
    /// Only SKILL may appear more than once: a posting naming five technologies can
    /// reasonably be asked about two of them. Every other topic has exactly one
    /// sensible question, and asking it twice produces the pair this caught in
    /// testing � "Have you had 2-5 years of professional experience?" followed by
    /// "Do you have at least 2 years of relevant professional experience?", which is
    /// one question wearing two hats and wastes a quarter of the gate.
    /// </summary>
    private static bool Repeats(IEnumerable<ScreeningQuestion> asked, string topic) =>
        topic != ScreeningTopics.Skill && asked.Any(q => q.Topic == topic);

    /// <summary>
    /// Whether a question is offering a menu rather than asking yes or no.
    ///
    /// "Do you prefer remote, hybrid, or onsite work?" is a multiple-choice question
    /// wearing a yes/no costume: every answer is "yes, one of those", and a decline
    /// then reads back as "you answered no to: do you prefer remote, hybrid or
    /// onsite work", which means nothing to the person who has to act on it.
    ///
    /// Detected by shape rather than wording — two or more commas before an "or" is
    /// a list of three or more alternatives. Kept deliberately narrow: "Are you based
    /// there, or willing to relocate?" has one comma and is a perfectly good binary
    /// question, so it passes.
    /// </summary>
    private static bool OffersAMenu(string question)
    {
        if (!question.Contains(" or ", StringComparison.OrdinalIgnoreCase)) return false;
        return question.Count(c => c == ',') >= 2;
    }

    /// <summary>
    /// Whether something already asked about covers this subject.
    ///
    /// Deduplicating on the whole sentence is not enough: "Do you have experience
    /// with Excel and SQL?" and "Do you have experience with Excel?" are different
    /// strings and the same question, and a live posting asked both. What matters is
    /// whether the SUBJECT has come up, so that is what is compared.
    /// </summary>
    private static bool AlreadyCovers(IEnumerable<ScreeningQuestion> asked, string subject)
    {
        var needle = Normalise(subject).Trim();
        if (needle.Length < 2) return true;

        return asked.Any(q =>
            Normalise(q.Question).Contains(needle, StringComparison.Ordinal) ||
            Normalise(q.Question).Equals(Normalise(subject), StringComparison.Ordinal));
    }

    private static IReadOnlyList<ScreeningQuestion> Renumber(IReadOnlyList<ScreeningQuestion> questions) =>
        questions.Select((q, i) => q with { Id = i + 1 }).ToList();

    private static bool Has(string? value) => !string.IsNullOrWhiteSpace(value);

    /// <summary>The extractor writes "Unspecified" where a posting said nothing.</summary>
    private static bool IsStated(string? value) => Has(value) && !IsPlaceholder(value!);

    private static bool IsPlaceholder(string value) =>
        value.Equals("Unspecified", StringComparison.OrdinalIgnoreCase);

    private static string Normalise(string text) =>
        new string((text ?? "").ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) ? c : ' ').ToArray())
            .Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Aggregate(new StringBuilder(), (b, w) => b.Append(w).Append(' '))
            .ToString();

    private sealed class RawSet
    {
        public List<RawQuestion>? Questions { get; set; }
    }

    private sealed class RawQuestion
    {
        public string? Question { get; set; }
        public string? Topic { get; set; }
        public string? Basis { get; set; }
    }
}
