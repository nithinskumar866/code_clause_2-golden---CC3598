using System.Globalization;
using System.Text.RegularExpressions;

namespace JobPortal.Api.Services.Chat;

/// <summary>
/// The two gates a turn passes through, both of them plain C#.
///
/// No model is involved in either direction, on purpose. A guardrail expressed as a
/// prompt instruction is a request; the career assistant already has a live example
/// of a model ignoring one, in <c>StreamOpeningAsync</c>, whose prompt says "do not
/// list the roles" and which produced a table anyway. Anything that must hold is
/// enforced after the model answers, in code it cannot argue with.
///
/// IN  — is this a question we are willing to take at all?
/// OUT — does the answer say anything the fact package does not support?
/// </summary>
public static class ChatGuardrails
{
    // --- Input -----------------------------------------------------------------
    /// <summary>
    /// Attempts to reach the system prompt, the rules, or another persona.
    ///
    /// Kept narrow deliberately. A broad filter here would reject real questions —
    /// "should I ignore the salary range?" is a legitimate thing for a candidate to
    /// ask — and a false refusal is far more visible to the user than a marginal
    /// jailbreak that the OUTPUT gate would catch anyway.
    /// </summary>
    private static readonly Regex Injection = new(
        @"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+)*" +
        @"(instruction|rule|prompt|guardrail|system|context)\w*\b" +
        @"|\b(system|developer)\s+(prompt|message)\b" +
        @"|\byou\s+are\s+now\b|\bact\s+as\s+(a|an)\b|\bpretend\s+(to\s+be|you)\b" +
        @"|\brepeat\s+(your|the)\s+(instructions|prompt|rules)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Asking about OTHER people. Two candidates on the same board must never see
    /// each other, and the hub already addresses every message to one connection —
    /// this closes the same hole at the question level.
    /// </summary>
    private static readonly Regex OtherPeople = new(
        @"\b(other|another|the other)\s+(candidate|applicant|user|person|people)s?\b" +
        @"|\bwho\s+else\s+(applied|is applying|has applied)\b" +
        @"|\b(list|show|give)\s+(me\s+)?(all\s+)?(candidates?|applicants?|resumes?|cvs?)\b" +
        @"|\bhow many\s+(candidates?|applicants?|people)\s+(applied|have applied)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>Fishing for infrastructure rather than for a job.</summary>
    private static readonly Regex Credentials = new(
        @"\b(api[_\s-]?key|access[_\s-]?token|password|secret|connection string|env(ironment)? var\w*|" +
        @"database (schema|dump|url)|sql\b|drop table|select \* from)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    public enum Refusal { None, Injection, OtherPeople, Credentials }

    public sealed record InputVerdict(Refusal Reason, string Message)
    {
        public bool Allowed => Reason == Refusal.None;
        public static readonly InputVerdict Ok = new(Refusal.None, "");
    }

    /// <summary>
    /// Decides whether a message is answerable at all.
    ///
    /// Note what is NOT rejected here: an off-topic but harmless question. "What's
    /// the weather" is refused later and more precisely, by the fact package having
    /// nothing to say about it, which produces a better answer than a blanket
    /// scope filter guessing at topics.
    /// </summary>
    public static InputVerdict InspectInput(string? message)
    {
        var text = message ?? "";
        if (string.IsNullOrWhiteSpace(text)) return InputVerdict.Ok;

        if (Injection.IsMatch(text))
        {
            return new InputVerdict(Refusal.Injection,
                "I can only help with the roles on this board and how your CV lines up against them.");
        }
        if (OtherPeople.IsMatch(text))
        {
            return new InputVerdict(Refusal.OtherPeople,
                "I can't discuss other candidates — I only see your own CV and the open roles.");
        }
        if (Credentials.IsMatch(text))
        {
            return new InputVerdict(Refusal.Credentials,
                "I can only answer questions about the job postings and your fit for them.");
        }
        return InputVerdict.Ok;
    }

    // --- Output ----------------------------------------------------------------
    private static readonly Regex NumberToken = new(@"\d[\d,]*(?:\.\d+)?%?", RegexOptions.Compiled);

    /// <summary>
    /// Numbers that carry no claim about a posting and would only create false
    /// positives: list markers, small counts, and years written in prose.
    /// </summary>
    private static bool Trivial(string token)
    {
        var bare = token.TrimEnd('%').Replace(",", "");
        if (!double.TryParse(bare, NumberStyles.Any, CultureInfo.InvariantCulture, out var value)) return true;
        return value <= 10;
    }

    public sealed record GroundingVerdict(IReadOnlyList<string> Unsupported)
    {
        public bool Grounded => Unsupported.Count == 0;
    }

    /// <summary>
    /// Checks that every number the answer states appears in the fact package.
    ///
    /// Numbers are the check because they are where an ungrounded answer does real
    /// damage and where it is unambiguously detectable: a salary, a fit percentage,
    /// a years-of-experience bar or a headcount that the posting never mentioned is
    /// a fabricated commitment to a candidate. Prose is not policed word by word —
    /// that would flag ordinary paraphrase — so this is a floor, not a proof.
    ///
    /// Values of 10 or less are ignored: they are overwhelmingly list positions and
    /// small counts, and flagging them would make the gate fire constantly and get
    /// switched off, which is the worst outcome for a safety check.
    /// </summary>
    public static GroundingVerdict Ground(string? answer, FactPackage package)
    {
        if (string.IsNullOrWhiteSpace(answer)) return new GroundingVerdict(Array.Empty<string>());

        var allowed = NumberToken.Matches(package.ToPrompt())
            .Select(m => Normalise(m.Value))
            .ToHashSet();

        var unsupported = NumberToken.Matches(answer)
            .Select(m => m.Value)
            .Where(token => !Trivial(token))
            .Where(token => !allowed.Contains(Normalise(token)))
            .Distinct()
            .ToList();

        return new GroundingVerdict(unsupported);
    }

    private static string Normalise(string token) => token.TrimEnd('%').Replace(",", "").TrimEnd('0').TrimEnd('.');

    /// <summary>
    /// Whether an answer makes a claim about the employer that the JD did not.
    ///
    /// This is the highest-risk question type in the assistant: a company is a real
    /// entity the model has training data about, so "where are they based" or "what's
    /// their leave policy" will be answered confidently and unverifiably about a real
    /// employer. If the posting carries no company text, the only safe answer is that
    /// it does not say — so any substantial reply to a company question is refused.
    /// </summary>
    public static bool CompanyClaimUnsupported(string? answer, FactPackage package, bool isCompanyQuestion)
    {
        if (!isCompanyQuestion || string.IsNullOrWhiteSpace(answer)) return false;

        var hasCompanyText = package.Companies.Values.Any(c => c.SaysAnything);
        if (hasCompanyText) return false;

        // Nothing in the package describes the employer. A short deflection is fine;
        // a paragraph means the model filled the gap from somewhere else.
        return answer.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length > 40;
    }

    private static readonly Regex CompanyQuestion = new(
        @"\b(company|employer|organisation|organization|firm|they|their)\b.*\b" +
        @"(culture|benefit|perk|polic\w+|leave|holiday|vacation|remote|office|located?|" +
        @"location|headquarter\w*|hq|work[- ]life|hours|insurance|based)\b" +
        @"|\b(who are they|what do they do|about the company|tell me about)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    public static bool LooksLikeCompanyQuestion(string? message) =>
        !string.IsNullOrWhiteSpace(message) && CompanyQuestion.IsMatch(message);
}
