using System.Text.RegularExpressions;
using JobPortal.Api.Contracts;

namespace JobPortal.Api.Services.Chat;

public enum ChatIntent
{
    /// <summary>Find or re-find jobs for the uploaded resume.</summary>
    Search,

    /// <summary>Narrow the previous result. Same search, fewer results.</summary>
    Refine,

    /// <summary>Ask about a result already produced. Answered from the stored
    /// result, never by searching again.</summary>
    Explain,

    /// <summary>
    /// Asking for guidance — which roles suit them, what to learn next. Answered
    /// from the stored result, like <see cref="Explain"/>, never by searching again.
    /// </summary>
    Advise,

    /// <summary>Small talk, or a question about the portal itself.</summary>
    Chitchat,
}

public record ParsedMessage(
    ChatIntent Intent,
    JobFilters Filters,
    /// <summary>1-based position the candidate referred to ("the second one"), if any.</summary>
    int? ReferencedRank,
    /// <summary>True when the message asks to clear constraints rather than add them.</summary>
    bool ResetFilters);

/// <summary>
/// Reads a candidate's message.
///
/// Deterministic, and it stays that way. A hallucinated CONSTRAINT is as damaging
/// as a hallucinated answer: it silently changes the question before the search
/// runs, and the candidate is shown a filtered list they never asked for with no
/// indication anything was dropped. So filters are only ever recognised from
/// patterns in the message itself, checked against
/// <see cref="JobBoardLexicon"/> — what the board actually contains.
/// </summary>
public class ChatQueryUnderstanding
{
    private static readonly Regex RemotePattern = new(
        @"\b(remote|work from home|wfh|anywhere)\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex HybridPattern = new(
        @"\bhybrid\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex OnsitePattern = new(
        @"\b(on[- ]?site|in[- ]office|in the office|relocat\w*)\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex ExplainPattern = new(
        @"\b(why|explain|how come|what do you mean|tell me more|more about|elaborate|reason|justify)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Asking for guidance rather than for a list.
    ///
    /// "What jobs suit me" and "what should I learn" are the two questions a
    /// candidate actually arrives with, and both are answerable from a result that
    /// has already been produced — which roles scored best, and which requirements
    /// came back Missing across them. Neither needs a fresh search, and running one
    /// would risk answering about a different ranking than the one on screen.
    /// </summary>
    private static readonly Regex AdvisePattern = new(
        @"\b(suit(s|able)?( me| for me)?|right for me|good fit|best fit|fit me|" +
        @"should i (learn|study|focus|improve|upskill)|what (skills?|should i)|" +
        @"which (skills?|jobs?|roles?) (should|do) i|" +
        @"improve|upskill|advice|advise|recommend|guidance|career)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex SearchPattern = new(
        @"\b(find|search|show|list|match|jobs?|roles?|positions?|openings?|opportunit\w+|apply|hiring)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex ResetPattern = new(
        @"\b(reset|clear|start over|remove (the )?filters?|show (me )?(all|everything)|never ?mind)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    // "over $100k", "at least 120000", "above 90k", "paying 100k+"
    private static readonly Regex SalaryPattern = new(
        @"(?:over|above|at least|minimum|min|more than|upwards of|paying|starting at|>)\s*[$£€₹]?\s*(?<n>\d[\d,\.]*)\s*(?<k>k\b)?",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    // "the second one", "#3", "number 1", "the first role"
    private static readonly Regex OrdinalWord = new(
        @"\b(?<word>first|second|third|fourth|fifth|last)\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex OrdinalNumber = new(
        @"(?:#|number\s+|option\s+|job\s+)(?<n>\d{1,2})\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex SeniorityPattern = new(
        @"\b(?<level>senior|junior|entry[- ]level|graduate|fresher|mid[- ]level|lead|principal|staff)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex EmploymentPattern = new(
        @"\b(?<type>full[- ]time|part[- ]time|contract|freelance|internship|intern)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    public ParsedMessage Parse(string message, JobFilters current, JobBoardLexicon lexicon)
    {
        var text = message ?? "";

        if (ResetPattern.IsMatch(text))
        {
            return new ParsedMessage(ChatIntent.Search, new JobFilters(), null, ResetFilters: true);
        }

        var filters = ExtractFilters(text, current, lexicon);
        var rank = ExtractRank(text);
        var changed = !ReferenceEquals(filters, current) && !Equals(filters, current);

        var intent = DetermineIntent(text, changed, rank);

        return new ParsedMessage(intent, filters, rank, ResetFilters: false);
    }

    private static ChatIntent DetermineIntent(string text, bool filtersChanged, int? rank)
    {
        // "Why is the second one only a 60% fit?" is about an answer already
        // given. Re-running retrieval for it would let the same question produce
        // two different rankings, and the explanation would then not match the
        // list the candidate is looking at.
        if (ExplainPattern.IsMatch(text) || rank is not null) return ChatIntent.Explain;

        // Checked before Search because "what jobs suit me" contains "jobs" and
        // would otherwise be read as a request to run the search again — throwing
        // away the very results the question is about.
        if (AdvisePattern.IsMatch(text)) return ChatIntent.Advise;

        if (SearchPattern.IsMatch(text)) return ChatIntent.Search;

        // "Remote only" is a complete message. It names no search verb, but it is
        // unambiguously an instruction to narrow.
        if (filtersChanged) return ChatIntent.Refine;

        return ChatIntent.Chitchat;
    }

    private static int? ExtractRank(string text)
    {
        var numeric = OrdinalNumber.Match(text);
        if (numeric.Success && int.TryParse(numeric.Groups["n"].Value, out var n) && n is > 0 and <= 20) return n;

        var word = OrdinalWord.Match(text);
        if (!word.Success) return null;

        return word.Groups["word"].Value.ToLowerInvariant() switch
        {
            "first" => 1,
            "second" => 2,
            "third" => 3,
            "fourth" => 4,
            "fifth" => 5,
            // "the last one" depends on how many were shown, so it is resolved by
            // the caller, which knows. -1 is that signal.
            "last" => -1,
            _ => null,
        };
    }

    /// <summary>
    /// Filters are ADDITIVE over the conversation. "Remote only" followed by
    /// "in Berlin" means both, because that is what a person means. A filter is
    /// only replaced when the new message states the same dimension again.
    /// </summary>
    private static JobFilters ExtractFilters(string text, JobFilters current, JobBoardLexicon lexicon)
    {
        var workMode = current.WorkMode;
        if (HybridPattern.IsMatch(text)) workMode = "Hybrid";
        else if (OnsitePattern.IsMatch(text)) workMode = "Onsite";
        else if (RemotePattern.IsMatch(text)) workMode = "Remote";

        var seniority = current.SeniorityLevel;
        var seniorityMatch = SeniorityPattern.Match(text);
        if (seniorityMatch.Success)
        {
            seniority = seniorityMatch.Groups["level"].Value.ToLowerInvariant() switch
            {
                "senior" => "Senior",
                "junior" or "entry-level" or "entry level" or "graduate" or "fresher" => "Junior",
                "mid-level" or "mid level" => "Mid",
                "lead" => "Lead",
                "principal" => "Principal",
                "staff" => "Staff",
                _ => seniority,
            };
        }

        var employment = current.EmploymentType;
        var employmentMatch = EmploymentPattern.Match(text);
        if (employmentMatch.Success)
        {
            employment = employmentMatch.Groups["type"].Value.ToLowerInvariant() switch
            {
                "full-time" or "full time" => "Full-time",
                "part-time" or "part time" => "Part-time",
                "contract" or "freelance" => "Contract",
                "internship" or "intern" => "Internship",
                _ => employment,
            };
        }

        var location = FindLexiconTerm(text, lexicon.Locations) ?? current.Location;
        var salary = ExtractSalary(text) ?? current.MinSalary;

        // Keywords come only from terms the board itself uses. Free text would
        // turn ordinary words in a sentence into hard constraints, and a
        // constraint nobody meant to state removes jobs without saying so.
        var keywords = current.Keywords?.ToList() ?? new List<string>();
        foreach (var skill in FindLexiconTerms(text, lexicon.Skills).Take(3))
        {
            if (!keywords.Contains(skill, StringComparer.OrdinalIgnoreCase)) keywords.Add(skill);
        }

        return new JobFilters(
            workMode,
            location,
            employment,
            seniority,
            salary,
            current.MaxYearsRequired,
            keywords.Count > 0 ? keywords : null);
    }

    private static decimal? ExtractSalary(string text)
    {
        var match = SalaryPattern.Match(text);
        if (!match.Success) return null;

        var digits = match.Groups["n"].Value.Replace(",", "").Replace(".", "");
        if (!decimal.TryParse(digits, out var value)) return null;

        if (match.Groups["k"].Success) value *= 1000;

        // "over 3 years" is not a salary. A figure this small in a pay context is
        // far more likely to be something else entirely.
        return value >= 1000 ? value : null;
    }

    private static string? FindLexiconTerm(string text, IReadOnlySet<string> vocabulary) =>
        FindLexiconTerms(text, vocabulary).FirstOrDefault();

    /// <summary>
    /// Terms from the board's vocabulary that appear in the message, longest
    /// first so "San Francisco" is preferred over a bare "San".
    /// </summary>
    private static IEnumerable<string> FindLexiconTerms(string text, IReadOnlySet<string> vocabulary)
    {
        foreach (var term in vocabulary.OrderByDescending(t => t.Length))
        {
            // Whole-word only. A substring test matches "Go" inside "Going" and
            // "R" inside every sentence, which is how a lexical filter starts
            // silently deleting results.
            if (Regex.IsMatch(text, $@"(?<![\w]){Regex.Escape(term)}(?![\w])", RegexOptions.IgnoreCase))
            {
                yield return term;
            }
        }
    }
}
