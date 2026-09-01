using System.Text;
using JobPortal.Api.Services.Llm;

namespace JobPortal.Api.Services.Chat;

/// <summary>
/// Writes the reply from a fact package, and refuses to ship one that strays from it.
///
/// WHY GENERATE AT ALL
/// -------------------
/// The composed replies this replaces were built by string concatenation, which is
/// why every follow-up read the same however it was asked. "Why only 50%?" and "what
/// would make this a strong match?" are different questions about the same facts, and
/// a template can only answer one of them.
///
/// WHY GENERATION IS STILL NOT TRUSTED
/// -----------------------------------
/// The model receives the fact package and nothing else — no CV, no other candidates,
/// no world knowledge it is invited to use. Then the answer is checked against the
/// package before it is kept:
///
///   * every number in it must appear in the package
///   * a company question answered at length, when the JD says nothing about the
///     employer, is discarded
///
/// A failed check is not an error. The deterministic sentence ships instead, which is
/// exactly what the assistant would have said before this class existed.
///
/// WHY IT DOES NOT STREAM TOKEN BY TOKEN
/// -------------------------------------
/// It cannot: a grounding check needs the whole answer, and text already pushed to the
/// browser cannot be recalled. So the model's reply is collected, checked, and only
/// then streamed out. The candidate still sees motion throughout, because the
/// orchestrator emits thoughts while this runs — the same trick the search turn
/// already uses.
/// </summary>
public class AnswerComposer
{
    private readonly IOllamaClient _llm;
    private readonly ILogger<AnswerComposer> _logger;

    public AnswerComposer(IOllamaClient llm, ILogger<AnswerComposer> logger)
    {
        _llm = llm;
        _logger = logger;
    }

    private const string System = """
        You answer a job-seeker's question about roles on one job board.

        You are given FACTS. They are the only thing you know.

        Rules:
        - Use ONLY the facts. Never add a number, a company detail, a requirement or a
          benefit that is not written there.
        - If the facts do not answer the question, say plainly that the posting does
          not say. Do not guess and do not fill the gap from general knowledge.
        - You know nothing about these companies beyond what the facts state, however
          familiar the name looks.
        - Speak to the candidate as "you". Be direct and specific. No preamble.
        - Two short paragraphs at most. Use a bullet list only for three or more items.
        - Never invent a job that is not in the facts.
        """;

    public sealed record Composition(string Text, bool Generated)
    {
        /// <summary>False when the model was absent, failed, or was overruled.</summary>
        public static Composition Deterministic(string text) => new(text, false);
    }

    /// <summary>
    /// Composes an answer, or returns <paramref name="fallback"/> unchanged.
    /// </summary>
    /// <param name="fallback">
    /// The deterministic sentence. Required, not optional — a composer with no
    /// fallback would have to choose between an ungrounded answer and no answer, and
    /// there must never be a path where the unsafe option is the only one available.
    /// </param>
    public async Task<Composition> ComposeAsync(
        string question,
        FactPackage package,
        string fallback,
        bool isCompanyQuestion = false,
        CancellationToken ct = default)
    {
        if (!_llm.ChatAvailable) return Composition.Deterministic(fallback);

        string? reply;
        try
        {
            reply = await _llm.ChatAsync(new[]
            {
                new ChatTurn("system", System),
                new ChatTurn("user", $"FACTS:\n{package.ToPrompt()}\n\nQUESTION: {question}"),
            }, jsonMode: false, ct: ct);
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Answer composition failed; using the deterministic reply.");
            return Composition.Deterministic(fallback);
        }

        if (string.IsNullOrWhiteSpace(reply)) return Composition.Deterministic(fallback);

        var answer = reply.Trim();

        var grounding = ChatGuardrails.Ground(answer, package);
        if (!grounding.Grounded)
        {
            // Worth a warning rather than a debug line: a model inventing figures at
            // a candidate is the failure this whole design exists to prevent, and it
            // should be visible in the logs when it starts happening.
            _logger.LogWarning(
                "Composed answer cited figures absent from the facts ({Numbers}); using the deterministic reply.",
                string.Join(", ", grounding.Unsupported));
            return Composition.Deterministic(fallback);
        }

        if (ChatGuardrails.CompanyClaimUnsupported(answer, package, isCompanyQuestion))
        {
            _logger.LogWarning("Composed a company answer the posting does not support; using the deterministic reply.");
            return Composition.Deterministic(fallback);
        }

        return new Composition(answer, Generated: true);
    }

    // --- Deterministic fallbacks --------------------------------------------------
    // One per action. These are what ships when there is no model, and what the
    // grounding gate falls back to — so they have to be genuine answers rather than
    // apologies. Each is assembled from the same package the model was given.

    public static string DescribeJobs(FactPackage package)
    {
        var text = new StringBuilder();
        foreach (var job in package.Jobs)
        {
            text.Append($"**{job.Title}**");
            if (!string.IsNullOrWhiteSpace(job.Company)) text.Append($" at {job.Company}");
            text.AppendLine();
            var bits = new List<string>();
            if (!string.IsNullOrWhiteSpace(job.Location)) bits.Add(job.Location);
            if (!string.IsNullOrWhiteSpace(job.WorkMode)) bits.Add(job.WorkMode);
            if (!string.IsNullOrWhiteSpace(job.EmploymentType)) bits.Add(job.EmploymentType);
            if (job.Salary is not null) bits.Add(job.Salary);
            if (job.Experience is not null) bits.Add(job.Experience);
            if (bits.Count > 0) text.AppendLine(string.Join(" · ", bits));
            if (job.Required.Count > 0) text.AppendLine($"Asks for: {string.Join(", ", job.Required)}");
            text.AppendLine();
        }
        return text.ToString().TrimEnd();
    }

    public static string DescribeCompany(FactPackage package)
    {
        var text = new StringBuilder();
        foreach (var job in package.Jobs)
        {
            if (!package.Companies.TryGetValue(job.JobId, out var company)) continue;

            if (!company.SaysAnything)
            {
                var name = string.IsNullOrWhiteSpace(company.Name) ? "the employer" : company.Name;
                text.AppendLine($"The posting for **{job.Title}** doesn't say anything about {name} " +
                                "beyond the role itself.");
                continue;
            }

            text.AppendLine($"What the **{job.Title}** posting says about {company.Name}:");
            foreach (var (heading, body) in company.Sections)
                text.AppendLine($"- *{heading}*: {body}");
        }
        return text.ToString().TrimEnd();
    }

    public static string DescribeGaps(FactPackage package)
    {
        if (package.Notes.TryGetValue("TO CLOSE", out var toClose))
            return $"To close for **{package.Jobs[0].Title}**: {toClose}";

        if (package.Notes.TryGetValue("MISSING MOST OFTEN", out var missing))
            return $"Across the roles on screen, what comes up most often: {missing}";

        if (package.Notes.TryGetValue("MOST DEMANDED AND NOT EVIDENCED", out var demand))
            return $"Ranked by how many open roles ask for it: {demand}";

        var gaps = package.Coverage.Where(c => c.Status == "Missing").Select(c => c.Requirement).Distinct();
        var list = string.Join(", ", gaps);
        return list.Length > 0
            ? $"Not evidenced on your CV: {list}"
            : "Every requirement these postings state is evidenced on your CV.";
    }

    public static string DescribeFit(FactPackage package)
    {
        var text = new StringBuilder();
        foreach (var (key, value) in package.Notes.Where(n => n.Key.StartsWith("FIT ")))
            text.AppendLine($"{value}");
        foreach (var (key, value) in package.Notes.Where(n => n.Key.StartsWith("STRENGTHS ")))
            text.AppendLine($"In your favour: {value}");
        foreach (var (key, value) in package.Notes.Where(n => n.Key.StartsWith("GAPS ")))
            text.AppendLine($"Counting against: {value}");

        return text.Length > 0 ? text.ToString().TrimEnd() : DescribeJobs(package);
    }
}
