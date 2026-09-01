namespace JobPortal.Api.Contracts;

/// <summary>
/// The API contract. The React client mirrors these shapes in
/// <c>frontend/src/types/index.ts</c>; this file is the source of truth and the
/// only coordination surface between the two halves of the portal.
/// </summary>
public record ApiResponse<T>(bool Success, string Message, T? Data)
{
    public static ApiResponse<T> Ok(T data, string message = "OK") => new(true, message, data);
    public static ApiResponse<T> Fail(string message) => new(false, message, default);
}

// -- jobs -------------------------------------------------------------------

public record JobSummaryDto(
    int Id,
    string Title,
    string Company,
    string Location,
    string WorkMode,
    string EmploymentType,
    string SeniorityLevel,
    double? MinYearsExperience,
    double? MaxYearsExperience,
    decimal? SalaryMin,
    decimal? SalaryMax,
    string? SalaryCurrency,
    IReadOnlyList<string> RequiredSkills,
    IReadOnlyList<string> PreferredSkills,
    string Summary,
    string? ApplyUrl,
    bool IsIndexed,
    DateTime CreatedAt);

public record JobDetailDto(
    JobSummaryDto Summary,
    IReadOnlyList<string> Responsibilities,
    IReadOnlyList<string> Qualifications,
    string RawText,
    string ExtractionMode,
    int? SourceJobDescriptionId);

/// <summary>A job posting typed straight into the portal rather than uploaded.</summary>
public record CreateJobRequest(
    string Title,
    string? Company,
    string? Location,
    string? WorkMode,
    string? EmploymentType,
    string? SeniorityLevel,
    double? MinYearsExperience,
    double? MaxYearsExperience,
    decimal? SalaryMin,
    decimal? SalaryMax,
    string? SalaryCurrency,
    string? Description,
    IReadOnlyList<string>? RequiredSkills,
    IReadOnlyList<string>? PreferredSkills,
    string? ApplyUrl);

public record JobUploadResultDto(JobDetailDto Job, bool Indexed, string EmbeddingModel, bool SemanticMatching);

/// <summary>
/// A follow-up the candidate can tap instead of typing.
///
/// Two jobs at once. It advertises what the assistant can actually do — nobody
/// guesses that "what should I learn across all these roles" is answerable — and it
/// is the ask-back when a question could not be pinned to a posting. Both cases are
/// the same object because both are "here are real things you can ask next", and
/// every one is built from the shortlist or the board, never invented.
/// </summary>
/// <param name="Label">What the chip reads.</param>
/// <param name="Message">What gets sent when it is tapped.</param>
public record FollowUpDto(string Label, string Message);

/// <summary>One file's outcome inside a bulk upload.</summary>
/// <param name="Error">Null on success. Why THIS file failed, in words the uploader
/// can act on — a batch that reports only "3 failed" tells them nothing about which
/// three or what to fix.</param>
public record JobUploadItemDto(
    string Filename,
    bool Success,
    JobSummaryDto? Job = null,
    bool Indexed = false,
    string? Error = null);

/// <summary>
/// The result of uploading several JDs at once.
///
/// Deliberately 200-with-per-file-status rather than a single pass/fail: in a batch
/// of thirty, two unreadable scans are normal and must not discard the twenty-eight
/// that parsed. The caller shows a per-file list and the uploader re-tries only the
/// failures.
/// </summary>
public record JobBulkUploadResultDto(
    int Total,
    int Succeeded,
    int Failed,
    IReadOnlyList<JobUploadItemDto> Items,
    string EmbeddingModel,
    bool SemanticMatching);

public record ImportResultDto(int Imported, int Skipped, int Failed, IReadOnlyList<string> Notes);

public record IndexStatusDto(
    int TotalJobs,
    int IndexedJobs,
    string ActiveModel,
    bool SemanticMatching,
    IReadOnlyDictionary<string, int> VectorsByModel);

// -- resumes ----------------------------------------------------------------

public record ResumeProfileDto(
    int Id,
    string Filename,
    string CandidateName,
    string Email,
    string Phone,
    string Location,
    string CurrentTitle,
    double? YearsExperience,
    IReadOnlyList<string> Skills,
    IReadOnlyList<string> Titles,
    IReadOnlyList<string> Education,
    string Summary,
    string ExtractionMode);

// -- matching ---------------------------------------------------------------

/// <summary>
/// One skill the job asks for, judged against the candidate.
/// <paramref name="Status"/> is Have | Transferable | Missing.
/// </summary>
public record SkillAssessmentDto(string Skill, string Status, string? EvidenceSkill, double Similarity);

/// <summary>
/// How a fit score was arrived at. Two values, and they are not interchangeable:
///
///   <c>computed</c> — <see cref="Services.Matching.MatchingService"/>'s weighted
///     arithmetic over four measured dimensions. Instant and reproducible.
///   <c>reasoned</c> — a language model judging the résumé against the posting's
///     own stated requirements. Slower, and the number is the model's own.
///
/// Carried on every match so nothing downstream has to infer which one it is
/// looking at. A percentage means a different thing under each, and a card that
/// did not say which would be presenting two incomparable numbers identically.
/// </summary>
public static class ScoringModes
{
    public const string Computed = "computed";
    public const string Reasoned = "reasoned";

    /// <summary>
    /// Chunked retrieval, then a model judging the retrieved passages per
    /// requirement. Unlike <see cref="Reasoned"/>, the model never sees a whole
    /// document — only the passages that matched, with their section labels.
    /// </summary>
    public const string Rag = "rag";

    private static readonly string[] Known = { Computed, Reasoned, Rag };

    /// <summary>
    /// Anything unrecognised falls back to the arithmetic, which always works.
    ///
    /// A set lookup rather than the two-way ternary this was: that shape silently
    /// collapsed every future mode into <see cref="Computed"/>, so adding one
    /// would have looked like it worked while scoring with the wrong engine.
    /// </summary>
    public static string Normalise(string? value)
    {
        var candidate = (value ?? "").Trim();
        return Known.FirstOrDefault(m => string.Equals(m, candidate, StringComparison.OrdinalIgnoreCase))
               ?? Computed;
    }
}

public record JobMatchDto(
    JobSummaryDto Job,
    double FitScore,
    string FitBand,
    double SemanticScore,
    double SkillScore,
    double TitleScore,
    double ExperienceScore,
    IReadOnlyList<SkillAssessmentDto> Skills,
    IReadOnlyList<string> Strengths,
    IReadOnlyList<string> Gaps,
    string RecruiterNote,
    /// <summary>See <see cref="ScoringModes"/>.</summary>
    string ScoringMode = ScoringModes.Computed,
    /// <summary>
    /// The reasoned judgement behind this match, when there is one.
    ///
    /// Null under <c>computed</c>. Present under <c>reasoned</c>, carrying the
    /// requirement-by-requirement evidence the score was justified with — which is
    /// the part a candidate can act on, and the part that makes the number
    /// auditable rather than merely stated.
    /// </summary>
    JobEvaluationDto? Evaluation = null);

/// <summary>
/// Filters the conversation has narrowed to. Every field is optional: a null
/// means the candidate has not expressed a preference, which is different from
/// having expressed no constraint.
/// </summary>
public record JobFilters(
    string? WorkMode = null,
    string? Location = null,
    string? EmploymentType = null,
    string? SeniorityLevel = null,
    decimal? MinSalary = null,
    double? MaxYearsRequired = null,
    IReadOnlyList<string>? Keywords = null)
{
    public bool IsEmpty =>
        WorkMode is null && Location is null && EmploymentType is null &&
        SeniorityLevel is null && MinSalary is null && MaxYearsRequired is null &&
        (Keywords is null || Keywords.Count == 0);
}

public record MatchResultDto(
    IReadOnlyList<JobMatchDto> Matches,
    int TotalCandidates,
    JobFilters AppliedFilters,
    bool SemanticMatching,
    string EmbeddingModel,
    /// <summary>See <see cref="ScoringModes"/>. Applies to every match in the list.</summary>
    string ScoringMode = ScoringModes.Computed,
    /// <summary>
    /// Identifies the shortlist this result belongs to, so a later, fuller version
    /// of the SAME shortlist replaces it rather than appearing as a new answer.
    ///
    /// Reasoned scoring takes tens of seconds per role, so the first few are sent
    /// as soon as they are judged and the rest arrive afterwards — possibly after
    /// the turn has already ended and the candidate has asked something else. The
    /// batch id is how the client knows which set of cards to update.
    /// </summary>
    string BatchId = "",
    /// <summary>How many more roles are still being judged. Zero when finished.</summary>
    int Pending = 0,
    /// <summary>False while a background pass is still filling this shortlist in.</summary>
    bool Complete = true);

// -- chat -------------------------------------------------------------------

/// <summary>
/// A command the bot issues to the UI alongside its prose: "Copilot Mode".
///
/// Kept separate from the message text so the client never has to parse prose to
/// decide what to do. <paramref name="Action"/> is a closed vocabulary the client
/// switches on; an unrecognised action is ignored rather than guessed at.
/// </summary>
public record UiActionDto(string Action, object? Payload);

/// <summary>
/// Strictly-typed copilot commands. The backend validates every command against
/// this schema before it is ever broadcast over SignalR. The client mirrors this
/// validation as defense in depth.
/// </summary>
public abstract record CopilotCommand
{
    public string CommandId { get; init; } = Guid.NewGuid().ToString("N");
}

/// <summary>Update job board filters.</summary>
public record UpdateFiltersCommand(
    string CommandId,
    UpdateFiltersPayload Payload) : CopilotCommand
{
    public UpdateFiltersCommand(UpdateFiltersPayload payload) : this(Guid.NewGuid().ToString("N"), payload) { }
}

public record UpdateFiltersPayload(
    bool? Remote = null,
    decimal? MinSalary = null,
    decimal? MaxSalary = null,
    string? Location = null,
    string? JobType = null);

/// <summary>Show specific jobs on the board.</summary>
public record ShowJobsCommand(
    string CommandId,
    ShowJobsPayload Payload) : CopilotCommand
{
    public ShowJobsCommand(ShowJobsPayload payload) : this(Guid.NewGuid().ToString("N"), payload) { }
}

public record ShowJobsPayload(IReadOnlyList<int> JobIds);

/// <summary>Navigate to a known route.</summary>
public record NavigateCommand(
    string CommandId,
    NavigatePayload Payload) : CopilotCommand
{
    public NavigateCommand(NavigatePayload payload) : this(Guid.NewGuid().ToString("N"), payload) { }
}

public record NavigatePayload(string Route);

/// <summary>
/// Validates a UiActionDto against the strict CopilotCommand schema.
/// Returns the validated command or null if invalid.
/// </summary>
public static class CopilotCommandValidator
{
    private static readonly HashSet<string> AllowedRoutes = new(StringComparer.OrdinalIgnoreCase)
    {
        "/",
        "/?view=job-board",
        "/?view=post-job",
        "/?view=applications",
        "/?view=settings"
    };

    private static readonly HashSet<string> AllowedActions = new(StringComparer.OrdinalIgnoreCase)
    {
        "update_filters",
        "show_jobs",
        "navigate",
        "resume_ready"
    };

    public static CopilotCommand? Validate(UiActionDto action, string sessionId, ILogger? logger = null)
    {
        if (!AllowedActions.Contains(action.Action))
        {
            logger?.LogWarning("Rejected command: unknown action {Action} for session {SessionId}", action.Action, sessionId);
            return null;
        }

        try
        {
            return action.Action switch
            {
                "update_filters" => ValidateUpdateFilters(action, sessionId, logger),
                "show_jobs" => ValidateShowJobs(action, sessionId, logger),
                "navigate" => ValidateNavigate(action, sessionId, logger),
                "resume_ready" => new UpdateFiltersCommand(new UpdateFiltersPayload()), // resume_ready has no payload validation needed
                _ => null
            };
        }
        catch (Exception ex)
        {
            logger?.LogWarning(ex, "Rejected command: validation failed for {Action} in session {SessionId}", action.Action, sessionId);
            return null;
        }
    }

    private static CopilotCommand? ValidateUpdateFilters(UiActionDto action, string sessionId, ILogger? logger)
    {
        if (action.Payload is not System.Text.Json.JsonElement element)
        {
            logger?.LogWarning("Rejected update_filters: payload is not a JSON object for session {SessionId}", sessionId);
            return null;
        }

        var payload = new UpdateFiltersPayload(
            Remote: element.TryGetProperty("remote", out var remote) && remote.ValueKind == System.Text.Json.JsonValueKind.True ? true : 
                    element.TryGetProperty("remote", out var remoteFalse) && remoteFalse.ValueKind == System.Text.Json.JsonValueKind.False ? false : null,
            MinSalary: element.TryGetProperty("minSalary", out var minSal) && minSal.ValueKind == System.Text.Json.JsonValueKind.Number ? minSal.GetDecimal() : null,
            MaxSalary: element.TryGetProperty("maxSalary", out var maxSal) && maxSal.ValueKind == System.Text.Json.JsonValueKind.Number ? maxSal.GetDecimal() : null,
            Location: element.TryGetProperty("location", out var loc) && loc.ValueKind == System.Text.Json.JsonValueKind.String ? loc.GetString() : null,
            JobType: element.TryGetProperty("jobType", out var jt) && jt.ValueKind == System.Text.Json.JsonValueKind.String ? jt.GetString() : null
        );

        // Strip unknown keys - only allow known properties
        return new UpdateFiltersCommand(payload);
    }

    private static CopilotCommand? ValidateShowJobs(UiActionDto action, string sessionId, ILogger? logger)
    {
        if (action.Payload is not System.Text.Json.JsonElement element)
        {
            logger?.LogWarning("Rejected show_jobs: payload is not a JSON object for session {SessionId}", sessionId);
            return null;
        }

        if (!element.TryGetProperty("jobIds", out var jobIdsElement) || jobIdsElement.ValueKind != System.Text.Json.JsonValueKind.Array)
        {
            logger?.LogWarning("Rejected show_jobs: jobIds array missing or invalid for session {SessionId}", sessionId);
            return null;
        }

        var jobIds = new List<int>();
        foreach (var item in jobIdsElement.EnumerateArray())
        {
            if (item.ValueKind == System.Text.Json.JsonValueKind.Number && item.TryGetInt32(out var id))
            {
                jobIds.Add(id);
            }
        }

        return new ShowJobsCommand(new ShowJobsPayload(jobIds));
    }

    private static CopilotCommand? ValidateNavigate(UiActionDto action, string sessionId, ILogger? logger)
    {
        if (action.Payload is not System.Text.Json.JsonElement element)
        {
            logger?.LogWarning("Rejected navigate: payload is not a JSON object for session {SessionId}", sessionId);
            return null;
        }

        if (!element.TryGetProperty("route", out var routeElement) || routeElement.ValueKind != System.Text.Json.JsonValueKind.String)
        {
            logger?.LogWarning("Rejected navigate: route string missing or invalid for session {SessionId}", sessionId);
            return null;
        }

        var route = routeElement.GetString() ?? "";
        if (!AllowedRoutes.Contains(route))
        {
            logger?.LogWarning("Rejected navigate: route {Route} not in allowlist for session {SessionId}", route, sessionId);
            return null;
        }

        return new NavigateCommand(new NavigatePayload(route));
    }
}

public record ChatMessageDto(string Role, string Content, DateTime CreatedAt);

public record ChatSessionDto(
    string SessionId,
    ResumeProfileDto? Resume,
    JobFilters Filters,
    IReadOnlyList<ChatMessageDto> Messages);

/// <summary>
/// One narration step in the bot's visible reasoning. This is the
/// "thought-process" stream: real stage transitions from the pipeline, not
/// decorative text on a timer.
/// </summary>
public record ThoughtDto(string Stage, string Text);

public record HealthDto(
    string Status,
    string Database,
    bool PythonTablesVisible,
    bool ChatModelConfigured,
    string ChatModel,
    bool EmbeddingModelConfigured,
    string EmbeddingModel,
    bool SemanticMatching,
    IReadOnlyList<string> AvailableModels,
    int JobCount,
    int IndexedJobCount);

// -- applications -----------------------------------------------------------

/// <summary>
/// One role in an apply-review, before anything is submitted.
///
/// The whole point of the preview is that the candidate sees exactly what will
/// be sent, per role, and can drop or edit any of it. Bulk apply without this
/// step is a button that mails strangers on your behalf.
/// </summary>
public record ApplicationPreviewItemDto(
    JobSummaryDto Job,
    double FitScore,
    string FitBand,
    string CoverLetter,
    string LetterMode,
    IReadOnlyList<SkillAssessmentDto> Skills,
    IReadOnlyList<string> Strengths,
    IReadOnlyList<string> Gaps,
    /// <summary>True when this CV has already been sent to this posting. Shown
    /// and excluded rather than hidden, so the count the candidate confirms is
    /// the count that gets sent.</summary>
    bool AlreadyApplied);

public record ApplicationPreviewDto(
    ResumeProfileDto Resume,
    IReadOnlyList<ApplicationPreviewItemDto> Items,
    int EligibleCount,
    int AlreadyAppliedCount,
    bool SemanticMatching,
    string EmbeddingModel,
    /// <summary>Set when the letters were composed without a model, so the UI can
    /// say so instead of implying they were written.</summary>
    bool LettersAreDeterministic);

/// <summary>
/// Ask for a review of what applying would send.
///
/// Either an explicit set of postings, or a fit floor ("everything above 70").
/// A floor with no ceiling is capped by <paramref name="Limit"/> — an unbounded
/// bulk apply is how one careless sentence reaches a hundred employers.
/// </summary>
public record ApplyPreviewRequest(
    int ResumeId,
    IReadOnlyList<int>? JobIds = null,
    double? MinFitScore = null,
    JobFilters? Filters = null,
    int? Limit = null,
    /// <summary>
    /// Whether to write the letters now.
    ///
    /// Default false, because a local model takes tens of seconds per letter and
    /// generating five before the review panel can open makes applying feel broken.
    /// The review opens instantly on composed letters and each one is upgraded on
    /// request — same output, not blocking the screen.
    /// </summary>
    bool DraftLetters = false);

/// <summary>
/// Submit. Letters are passed back verbatim from the preview so that what the
/// candidate approved is what is stored — regenerating here would send text
/// nobody read, and a model is not deterministic between two calls.
/// </summary>
public record ApplySubmitRequest(
    int ResumeId,
    IReadOnlyList<ApplySubmitItem> Items);

public record ApplySubmitItem(int JobId, string? CoverLetter);

public record ApplySubmitResultDto(
    IReadOnlyList<ApplicationDto> Submitted,
    IReadOnlyList<string> Skipped,
    int SubmittedCount,
    int SkippedCount);

/// <summary>One stored application, as the recruiter and the candidate see it.</summary>
public record ApplicationDto(
    int Id,
    JobSummaryDto Job,
    ResumeProfileDto Candidate,
    string Status,
    string CoverLetter,
    string LetterMode,
    double FitScore,
    string FitBand,
    double SemanticScore,
    double SkillScore,
    double TitleScore,
    double ExperienceScore,
    IReadOnlyList<SkillAssessmentDto> Skills,
    IReadOnlyList<string> Strengths,
    IReadOnlyList<string> Gaps,
    string RecruiterNote,
    bool SemanticMatching,
    string EmbeddingModel,
    int? AcceptedAnalysisId,
    DateTime? AcceptedAt,
    bool HasResumeFile,
    DateTime CreatedAt);

/// <summary>
/// Records a decision on an applicant.
///
/// <paramref name="AnalysisId"/> is supplied when a recruiter accepts someone into
/// the hiring pipeline: the analysis row is created by the PYTHON backend, and its
/// id is reported back here. This side never writes the recruiter platform's tables.
/// </summary>
public record ApplicationStatusRequest(string Status, int? AnalysisId = null);

/// <summary>Writes one letter, for a role already being reviewed.</summary>
public record DraftLetterRequest(int ResumeId, int JobId);

public record DraftLetterDto(int JobId, string CoverLetter, string LetterMode);

// -- browsing without a resume ----------------------------------------------

/// <summary>
/// Roles offered to someone who has not uploaded a CV.
///
/// Carries no fit score, and that is the point: a percentage is a claim about a
/// person, and there is no person here yet. <paramref name="Ranked"/> says whether
/// the order means relevance or is simply newest-first, so the UI never presents
/// an arbitrary sequence as if it were a ranking.
/// </summary>
public record JobSuggestionResultDto(
    IReadOnlyList<JobSummaryDto> Jobs,
    int TotalMatching,
    JobFilters AppliedFilters,
    bool SemanticMatching,
    string EmbeddingModel,
    bool Ranked);

// -- LLM evaluation ---------------------------------------------------------

/// <summary>
/// What KIND of thing a job description is asking for.
///
/// A closed vocabulary, because the whole point is that these are not worth the
/// same. A posting's responsibilities are the duties of the role, not a checklist
/// the applicant must already have ticked — "manage the end-to-end recruitment
/// process" is a description of the job, and scoring its absence like a missing
/// mandatory tool is what dropped a seven-year recruiter to 65% on a recruiting
/// role. Separating the kinds is what lets the weighting say so.
/// </summary>
public static class RequirementKinds
{
    /// <summary>Explicitly required or must-have. The only kind that can trigger a knockout.</summary>
    public const string MustHave = "must_have";
    /// <summary>A named technical skill or tool the role turns on, short of mandatory.</summary>
    public const string CoreSkill = "core_skill";
    /// <summary>Years, seniority or domain background.</summary>
    public const string Experience = "experience";
    /// <summary>A duty of the role rather than a qualification for it.</summary>
    public const string Responsibility = "responsibility";
    public const string Education = "education";
    /// <summary>Stated as a plus, preferred or bonus.</summary>
    public const string NiceToHave = "nice_to_have";

    public static readonly string[] All =
        { MustHave, CoreSkill, Experience, Responsibility, Education, NiceToHave };

    /// <summary>Anything the model invents falls back to the middle of the range.</summary>
    public static string Normalise(string? value)
    {
        var v = (value ?? "").Trim().ToLowerInvariant().Replace(' ', '_').Replace('-', '_');
        return All.Contains(v) ? v : CoreSkill;
    }

    public static string Label(string kind) => kind switch
    {
        MustHave => "Must-have",
        CoreSkill => "Core skill",
        Experience => "Experience",
        Responsibility => "Responsibility",
        Education => "Education",
        NiceToHave => "Nice to have",
        _ => kind,
    };
}

/// <summary>How well one requirement is backed up. STRONG | WEAK | MISSING.</summary>
public static class MatchLevels
{
    /// <summary>Demonstrated in work history. Full credit.</summary>
    public const string Strong = "STRONG";
    /// <summary>Claimed in a skills list, a project, education or a summary. Half credit.</summary>
    public const string Weak = "WEAK";
    public const string Missing = "MISSING";

    public static string Normalise(string? value) => (value ?? "").Trim().ToUpperInvariant() switch
    {
        "STRONG" => Strong,
        "WEAK" => Weak,
        _ => Missing,
    };

    /// <summary>The credit each level earns, before weighting.</summary>
    public static double Credit(string level) => level switch
    {
        Strong => 1.0,
        Weak => 0.5,
        _ => 0.0,
    };
}

/// <summary>
/// One atomised requirement, judged.
///
/// <paramref name="Quote"/> is a phrase verified to exist in the résumé — the
/// model may judge, but it may not invent the words it judges from.
/// </summary>
public record EvaluationRequirementDto(
    string Requirement,
    string Quote,
    /// <summary>See <see cref="MatchLevels"/>.</summary>
    string MatchLevel,
    /// <summary>See <see cref="RequirementKinds"/>.</summary>
    string Kind,
    /// <summary>Which part of the résumé the quote came from, when it could be located.</summary>
    string Where = "",
    /// <summary>
    /// One sentence on why THIS requirement got THIS verdict.
    ///
    /// Tied to a single requirement that carries its own verdict and quote, which
    /// is what makes it safe to show: a sentence claiming the candidate has
    /// something is contradicted on the same row when the row says MISSING. Free
    /// prose about the candidate as a whole has no such check, which is why the
    /// card no longer takes any.
    /// </summary>
    string Reasoning = "");

/// <summary>
/// Whether a hard dealbreaker fired, and what it was.
///
/// Deliberately narrow. A knockout caps the score outright, so it may only be
/// raised by a genuinely absent MUST-HAVE or an unmet years requirement — never
/// by a responsibility the candidate's role plainly covers.
/// </summary>
public record EvaluationKnockoutDto(
    bool MissingMandatorySkills,
    bool MissingYearsOfExperience,
    IReadOnlyList<string> Reasons,
    bool MissingRequiredEducation = false)
{
    public bool Fired => MissingMandatorySkills || MissingYearsOfExperience || MissingRequiredEducation;
}

/// <summary>
/// One weighting, over a <see cref="RequirementKinds"/> value. They sum to 100.
/// </summary>
public record EvaluationWeightDto(string Criterion, int Weight);

/// <summary>One requirement the candidate can back up, and the wording that proves it.</summary>
public record EvaluationMatchDto(string Requirement, string Evidence, string Where);

public record EvaluationGapDto(string Requirement, string Why);

/// <summary>
/// A reasoned evaluation of one résumé against one posting.
///
/// The counterpart to <see cref="JobMatchDto"/>: that carries a computed fit with
/// four weighted dimensions, this carries a judgement with its evidence attached.
/// Both can be shown for the same role, and they are allowed to disagree — one is
/// arithmetic over structure, the other is reading comprehension.
/// </summary>
public record JobEvaluationDto(
    int JobId,
    /// <summary>
    /// The authoritative percentage, computed from the per-requirement verdicts and
    /// the weighting — not stated by the model.
    ///
    /// The split exists because the two halves of this job need different things.
    /// Deciding "is this requirement proven in work history?" is reading
    /// comprehension, which a small model does adequately. Turning six such verdicts
    /// into a number out of a hundred is arithmetic, which it does badly and
    /// inconsistently. So it judges and we count.
    /// </summary>
    int OverallMatch,
    string Category,
    string Reasoning,
    string ExecutiveSummary,
    IReadOnlyList<EvaluationWeightDto> Weights,
    IReadOnlyList<EvaluationMatchDto> HighConfidence,
    IReadOnlyList<EvaluationMatchDto> Partial,
    IReadOnlyList<EvaluationGapDto> Gaps,
    string Decision,
    string? AlternateRole,
    /// <summary>Which prompt produced this. Two scores from different prompts are
    /// not comparable, and a stored one that does not say cannot be audited.</summary>
    string PromptVersion,
    /// <summary>Every requirement the posting was broken into, with its verdict.</summary>
    IReadOnlyList<EvaluationRequirementDto>? Requirements = null,
    /// <summary>
    /// The percentage the model stated for itself, kept beside the computed one.
    ///
    /// Shown rather than discarded: where the two disagree sharply, one of them is
    /// wrong about this candidate, and that is worth seeing while the prompt is
    /// still being calibrated.
    /// </summary>
    int ModelMatch = 0,
    EvaluationKnockoutDto? Knockout = null,
    /// <summary>The model's one-line defence of its own verdict.</summary>
    string Justification = "");

public record EvaluateRequest(int ResumeId, int JobId);
