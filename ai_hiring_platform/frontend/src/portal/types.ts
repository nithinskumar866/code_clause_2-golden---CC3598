/**
 * Mirror of the job-portal contract in
 * `job_portal/backend/JobPortal.Api/Contracts/Contracts.cs`.
 *
 * That backend owns these shapes; this file follows. When the two disagree the
 * backend is right — change it there first, then mirror the change here.
 *
 * Deliberately separate from `src/types/index.ts`, which mirrors the Python
 * recruiter API. Two backends, two contracts: merging them into one file would
 * make a change on either side look like a change to the whole app's types.
 */

export interface ApiResponse<T> {
  success: boolean;
  message: string;
  data: T | null;
}

// -- jobs -------------------------------------------------------------------

export interface JobSummary {
  id: number;
  title: string;
  company: string;
  location: string;
  workMode: string;
  employmentType: string;
  seniorityLevel: string;
  minYearsExperience: number | null;
  maxYearsExperience: number | null;
  salaryMin: number | null;
  salaryMax: number | null;
  salaryCurrency: string | null;
  requiredSkills: string[];
  preferredSkills: string[];
  summary: string;
  applyUrl: string | null;
  isIndexed: boolean;
  createdAt: string;
}

export interface JobDetail {
  summary: JobSummary;
  responsibilities: string[];
  qualifications: string[];
  rawText: string;
  extractionMode: string;
  sourceJobDescriptionId: number | null;
}

export interface CreateJobRequest {
  title: string;
  company?: string;
  location?: string;
  workMode?: string;
  employmentType?: string;
  seniorityLevel?: string;
  minYearsExperience?: number | null;
  maxYearsExperience?: number | null;
  salaryMin?: number | null;
  salaryMax?: number | null;
  salaryCurrency?: string;
  description?: string;
  requiredSkills?: string[];
  preferredSkills?: string[];
  applyUrl?: string;
}

export interface JobUploadResult {
  job: JobDetail;
  indexed: boolean;
  embeddingModel: string;
  semanticMatching: boolean;
}

/**
 * A question the assistant offers as a tappable chip.
 *
 * `label` is what it reads; `message` is what gets sent. They differ because a chip
 * has to fit on one line while the message it sends should be unambiguous on its own
 * once the transcript has scrolled.
 */
export interface FollowUp {
  label: string;
  message: string;
}

/** One file's outcome in a bulk upload. `error` is null when `success` is true. */
export interface JobUploadItem {
  filename: string;
  success: boolean;
  job: JobSummary | null;
  indexed: boolean;
  error: string | null;
}

/**
 * Mirrors JobBulkUploadResultDto. Partial success is the normal case, not an edge
 * case — a couple of unreadable scans in a batch of thirty must not discard the rest.
 */
export interface JobBulkUploadResult {
  total: number;
  succeeded: number;
  failed: number;
  items: JobUploadItem[];
  embeddingModel: string;
  semanticMatching: boolean;
}

export interface ImportResult {
  imported: number;
  skipped: number;
  failed: number;
  notes: string[];
}

export interface IndexStatus {
  totalJobs: number;
  indexedJobs: number;
  activeModel: string;
  semanticMatching: boolean;
  vectorsByModel: Record<string, number>;
}

// -- resumes ----------------------------------------------------------------

export interface ResumeProfile {
  id: number;
  filename: string;
  candidateName: string;
  email: string;
  phone: string;
  location: string;
  currentTitle: string;
  yearsExperience: number | null;
  skills: string[];
  titles: string[];
  education: string[];
  summary: string;
  extractionMode: string;
}

// -- matching ---------------------------------------------------------------

export type SkillStatus = 'Have' | 'Transferable' | 'Missing';

export interface SkillAssessment {
  skill: string;
  status: SkillStatus;
  evidenceSkill: string | null;
  similarity: number;
}

/**
 * How a fit score was arrived at.
 *
 * `computed` is the weighted arithmetic over four measured dimensions — instant
 * and reproducible. `reasoned` is a language model judging the CV against the
 * posting's own stated requirements, weighted the way that role warrants; slower,
 * and the percentage is the model's own.
 *
 * The two numbers are NOT comparable, which is why every card says which one it
 * is showing rather than presenting a bare percentage.
 */
export type ScoringMode = 'computed' | 'reasoned' | 'rag';

/** One weighting the evaluator chose for THIS posting. They sum to 100. */
export interface EvaluationWeight {
  criterion: string;
  weight: number;
}

/** A requirement the CV can back up, with the wording that proves it. */
export interface EvaluationMatch {
  requirement: string;
  /** A phrase verified to exist in the CV. Never the model's paraphrase. */
  evidence: string;
  /** Where in the document it was found — "work experience", "skills list". */
  where: string;
}

export interface EvaluationGap {
  requirement: string;
  why: string;
}

/**
 * A reasoned judgement of one CV against one posting.
 *
 * Mirrors `JobEvaluationDto`. Present only on matches whose `scoringMode` is
 * `reasoned` — under `computed` there is no judgement to show, and rendering an
 * empty one would imply the evaluator ran and found nothing.
 */
/**
 * What kind of thing the posting was asking for.
 *
 * These are not worth the same, and saying so is the point. A `responsibility`
 * is a duty of the role rather than a qualification for it — "manage the
 * end-to-end recruitment process" describes the job on offer, and scoring its
 * absence like a missing mandatory tool is what dropped a seven-year recruiter
 * to 65% on a recruiting role.
 */
export type RequirementKind =
  | 'must_have' | 'core_skill' | 'experience' | 'responsibility' | 'education' | 'nice_to_have';

/** How well one requirement is backed up. */
export type MatchLevel = 'STRONG' | 'WEAK' | 'MISSING';

/** One atomised requirement from the posting, judged. */
export interface EvaluationRequirement {
  requirement: string;
  /** A phrase verified to exist in the CV. Empty when nothing supports it. */
  quote: string;
  matchLevel: MatchLevel;
  kind: RequirementKind;
  where: string;
  /**
   * One sentence on why this row got this verdict.
   *
   * Safe to render because it sits next to its own verdict and quote: a sentence
   * crediting the candidate with something is contradicted in place when the row
   * says MISSING. The card takes no free-floating prose from the model for
   * exactly that reason — there is nothing to contradict it.
   */
  reasoning: string;
}

/**
 * A hard dealbreaker, if one fired. Caps the score at 35.
 *
 * Narrow by construction: only an absent must-have or an unmet years minimum can
 * raise it. A missing responsibility never can.
 */
export interface EvaluationKnockout {
  missingMandatorySkills: boolean;
  missingYearsOfExperience: boolean;
  missingRequiredEducation: boolean;
  reasons: string[];
}

export interface JobEvaluation {
  jobId: number;
  /** The authoritative score, computed from the per-requirement verdicts. */
  overallMatch: number;
  /**
   * What the model said the score was, when it was asked for one.
   *
   * Zero means it was not — the auditor prompt forbids the model from producing
   * any number, so there is no second opinion to show and the card must not
   * render a 0% the model never claimed.
   */
  modelMatch: number;
  category: string;
  reasoning: string;
  justification: string;
  executiveSummary: string;
  weights: EvaluationWeight[];
  requirements: EvaluationRequirement[] | null;
  knockout: EvaluationKnockout | null;
  highConfidence: EvaluationMatch[];
  partial: EvaluationMatch[];
  gaps: EvaluationGap[];
  decision: string;
  alternateRole: string | null;
  promptVersion: string;
}

export interface JobMatch {
  job: JobSummary;
  fitScore: number;
  fitBand: string;
  semanticScore: number;
  skillScore: number;
  titleScore: number;
  experienceScore: number;
  skills: SkillAssessment[];
  strengths: string[];
  gaps: string[];
  recruiterNote: string;
  scoringMode: ScoringMode;
  /** Null under `computed`, and when the evaluator was unreachable for this role. */
  evaluation: JobEvaluation | null;
}

export interface JobFilters {
  workMode?: string | null;
  location?: string | null;
  employmentType?: string | null;
  seniorityLevel?: string | null;
  minSalary?: number | null;
  maxYearsRequired?: number | null;
  keywords?: string[] | null;
}

export interface MatchResult {
  matches: JobMatch[];
  totalCandidates: number;
  appliedFilters: JobFilters;
  semanticMatching: boolean;
  embeddingModel: string;
  scoringMode: ScoringMode;
  /**
   * Identifies the shortlist this belongs to.
   *
   * Reasoned scoring judges roles one at a time and keeps going after the turn
   * has ended, so a later, fuller version of the same shortlist arrives while the
   * candidate may already have asked something else. Matching on the batch id is
   * how those updates land on the right cards instead of appearing as a new answer.
   */
  batchId: string;
  /** Roles still being judged. Zero when finished. */
  pending: number;
  complete: boolean;
}

// -- chat -------------------------------------------------------------------

/**
 * Copilot Mode. The backend sends these alongside the prose so the UI never has
 * to parse a sentence to decide what to do. An unrecognised action is ignored,
 * not guessed at.
 */
export interface UiAction {
  action: 'update_filters' | 'show_jobs' | 'resume_ready' | string;
  payload: unknown;
}

export interface Thought {
  stage: string;
  text: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | string;
  content: string;
  createdAt: string;
}

export interface ChatSession {
  sessionId: string;
  resume: ResumeProfile | null;
  filters: JobFilters;
  messages: ChatMessage[];
}

export interface Health {
  status: string;
  database: string;
  pythonTablesVisible: boolean;
  chatModelConfigured: boolean;
  chatModel: string;
  embeddingModelConfigured: boolean;
  embeddingModel: string;
  semanticMatching: boolean;
  availableModels: string[];
  jobCount: number;
  indexedJobCount: number;
}

// -- applications -----------------------------------------------------------

export interface ApplicationPreviewItem {
  job: JobSummary;
  fitScore: number;
  fitBand: string;
  coverLetter: string;
  letterMode: string;
  skills: SkillAssessment[];
  strengths: string[];
  gaps: string[];
  /** Already sent from this CV. Shown and excluded, never silently dropped. */
  alreadyApplied: boolean;
}

export interface ApplicationPreview {
  resume: ResumeProfile;
  items: ApplicationPreviewItem[];
  eligibleCount: number;
  alreadyAppliedCount: number;
  semanticMatching: boolean;
  embeddingModel: string;
  lettersAreDeterministic: boolean;
}

export interface ApplyPreviewRequest {
  resumeId: number;
  jobIds?: number[];
  minFitScore?: number;
  filters?: JobFilters;
  limit?: number;
}

export interface ApplySubmitItem {
  jobId: number;
  coverLetter?: string;
}

export interface ApplySubmitResult {
  submitted: Application[];
  skipped: string[];
  submittedCount: number;
  skippedCount: number;
}

export type ApplicationStatus = 'Submitted' | 'Accepted' | 'Declined';

export interface Application {
  id: number;
  job: JobSummary;
  candidate: ResumeProfile;
  status: ApplicationStatus | string;
  coverLetter: string;
  letterMode: string;
  fitScore: number;
  fitBand: string;
  semanticScore: number;
  skillScore: number;
  titleScore: number;
  experienceScore: number;
  skills: SkillAssessment[];
  strengths: string[];
  gaps: string[];
  recruiterNote: string;
  semanticMatching: boolean;
  embeddingModel: string;
  acceptedAnalysisId: number | null;
  acceptedAt: string | null;
  hasResumeFile: boolean;
  createdAt: string;
}

// -- pre-application screening ----------------------------------------------

/**
 * What a screening question is about.
 *
 * Mirrors `ScreeningTopics`. Only `skill` may appear more than once in a set —
 * every other topic has exactly one sensible question.
 */
export type ScreeningTopic =
  | 'location' | 'work_mode' | 'skill' | 'experience' | 'employment_type';

export interface ScreeningQuestion {
  id: number;
  question: string;
  topic: ScreeningTopic;
  /** The phrase from the posting that prompted it — the answer to "why ask me this?". */
  basedOn: string;
}

/**
 * Someone to talk to about this kind of role.
 *
 * `relevant` marks the one whose remit matches THIS posting — the backend routes
 * on the job's work mode, so an onsite role leads with the onsite contact rather
 * than making the applicant work out which of two links applies to them.
 */
export interface ScreeningContact {
  name: string;
  url: string;
  /** What they cover, in the applicant's words: "remote roles". */
  handles: string;
  relevant: boolean;
}

export interface ScreeningQuestionSet {
  jobId: number;
  jobTitle: string;
  questions: ScreeningQuestion[];
  /** False when the model was unreachable and the deterministic set stood in. */
  generated: boolean;
  /** Most relevant to this posting first. May be empty if none are configured. */
  contacts: ScreeningContact[];
}

export interface ScreeningAnswer {
  questionId: number;
  yes: boolean;
}

export interface ScreeningResult {
  passed: boolean;
  declined: ScreeningQuestion[];
  unanswered: number[];
  message: string;
}

// -- browsing without a resume ----------------------------------------------

/**
 * Roles offered before any CV exists.
 *
 * No fit score, deliberately: a percentage is a claim about a person, and there
 * is nobody to make it about yet. `ranked` says whether the order means
 * relevance or is simply newest-first, so the UI never presents an arbitrary
 * sequence as a ranking.
 */
export interface JobSuggestionResult {
  jobs: JobSummary[];
  totalMatching: number;
  appliedFilters: JobFilters;
  semanticMatching: boolean;
  embeddingModel: string;
  ranked: boolean;
}

export interface DraftLetterResult {
  jobId: number;
  coverLetter: string;
  letterMode: string;
}
