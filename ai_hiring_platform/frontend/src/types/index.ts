export interface ResumeMetadata {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
}

export interface JobDescriptionMetadata {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
}

export interface SystemStatusData {
  backend: string;
  database: string;
  llmProvider: string;
  vectorStore: string;
  embeddingModel: string;
  agentStatus: string;
}

/**
 * A stored resume or job-description record as returned by the list endpoints.
 * Shared by the upload pages and the analysis selectors.
 */
export interface FileRecord {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
}

/** Result of a single document upload (resume or job description). */
export type UploadResult = FileRecord;

/** Result of one file within a bulk upload batch. */
export interface BulkUploadFileResult {
  filename: string;
  success: boolean;
  data?: UploadResult;
  error?: string;
  /** The file's content already exists in the pool — a correct skip, not a failure. */
  duplicate?: boolean;
  existing_id?: number;
  existing_filename?: string;
}

/** Result of a bulk upload request. */
export interface BulkUploadResult {
  success_count: number;
  failed_count: number;
  /** Files skipped because the same content is already indexed. */
  duplicate_count?: number;
  total: number;
  results: BulkUploadFileResult[];
}

/* ------------------------------------------------------------------ *
 * Hiring analysis report — mirror of backend schemas/analysis.py.
 * The frontend only mirrors this contract; it never defines it.
 * ------------------------------------------------------------------ */

export interface RequirementFit {
  requirement: string;
  category: string;
  status: string; // "Matched" | "Partial" | "Missing"
  matched_evidence: string;
  explanation: string;
  limitations: string;
  confidence: number;
  // F2 — requirement prioritization (derived from JD wording). Nullable for
  // backward-compat with reports persisted before the field existed.
  importance?: 'must' | 'nice' | null;
  weight?: number | null;
  /** Scannable one-line verdict shown on the collapsed row. Null on older reports. */
  evidence_summary?: string | null;
}

export interface LearningRoadmapItem {
  skill: string;
  estimated_time: string;
  reason: string;
}

/**
 * F1 — deterministic keyword-stuffing / over-claim detection. A skill only counts
 * as demonstrated when it appears in a narrative (Experience/Projects) section;
 * skills that surface only in listing sections are over-claimed. Computed
 * algorithmically from retrieved evidence — never by the LLM.
 */
export interface AuthenticityAssessment {
  credibility_score: number; // 0-100 (corroboration as %)
  keyword_stuffing_risk: 'Low' | 'Medium' | 'High';
  over_claimed_skills: string[]; // listed but never demonstrated
  corroboration_ratio: number; // 0.0-1.0
  explanation: string;
}

/**
 * F3 — deterministic candidate identity + seniority fit vs the JD's stated
 * experience requirement. All fields nullable — render only what is present.
 */
export interface CandidateProfile {
  name: string | null;
  title: string | null;
  total_years: number | null;
  seniority_level: 'Junior' | 'Mid' | 'Senior' | 'Lead' | null;
  required_years: number | null;
  seniority_fit: 'Below' | 'Meets' | 'Exceeds' | 'Unknown' | null;
  explanation: string;
}

export interface MatchRecord {
  chunk: string;
  section: string;
  score: number;
  confidence: number;
  page: number;
  filename: string;
  chunk_id: number;
}

export interface RetrievalResult {
  requirement: string;
  matches: MatchRecord[];
  error?: string;
}

/* ------------------------------------------------------------------ *
 * Match Score — mirror of backend schemas/analysis.py :: MatchScore.
 * ------------------------------------------------------------------ */

export interface MatchParameter {
  key: string;
  label: string;
  /** Share of the final score this parameter carries, 0.0-1.0. */
  weight: number;
  /** This parameter's own score, 0-100, to one decimal. */
  score: number;
  /** score x weight — the points this parameter contributed. */
  contribution: number;
  /** Why it scored what it did, in recruiter language. */
  basis: string;
  /** The job description never stated this requirement, so it scored mid-range. */
  neutral: boolean;
}

export interface MatchScore {
  score: number;
  band: string;
  parameters: MatchParameter[];
}

export interface AnalysisReport {
  analysis_id: number;
  candidate_id: number;
  resume_id: number;
  jd_id: number;
  retrieval_results: RetrievalResult[];

  /** The Match Score, 0-100 to one decimal. */
  overall_score: number;
  /** Its nine-parameter decomposition. Absent on reports evaluated before Match Score. */
  match_score?: MatchScore | null;
  coverage_score: number;
  experience_score: number;
  project_score: number;
  confidence_score: number;
  quality_score: number;

  summary: string;
  requirements: RequirementFit[];

  strengths: string[];
  weaknesses: string[];
  skill_relationships: string[];

  missing_skills: string[];
  learning_roadmap: LearningRoadmapItem[];
  interview_questions: string[];

  recruiter_recommendation: string;
  rejection_email: string | null;

  // F1/F3 — additive, nullable. Present on reports produced after these slices.
  authenticity?: AuthenticityAssessment | null;
  candidate_profile?: CandidateProfile | null;
}

/* ------------------------------------------------------------------ *
 * Multi-candidate ranking — mirror of backend schemas/ranking.py.
 * ------------------------------------------------------------------ */

export interface RankingEntry {
  rank: number;
  analysis_id: number | null;
  resume_id: number;
  resume_filename: string;
  overall_score: number;
  coverage_score: number;
  experience_score: number;
  project_score: number;
  quality_score: number;
  confidence_score: number;
  recruiter_recommendation: string;
  seniority_fit: string | null;
  credibility_score: number | null;
  keyword_stuffing_risk: string | null;
  matched_count: number;
  partial_count: number;
  missing_count: number;
  top_missing_skills: string[];
  error: string | null;
}

export interface RankingResponse {
  jd_id: number;
  jd_filename: string;
  candidate_count: number;
  evaluated_count: number;
  top_candidate: RankingEntry | null;
  entries: RankingEntry[];
}

/* ------------------------------------------------------------------ *
 * Analytics — mirror of backend schemas/analytics.py.
 * ------------------------------------------------------------------ */

export interface OverallStatistics {
  total_analyses: number;
  selected: number;
  borderline: number;
  rejected: number;
  average_overall_score: number;
  average_experience_score: number;
  average_project_score: number;
  average_quality_score: number;
  average_coverage_score: number;
}

export interface DashboardOverview {
  total_analyses: number;
  selected_count: number;
  borderline_count: number;
  rejected_count: number;
  average_overall_score: number;
  average_skill_score: number;
  average_experience_score: number;
  average_project_score: number;
  average_quality_score: number;
}

export interface ScoreBucket {
  label: string;
  min: number;
  max: number;
  count: number;
}
export interface ScoreDistribution {
  total_analyses: number;
  ranges: ScoreBucket[];
}

export interface RecommendationBucket {
  label: string;
  count: number;
  percentage: number;
}
export interface RecommendationDistribution {
  total_analyses: number;
  distribution: RecommendationBucket[];
}

export interface TrendPoint {
  period: string;
  count: number;
}
export interface TrendData {
  daily: TrendPoint[];
  weekly: TrendPoint[];
  monthly: TrendPoint[];
}

export interface TopItem {
  name: string;
  count: number;
}

export interface SkillCount {
  skill: string;
  count: number;
}
export interface SkillFrequency {
  top_matched: SkillCount[];
  top_missing: SkillCount[];
}

/* ------------------------------------------------------------------ *
 * Recruiter workflow + notes — mirror of backend schemas/status.py & note.py.
 * ------------------------------------------------------------------ */

/** Ordered candidate pipeline stages (mirror of backend WORKFLOW_ORDER). */
export const WORKFLOW_STAGES = [
  'Applied',
  'Screening',
  'Reviewed',
  'Interview Scheduled',
  'Interview Completed',
  'Selected',
  'Rejected',
  'Offer Sent',
] as const;
export type WorkflowStatus = (typeof WORKFLOW_STAGES)[number];

export interface Note {
  id: number;
  analysis_id: number;
  text: string;
  author: string;
  created_at: string;
  updated_at: string;
}

/* AI Recruiter — mirror of backend schemas/interview.py. */
export interface InterviewQA {
  question: string;
  ideal_answer: string;
  evidence: string;
  confidence: number;
  missing_information: string;
  follow_up_questions: string[];
  recruiter_evaluation: string;
}

export interface InterviewSimulation {
  analysis_id: number;
  generated_by: string; // "llm" | "deterministic"
  items: InterviewQA[];
}

/** Recruiter recommendation buckets accepted by the history filter. */
export type RecommendationValue = 'Selected' | 'Borderline' | 'Rejected';

/** Result orderings accepted by the history endpoint. */
export type HistorySortValue = 'newest' | 'oldest' | 'highest_score' | 'lowest_score';

/**
 * Query parameters for GET /analysis/history. Keys map 1:1 to the backend
 * query parameter names — the single source of truth for the contract.
 */
export interface HistoryQuery {
  resume_filename?: string;
  jd_filename?: string;
  recommendation?: RecommendationValue;
  min_score?: number;
  max_score?: number;
  date_from?: string; // YYYY-MM-DD
  date_to?: string; // YYYY-MM-DD
  sort?: HistorySortValue;
  page?: number;
  page_size?: number;
}

/** Pagination metadata returned in the response envelope's `meta` field. */
export interface HistoryPageMeta {
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
}

/** A page of history results: the rows plus pagination metadata. */
export interface HistoryPage {
  items: HistoryRecord[];
  meta: HistoryPageMeta;
}

/**
 * A summary row for a previously stored analysis, as returned by the
 * history list endpoint. The full report is fetched separately by id.
 */
export interface HistoryRecord {
  id: number; // analysis_id
  created_at: string; // ISO timestamp
  resume_id: number;
  jd_id: number;
  resume_filename: string;
  jd_filename: string;
  overall_score: number;
  recruiter_recommendation: string;
  summary: string;
}

/* ------------------------------------------------------------------------- *
 * Recruiter chatbot — mirrors backend/app/schemas/chat.py
 * ------------------------------------------------------------------------- */

/** One retrieved resume chunk backing a claim — the audit trail for a match. */
export interface ChatEvidence {
  skill: string;
  text: string;
  section: string;
  page: number;
  filename: string;
  similarity: number;
  /** The chunk contains the skill verbatim, rather than merely being semantically near it. */
  literal: boolean;
}

/** Decomposition of the match percentage, so the number is explainable. */
export interface ChatScoreBreakdown {
  skill_coverage: number;
  evidence_strength: number;
  experience_fit: number;
}

export interface ChatCandidate {
  resume_id: number;
  name: string | null;
  title: string | null;
  filename: string | null;
  email: string | null;
  phone: string | null;
  location: string | null;
  total_years: number | null;
  seniority_level: string | null;
  match_percentage: number;
  breakdown: ChatScoreBreakdown;
  matched_skills: string[];
  missing_skills: string[];
  demonstrated_skills: string[];
  listed_only_skills: string[];
  /** Per-skill substantiation 0-100 and the resume sections that proved it. */
  skill_depth: Record<string, number>;
  skill_sections: Record<string, string[]>;
  all_skills: string[];
  sections_present: string[];
  experience_note: string;
  /** One-line headline judgement, e.g. "Strong fit — 88%, proven in real work". */
  verdict: string;
  /** Card body as scannable points rather than a paragraph. */
  highlights: string[];
  /** Whether this candidate satisfies a location the recruiter asked for. */
  location_match: boolean;
  /** How to say it — "based in T Nagar, as you asked" / "listed in Bangalore, not T Nagar". */
  location_note: string;
  reasoning: string;
  evidence: ChatEvidence[];
}

/**
 * A direct answer to a direct question about one candidate.
 *
 * `found: false` means the resume genuinely does not state the detail. That is a real
 * answer — substituting a pool-wide search would hand back a different person.
 */
export interface ChatFact {
  resume_id: number;
  name: string | null;
  attribute: string;
  value: string | null;
  found: boolean;
  evidence: ChatEvidence[];
}

/** One concrete choice offered alongside a question back. */
export interface ChatClarificationOption {
  label: string;
  query: string;
  action: 'ask' | 'navigate';
  resume_id: number | null;
}

/**
 * A question the assistant asked back, with the options that answer it.
 *
 * Raised only when it genuinely cannot proceed — several real people share the name,
 * or every constraint combination is empty. Each option carries a measured outcome.
 */
export interface ChatClarification {
  question: string;
  options: ChatClarificationOption[];
  /** True when "I don't know which — show me all" is a sensible reply. */
  allow_all: boolean;
  all_label: string;
}

/** What SHAPE an answer is, so the UI renders it as what it actually is. */
export type ChatAnswerType =
  | 'candidates'
  | 'comparison'
  | 'fact'
  | 'explanation'
  | 'clarification'
  | 'refusal'
  | 'empty';

/** A suggested next step shown under an answer. */
export interface ChatSuggestion {
  label: string;
  query: string;
  /** `ask` re-queries the assistant; `navigate` hands off to AI Analysis. */
  action: 'ask' | 'navigate';
  resume_id: number | null;
}

/** How the backend understood the question — shown so recruiters can see the parse. */
export interface ChatIntent {
  kind: string;
  skills: string[];
  min_years: number | null;
  /** Places named in the question that the pool actually knows about. */
  places: string[];
  /** The detail a fact question asked for ("location", "initials"), if any. */
  attribute: string | null;
  named_candidates: string[];
  /** [typed, corrected] pairs, e.g. [["javaa", "java"]]. */
  corrections: string[][];
  unresolved_name: string | null;
  /** A place the recruiter named that no resume mentions — reported, never dropped. */
  unmatched_place: string | null;
  is_followup: boolean;
  /** "rules" or "rules+llm" — whether the language model helped read the question. */
  parsed_by: string;
}

export interface ChatResponse {
  answer: string;
  answer_type: ChatAnswerType;
  candidates: ChatCandidate[];
  fact: ChatFact | null;
  clarification: ChatClarification | null;
  refused: boolean;
  refusal_category: string | null;
  /** The assistant asked a question back instead of guessing. */
  needs_clarification: boolean;
  intent: ChatIntent | null;
  suggestions: ChatSuggestion[];
  diagnostics: Record<string, unknown>;
  elapsed_ms: number;
}

export interface CorpusStatus {
  indexed_resumes: number;
  indexed_chunks: number;
  database_resumes: number | null;
  in_sync: boolean;
  cached: boolean;
}

export interface CorpusSyncResult {
  added: number;
  updated: number;
  removed: number;
  unchanged: number;
  total_resumes: number;
  total_chunks: number;
}

/** A single rendered turn in the chat transcript. */
export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  answerType?: ChatAnswerType;
  fact?: ChatFact | null;
  clarification?: ChatClarification | null;
  candidates?: ChatCandidate[];
  intent?: ChatIntent | null;
  suggestions?: ChatSuggestion[];
  diagnostics?: Record<string, unknown>;
  refused?: boolean;
  needsClarification?: boolean;
  elapsedMs?: number;
}

/* ---------------------------------------------------------------------------
 * Shared embedding store + model comparison
 * ------------------------------------------------------------------------- */

/** One embedding model the platform can index and search with. */
export interface EngineInfo {
  name: string;
  model: string | null;
  dimension: number | null;
  location: string | null;
  /** False when a remote endpoint is unreachable — it cannot be indexed with. */
  available: boolean;
  configured: boolean;
  /** Each model carries its own cosine floor, calibrated to equal selectivity. */
  min_similarity: number | null;
}

export interface IndexProgressInfo {
  model: string;
  state: 'idle' | 'queued' | 'running' | 'done' | 'error';
  done: number;
  total: number;
  embedded: number;
  reused: number;
  percent: number;
  elapsed_seconds: number;
  error: string | null;
}

/** How much of the shared chunk set one model actually holds. */
export interface ModelCoverage {
  model: string;
  indexed_resumes: number;
  indexed_chunks: number;
  dimension: number | null;
  in_sync: boolean;
  progress: IndexProgressInfo;
}

/**
 * One document set, several models indexing it.
 *
 * `comparable_resumes` is the intersection — the only population on which a
 * multi-model comparison is honest. `models_aligned` goes false the moment they
 * diverge, which is the condition that used to go unnoticed.
 */
export interface StoreCoverage {
  documents: { resumes: number; chunks: number };
  database_resumes: number | null;
  documents_in_sync: boolean;
  models: ModelCoverage[];
  comparable_resumes: number;
  models_aligned: boolean;
  engines: EngineInfo[];
}

export interface ComparisonCandidate {
  resume_id: number;
  name: string | null;
  title: string | null;
  match_percentage: number;
  total_years: number | null;
  matched_skills: string[];
  missing_skills: string[];
  top_evidence: string | null;
  section: string | null;
  similarity: number | null;
}

/** What one model made of the question. */
export interface ModelAnswer {
  model: string;
  dimension: number | null;
  available: boolean;
  answer: string;
  /** Only present when the LLM toggle is on. */
  llm_answer: string | null;
  answer_type: string;
  candidates: ComparisonCandidate[];
  elapsed_ms: number;
  indexed_resumes: number;
  error: string | null;
}

/**
 * How much the models agree.
 *
 * There is no labelled ground truth in a resume pool, so the honest measurement is
 * agreement rather than correctness: unanimity means the choice of model is not
 * buying anything for this question, and divergence is where a human should look.
 */
export interface ComparisonAgreement {
  top1_unanimous: boolean;
  top1_by_model: Record<string, string | null>;
  /** Jaccard overlap of returned resume sets, keyed "modelA|modelB". */
  overlap: Record<string, number>;
  common_resume_ids: number[];
}

export interface ComparisonResponse {
  question: string;
  models: ModelAnswer[];
  agreement: ComparisonAgreement;
  compared_over_resumes: number;
  fair_mode: boolean;
  warnings: string[];
}

/* -------------------------------------------------------------------------
 * Document management — the working set
 *
 * Uploading a resume and paying to embed it are separate acts, and at 300+ CVs the
 * difference is expensive. The WORKING SET is the subset currently parsed into the
 * shared document layer, and therefore the subset that can be embedded and searched.
 * It is deliberately narrower than "everything uploaded".
 * ------------------------------------------------------------------------- */

export interface ManagedResume {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
  /** False when the row survives but the file behind it does not. */
  file_present: boolean;
  /** Parsed and chunked, and therefore eligible for embedding. */
  in_working_set: boolean;
  chunks: number;
  /** Deterministic profile fields — available only once the resume has been parsed. */
  name: string | null;
  title: string | null;
  total_years: number | null;
  location: string | null;
  /** {model -> holds vectors for this resume}. A model can legitimately be behind. */
  models: Record<string, boolean>;
}

export interface ManagedResumeList {
  resumes: ManagedResume[];
  total: number;
  in_working_set: number;
  models: Record<string, number>;
}

/** A JD carries no index state: it is parsed at analysis time and never embedded. */
export interface ManagedJob {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
  file_present: boolean;
  analyses: number;
}

export interface ManagedJobList {
  jobs: ManagedJob[];
  total: number;
}

export interface WorkingSetResult {
  added: number;
  unchanged: number;
  removed: number;
  failed: string[];
  chunks: number;
  working_set: number;
  /** Models realigned after a removal — a file write, not an embedding run. */
  realigned: string[];
}

export interface DeleteResult {
  deleted: number;
  files_removed: number;
}


/** One section of a parsed resume, as the viewer renders it. */
export interface ResumeSection {
  section: string;
  page: number;
  text: string;
}

/**
 * A resume as the platform actually holds it.
 *
 * Served from the document layer rather than re-parsed, so what a recruiter reads is
 * exactly the text the search matched on.
 */
export interface ResumeContent {
  resume_id: number;
  filename: string | null;
  in_working_set: boolean;
  profile: Record<string, unknown>;
  sections: ResumeSection[];
  chunks: number;
}


/* ------------------------------------------------------------------ Prompt Lab --
 * Mirrors backend `schemas/prompt_lab.py` (Golden Rule 9: the backend owns the shape).
 *
 * A case carries what the turn PERMITS, not what the prompt says, which is what lets
 * the same suite grade a rewritten prompt without itself being rewritten.
 */

export interface CaseExpectations {
  max_lines: number | null;
  max_chars: number | null;
  link_allowed: boolean;
  link_required: boolean;
  years_known: boolean;
  expected_years: string | null;
  jobs_requested: boolean;
  list_allowed: boolean;
  skills_requested: boolean;
  must_contain: string[];
  must_not_contain: string[];
}

export interface PromptCase {
  id?: string;
  name?: string;
  question: string;
  context: string;
  history: { role: string; content: string }[];
  expectations: CaseExpectations;
}

export interface PromptVariant {
  label: string;
  prompt: string;
}

/** pass | fail | na — `na` means the rule had nothing to say about this turn. */
export type RuleStatus = 'pass' | 'fail' | 'na';

export interface RuleVerdict {
  rule: string;
  title: string;
  status: RuleStatus;
  reason: string;
}

export interface PromptCaseResult {
  case_id: string;
  name: string;
  question: string;
  answer: string;
  error: string | null;
  latency_ms: number;
  rules: RuleVerdict[];
  passed: number;
  failed: number;
  not_applicable: number;
  score: number | null;
  violations: string[];
}

export interface RuleBreakdown {
  rule: string;
  title: string;
  passed: number;
  failed: number;
  not_applicable: number;
  score: number | null;
}

export interface VariantResult {
  label: string;
  prompt_hash: string;
  cases: PromptCaseResult[];
  total_cases: number;
  clean_cases: number;
  passed: number;
  failed: number;
  score: number | null;
  by_rule: RuleBreakdown[];
  errors: string[];
}

export interface RuleDelta {
  rule: string;
  title: string;
  baseline_score: number | null;
  variant_score: number | null;
  delta: number | null;
}

export interface PromptRunResult {
  variants: VariantResult[];
  deltas: RuleDelta[];
  llm_available: boolean;
  model: string;
}

export interface PromptScoreResult {
  rules: RuleVerdict[];
  passed: number;
  failed: number;
  not_applicable: number;
  score: number | null;
  violations: string[];
}

export interface PromptRuleInfo {
  id: string;
  title: string;
  description: string;
}

export interface PromptSuite {
  id: number;
  name: string;
  description: string;
  prompt: string;
  cases: PromptCase[];
  created_at: string;
  updated_at: string;
}

export interface StarterSuite {
  name: string;
  prompt: string;
  cases: PromptCase[];
}

export interface PromptRunSummary {
  id: number;
  suite_id: number | null;
  label: string;
  prompt_hash: string;
  model: string;
  total_cases: number;
  clean_cases: number;
  passed: number;
  failed: number;
  score: number | null;
  created_at: string;
}

export interface PromptRunDetail extends PromptRunSummary {
  results: VariantResult;
}

// ---------------------------------------------------------------------------
// Company knowledge base (`/api/v1/company-chat`) — mirrors
// `backend/app/schemas/company_chat.py`.
//
// A separate, opt-in corpus of employer records held in Qdrant. Deliberately NOT
// merged with the candidate chat types: a company and a candidate share no fields,
// and one union type would force every branch of the UI to check which half it holds.
// ---------------------------------------------------------------------------

/** One retrieved field of one company — the audit trail behind a claim. */
export interface CompanyFieldMatch {
  field: string;
  label: string;
  text: string;
  score: number;
}

/** One company, reassembled from the field-level hits that matched. */
export interface CompanyResult {
  company_id: string;
  company_name: string;
  relevance: number;
  best_field: string;
  matches: CompanyFieldMatch[];
  /** The whole stored row, so any field can be shown without another request. */
  fields: Record<string, string>;
  industries: string[];
  people: Record<string, unknown>[];
}

export interface CompanyChatResponse {
  answer: string;
  companies: CompanyResult[];
  /** 'company' when the question named one, 'open' for a pool-wide search. */
  mode: string;
  /** 'llm' when a model phrased the answer. The facts are identical either way. */
  engine: string;
  refused: boolean;
  refusal_category: string | null;
  is_followup: boolean;
  stats: Record<string, unknown>;
  elapsed_ms: number;
}

export interface CompanyStoreStatus {
  configured: boolean;
  reachable: boolean;
  collection: string;
  points: number;
  companies: number;
  vector_size: number;
  detail: string;
}

/** One rendered turn of the company conversation. */
export interface CompanyTurn {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  companies?: CompanyResult[];
  mode?: string;
  engine?: string;
  refused?: boolean;
  isFollowup?: boolean;
  elapsedMs?: number;
}

// ---------------------------------------------------------------------------
// Web company intelligence — mirrors `company_intel/app/api/schemas.py`.
//
// A SEPARATE SERVICE, not part of this backend. It crawls company websites into its
// own Qdrant collections and answers from them; the browser reaches it directly at
// `VITE_COMPANY_INTEL_URL`. Kept apart from the `Company*` types above on purpose —
// those describe rows of a spreadsheet, these describe pages of a website, and the
// citation for each is a different thing (a stored field vs a live URL).
// ---------------------------------------------------------------------------

/** One passage retrieved from one crawled page — the audit trail behind a claim. */
export interface WebMatch {
  page_url: string;
  page_title: string;
  /** home · about · products · services · clients · careers · leadership · news · contact · other */
  page_type: string;
  /** The heading path the passage sat under, e.g. "Products > CRM". */
  section: string;
  text: string;
  score: number;
}

/** One company, reassembled from the page-level hits that matched. */
export interface WebCompanyResult {
  company_id: string;
  company_name: string;
  relevance: number;
  matches: WebMatch[];
}

/** A numbered source behind the answer. The answer text refers to these as [1], [2]. */
export interface WebCitation {
  n: number;
  company_id: string;
  company_name: string;
  page_url: string;
  page_title: string;
  page_type: string;
  section: string;
  score: number;
}

export interface WebChatResponse {
  answer: string;
  refused: boolean;
  /** 'company' when the question named one, 'pool' for a search across all, 'none' if refused. */
  scope: string;
  company: Record<string, unknown> | null;
  companies: WebCompanyResult[];
  citations: WebCitation[];
  /** Passages the answer was WRITTEN from. `companies` and `citations` match this. */
  evidence_count: number;
  /** Passages that cleared retrieval before the prompt budget trimmed them. */
  retrieved_count: number;
  intent: string;
  /** false when no LLM is configured — the answer is then quoted directly from sources. */
  llm_used: boolean;
  duration_seconds: number;
}

/** A registered company and where its content is crawled from. */
export interface WebCompany {
  company_id: string;
  name: string;
  domain: string;
  seed_urls: string[];
  /** Stored and shown as links only. This service never fetches LinkedIn. */
  linkedin_urls: string[];
  allow_patterns: string[];
  deny_patterns: string[];
  enabled: boolean;
  /** Fetch this company's pages through a headless browser. Opt-in; slow. */
  render: boolean;
  /**
   * Set by the crawler when most of a company's pages turned out to hold identical
   * content — the signature of a site that renders in the browser. Diagnosis, not
   * configuration: it explains why a corpus is thin.
   */
  client_rendered: boolean;
  created_at: number | null;
  updated_at: number | null;
}

/** What has actually been crawled for one company. */
export interface WebCompanyStatus {
  company: WebCompany;
  pages_known: number;
  pages_by_status: Record<string, number>;
  pages_by_type: Record<string, number>;
  chunks: number;
  last_crawled_at: number | null;
  next_due_at: number | null;
  failures: { url: string; error: string; fail_count: number }[];
}

/** Whether the crawler service is up, and what its store holds. */
export interface WebHealth {
  status: string;
  qdrant: {
    configured: boolean;
    reachable: boolean;
    collections: Record<string, number | null>;
    detail: string;
  };
  embedding_model: string;
  embedding_dim: number;
  /** 'fastembed' (local CPU) or 'ollama' (remote GPU endpoint). */
  embedding_provider: string;
  /** False when a remote embedding endpoint is asleep — crawling AND search both stop. */
  embedding_reachable: boolean;
  embedding_detail: string;
  /** 'none' | 'ollama' | 'openai'. 'none' means answers quote the sources verbatim. */
  llm_provider: string;
  llm_configured: boolean;
  companies: number;
  pages_due: number;
}

/** One rendered turn of the web conversation. */
export interface WebTurn {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  companies?: WebCompanyResult[];
  citations?: WebCitation[];
  scope?: string;
  refused?: boolean;
  llmUsed?: boolean;
  evidenceCount?: number;
  durationSeconds?: number;
}
