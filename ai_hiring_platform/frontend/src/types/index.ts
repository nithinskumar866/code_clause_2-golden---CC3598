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

export interface AnalysisReport {
  analysis_id: number;
  candidate_id: number;
  resume_id: number;
  jd_id: number;
  retrieval_results: RetrievalResult[];

  overall_score: number;
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
