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
}

export interface LearningRoadmapItem {
  skill: string;
  estimated_time: string;
  reason: string;
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
