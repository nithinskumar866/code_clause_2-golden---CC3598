/**
 * Mirror of the backend contract in
 * `backend/JobPortal.Api/Contracts/Contracts.cs`.
 *
 * The backend owns these shapes; this file follows. When the two disagree the
 * backend is right — change it there first, then mirror the change here.
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
}

// -- chat -------------------------------------------------------------------

/**
 * Copilot Mode. The backend sends these alongside the prose so the UI never has
 * to parse a sentence to decide what to do. An unrecognised action is ignored,
 * not guessed at.
 */
export interface UiAction {
  action: 'update_filters' | 'show_jobs' | 'resume_ready' | 'navigate' | string;
  payload: unknown;
  commandId?: string;
}

/**
 * Strictly-typed copilot commands (mirrors backend CopilotCommand schema).
 * Client validates these as defense in depth.
 */
export type CopilotCommand =
  | { type: 'update_filters'; payload: UpdateFiltersPayload; commandId: string }
  | { type: 'show_jobs'; payload: ShowJobsPayload; commandId: string }
  | { type: 'navigate'; payload: NavigatePayload; commandId: string }
  | { type: 'resume_ready'; payload: Record<string, never>; commandId: string };

export interface UpdateFiltersPayload {
  remote?: boolean;
  minSalary?: number;
  maxSalary?: number;
  location?: string;
  jobType?: string;
}

export interface ShowJobsPayload {
  jobIds: number[];
}

export interface NavigatePayload {
  route: string;
}

/**
 * Known routes for navigate action - must match backend allowlist.
 */
export const ALLOWED_ROUTES = [
  '/',
  '/?view=job-board',
  '/?view=post-job',
  '/?view=applications',
  '/?view=settings'
] as const;

export type AllowedRoute = typeof ALLOWED_ROUTES[number];

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
