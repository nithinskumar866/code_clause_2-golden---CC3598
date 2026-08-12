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
