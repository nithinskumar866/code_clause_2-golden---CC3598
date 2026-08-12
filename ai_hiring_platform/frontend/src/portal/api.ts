import type {
  ApiResponse, CreateJobRequest, Health, ImportResult, IndexStatus,
  JobDetail, JobFilters, JobSummary, JobUploadResult, MatchResult,
  ResumeProfile, ChatSession,
  Application, ApplicationPreview, ApplyPreviewRequest, ApplySubmitItem, ApplySubmitResult,
  JobSuggestionResult, DraftLetterResult,
} from './types';

/**
 * The .NET job-portal API.
 *
 * A second base URL, separate from `VITE_API_BASE` (the Python recruiter API):
 * the two backends are different processes on different ports and either one can
 * move without the other. Sharing a single base would make the job portal
 * unreachable the moment the Python API is deployed somewhere else.
 */
export const PORTAL_API_BASE: string =
  (import.meta.env.VITE_PORTAL_API_BASE as string | undefined)?.replace(/\/$/, '') ??
  'http://localhost:5290';

/**
 * Unwraps the envelope every portal endpoint returns.
 *
 * `success: false` carries a message written for the person reading it, so it is
 * thrown as-is rather than replaced with a generic failure string.
 */
async function unwrap<T>(response: Response): Promise<T> {
  let body: ApiResponse<T> | null = null;
  try {
    body = (await response.json()) as ApiResponse<T>;
  } catch {
    throw new Error(`The job portal API returned ${response.status} with no readable body.`);
  }

  if (!response.ok || !body.success || body.data === null) {
    throw new Error(body?.message || `Request failed with ${response.status}.`);
  }
  return body.data;
}

const json = { 'Content-Type': 'application/json' };

export const portalApi = {
  health: (): Promise<Health> =>
    fetch(`${PORTAL_API_BASE}/api/health`).then(unwrap<Health>),

  // -- jobs ---------------------------------------------------------------

  listJobs: (params: Record<string, string | number | undefined> = {}): Promise<JobSummary[]> => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== '') query.set(key, String(value));
    }
    return fetch(`${PORTAL_API_BASE}/api/jobs?${query}`).then(unwrap<JobSummary[]>);
  },

  getJob: (id: number): Promise<JobDetail> =>
    fetch(`${PORTAL_API_BASE}/api/jobs/${id}`).then(unwrap<JobDetail>),

  uploadJob: (file: File): Promise<JobUploadResult> => {
    const form = new FormData();
    form.append('file', file);
    // No Content-Type header: the browser must set the multipart boundary, and
    // setting it by hand produces a request the server cannot parse.
    return fetch(`${PORTAL_API_BASE}/api/jobs/upload`, { method: 'POST', body: form })
      .then(unwrap<JobUploadResult>);
  },

  createJob: (request: CreateJobRequest): Promise<JobDetail> =>
    fetch(`${PORTAL_API_BASE}/api/jobs`, {
      method: 'POST', headers: json, body: JSON.stringify(request),
    }).then(unwrap<JobDetail>),

  deleteJob: (id: number): Promise<string> =>
    fetch(`${PORTAL_API_BASE}/api/jobs/${id}`, { method: 'DELETE' }).then(unwrap<string>),

  importFromPlatform: (): Promise<ImportResult> =>
    fetch(`${PORTAL_API_BASE}/api/jobs/import-from-platform`, { method: 'POST' })
      .then(unwrap<ImportResult>),

  indexStatus: (): Promise<IndexStatus> =>
    fetch(`${PORTAL_API_BASE}/api/jobs/index-status`).then(unwrap<IndexStatus>),

  reindex: (force = false): Promise<IndexStatus> =>
    fetch(`${PORTAL_API_BASE}/api/jobs/reindex?force=${force}`, { method: 'POST' })
      .then(unwrap<IndexStatus>),

  // -- resumes and chat ---------------------------------------------------

  uploadResume: (file: File): Promise<ResumeProfile> => {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${PORTAL_API_BASE}/api/resumes/upload`, { method: 'POST', body: form })
      .then(unwrap<ResumeProfile>);
  },

  getSession: (sessionId: string): Promise<ChatSession> =>
    fetch(`${PORTAL_API_BASE}/api/chat/${encodeURIComponent(sessionId)}`).then(unwrap<ChatSession>),

  matches: (resumeId: number, filters: JobFilters, top?: number): Promise<MatchResult> =>
    fetch(`${PORTAL_API_BASE}/api/resumes/${resumeId}/matches${top ? `?top=${top}` : ''}`, {
      method: 'POST', headers: json, body: JSON.stringify(filters),
    }).then(unwrap<MatchResult>),
};

/**
 * Applying to postings on this board.
 *
 * Preview and submit are separate calls on purpose: preview does all the work
 * and stores nothing, so the candidate can see exactly what would be sent before
 * any of it is. Submitting is outward-facing and cannot be undone from here.
 */
export const applicationApi = {
  preview: (request: ApplyPreviewRequest): Promise<ApplicationPreview> =>
    fetch(`${PORTAL_API_BASE}/api/applications/preview`, {
      method: 'POST', headers: json, body: JSON.stringify(request),
    }).then(unwrap<ApplicationPreview>),

  submit: (resumeId: number, items: ApplySubmitItem[]): Promise<ApplySubmitResult> =>
    fetch(`${PORTAL_API_BASE}/api/applications`, {
      method: 'POST', headers: json, body: JSON.stringify({ resumeId, items }),
    }).then(unwrap<ApplySubmitResult>),

  list: (params: { jobId?: number; status?: string } = {}): Promise<Application[]> => {
    const query = new URLSearchParams();
    if (params.jobId !== undefined) query.set('jobId', String(params.jobId));
    if (params.status) query.set('status', params.status);
    return fetch(`${PORTAL_API_BASE}/api/applications?${query}`).then(unwrap<Application[]>);
  },

  setStatus: (id: number, status: string, analysisId?: number): Promise<Application> =>
    fetch(`${PORTAL_API_BASE}/api/applications/${id}/status`, {
      method: 'PATCH', headers: json, body: JSON.stringify({ status, analysisId }),
    }).then(unwrap<Application>),

  /**
   * Writes one letter for a role already under review.
   *
   * Separate from preview because a local model takes tens of seconds per letter:
   * the panel opens immediately on composed letters and upgrades them one at a time.
   */
  draftLetter: (resumeId: number, jobId: number): Promise<DraftLetterResult> =>
    fetch(`${PORTAL_API_BASE}/api/applications/draft-letter`, {
      method: 'POST', headers: json, body: JSON.stringify({ resumeId, jobId }),
    }).then(unwrap<DraftLetterResult>),

  /** The CV as uploaded. A URL rather than a fetch so the browser downloads it. */
  resumeUrl: (id: number): string => `${PORTAL_API_BASE}/api/applications/${id}/resume`,
};

/** Roles to open with, before a CV exists. */
export const suggestJobs = (query?: string, top = 4): Promise<JobSuggestionResult> => {
  const params = new URLSearchParams();
  if (query) params.set('query', query);
  params.set('top', String(top));
  return fetch(`${PORTAL_API_BASE}/api/jobs/suggestions?${params}`).then(unwrap<JobSuggestionResult>);
};

/**
 * Clears the conversation, keeping the CV attached.
 *
 * "New chat" drops the context, not the person — making someone re-upload their
 * resume to ask a fresh question would be worse and slower.
 */
export const resetConversation = (sessionId: string): Promise<ChatSession> =>
  fetch(`${PORTAL_API_BASE}/api/chat/${encodeURIComponent(sessionId)}/messages`, {
    method: 'DELETE',
  }).then(unwrap<ChatSession>);
