import type {
  ApiResponse, CreateJobRequest, Health, ImportResult, IndexStatus,
  JobDetail, JobFilters, JobSummary, JobUploadResult, MatchResult,
  ResumeProfile, ChatSession,
} from '../types';

/**
 * The .NET API base. Configured through Vite so a deployed build can point at a
 * different host without a code change.
 */
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ?? 'http://localhost:5290';

/**
 * Unwraps the envelope every endpoint returns.
 *
 * `success: false` carries a message written for the person reading it, so it is
 * thrown as-is rather than replaced with a generic failure string.
 */
async function unwrap<T>(response: Response): Promise<T> {
  let body: ApiResponse<T> | null = null;
  try {
    body = (await response.json()) as ApiResponse<T>;
  } catch {
    throw new Error(`The server returned ${response.status} with no readable body.`);
  }

  if (!response.ok || !body.success || body.data === null) {
    throw new Error(body?.message || `Request failed with ${response.status}.`);
  }
  return body.data;
}

const json = { 'Content-Type': 'application/json' };

export const api = {
  health: (): Promise<Health> =>
    fetch(`${API_BASE}/api/health`).then(unwrap<Health>),

  // -- jobs ---------------------------------------------------------------

  listJobs: (params: Record<string, string | number | undefined> = {}): Promise<JobSummary[]> => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== '') query.set(key, String(value));
    }
    return fetch(`${API_BASE}/api/jobs?${query}`).then(unwrap<JobSummary[]>);
  },

  getJob: (id: number): Promise<JobDetail> =>
    fetch(`${API_BASE}/api/jobs/${id}`).then(unwrap<JobDetail>),

  uploadJob: (file: File): Promise<JobUploadResult> => {
    const form = new FormData();
    form.append('file', file);
    // No Content-Type header: the browser must set the multipart boundary, and
    // setting it by hand produces a request the server cannot parse.
    return fetch(`${API_BASE}/api/jobs/upload`, { method: 'POST', body: form })
      .then(unwrap<JobUploadResult>);
  },

  createJob: (request: CreateJobRequest): Promise<JobDetail> =>
    fetch(`${API_BASE}/api/jobs`, {
      method: 'POST', headers: json, body: JSON.stringify(request),
    }).then(unwrap<JobDetail>),

  deleteJob: (id: number): Promise<string> =>
    fetch(`${API_BASE}/api/jobs/${id}`, { method: 'DELETE' }).then(unwrap<string>),

  importFromPlatform: (): Promise<ImportResult> =>
    fetch(`${API_BASE}/api/jobs/import-from-platform`, { method: 'POST' }).then(unwrap<ImportResult>),

  indexStatus: (): Promise<IndexStatus> =>
    fetch(`${API_BASE}/api/jobs/index-status`).then(unwrap<IndexStatus>),

  reindex: (force = false): Promise<IndexStatus> =>
    fetch(`${API_BASE}/api/jobs/reindex?force=${force}`, { method: 'POST' }).then(unwrap<IndexStatus>),

  // -- resumes and chat ---------------------------------------------------

  uploadResume: (file: File): Promise<ResumeProfile> => {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${API_BASE}/api/resumes/upload`, { method: 'POST', body: form })
      .then(unwrap<ResumeProfile>);
  },

  getSession: (sessionId: string): Promise<ChatSession> =>
    fetch(`${API_BASE}/api/chat/${encodeURIComponent(sessionId)}`).then(unwrap<ChatSession>),

  matches: (resumeId: number, filters: JobFilters, top?: number): Promise<MatchResult> =>
    fetch(`${API_BASE}/api/resumes/${resumeId}/matches${top ? `?top=${top}` : ''}`, {
      method: 'POST', headers: json, body: JSON.stringify(filters),
    }).then(unwrap<MatchResult>),
};
