import { api } from './client';
import type {
  DeleteResult,
  ResumeContent,
  ManagedJobList,
  ManagedResumeList,
  WorkingSetResult,
} from '../types';

/**
 * Document management — what is uploaded, what is searchable, and what to remove.
 *
 * The distinction that matters here: uploading a resume and paying to embed it are
 * separate acts. `workingSetAdd` chunks a selection so it *can* be embedded;
 * `startIndexing` (in `api/embeddings`) is what actually embeds it.
 */

/** Every uploaded resume with its working-set and per-model indexing state. */
export async function listManagedResumes(): Promise<ManagedResumeList> {
  const res = await api.get('/documents/resumes');
  if (res.data?.success) return res.data.data as ManagedResumeList;
  throw new Error(res.data?.message || 'Could not load resumes');
}

/** Every uploaded job description. JDs are never chunked, so they carry no index state. */
export async function listManagedJobs(): Promise<ManagedJobList> {
  const res = await api.get('/documents/jobs');
  if (res.data?.success) return res.data.data as ManagedJobList;
  throw new Error(res.data?.message || 'Could not load job descriptions');
}

/** Parse and chunk the selected resumes so they become eligible for embedding. */
export async function workingSetAdd(ids: number[]): Promise<WorkingSetResult> {
  const res = await api.post('/documents/resumes/working-set/add', { ids });
  if (res.data?.success) return res.data.data as WorkingSetResult;
  throw new Error(res.data?.message || 'Could not add to the index');
}

/**
 * Drop the selected resumes' chunks and vectors. Files and database rows survive, so
 * this is reversible — re-adding costs one parse and no embedding of anything else.
 */
export async function workingSetRemove(ids: number[]): Promise<WorkingSetResult> {
  const res = await api.post('/documents/resumes/working-set/remove', { ids });
  if (res.data?.success) return res.data.data as WorkingSetResult;
  throw new Error(res.data?.message || 'Could not remove from the index');
}

/** Permanently delete resumes: vectors, chunks, database row and file. Not reversible. */
export async function deleteResumes(ids: number[]): Promise<DeleteResult> {
  const res = await api.post('/documents/resumes/delete', { ids });
  if (res.data?.success) return res.data.data as DeleteResult;
  throw new Error(res.data?.message || 'Could not delete resumes');
}

/** Permanently delete job descriptions. */
export async function deleteJobs(ids: number[]): Promise<DeleteResult> {
  const res = await api.post('/documents/jobs/delete', { ids });
  if (res.data?.success) return res.data.data as DeleteResult;
  throw new Error(res.data?.message || 'Could not delete job descriptions');
}


/** The parsed resume, section by section — what the viewer renders. */
export async function getResumeContent(resumeId: number): Promise<ResumeContent> {
  const res = await api.get(`/documents/resumes/${resumeId}/content`);
  if (res.data?.success) return res.data.data as ResumeContent;
  throw new Error(res.data?.message || 'Could not load the resume');
}

/** URL of the ORIGINAL uploaded PDF/DOCX, for download or opening in a new tab. */
export function resumeFileUrl(resumeId: number): string {
  return `${api.defaults.baseURL || ''}/documents/resumes/${resumeId}/file`;
}
