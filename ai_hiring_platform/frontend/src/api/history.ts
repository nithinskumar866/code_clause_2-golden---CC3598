import axios from 'axios';
import type { AnalysisReport, HistoryRecord } from '../types';

/**
 * Analysis-history API client. Consumes only the stable backend contract:
 *   GET    /analysis/history       → list summaries
 *   GET    /analysis/history/{id}  → full report
 *   DELETE /analysis/history/{id}  → remove one
 *   DELETE /analysis/history       → clear all
 * The base URL is centralized here so it is the single place to adjust.
 */
const HISTORY_URL = 'http://localhost:8000/api/v1/analysis/history';

/**
 * The raw list-item shape emitted by the backend `AnalysisHistoryItem` schema.
 * The frontend model (`HistoryRecord`) uses different field names, so this file
 * is the single adapter between them. Backend fields are the source of truth
 * (backend owns the contract); they are never renamed upstream.
 */
interface RawHistoryItem {
  analysis_id: number;
  timestamp: string;
  resume_id: number;
  job_description_id: number;
  resume_filename: string;
  jd_filename: string;
  overall_score: number;
  recruiter_recommendation: string;
  summary: string;
}

/** Adapt a backend history row to the frontend `HistoryRecord` model. */
function toHistoryRecord(raw: RawHistoryItem): HistoryRecord {
  return {
    id: raw.analysis_id,
    created_at: raw.timestamp,
    resume_id: raw.resume_id,
    jd_id: raw.job_description_id,
    resume_filename: raw.resume_filename,
    jd_filename: raw.jd_filename,
    overall_score: raw.overall_score,
    recruiter_recommendation: raw.recruiter_recommendation,
    summary: raw.summary,
  };
}

export async function fetchHistory(): Promise<HistoryRecord[]> {
  const res = await axios.get(HISTORY_URL);
  if (res.data && res.data.success) {
    return (res.data.data as RawHistoryItem[]).map(toHistoryRecord);
  }
  throw new Error(res.data?.message || 'Failed to load history');
}

export async function fetchHistoryReport(id: number): Promise<AnalysisReport> {
  const res = await axios.get(`${HISTORY_URL}/${id}`);
  if (res.data && res.data.success) {
    // The detail envelope carries identifiers at the top level and the full
    // hiring report under `report`. The stored report is a bare HiringReport,
    // so it omits analysis_id/resume_id/jd_id and retrieval_results. Lift the
    // real identifiers from the envelope and default absent raw matches to an
    // empty list (history does not persist raw RAG matches — do not fabricate).
    const data = res.data.data ?? {};
    const report = data.report ?? data ?? {};
    return {
      ...report,
      analysis_id: report.analysis_id ?? data.analysis_id ?? id,
      resume_id: report.resume_id ?? data.resume_id,
      jd_id: report.jd_id ?? data.job_description_id,
      retrieval_results: Array.isArray(report.retrieval_results) ? report.retrieval_results : [],
    } as AnalysisReport;
  }
  throw new Error(res.data?.message || 'Failed to load report');
}

export async function deleteHistoryItem(id: number): Promise<void> {
  const res = await axios.delete(`${HISTORY_URL}/${id}`);
  if (!(res.data && res.data.success)) {
    throw new Error(res.data?.message || 'Failed to delete analysis');
  }
}

export async function clearHistory(): Promise<void> {
  const res = await axios.delete(HISTORY_URL);
  if (!(res.data && res.data.success)) {
    throw new Error(res.data?.message || 'Failed to clear history');
  }
}
