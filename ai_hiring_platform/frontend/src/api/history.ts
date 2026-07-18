import axios from 'axios';
import { API_V1 } from './client';
import type { AnalysisReport, HistoryRecord, HistoryQuery, HistoryPage, HistoryPageMeta } from '../types';

/**
 * Analysis-history API client. Consumes only the stable backend contract:
 *   GET    /analysis/history       → list summaries
 *   GET    /analysis/history/{id}  → full report
 *   DELETE /analysis/history/{id}  → remove one
 *   DELETE /analysis/history       → clear all
 * The base URL is centralized here so it is the single place to adjust.
 */
const HISTORY_URL = `${API_V1}/analysis/history`;

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

/**
 * Search/filter/sort/paginate history. Every field of `query` maps directly
 * to a backend query parameter; params are passed via axios (never hand-built
 * query strings) and undefined fields are omitted. Returns the rows plus
 * pagination metadata from the response envelope's `meta` field.
 */
export async function fetchHistory(query: HistoryQuery = {}): Promise<HistoryPage> {
  const params: Record<string, string | number> = {};
  if (query.resume_filename) params.resume_filename = query.resume_filename;
  if (query.jd_filename) params.jd_filename = query.jd_filename;
  if (query.recommendation) params.recommendation = query.recommendation;
  if (query.min_score !== undefined) params.min_score = query.min_score;
  if (query.max_score !== undefined) params.max_score = query.max_score;
  if (query.date_from) params.date_from = query.date_from;
  if (query.date_to) params.date_to = query.date_to;
  if (query.sort) params.sort = query.sort;
  if (query.page !== undefined) params.page = query.page;
  if (query.page_size !== undefined) params.page_size = query.page_size;

  const res = await axios.get(HISTORY_URL, { params });
  if (res.data && res.data.success) {
    const items = (res.data.data as RawHistoryItem[]).map(toHistoryRecord);
    const raw = res.data.meta;
    // Tolerate an older backend that omits `meta` (returns all rows unpaged).
    const meta: HistoryPageMeta = raw
      ? {
          total_count: raw.total_count,
          page: raw.page,
          page_size: raw.page_size,
          total_pages: raw.total_pages,
        }
      : { total_count: items.length, page: 1, page_size: items.length || 1, total_pages: 1 };
    return { items, meta };
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
