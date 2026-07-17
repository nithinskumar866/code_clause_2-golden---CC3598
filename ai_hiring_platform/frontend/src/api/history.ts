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

export async function fetchHistory(): Promise<HistoryRecord[]> {
  const res = await axios.get(HISTORY_URL);
  if (res.data && res.data.success) {
    return res.data.data as HistoryRecord[];
  }
  throw new Error(res.data?.message || 'Failed to load history');
}

export async function fetchHistoryReport(id: number): Promise<AnalysisReport> {
  const res = await axios.get(`${HISTORY_URL}/${id}`);
  if (res.data && res.data.success) {
    const data = res.data.data;
    // Tolerate either { report } (as the evaluate endpoint returns) or a bare report.
    return (data?.report ?? data) as AnalysisReport;
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
