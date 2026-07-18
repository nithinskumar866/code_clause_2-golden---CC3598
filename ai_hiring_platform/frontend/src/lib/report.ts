import type { AnalysisReport } from '../types';
import { api } from '../api/client';

/** Trigger a client-side download of the report as a pretty-printed JSON file. */
export function downloadReportJSON(report: AnalysisReport): void {
  const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(report, null, 2))}`;
  const anchor = document.createElement('a');
  anchor.setAttribute('href', jsonString);
  anchor.setAttribute('download', `explainable_hiring_report_${report.analysis_id ?? 'report'}.json`);
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

/**
 * Download the server-rendered PDF for a stored analysis. Fetches the bytes as a
 * blob (so failures surface as errors rather than a broken tab) and triggers a
 * browser download. Throws on failure so the caller can toast it.
 */
export async function downloadReportPDF(analysisId: number): Promise<void> {
  const res = await api.get(`/analysis/${analysisId}/export/pdf`, { responseType: 'blob' });
  const url = URL.createObjectURL(res.data as Blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `report_analysis_${analysisId}.pdf`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
