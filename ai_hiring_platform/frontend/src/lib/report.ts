import type { AnalysisReport } from '../types';

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
