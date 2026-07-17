import { useState, type FC } from 'react';
import type { AnalysisReport } from '../../types';
import { ReportHeader } from './ReportHeader';
import type { ReportViewMode } from './ReportHeader';
import { HiringReportView } from './HiringReportView';
import { RawRagView } from './RawRagView';
import { downloadReportJSON } from '../../lib/report';

interface ReportViewerProps {
  report: AnalysisReport;
}

/**
 * Reusable report viewer: header (identifiers + JSON export + view toggle)
 * over either the recruiter hiring report or the raw RAG matches.
 * Shared by the Analysis page and the History report modal.
 */
export const ReportViewer: FC<ReportViewerProps> = ({ report }) => {
  const [viewMode, setViewMode] = useState<ReportViewMode>('report');

  return (
    <div className="space-y-6">
      <ReportHeader
        report={report}
        viewMode={viewMode}
        onChangeView={setViewMode}
        onDownload={() => downloadReportJSON(report)}
      />

      {viewMode === 'report' ? (
        <HiringReportView report={report} />
      ) : (
        <RawRagView results={report.retrieval_results} />
      )}
    </div>
  );
};
