import type { FC } from 'react';
import { Download } from 'lucide-react';
import type { AnalysisReport } from '../../types';

export type ReportViewMode = 'report' | 'raw_rag';

interface ReportHeaderProps {
  report: AnalysisReport;
  viewMode: ReportViewMode;
  onChangeView: (mode: ReportViewMode) => void;
  onDownload: () => void;
}

/** Report title bar: identifiers, JSON export and the report / raw-RAG toggle. */
export const ReportHeader: FC<ReportHeaderProps> = ({ report, viewMode, onChangeView, onDownload }) => (
  <div className="flex items-center justify-between border-b border-white/5 pb-4">
    <div>
      <h2 className="text-lg font-medium text-white">Recruiter Assessment Profile</h2>
      <p className="text-xs text-gray-400 mt-1">
        Analysis ID: #{report.analysis_id} • Candidate Resume ID: #{report.resume_id} • Job ID: #{report.jd_id}
      </p>
    </div>

    <div className="flex items-center gap-3">
      <button
        onClick={onDownload}
        className="flex items-center gap-1 text-xs border border-white/10 rounded-lg px-2.5 py-1 text-gray-300 hover:text-white hover:bg-white/5 transition"
        title="Download Report JSON"
      >
        <Download className="h-3.5 w-3.5" /> JSON Report
      </button>

      <div className="flex rounded-lg border border-white/10 bg-black/20 p-0.5">
        <button
          onClick={() => onChangeView('report')}
          className={`px-3 py-1 text-xs font-semibold rounded-md transition ${
            viewMode === 'report' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          Hiring Report
        </button>
        <button
          onClick={() => onChangeView('raw_rag')}
          className={`px-3 py-1 text-xs font-semibold rounded-md transition ${
            viewMode === 'raw_rag' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          Raw RAG Matches
        </button>
      </div>
    </div>
  </div>
);
