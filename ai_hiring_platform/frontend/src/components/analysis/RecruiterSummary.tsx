import type { FC } from 'react';
import { Award, FileText } from 'lucide-react';

interface RecruiterSummaryProps {
  recommendation: string;
  summary: string;
}

/** Recruiter recommendation chip + executive summary text. */
export const RecruiterSummary: FC<RecruiterSummaryProps> = ({ recommendation, summary }) => (
  <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
    <div className="md:col-span-1 rounded-xl border border-white/5 bg-card p-5 flex flex-col items-center justify-center text-center">
      <Award className="h-7 w-7 text-indigo-400 mb-2" />
      <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider mb-1">
        Recruiter recommendation
      </span>
      <span className="text-xs font-bold text-indigo-300 leading-tight">{recommendation}</span>
    </div>

    <div className="md:col-span-3 rounded-xl border border-white/5 bg-card p-5 space-y-2">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-400 uppercase tracking-wider">
        <FileText className="h-4 w-4" /> Recruiter summary
      </div>
      <p className="text-sm text-gray-300 leading-relaxed font-sans font-normal">{summary}</p>
    </div>
  </div>
);
