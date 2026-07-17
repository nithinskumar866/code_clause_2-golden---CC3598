import type { FC } from 'react';
import { Clock, FileText, Briefcase, Trash2 } from 'lucide-react';
import type { HistoryRecord } from '../../types';
import { getScoreColor, getScoreLabel, classifyFit, FIT_CATEGORY_STYLE } from '../analysis/scoreColors';

interface HistoryCardProps {
  record: HistoryRecord;
  onOpen: (record: HistoryRecord) => void;
  onDelete: (record: HistoryRecord) => void;
}

const formatTimestamp = (iso: string): string => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? 'Unknown date' : date.toLocaleString();
};

/** Summary card for a single stored analysis. Opens the full report on click. */
export const HistoryCard: FC<HistoryCardProps> = ({ record, onOpen, onDelete }) => {
  const category = classifyFit(record.overall_score);

  return (
    <div
      onClick={() => onOpen(record)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onOpen(record);
        }
      }}
      role="button"
      tabIndex={0}
      className="group flex cursor-pointer flex-col gap-4 rounded-xl border border-white/5 bg-card p-5 text-left transition hover:border-white/15 hover:bg-white/[0.02] focus:outline-none focus:ring-1 focus:ring-indigo-500"
    >
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-[11px] text-gray-500">
          <Clock className="h-3.5 w-3.5" /> {formatTimestamp(record.created_at)}
        </span>
        <span
          className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${FIT_CATEGORY_STYLE[category]}`}
        >
          {category}
        </span>
      </div>

      <div className="flex items-center gap-4">
        <div
          className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-full border-2 text-lg font-extrabold ${getScoreColor(record.overall_score)}`}
        >
          {record.overall_score}
        </div>
        <div className="min-w-0 space-y-1">
          <p className="flex items-center gap-1.5 text-sm font-medium text-white">
            <FileText className="h-3.5 w-3.5 shrink-0 text-indigo-400" />
            <span className="truncate">{record.resume_filename}</span>
          </p>
          <p className="flex items-center gap-1.5 text-xs text-gray-400">
            <Briefcase className="h-3.5 w-3.5 shrink-0 text-gray-500" />
            <span className="truncate">{record.jd_filename}</span>
          </p>
          <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">
            {getScoreLabel(record.overall_score)}
          </p>
        </div>
      </div>

      <div className="space-y-1">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-indigo-400">
          {record.recruiter_recommendation}
        </p>
        <p className="line-clamp-2 text-xs leading-relaxed text-gray-400">{record.summary}</p>
      </div>

      <div className="flex items-center justify-between border-t border-white/5 pt-3">
        <span className="text-[11px] font-semibold text-indigo-400 group-hover:text-indigo-300">
          View full report →
        </span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete(record);
          }}
          className="flex items-center gap-1 rounded-lg border border-white/10 px-2 py-1 text-[11px] font-medium text-gray-400 transition hover:border-rose-500/30 hover:bg-rose-500/10 hover:text-rose-400"
          aria-label={`Delete analysis for ${record.resume_filename}`}
        >
          <Trash2 className="h-3.5 w-3.5" /> Delete
        </button>
      </div>
    </div>
  );
};
