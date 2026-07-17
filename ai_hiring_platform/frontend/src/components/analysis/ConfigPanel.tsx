import type { FC } from 'react';
import { Layers, AlertCircle, RefreshCw, Play } from 'lucide-react';
import type { FileRecord } from '../../types';

interface ConfigPanelProps {
  resumes: FileRecord[];
  jds: FileRecord[];
  selectedResume: string;
  selectedJd: string;
  onSelectResume: (id: string) => void;
  onSelectJd: (id: string) => void;
  loadingLists: boolean;
  evaluating: boolean;
  error: string | null;
  onEvaluate: () => void;
  onRefresh: () => void;
}

/** Left-hand panel: pick a resume + JD and trigger the evaluation workflow. */
export const ConfigPanel: FC<ConfigPanelProps> = ({
  resumes,
  jds,
  selectedResume,
  selectedJd,
  onSelectResume,
  onSelectJd,
  loadingLists,
  evaluating,
  error,
  onEvaluate,
  onRefresh,
}) => {
  const listsEmpty = resumes.length === 0 && jds.length === 0;
  const showSkeleton = loadingLists && listsEmpty;

  return (
    <div className="lg:col-span-1 rounded-xl border border-white/5 bg-card p-6 h-fit space-y-6">
      <div className="flex items-center gap-2 pb-2 border-b border-white/5">
        <Layers className="h-5 w-5 text-indigo-400" />
        <h2 className="text-base font-semibold text-white">Configure Analysis</h2>
      </div>

      {error && (
        <div className="flex items-start gap-2.5 rounded-lg border border-rose-500/20 bg-rose-500/10 p-3 text-xs text-rose-400">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {showSkeleton ? (
        <div className="space-y-4">
          {[0, 1].map((i) => (
            <div key={i} className="space-y-2">
              <div className="h-3 w-28 rounded bg-white/10 animate-pulse" />
              <div className="h-9 w-full rounded-lg bg-white/5 animate-pulse" />
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {/* Candidate Resume Selector */}
          <div>
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Candidate Resume
            </label>
            <select
              value={selectedResume}
              onChange={(e) => onSelectResume(e.target.value)}
              disabled={loadingLists || evaluating}
              className="w-full rounded-lg border border-white/10 bg-black/40 px-3.5 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
            >
              <option value="">-- Choose Candidate --</option>
              {resumes.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.filename} (ID: #{r.id})
                </option>
              ))}
            </select>
          </div>

          {/* Job Description Selector */}
          <div>
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Job Description
            </label>
            <select
              value={selectedJd}
              onChange={(e) => onSelectJd(e.target.value)}
              disabled={loadingLists || evaluating}
              className="w-full rounded-lg border border-white/10 bg-black/40 px-3.5 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
            >
              <option value="">-- Choose Job Description --</option>
              {jds.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.filename} (ID: #{j.id})
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      <div className="pt-4 border-t border-white/5 flex gap-2">
        <button
          onClick={onEvaluate}
          disabled={evaluating || !selectedResume || !selectedJd}
          className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition shadow-lg shadow-indigo-600/10"
        >
          {evaluating ? (
            <>
              <RefreshCw className="h-4 w-4 animate-spin" />
              Running LangGraph Workflow...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 fill-current" />
              Evaluate Candidate Fit
            </>
          )}
        </button>

        <button
          onClick={onRefresh}
          disabled={loadingLists || evaluating}
          className="p-2.5 rounded-lg border border-white/10 text-gray-400 hover:text-white hover:bg-white/5 transition"
          title="Refresh Files List"
        >
          <RefreshCw className={`h-4 w-4 ${loadingLists ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </div>
  );
};
