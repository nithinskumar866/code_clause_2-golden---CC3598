import { useMemo, type FC } from 'react';
import { Layers, RefreshCw, Play } from 'lucide-react';
import type { FileRecord } from '../../types';
import { Banner } from '../ui/Banner';
import { SearchableSelect, type SearchableOption } from '../ui/SearchableSelect';

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

  // A native dropdown is fine for five files and unusable for six hundred, which is
  // what the pool actually holds — so both pickers are searchable. The id stays in the
  // hint rather than the label so two candidates with similar filenames are still
  // distinguishable at a glance.
  const toOptions = (files: FileRecord[]): SearchableOption[] =>
    files.map((f) => ({ value: String(f.id), label: f.filename, hint: `ID #${f.id}` }));

  const resumeOptions = useMemo(() => toOptions(resumes), [resumes]);
  const jdOptions = useMemo(() => toOptions(jds), [jds]);

  return (
    <div className="lg:col-span-1 rounded-xl border border-white/5 bg-card p-6 h-fit space-y-6">
      <div className="flex items-center gap-2 pb-2 border-b border-white/5">
        <Layers className="h-5 w-5 text-indigo-400" />
        <h2 className="text-base font-semibold text-white">Configure Analysis</h2>
      </div>

      {error && <Banner variant="error">{error}</Banner>}

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
          <SearchableSelect
            label="Candidate Resume"
            placeholder="Search candidates…"
            options={resumeOptions}
            value={selectedResume}
            onChange={onSelectResume}
            disabled={loadingLists || evaluating}
            emptyMessage="No resume matches that."
          />

          <SearchableSelect
            label="Job Description"
            placeholder="Search job descriptions…"
            options={jdOptions}
            value={selectedJd}
            onChange={onSelectJd}
            disabled={loadingLists || evaluating}
            emptyMessage="No job description matches that."
          />
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
