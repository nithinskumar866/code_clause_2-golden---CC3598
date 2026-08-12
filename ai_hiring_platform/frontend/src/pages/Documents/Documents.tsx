import { useCallback, useEffect, useMemo, useState, type FC } from 'react';
import {
  FolderCog, Search, Trash2, DatabaseZap, DatabaseBackup, Loader2,
  CheckCircle2, AlertTriangle, FileText, Briefcase, MousePointerClick,
} from 'lucide-react';
import type { ManagedJob, ManagedResume } from '../../types';
import {
  deleteJobs, deleteResumes, listManagedJobs, listManagedResumes,
  workingSetAdd, workingSetRemove,
} from '../../api/documents';
import { getIndexProgress, startIndexing } from '../../api/embeddings';
import { useGallerySelection } from '../../hooks/useGallerySelection';
import { PageHeader } from '../../components/ui/PageHeader';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Spinner } from '../../components/ui/Spinner';

const MODELS = [
  { id: 'bge', label: 'BGE', hint: 'fast · always available' },
  { id: 'mxbai', label: 'mxbai', hint: 'slow · ~1.9 chunks/s' },
  { id: 'gpu', label: 'nomic', hint: 'your GPU endpoint' },
];

type Tab = 'resumes' | 'jobs';

const fmtDate = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleDateString();
};

/**
 * Documents — choose what is searchable, and remove what is not wanted.
 *
 * THE IDEA THIS SCREEN EXISTS TO EXPRESS
 * Uploading a resume and paying to embed it are different decisions. With 300+ CVs the
 * difference is roughly 40 minutes of CPU for the slow model, so the platform keeps a
 * WORKING SET — the resumes actually parsed and chunked — and this is where a recruiter
 * decides who is in it.
 *
 *     uploaded  ──select──▶  working set  ──index──▶  searchable by a model
 *
 * Selection uses the gestures people already have in their fingers from photo galleries
 * (see `useGallerySelection`): click, ctrl-click, shift-click and press-and-drag to
 * sweep. Picking 60 of 300 with individual checkboxes is 60 precise clicks, and nobody
 * does that twice.
 *
 * Two removals are offered, deliberately not merged: taking an expensive model off a
 * candidate must not mean losing the candidate.
 */
export const Documents: FC = () => {
  const [tab, setTab] = useState<Tab>('resumes');
  const [resumes, setResumes] = useState<ManagedResume[]>([]);
  const [jobs, setJobs] = useState<ManagedJob[]>([]);
  const [counts, setCounts] = useState<{ total: number; working: number; models: Record<string, number> }>(
    { total: 0, working: 0, models: {} },
  );
  const [query, setQuery] = useState('');
  const [onlyUnindexed, setOnlyUnindexed] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [indexModels, setIndexModels] = useState<string[]>(['bge']);
  const [progress, setProgress] = useState<Record<string, { state: string; percent: number }>>({});

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [r, j] = await Promise.all([listManagedResumes(), listManagedJobs()]);
      setResumes(r.resumes);
      setCounts({ total: r.total, working: r.in_working_set, models: r.models });
      setJobs(j.jobs);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load documents.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Poll only while something is embedding — the list changes as models catch up.
  useEffect(() => {
    const running = Object.values(progress).some((p) => p.state === 'running' || p.state === 'queued');
    if (!running) return undefined;
    const t = setInterval(() => {
      void getIndexProgress()
        .then((p) => {
          setProgress(p as never);
          if (!Object.values(p).some((x) => x.state === 'running' || x.state === 'queued')) {
            void load();
          }
        })
        .catch(() => undefined);
    }, 1500);
    return () => clearInterval(t);
  }, [progress, load]);

  const visibleResumes = useMemo(() => {
    const q = query.trim().toLowerCase();
    return resumes.filter((r) => {
      if (onlyUnindexed && r.in_working_set) return false;
      if (!q) return true;
      return (
        r.filename.toLowerCase().includes(q) ||
        (r.name || '').toLowerCase().includes(q) ||
        (r.location || '').toLowerCase().includes(q) ||
        (r.title || '').toLowerCase().includes(q)
      );
    });
  }, [resumes, query, onlyUnindexed]);

  const visibleJobs = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? jobs.filter((j) => j.filename.toLowerCase().includes(q)) : jobs;
  }, [jobs, query]);

  const ids = useMemo(
    () => (tab === 'resumes' ? visibleResumes.map((r) => r.id) : visibleJobs.map((j) => j.id)),
    [tab, visibleResumes, visibleJobs],
  );
  const sel = useGallerySelection(ids);

  const run = async (label: string, fn: () => Promise<string>) => {
    setBusy(label);
    setError(null);
    setNotice(null);
    try {
      setNotice(await fn());
      await load();
      sel.clear();
    } catch (e) {
      setError(e instanceof Error ? e.message : `${label} failed.`);
    } finally {
      setBusy(null);
    }
  };

  const selectedIds = () => Array.from(sel.selected);

  const onAdd = () =>
    run('add', async () => {
      const r = await workingSetAdd(selectedIds());
      // "0 resume(s) chunked" reads as a broken button. Chunking is idempotent, so a
      // selection that is already in the working set is a no-op — say that plainly and
      // point at the step that actually remains, instead of reporting a zero.
      if (r.added === 0) {
        return `Nothing to chunk — all ${selectedIds().length} selected resume(s) are already `
          + `in the working set. ${r.working_set} total. Use "Index" to embed them.`;
      }
      return `${r.added} resume(s) chunked. ${r.working_set} now in the working set — index them to make them searchable.`;
    });

  const onRemove = () =>
    run('remove', async () => {
      const r = await workingSetRemove(selectedIds());
      return `${r.removed} resume(s) removed from the index. Files kept — you can add them back any time.`;
    });

  const onIndex = () =>
    run('index', async () => {
      const queued = await startIndexing(indexModels, false);
      const p = await getIndexProgress();
      setProgress(p as never);
      return `Indexing started for ${queued.join(', ')}. This runs in the background.`;
    });

  const onDelete = () => {
    const n = sel.count;
    const what = tab === 'resumes' ? 'resume' : 'job description';
    if (!window.confirm(
      `Permanently delete ${n} ${what}${n === 1 ? '' : 's'}?\n\n` +
      `This removes the file, its database row and any vectors. It cannot be undone.\n\n` +
      (tab === 'resumes'
        ? 'To just make them non-searchable while keeping the files, use "Remove from index" instead.'
        : ''),
    )) return;
    void run('delete', async () => {
      const r = tab === 'resumes'
        ? await deleteResumes(selectedIds())
        : await deleteJobs(selectedIds());
      return `${r.deleted} ${what}(s) deleted permanently.`;
    });
  };

  const toggleModel = (id: string) =>
    setIndexModels((prev) => (prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]));

  return (
    <div className="space-y-5">
      <PageHeader
        title="Documents"
        description="Choose which resumes are chunked and embedded — and remove what you don't need."
        icon={<FolderCog className="h-5 w-5" />}
      />

      {/* The funnel, stated plainly: uploaded -> working set -> per model. */}
      <Card className="flex flex-wrap items-center gap-x-6 gap-y-2 p-3 text-xs text-gray-400">
        <span>
          <strong className="text-white">{counts.total}</strong> uploaded
        </span>
        <span className="text-gray-600">→</span>
        <span>
          <strong className="text-white">{counts.working}</strong> chunked (working set)
        </span>
        <span className="text-gray-600">→</span>
        {MODELS.map((m) => (
          <span key={m.id} className="inline-flex items-center gap-1.5">
            {m.label}
            <strong className="text-white">{counts.models[m.id] ?? 0}</strong>
            {progress[m.id]?.state === 'running' && (
              <Loader2 className="h-3 w-3 animate-spin text-indigo-400" />
            )}
          </span>
        ))}
      </Card>

      {/* Tabs + filter */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center rounded-lg border border-white/10 p-0.5" role="tablist">
          {([['resumes', 'Resumes', FileText], ['jobs', 'Job Descriptions', Briefcase]] as const).map(
            ([id, label, Icon]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={tab === id}
                onClick={() => { setTab(id); sel.clear(); }}
                className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                  tab === id ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ),
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {tab === 'resumes' && (
            <label className="inline-flex cursor-pointer items-center gap-1.5 text-[11px] text-gray-400">
              <input
                type="checkbox"
                checked={onlyUnindexed}
                onChange={(e) => setOnlyUnindexed(e.target.checked)}
                className="h-3 w-3 accent-indigo-500"
              />
              Only not-yet-chunked
            </label>
          )}
          <div className="flex items-center gap-1.5 rounded-lg border border-white/10 px-2 py-1">
            <Search className="h-3.5 w-3.5 text-gray-500" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by name, file, location…"
              className="w-52 bg-transparent text-xs text-white placeholder:text-gray-600 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* How to select — stated once, because the gestures are invisible otherwise. */}
      <p className="flex items-center gap-1.5 pl-0.5 text-[11px] text-gray-500">
        <MousePointerClick className="h-3 w-3" />
        Click to select · drag across to sweep · Shift-click for a range · Ctrl/Cmd-click to
        add one · Ctrl/Cmd-A for all · Esc to clear
      </p>

      {notice && (
        <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
          {notice}
        </div>
      )}
      {error && (
        <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* Action bar — appears only with a selection, so it never nags. */}
      {sel.count > 0 && (
        <Card className="sticky top-2 z-10 flex flex-wrap items-center gap-2 border-indigo-500/30 bg-indigo-500/[0.07] p-2.5">
          <span className="px-1 text-xs font-semibold text-white">{sel.count} selected</span>
          <Button variant="ghost" size="sm" onClick={sel.selectAll}>All ({ids.length})</Button>
          <Button variant="ghost" size="sm" onClick={sel.invert}>Invert</Button>
          <Button variant="ghost" size="sm" onClick={sel.clear}>Clear</Button>

          <span className="mx-1 h-4 w-px bg-white/10" />

          {tab === 'resumes' && (
            <>
              <Button
                size="sm"
                leftIcon={<DatabaseZap className="h-3.5 w-3.5" />}
                loading={busy === 'add'}
                onClick={onAdd}
              >
                Add to index
              </Button>
              <Button
                variant="secondary"
                size="sm"
                leftIcon={<DatabaseBackup className="h-3.5 w-3.5" />}
                loading={busy === 'remove'}
                onClick={onRemove}
              >
                Remove from index
              </Button>
            </>
          )}

          <Button
            variant="secondary"
            size="sm"
            leftIcon={<Trash2 className="h-3.5 w-3.5" />}
            loading={busy === 'delete'}
            onClick={onDelete}
            className="!border-rose-500/40 !text-rose-300 hover:!bg-rose-500/10"
          >
            Delete permanently
          </Button>
        </Card>
      )}

      {/* Which models to embed with. Separate from selection on purpose: you index the
          whole working set, and the expensive model is the one you ration. */}
      {tab === 'resumes' && (
        <Card className="flex flex-wrap items-center gap-2 p-2.5">
          <span className="px-1 text-[11px] text-gray-400">Embed the working set with:</span>
          {MODELS.map((m) => (
            <button
              key={m.id}
              type="button"
              title={m.hint}
              onClick={() => toggleModel(m.id)}
              className={`rounded-lg border px-2.5 py-1 text-[11px] transition ${
                indexModels.includes(m.id)
                  ? 'border-indigo-500/50 bg-indigo-500/15 text-indigo-200'
                  : 'border-white/10 text-gray-400 hover:text-white'
              }`}
            >
              {m.label} <span className="text-gray-600">· {m.hint}</span>
            </button>
          ))}
          <Button
            size="sm"
            loading={busy === 'index'}
            disabled={!indexModels.length}
            onClick={onIndex}
          >
            Index now
          </Button>
        </Card>
      )}

      {/* The list */}
      {loading ? (
        <div className="flex items-center gap-2 p-6 text-xs text-gray-400">
          <Spinner /> Loading documents…
        </div>
      ) : (
        <div
          className={`grid gap-1.5 ${sel.dragging ? 'select-none' : ''}`}
          role="listbox"
          aria-multiselectable="true"
        >
          {tab === 'resumes'
            ? visibleResumes.map((r, i) => (
                <ResumeRow key={r.id} resume={r} selected={sel.isSelected(r.id)} {...sel.itemProps(r.id, i)} />
              ))
            : visibleJobs.map((j, i) => (
                <JobRow key={j.id} job={j} selected={sel.isSelected(j.id)} {...sel.itemProps(j.id, i)} />
              ))}
          {!ids.length && (
            <p className="p-6 text-center text-xs text-gray-500">Nothing matches that filter.</p>
          )}
        </div>
      )}
    </div>
  );
};

interface RowProps {
  selected: boolean;
  onPointerDown: (e: React.PointerEvent) => void;
  onPointerEnter: () => void;
  onClick: (e: React.MouseEvent) => void;
}

const rowClass = (selected: boolean) =>
  `flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2 transition ${
    selected
      ? 'border-indigo-500/60 bg-indigo-500/15'
      : 'border-white/10 bg-white/[0.02] hover:border-white/20'
  }`;

const ResumeRow: FC<RowProps & { resume: ManagedResume }> = ({ resume: r, selected, ...handlers }) => (
  <div {...handlers} role="option" aria-selected={selected} className={rowClass(selected)}>
    <span
      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border ${
        selected ? 'border-indigo-400 bg-indigo-500' : 'border-white/25'
      }`}
    >
      {selected && <CheckCircle2 className="h-3 w-3 text-white" />}
    </span>

    <div className="min-w-0 flex-1">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <span className="truncate text-xs font-semibold text-white">
          {r.name || r.filename}
        </span>
        {r.title && <span className="truncate text-[11px] text-gray-400">· {r.title}</span>}
        {!r.file_present && (
          <Badge tone="danger">
            <AlertTriangle className="mr-1 inline h-3 w-3" />file missing
          </Badge>
        )}
      </div>
      <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-[10px] text-gray-500">
        <span className="truncate">{r.filename}</span>
        <span>{fmtDate(r.upload_time)}</span>
        {r.location && <span>{r.location}</span>}
        {r.total_years !== null && <span>{r.total_years} yrs</span>}
      </div>
    </div>

    <div className="flex shrink-0 items-center gap-1.5">
      {r.in_working_set ? (
        <Badge tone="info">{r.chunks} chunks</Badge>
      ) : (
        <Badge tone="warning">not chunked</Badge>
      )}
      {MODELS.map((m) => (
        <span
          key={m.id}
          title={`${m.label}: ${r.models[m.id] ? 'indexed' : 'not indexed'}`}
          className={`rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase ${
            r.models[m.id] ? 'bg-emerald-500/15 text-emerald-300' : 'bg-white/5 text-gray-600'
          }`}
        >
          {m.label}
        </span>
      ))}
    </div>
  </div>
);

const JobRow: FC<RowProps & { job: ManagedJob }> = ({ job: j, selected, ...handlers }) => (
  <div {...handlers} role="option" aria-selected={selected} className={rowClass(selected)}>
    <span
      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border ${
        selected ? 'border-indigo-400 bg-indigo-500' : 'border-white/25'
      }`}
    >
      {selected && <CheckCircle2 className="h-3 w-3 text-white" />}
    </span>
    <div className="min-w-0 flex-1">
      <span className="truncate text-xs font-semibold text-white">{j.filename}</span>
      <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-[10px] text-gray-500">
        <span>{fmtDate(j.upload_time)}</span>
        <span>{j.analyses} analysis(es)</span>
        {!j.file_present && <span className="text-rose-400">file missing</span>}
      </div>
    </div>
    {/* A JD is parsed for its requirements at analysis time and never embedded, so
        there is no working-set decision to show here. */}
    <Badge tone="neutral">not embedded by design</Badge>
  </div>
);
