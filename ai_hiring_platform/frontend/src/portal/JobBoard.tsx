import { useCallback, useEffect, useMemo, useState, type FC } from 'react';
import { Building2, Download, Trash2 } from 'lucide-react';
import { portalApi } from './api';
import { usePortalState } from './portal-context';
import type { JobSummary } from './types';
import { PageHeader } from '../components/ui/PageHeader';
import { Button } from '../components/ui/Button';
import { Banner } from '../components/ui/Banner';
import { EmptyState } from '../components/ui/EmptyState';
import { Select } from '../components/ui/Select';
import { Spinner } from '../components/ui/Spinner';

const WORK_MODES = ['', 'Remote', 'Hybrid', 'Onsite'];
const SENIORITIES = ['', 'Junior', 'Mid', 'Senior', 'Lead', 'Staff', 'Principal'];
const asOptions = (values: string[], anyLabel = 'Any') =>
  values.map(value => ({ value, label: value || anyLabel }));

/**
 * Browsing the open roles.
 *
 * Filters here are exact and deterministic — this is a catalogue, not a search.
 * Semantic ranking lives in the assistant, where there is a resume to rank
 * against; ranking a browse by similarity to nothing in particular would just be
 * an arbitrary order presented as relevance.
 *
 * The filter state is NOT owned by this page. It lives in PortalState so the
 * chat popup can drive it: "show me remote roles over $120k" has to change this
 * grid, not just produce a sentence about it.
 */
export const JobBoard: FC = () => {
  const {
    filters, setFilters, spotlight, clearSpotlight, lastCommand,
    engage, reachable, openChat,
  } = usePortalState();

  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Tells the provider the portal API is now in use, so it starts reporting on it.
  useEffect(engage, [engage]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setJobs(await portalApi.listJobs({
        search,
        workMode: filters.workMode ?? '',
        location: filters.location ?? '',
        seniority: filters.seniorityLevel ?? '',
        minSalary: filters.minSalary ?? undefined,
      }));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'The board could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, [search, filters]);

  // Debounced: typing in the search box should not fire a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => { void load(); }, 250);
    return () => clearTimeout(timer);
  }, [load]);

  /**
   * When the assistant has surfaced a shortlist, the grid shows exactly those
   * roles in exactly that order — its ranking, not the board's default recency
   * order. Re-sorting them here would quietly disagree with the list in the chat.
   */
  const visible = useMemo(() => {
    if (!spotlight?.length) return jobs;

    const rank = new Map(spotlight.map((id, index) => [id, index]));
    return jobs
      .filter(job => rank.has(job.id))
      .sort((a, b) => rank.get(a.id)! - rank.get(b.id)!);
  }, [jobs, spotlight]);

  const update = (patch: Partial<typeof filters>) => {
    // A manual change is the user overriding the assistant, so its shortlist
    // stops applying — otherwise the grid would ignore what they just picked.
    clearSpotlight();
    setFilters({ ...filters, ...patch });
  };

  const importFromPlatform = async () => {
    setNotice(null);
    try {
      const result = await portalApi.importFromPlatform();
      setNotice(
        `Imported ${result.imported}, skipped ${result.skipped}, failed ${result.failed}.` +
        (result.notes.length ? ` ${result.notes[0]}` : ''));
      await load();
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : 'Import failed.');
    }
  };

  const remove = async (id: number) => {
    try {
      await portalApi.deleteJob(id);
      await load();
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'The job could not be removed.');
    }
  };

  const driven = Boolean(spotlight?.length) || Boolean(lastCommand);

  return (
    <div className="space-y-6">
      <PageHeader
        icon={<Building2 className="h-5 w-5" />}
        title="Job Board"
        description="Open roles on the candidate-facing portal. Ask the assistant to filter these for you, or browse them here."
        actions={
          <Button variant="secondary" leftIcon={<Download className="h-4 w-4" />} onClick={importFromPlatform}>
            Import platform JDs
          </Button>
        }
      />

      {reachable === false && (
        <Banner variant="error" title="Job portal API unreachable">
          Start it with <code>dotnet run</code> in <code>job_portal/backend/JobPortal.Api</code>.
        </Banner>
      )}
      {error && <Banner variant="error" onDismiss={() => setError(null)}>{error}</Banner>}
      {notice && <Banner variant="success" onDismiss={() => setNotice(null)}>{notice}</Banner>}

      {/* The board must say when it is not showing everything. A silently
          filtered grid is indistinguishable from an empty job board. */}
      {driven && (
        <Banner variant="info" title="Filtered by the assistant">
          <div className="flex flex-wrap items-center gap-3">
            <span>
              {lastCommand ?? 'Showing its shortlist'}
              {spotlight?.length ? ` · its ${spotlight.length} match(es), best first` : ''}
            </span>
            <button
              type="button"
              onClick={() => { clearSpotlight(); setFilters({}); }}
              className="ml-auto rounded-md border border-white/10 px-2 py-1 text-[11px] font-semibold text-white transition hover:bg-white/10"
            >
              Show all roles
            </button>
          </div>
        </Banner>
      )}

      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-white/5 bg-card p-4">
        <label className="min-w-[220px] flex-1 block">
          <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-gray-400">
            Search
          </span>
          <input
            value={search}
            onChange={event => setSearch(event.target.value)}
            placeholder="Title, company or skill"
            className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white placeholder:text-gray-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </label>

        <Select
          label="Work mode"
          options={asOptions(WORK_MODES)}
          value={filters.workMode ?? ''}
          onChange={event => update({ workMode: event.target.value || null })}
        />
        <Select
          label="Seniority"
          options={asOptions(SENIORITIES)}
          value={filters.seniorityLevel ?? ''}
          onChange={event => update({ seniorityLevel: event.target.value || null })}
        />
        <label className="block">
          <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-gray-400">
            Min salary
          </span>
          <input
            type="number"
            min="0"
            step="10000"
            value={filters.minSalary ?? ''}
            onChange={event => update({ minSalary: event.target.value ? Number(event.target.value) : null })}
            className="w-32 rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </label>
      </div>

      {loading && visible.length === 0 && (
        <div className="flex justify-center py-16"><Spinner className="h-6 w-6" /></div>
      )}

      {!loading && visible.length === 0 && (
        <EmptyState
          icon={<Building2 className="h-8 w-8" />}
          title={driven ? 'Nothing matches that filter' : 'No roles posted yet'}
          description={
            driven
              ? 'Nothing on the board matches what the assistant filtered to.'
              : "Post a role, or import the recruiter platform's job descriptions."
          }
          action={
            driven
              ? <Button variant="secondary" onClick={() => { clearSpotlight(); setFilters({}); }}>Show all roles</Button>
              : <Button onClick={openChat}>Ask the assistant</Button>
          }
        />
      )}

      {visible.length > 0 && (
        <>
          <p className="text-sm text-gray-400">{visible.length} open role(s).</p>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {visible.map(job => (
              <article key={job.id} className="flex flex-col gap-3 rounded-xl border border-white/5 bg-card p-4">
                <h3 className="text-base font-semibold text-white">{job.title}</h3>
                <p className="text-xs text-gray-400">
                  {[
                    job.company,
                    job.location,
                    job.workMode,
                    job.employmentType !== 'Unspecified' ? job.employmentType : null,
                  ].filter(Boolean).join(' · ')}
                </p>

                {job.summary && (
                  <p className="line-clamp-3 text-sm leading-relaxed text-gray-400">{job.summary}</p>
                )}

                <div className="flex flex-wrap gap-1.5">
                  {job.requiredSkills.slice(0, 8).map(skill => (
                    <span
                      key={skill}
                      className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[10px] font-medium text-gray-300"
                    >
                      {skill}
                    </span>
                  ))}
                </div>

                <div className="mt-auto flex items-center gap-2 border-t border-white/5 pt-3">
                  {/* An unindexed job cannot be matched at all, so it is called
                      out rather than left looking identical to one that can. */}
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${job.isIndexed ? 'bg-emerald-400' : 'bg-amber-400'}`}
                    aria-hidden="true"
                  />
                  <span className="text-[11px] text-gray-500">
                    {job.isIndexed ? 'searchable' : 'not indexed'}
                  </span>
                  <button
                    type="button"
                    onClick={() => remove(job.id)}
                    aria-label={`Remove ${job.title}`}
                    className="ml-auto rounded-md p-1.5 text-gray-500 transition hover:bg-rose-500/10 hover:text-rose-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </article>
            ))}
          </div>
        </>
      )}
    </div>
  );
};
