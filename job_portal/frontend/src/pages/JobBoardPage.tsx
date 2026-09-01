import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';
import { usePortalState } from '../state/PortalState';
import type { JobSummary } from '../types';

/**
 * Browsing the board.
 *
 * Filters here are exact and deterministic — this is a catalogue, not a search.
 * Semantic ranking lives in the chat, where there is a resume to rank against;
 * ranking a browse by similarity to nothing in particular would just be an
 * arbitrary order presented as relevance.
 *
 * The filter state is NOT owned by this page. It lives in PortalState so the chat
 * can drive it: "show me remote roles over $120k" has to change this grid, not
 * just produce a sentence about it.
 */
export function JobBoardPage() {
  const { filters, setFilters, spotlight, clearSpotlight, lastCommand } = usePortalState();

  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setJobs(await api.listJobs({
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
   * When the bot has surfaced a shortlist, the grid shows exactly those roles in
   * exactly that order — its ranking, not the board's default recency order.
   * Re-sorting them here would quietly disagree with the list in the chat.
   */
  const visible = useMemo(() => {
    if (!spotlight?.length) return jobs;

    const rank = new Map(spotlight.map((id, index) => [id, index]));
    return jobs
      .filter(job => rank.has(job.id))
      .sort((a, b) => rank.get(a.id)! - rank.get(b.id)!);
  }, [jobs, spotlight]);

  const update = (patch: Partial<typeof filters>) => {
    // A manual change is the user overriding the bot, so its shortlist stops
    // applying — otherwise the grid would silently ignore what they just picked.
    clearSpotlight();
    setFilters({ ...filters, ...patch });
  };

  const importFromPlatform = async () => {
    setNotice(null);
    try {
      const result = await api.importFromPlatform();
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
      await api.deleteJob(id);
      await load();
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'The job could not be removed.');
    }
  };

  const driven = Boolean(spotlight?.length) || Boolean(lastCommand);

  return (
    <div className="page">
      <h1>Job board</h1>
      <p className="lede">{visible.length} open role(s).</p>

      {error && <div className="banner error">{error}</div>}
      {notice && <div className="banner">{notice}</div>}

      {/* The board must say when it is not showing everything. A silently
          filtered grid is indistinguishable from an empty job board. */}
      {driven && (
        <div className="banner" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span>
            <strong>Filtered by the assistant</strong>
            {lastCommand ? ` — ${lastCommand}` : ''}
            {spotlight?.length ? ` · showing its ${spotlight.length} match(es), best first` : ''}
          </span>
          <button style={{ marginLeft: 'auto', padding: '4px 10px', fontSize: 13 }}
                  onClick={() => { clearSpotlight(); setFilters({}); }}>
            Show all roles
          </button>
        </div>
      )}

      <div className="toolbar">
        <div className="field" style={{ flex: 1, minWidth: 220 }}>
          <label>Search</label>
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Title, company or skill" />
        </div>
        <div className="field">
          <label>Work mode</label>
          <select value={filters.workMode ?? ''} onChange={e => update({ workMode: e.target.value || null })}>
            <option value="">Any</option>
            <option>Remote</option><option>Hybrid</option><option>Onsite</option>
          </select>
        </div>
        <div className="field">
          <label>Seniority</label>
          <select value={filters.seniorityLevel ?? ''}
                  onChange={e => update({ seniorityLevel: e.target.value || null })}>
            <option value="">Any</option>
            <option>Junior</option><option>Mid</option><option>Senior</option>
            <option>Lead</option><option>Staff</option><option>Principal</option>
          </select>
        </div>
        <div className="field">
          <label>Min salary</label>
          <input type="number" min="0" step="10000" value={filters.minSalary ?? ''}
                 onChange={e => update({ minSalary: e.target.value ? Number(e.target.value) : null })} />
        </div>
        <button onClick={importFromPlatform}>Import from recruiter platform</button>
      </div>

      {loading && visible.length === 0 && <div className="empty">Loading…</div>}
      {!loading && visible.length === 0 && (
        <div className="empty">
          {driven
            ? 'Nothing on the board matches what the assistant filtered to.'
            : "No roles yet. Post one, or import the recruiter platform's job descriptions."}
        </div>
      )}

      <div className="job-grid">
        {visible.map(job => (
          <article className="card job-card" key={job.id}>
            <h3>{job.title}</h3>
            <div className="meta">
              {job.company && <span>{job.company}</span>}
              {job.location && <span>· {job.location}</span>}
              <span>· {job.workMode}</span>
              {job.employmentType !== 'Unspecified' && <span>· {job.employmentType}</span>}
            </div>
            {job.summary && <p className="summary">{job.summary}</p>}

            <div className="chips">
              {job.requiredSkills.slice(0, 8).map(skill => <span key={skill} className="chip">{skill}</span>)}
            </div>

            <div className="meta" style={{ marginTop: 'auto', paddingTop: 8 }}>
              {/* An unindexed job cannot be matched at all, so it is called out
                  rather than left looking identical to one that can. */}
              <span className={`dot ${job.isIndexed ? 'ok' : 'warn'}`} />
              <span>{job.isIndexed ? 'searchable' : 'not indexed'}</span>
              <button style={{ marginLeft: 'auto', padding: '3px 9px', fontSize: 12.5 }}
                      onClick={() => remove(job.id)}>Remove</button>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
