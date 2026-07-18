import { useEffect, useMemo, useState, type FC } from 'react';
import { Trophy, Users, Download, RefreshCw, Play, AlertTriangle, CheckSquare, Square } from 'lucide-react';
import { api } from '../../api/client';
import { rankCandidates } from '../../api/ranking';
import type { FileRecord, RankingEntry, RankingResponse } from '../../types';
import { PageHeader } from '../../components/ui/PageHeader';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Select } from '../../components/ui/Select';
import { Badge } from '../../components/ui/Badge';
import type { BadgeTone } from '../../components/ui/Badge';
import { EmptyState } from '../../components/ui/EmptyState';
import { ErrorState } from '../../components/ui/ErrorState';
import { Skeleton } from '../../components/common/Skeleton';
import { getScoreColor } from '../../components/analysis/scoreColors';
import { useToast } from '../../components/ui/toast-context';

type SortKey = 'rank' | 'overall_score' | 'coverage_score' | 'quality_score' | 'resume_filename';

const RECO_TONE = (reco: string): BadgeTone =>
  /highly|proceed to technical|selected/i.test(reco)
    ? 'success'
    : /conditional|interview|borderline/i.test(reco)
      ? 'warning'
      : 'danger';

const FIT_TONE = (fit: string | null): BadgeTone =>
  fit === 'Exceeds' ? 'info' : fit === 'Meets' ? 'success' : fit === 'Below' ? 'danger' : 'neutral';

const csvEscape = (v: unknown): string => {
  const s = String(v ?? '');
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

function exportCsv(result: RankingResponse) {
  const cols: Array<[string, (e: RankingEntry) => unknown]> = [
    ['rank', (e) => e.rank],
    ['resume', (e) => e.resume_filename],
    ['overall', (e) => e.overall_score],
    ['coverage', (e) => e.coverage_score],
    ['experience', (e) => e.experience_score],
    ['project', (e) => e.project_score],
    ['quality', (e) => e.quality_score],
    ['confidence', (e) => e.confidence_score],
    ['seniority_fit', (e) => e.seniority_fit ?? ''],
    ['credibility', (e) => e.credibility_score ?? ''],
    ['stuffing_risk', (e) => e.keyword_stuffing_risk ?? ''],
    ['matched', (e) => e.matched_count],
    ['missing', (e) => e.missing_count],
    ['recommendation', (e) => e.recruiter_recommendation],
    ['error', (e) => e.error ?? ''],
  ];
  const header = cols.map(([h]) => h).join(',');
  const rows = result.entries.map((e) => cols.map(([, f]) => csvEscape(f(e))).join(','));
  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ranking_jd_${result.jd_id}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export const Ranking: FC = () => {
  const [resumes, setResumes] = useState<FileRecord[]>([]);
  const [jds, setJds] = useState<FileRecord[]>([]);
  const [selectedJd, setSelectedJd] = useState<string>('');
  const [selectedResumes, setSelectedResumes] = useState<Set<number>>(new Set());

  const [loadingLists, setLoadingLists] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [ranking, setRanking] = useState(false);
  const [result, setResult] = useState<RankingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [sortKey, setSortKey] = useState<SortKey>('rank');
  const [minScore, setMinScore] = useState(0);
  const [recoFilter, setRecoFilter] = useState<string>('all');

  const toast = useToast();

  const fetchLists = async () => {
    setLoadingLists(true);
    setListError(null);
    try {
      const [r, j] = await Promise.all([api.get('/resume'), api.get('/job')]);
      if (r.data?.success) setResumes(r.data.data);
      if (j.data?.success) setJds(j.data.data);
    } catch {
      setListError('Failed to load resumes and job descriptions. Ensure the backend is running.');
    } finally {
      setLoadingLists(false);
    }
  };

  useEffect(() => {
    fetchLists();
  }, []);

  const toggleResume = (id: number) =>
    setSelectedResumes((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const allSelected = resumes.length > 0 && selectedResumes.size === resumes.length;
  const toggleAll = () =>
    setSelectedResumes(allSelected ? new Set() : new Set(resumes.map((r) => r.id)));

  const runRanking = async () => {
    if (!selectedJd || selectedResumes.size === 0) return;
    setRanking(true);
    setError(null);
    setResult(null);
    try {
      const data = await rankCandidates(Number(selectedJd), [...selectedResumes]);
      setResult(data);
      toast.success(
        'Ranking complete',
        data.top_candidate
          ? `Top: ${data.top_candidate.resume_filename} (${data.top_candidate.overall_score}%)`
          : 'No candidate could be evaluated.',
      );
    } catch (err: any) {
      const msg = err?.response?.data?.message || err?.message || 'Ranking failed';
      setError(msg);
      toast.error(msg);
    } finally {
      setRanking(false);
    }
  };

  const visibleEntries = useMemo(() => {
    if (!result) return [];
    let list = [...result.entries];
    if (minScore > 0) list = list.filter((e) => e.overall_score >= minScore);
    if (recoFilter !== 'all') list = list.filter((e) => RECO_TONE(e.recruiter_recommendation) === recoFilter);
    list.sort((a, b) => {
      if (sortKey === 'resume_filename') return a.resume_filename.localeCompare(b.resume_filename);
      if (sortKey === 'rank') return a.rank - b.rank;
      return (b[sortKey] as number) - (a[sortKey] as number);
    });
    return list;
  }, [result, minScore, recoFilter, sortKey]);

  return (
    <div className="space-y-8 animate-fadeIn">
      <PageHeader
        title="Candidate Ranking"
        description="Score many resumes against one job description and compare them on a single leaderboard."
      />

      {/* Configuration */}
      <Card className="p-6 space-y-5">
        {listError ? (
          <ErrorState message={listError} onRetry={fetchLists} />
        ) : (
          <>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <Select
                label="Job description"
                value={selectedJd}
                onChange={(e) => setSelectedJd(e.target.value)}
                disabled={loadingLists || jds.length === 0}
                className="w-full sm:w-80"
                options={[
                  { value: '', label: jds.length ? 'Select a job description…' : 'No job descriptions uploaded' },
                  ...jds.map((j) => ({ value: String(j.id), label: `#${j.id} · ${j.filename}` })),
                ]}
              />
              <div className="flex items-center gap-2">
                <Button variant="secondary" size="sm" leftIcon={<RefreshCw className="h-4 w-4" />} onClick={fetchLists}>
                  Refresh
                </Button>
                <Button
                  leftIcon={<Play className="h-4 w-4" />}
                  loading={ranking}
                  disabled={!selectedJd || selectedResumes.size === 0}
                  onClick={runRanking}
                >
                  Rank {selectedResumes.size > 0 ? `${selectedResumes.size} ` : ''}candidate
                  {selectedResumes.size === 1 ? '' : 's'}
                </Button>
              </div>
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                  Resumes ({selectedResumes.size} selected)
                </span>
                {resumes.length > 0 && (
                  <button
                    onClick={toggleAll}
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-indigo-300 hover:text-indigo-200"
                  >
                    {allSelected ? <CheckSquare className="h-3.5 w-3.5" /> : <Square className="h-3.5 w-3.5" />}
                    {allSelected ? 'Clear all' : 'Select all'}
                  </button>
                )}
              </div>
              {loadingLists ? (
                <Skeleton className="h-24 w-full" />
              ) : resumes.length === 0 ? (
                <p className="text-xs italic text-gray-500">No resumes uploaded yet.</p>
              ) : (
                <div className="grid max-h-56 grid-cols-1 gap-1.5 overflow-y-auto sm:grid-cols-2 lg:grid-cols-3">
                  {resumes.map((r) => {
                    const checked = selectedResumes.has(r.id);
                    return (
                      <button
                        key={r.id}
                        onClick={() => toggleResume(r.id)}
                        className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs transition ${
                          checked
                            ? 'border-indigo-500/40 bg-indigo-500/10 text-white'
                            : 'border-white/10 bg-black/20 text-gray-300 hover:bg-white/5'
                        }`}
                      >
                        {checked ? (
                          <CheckSquare className="h-4 w-4 shrink-0 text-indigo-400" />
                        ) : (
                          <Square className="h-4 w-4 shrink-0 text-gray-500" />
                        )}
                        <span className="truncate">{r.filename}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </>
        )}
      </Card>

      {/* Results */}
      {error ? (
        <ErrorState title="Ranking failed" message={error} onRetry={runRanking} />
      ) : ranking ? (
        <div className="space-y-4">
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : result ? (
        result.entries.length === 0 ? (
          <EmptyState icon={<Users className="h-9 w-9" />} title="No candidates ranked" description="Select resumes and run again." />
        ) : (
          <div className="space-y-6">
            {/* Top candidate */}
            {result.top_candidate && (
              <Card className="p-6">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-4">
                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-full border border-amber-400/40 bg-amber-400/10 text-amber-300">
                      <Trophy className="h-7 w-7" />
                    </div>
                    <div className="min-w-0">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-amber-400">
                        Top candidate for {result.jd_filename}
                      </span>
                      <h3 className="truncate text-lg font-bold text-white">{result.top_candidate.resume_filename}</h3>
                      <p className="text-xs text-gray-400">{result.top_candidate.recruiter_recommendation}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    <div className="text-center">
                      <div className={`rounded-lg border px-3 py-1.5 text-2xl font-bold ${getScoreColor(result.top_candidate.overall_score)}`}>
                        {result.top_candidate.overall_score}%
                      </div>
                      <span className="mt-1 block text-[10px] uppercase tracking-wider text-gray-500">Overall</span>
                    </div>
                    <div className="text-xs text-gray-400">
                      {result.evaluated_count}/{result.candidate_count} evaluated
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* Controls */}
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div className="flex flex-wrap items-end gap-3">
                <Select
                  label="Sort by"
                  srLabel
                  value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SortKey)}
                  options={[
                    { value: 'rank', label: 'Rank' },
                    { value: 'overall_score', label: 'Overall score' },
                    { value: 'coverage_score', label: 'Coverage' },
                    { value: 'quality_score', label: 'Quality' },
                    { value: 'resume_filename', label: 'Filename' },
                  ]}
                />
                <Select
                  label="Recommendation"
                  srLabel
                  value={recoFilter}
                  onChange={(e) => setRecoFilter(e.target.value)}
                  options={[
                    { value: 'all', label: 'All recommendations' },
                    { value: 'success', label: 'Strong' },
                    { value: 'warning', label: 'Conditional' },
                    { value: 'danger', label: 'Weak' },
                  ]}
                />
                <label className="block">
                  <span className="sr-only">Minimum score</span>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={minScore}
                    onChange={(e) => setMinScore(Math.max(0, Math.min(100, Number(e.target.value) || 0)))}
                    className="w-28 rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                    placeholder="Min score"
                    aria-label="Minimum overall score"
                  />
                </label>
              </div>
              <Button variant="secondary" size="sm" leftIcon={<Download className="h-4 w-4" />} onClick={() => exportCsv(result)}>
                Export CSV
              </Button>
            </div>

            {/* Leaderboard */}
            <Card className="overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-left text-xs">
                  <thead className="border-b border-white/5 text-[10px] uppercase tracking-wider text-gray-500">
                    <tr>
                      <th className="px-4 py-3">#</th>
                      <th className="px-4 py-3">Candidate</th>
                      <th className="px-4 py-3">Overall</th>
                      <th className="px-4 py-3">Coverage</th>
                      <th className="px-4 py-3">Exp</th>
                      <th className="px-4 py-3">Proj</th>
                      <th className="px-4 py-3">Quality</th>
                      <th className="px-4 py-3">Seniority</th>
                      <th className="px-4 py-3">Credibility</th>
                      <th className="px-4 py-3">Reqs (M/P/×)</th>
                      <th className="px-4 py-3">Recommendation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {visibleEntries.map((e) => (
                      <tr key={e.resume_id} className={e.error ? 'opacity-60' : 'hover:bg-white/[0.02]'}>
                        <td className="px-4 py-3 font-bold text-gray-400">{e.rank}</td>
                        <td className="px-4 py-3">
                          <span className="block max-w-[200px] truncate font-medium text-white">{e.resume_filename}</span>
                          {e.error && (
                            <span className="mt-0.5 inline-flex items-center gap-1 text-[10px] text-rose-400">
                              <AlertTriangle className="h-3 w-3" /> {e.error}
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {e.error ? (
                            <span className="text-gray-600">—</span>
                          ) : (
                            <span className={`rounded-md border px-2 py-0.5 font-bold ${getScoreColor(e.overall_score)}`}>
                              {e.overall_score}%
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-gray-300">{e.error ? '—' : `${e.coverage_score}%`}</td>
                        <td className="px-4 py-3 text-gray-300">{e.error ? '—' : `${e.experience_score}%`}</td>
                        <td className="px-4 py-3 text-gray-300">{e.error ? '—' : `${e.project_score}%`}</td>
                        <td className="px-4 py-3 text-gray-300">{e.error ? '—' : `${e.quality_score}%`}</td>
                        <td className="px-4 py-3">
                          {e.seniority_fit ? <Badge tone={FIT_TONE(e.seniority_fit)}>{e.seniority_fit}</Badge> : <span className="text-gray-600">—</span>}
                        </td>
                        <td className="px-4 py-3 text-gray-300">
                          {e.credibility_score != null ? `${e.credibility_score}%` : '—'}
                        </td>
                        <td className="px-4 py-3 text-gray-400">
                          {e.error ? '—' : `${e.matched_count}/${e.partial_count}/${e.missing_count}`}
                        </td>
                        <td className="px-4 py-3">
                          {e.error ? (
                            <span className="text-gray-600">—</span>
                          ) : (
                            <Badge tone={RECO_TONE(e.recruiter_recommendation)}>{e.recruiter_recommendation}</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {visibleEntries.length === 0 && (
                <div className="px-4 py-8 text-center text-xs text-gray-500">No candidates match the current filters.</div>
              )}
            </Card>
          </div>
        )
      ) : (
        <EmptyState
          icon={<Trophy className="h-9 w-9" />}
          title="No ranking yet"
          description="Pick a job description and one or more resumes, then run the ranking."
        />
      )}
    </div>
  );
};
