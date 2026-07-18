import { useCallback, useEffect, useMemo, useState, type FC, type ReactNode } from 'react';
import { BarChart3, Users, TrendingUp, Award, CheckCircle2, XCircle, FileText, Briefcase } from 'lucide-react';
import {
  fetchOverview,
  fetchScoreDistribution,
  fetchRecommendationDistribution,
  fetchTrends,
  fetchTopResumes,
  fetchTopJobs,
  fetchSkillFrequency,
} from '../../api/analytics';
import type {
  OverallStatistics,
  ScoreDistribution,
  RecommendationDistribution,
  TrendData,
  TopItem,
  SkillFrequency,
} from '../../types';
import { PageHeader } from '../../components/ui/PageHeader';
import { Card } from '../../components/ui/Card';
import { StatCard } from '../../components/ui/StatCard';
import { EmptyState } from '../../components/ui/EmptyState';
import { ErrorState } from '../../components/ui/ErrorState';
import { Skeleton } from '../../components/common/Skeleton';

interface AllData {
  overview: OverallStatistics;
  scores: ScoreDistribution;
  recos: RecommendationDistribution;
  trends: TrendData;
  topResumes: TopItem[];
  topJobs: TopItem[];
  skills: SkillFrequency;
}

const RECO_COLOR: Record<string, string> = {
  Selected: 'bg-emerald-500',
  Borderline: 'bg-amber-500',
  Rejected: 'bg-rose-500',
};

/** Horizontal bar row (label · bar · value) — the shared primitive for every chart here. */
const BarRow: FC<{ label: string; value: number; max: number; barClass?: string; suffix?: string }> = ({
  label,
  value,
  max,
  barClass = 'bg-indigo-500',
  suffix = '',
}) => (
  <div className="space-y-1">
    <div className="flex justify-between text-xs text-gray-300">
      <span className="truncate pr-2">{label}</span>
      <span className="font-semibold shrink-0">
        {value}
        {suffix}
      </span>
    </div>
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
      <div className={`h-full ${barClass}`} style={{ width: `${max > 0 ? (value / max) * 100 : 0}%` }} />
    </div>
  </div>
);

const Panel: FC<{ title: string; icon: ReactNode; children: ReactNode }> = ({ title, icon, children }) => (
  <Card className="p-6 space-y-4">
    <h3 className="flex items-center gap-2 border-b border-white/5 pb-2 text-sm font-semibold text-white">
      <span className="text-indigo-400">{icon}</span>
      {title}
    </h3>
    {children}
  </Card>
);

export const Analytics: FC = () => {
  const [data, setData] = useState<AllData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [trendMode, setTrendMode] = useState<'daily' | 'weekly' | 'monthly'>('daily');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [overview, scores, recos, trends, topResumes, topJobs, skills] = await Promise.all([
        fetchOverview(),
        fetchScoreDistribution(),
        fetchRecommendationDistribution(),
        fetchTrends(),
        fetchTopResumes(5),
        fetchTopJobs(5),
        fetchSkillFrequency(8),
      ]);
      setData({ overview, scores, recos, trends, topResumes, topJobs, skills });
    } catch {
      setError('Failed to load analytics. Ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const trendSeries = useMemo(() => (data ? data.trends[trendMode] : []), [data, trendMode]);
  const trendMax = useMemo(() => Math.max(1, ...trendSeries.map((t) => t.count)), [trendSeries]);

  if (loading) {
    return (
      <div className="space-y-8">
        <PageHeader title="Analytics" description="Hiring analytics across all evaluations." />
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28 w-full" />
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-8">
        <PageHeader title="Analytics" description="Hiring analytics across all evaluations." />
        <ErrorState message={error} onRetry={load} />
      </div>
    );
  }

  if (!data || data.overview.total_analyses === 0) {
    return (
      <div className="space-y-8">
        <PageHeader title="Analytics" description="Hiring analytics across all evaluations." />
        <EmptyState
          icon={<BarChart3 className="h-9 w-9" />}
          title="No analytics yet"
          description="Run some candidate evaluations to populate hiring analytics."
        />
      </div>
    );
  }

  const { overview, scores, recos, topResumes, topJobs, skills } = data;
  const passPct =
    overview.total_analyses > 0 ? Math.round((overview.selected / overview.total_analyses) * 100) : 0;
  const rejectPct =
    overview.total_analyses > 0 ? Math.round((overview.rejected / overview.total_analyses) * 100) : 0;
  const scoreMax = Math.max(1, ...scores.ranges.map((r) => r.count));
  const resumeMax = Math.max(1, ...topResumes.map((r) => r.count));
  const jobMax = Math.max(1, ...topJobs.map((j) => j.count));
  const matchedMax = Math.max(1, ...skills.top_matched.map((s) => s.count));
  const missingMax = Math.max(1, ...skills.top_missing.map((s) => s.count));

  return (
    <div className="space-y-8 animate-fadeIn">
      <PageHeader title="Analytics" description="Hiring analytics across all evaluations." />

      {/* Headline stats */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={<Users className="h-6 w-6" />} label="Total analyses" value={String(overview.total_analyses)} />
        <StatCard
          icon={<Award className="h-6 w-6" />}
          iconClass="bg-emerald-500/10 text-emerald-400"
          label="Average score"
          value={`${overview.average_overall_score}%`}
        />
        <StatCard
          icon={<CheckCircle2 className="h-6 w-6" />}
          iconClass="bg-emerald-500/10 text-emerald-400"
          label="Pass rate (Selected)"
          value={`${passPct}%`}
        />
        <StatCard
          icon={<XCircle className="h-6 w-6" />}
          iconClass="bg-rose-500/10 text-rose-400"
          label="Reject rate"
          value={`${rejectPct}%`}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Recommendation distribution */}
        <Panel title="Hiring recommendations" icon={<Award className="h-4 w-4" />}>
          <div className="space-y-3">
            {recos.distribution.map((b) => (
              <BarRow
                key={b.label}
                label={`${b.label} (${b.count})`}
                value={b.percentage}
                max={100}
                suffix="%"
                barClass={RECO_COLOR[b.label] ?? 'bg-indigo-500'}
              />
            ))}
          </div>
        </Panel>

        {/* Score distribution */}
        <Panel title="Score distribution" icon={<BarChart3 className="h-4 w-4" />}>
          <div className="space-y-3">
            {scores.ranges.map((r) => (
              <BarRow key={r.label} label={r.label} value={r.count} max={scoreMax} />
            ))}
          </div>
        </Panel>

        {/* Average sub-scores */}
        <Panel title="Average sub-scores" icon={<TrendingUp className="h-4 w-4" />}>
          <div className="space-y-3">
            <BarRow label="Coverage" value={overview.average_coverage_score} max={100} suffix="%" barClass="bg-indigo-500" />
            <BarRow label="Experience" value={overview.average_experience_score} max={100} suffix="%" barClass="bg-sky-500" />
            <BarRow label="Project" value={overview.average_project_score} max={100} suffix="%" barClass="bg-violet-500" />
            <BarRow label="Quality" value={overview.average_quality_score} max={100} suffix="%" barClass="bg-emerald-500" />
          </div>
        </Panel>

        {/* Trends */}
        <Panel title="Activity trends" icon={<TrendingUp className="h-4 w-4" />}>
          <div className="mb-2 flex gap-1 rounded-lg border border-white/10 bg-black/20 p-0.5 text-xs">
            {(['daily', 'weekly', 'monthly'] as const).map((m) => (
              <button
                key={m}
                onClick={() => setTrendMode(m)}
                className={`flex-1 rounded-md px-2 py-1 font-semibold capitalize transition ${
                  trendMode === m ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
          {trendSeries.length === 0 ? (
            <p className="py-6 text-center text-xs text-gray-500">No activity in this window.</p>
          ) : (
            <div className="flex h-40 items-end gap-1">
              {trendSeries.map((t) => (
                <div key={t.period} className="flex flex-1 flex-col items-center gap-1" title={`${t.period}: ${t.count}`}>
                  <div className="flex w-full items-end justify-center" style={{ height: '128px' }}>
                    <div
                      className="w-full rounded-t bg-indigo-500/80"
                      style={{ height: `${(t.count / trendMax) * 100}%`, minHeight: t.count > 0 ? '3px' : '0' }}
                    />
                  </div>
                  <span className="w-full truncate text-center text-[8px] text-gray-600">{t.period.slice(5)}</span>
                </div>
              ))}
            </div>
          )}
        </Panel>

        {/* Most common missing skills */}
        <Panel title="Most common missing skills" icon={<XCircle className="h-4 w-4" />}>
          {skills.top_missing.length === 0 ? (
            <p className="py-4 text-center text-xs text-gray-500">No missing-skill data yet.</p>
          ) : (
            <div className="space-y-3">
              {skills.top_missing.map((s) => (
                <BarRow key={s.skill} label={s.skill} value={s.count} max={missingMax} barClass="bg-rose-500" />
              ))}
            </div>
          )}
        </Panel>

        {/* Most frequently matched skills */}
        <Panel title="Most frequently matched skills" icon={<CheckCircle2 className="h-4 w-4" />}>
          {skills.top_matched.length === 0 ? (
            <p className="py-4 text-center text-xs text-gray-500">No matched-skill data yet.</p>
          ) : (
            <div className="space-y-3">
              {skills.top_matched.map((s) => (
                <BarRow key={s.skill} label={s.skill} value={s.count} max={matchedMax} barClass="bg-emerald-500" />
              ))}
            </div>
          )}
        </Panel>

        {/* Top resumes */}
        <Panel title="Most-analysed resumes" icon={<FileText className="h-4 w-4" />}>
          {topResumes.length === 0 ? (
            <p className="py-4 text-center text-xs text-gray-500">No data yet.</p>
          ) : (
            <div className="space-y-3">
              {topResumes.map((r) => (
                <BarRow key={r.name} label={r.name} value={r.count} max={resumeMax} barClass="bg-sky-500" />
              ))}
            </div>
          )}
        </Panel>

        {/* Top jobs */}
        <Panel title="Most-analysed job descriptions" icon={<Briefcase className="h-4 w-4" />}>
          {topJobs.length === 0 ? (
            <p className="py-4 text-center text-xs text-gray-500">No data yet.</p>
          ) : (
            <div className="space-y-3">
              {topJobs.map((j) => (
                <BarRow key={j.name} label={j.name} value={j.count} max={jobMax} barClass="bg-violet-500" />
              ))}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
};
