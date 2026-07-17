import { useEffect, useState, type FC } from 'react';
import { Server, Database, Cpu, Boxes, Sparkles, Activity, ArrowRight } from 'lucide-react';
import axios from 'axios';
import type { HistoryRecord } from '../../types';
import type { PageId } from '../../components/layout/navConfig';
import { fetchHistory } from '../../api/history';
import { getScoreColor } from '../../components/analysis/scoreColors';
import { PageHeader } from '../../components/ui/PageHeader';
import { StatCard } from '../../components/ui/StatCard';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import type { BadgeTone } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { EmptyState } from '../../components/ui/EmptyState';
import { ErrorState } from '../../components/ui/ErrorState';
import { Skeleton } from '../../components/common/Skeleton';

type ConnState = 'Connected' | 'Disconnected' | 'Checking';

interface DashboardProps {
  onNavigate?: (id: PageId) => void;
}

const toneFor = (state: ConnState): BadgeTone =>
  state === 'Connected' ? 'success' : state === 'Disconnected' ? 'danger' : 'warning';

const RECENT_LIMIT = 5;

export const Dashboard: FC<DashboardProps> = ({ onNavigate }) => {
  const [backend, setBackend] = useState<ConnState>('Checking');
  const [database, setDatabase] = useState<ConnState>('Checking');

  const [recent, setRecent] = useState<HistoryRecord[]>([]);
  const [recentLoading, setRecentLoading] = useState(true);
  const [recentError, setRecentError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const checkHealth = async () => {
      try {
        const res = await axios.get('http://localhost:8000/api/v1/health');
        if (!active) return;
        if (res.data && res.data.success) {
          setBackend('Connected');
          setDatabase(res.data.data?.database === 'connected' ? 'Connected' : 'Disconnected');
        } else {
          setBackend('Disconnected');
          setDatabase('Disconnected');
        }
      } catch {
        if (!active) return;
        setBackend('Disconnected');
        setDatabase('Disconnected');
      }
    };
    checkHealth();
    const id = setInterval(checkHealth, 10000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const loadRecent = async () => {
    setRecentLoading(true);
    setRecentError(null);
    try {
      const { items } = await fetchHistory({ sort: 'newest', page_size: RECENT_LIMIT });
      setRecent(items);
    } catch {
      setRecentError('Failed to load recent evaluations. Ensure the backend is running.');
    } finally {
      setRecentLoading(false);
    }
  };

  useEffect(() => {
    loadRecent();
  }, []);

  return (
    <div className="space-y-8 animate-fadeIn">
      <PageHeader
        title="Dashboard"
        description="Platform health and your most recent candidate evaluations at a glance."
        actions={
          onNavigate && (
            <Button leftIcon={<Sparkles className="h-4 w-4" />} onClick={() => onNavigate('analysis')}>
              New analysis
            </Button>
          )
        }
      />

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={<Server className="h-6 w-6" />}
          label="Backend API"
          value="FastAPI Service"
          badge={<Badge tone={toneFor(backend)}>{backend}</Badge>}
        />
        <StatCard
          icon={<Database className="h-6 w-6" />}
          iconClass="bg-cyan-500/10 text-cyan-400"
          label="Database"
          value="SQLite"
          badge={<Badge tone={toneFor(database)}>{database}</Badge>}
        />
        <StatCard
          icon={<Cpu className="h-6 w-6" />}
          iconClass="bg-emerald-500/10 text-emerald-400"
          label="Embeddings"
          value="BGE-small (local)"
          badge={<Badge tone="success">Active</Badge>}
        />
        <StatCard
          icon={<Boxes className="h-6 w-6" />}
          label="Vector Store"
          value="FAISS (local)"
          badge={<Badge tone="success">Active</Badge>}
        />
      </div>

      <Card className="p-6">
        <div className="flex items-center justify-between border-b border-white/5 pb-4">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-indigo-400" />
            <h2 className="text-base font-semibold text-white">Recent evaluations</h2>
          </div>
          {onNavigate && !recentLoading && !recentError && recent.length > 0 && (
            <Button variant="ghost" size="sm" onClick={() => onNavigate('history')}>
              View all <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>

        <div className="pt-4">
          {recentLoading ? (
            <ul className="space-y-2">
              {[0, 1, 2].map((i) => (
                <li key={i}>
                  <Skeleton className="h-14 w-full" />
                </li>
              ))}
            </ul>
          ) : recentError ? (
            <ErrorState message={recentError} onRetry={loadRecent} />
          ) : recent.length === 0 ? (
            <EmptyState
              icon={<Activity className="h-9 w-9" />}
              title="No evaluations yet"
              description="Run your first candidate evaluation to see it appear here."
              action={
                onNavigate && (
                  <Button variant="secondary" onClick={() => onNavigate('analysis')}>
                    Go to AI Analysis
                  </Button>
                )
              }
            />
          ) : (
            <ul className="divide-y divide-white/5">
              {recent.map((r) => (
                <li key={r.id}>
                  <button
                    onClick={() => onNavigate?.('history')}
                    className="flex w-full items-center gap-4 rounded-lg px-2 py-3 text-left transition hover:bg-white/[0.02] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                  >
                    <span
                      className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full border text-sm font-bold ${getScoreColor(
                        r.overall_score,
                      )}`}
                    >
                      {r.overall_score}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-white">{r.resume_filename}</span>
                      <span className="block truncate text-xs text-gray-500">{r.jd_filename}</span>
                    </span>
                    <span className="hidden max-w-[40%] truncate text-xs font-medium text-indigo-300 sm:block">
                      {r.recruiter_recommendation}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Card>
    </div>
  );
};
