import { useEffect, useMemo, useState, type FC } from 'react';
import { RefreshCw, Inbox, AlertCircle, Search } from 'lucide-react';
import type { AnalysisReport, HistoryRecord } from '../../types';
import { fetchHistory, fetchHistoryReport, deleteHistoryItem, clearHistory } from '../../api/history';
import { classifyFit } from '../../components/analysis/scoreColors';
import { HistoryCard } from '../../components/history/HistoryCard';
import { HistoryToolbar } from '../../components/history/HistoryToolbar';
import type { HistoryFilter, HistorySort } from '../../components/history/HistoryToolbar';
import { ReportModal } from '../../components/history/ReportModal';
import { ConfirmDialog } from '../../components/common/ConfirmDialog';
import { Skeleton } from '../../components/common/Skeleton';

type PendingAction = { kind: 'delete'; record: HistoryRecord } | { kind: 'clear' };

const dateValue = (iso: string): number => {
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? 0 : t;
};

const HistoryCardSkeleton: FC = () => (
  <div className="space-y-4 rounded-xl border border-white/5 bg-card p-5">
    <div className="flex justify-between">
      <Skeleton className="h-3 w-32" />
      <Skeleton className="h-4 w-16" />
    </div>
    <div className="flex items-center gap-4">
      <Skeleton className="h-14 w-14 rounded-full" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/2" />
      </div>
    </div>
    <Skeleton className="h-3 w-full" />
    <Skeleton className="h-3 w-5/6" />
  </div>
);

const EmptyState: FC = () => (
  <div className="flex flex-col items-center justify-center rounded-xl border border-white/5 bg-card px-6 py-24 text-center">
    <Inbox className="mb-4 h-10 w-10 text-gray-500" />
    <h3 className="text-base font-medium text-white">No analyses yet</h3>
    <p className="mt-1 max-w-sm text-xs text-gray-400">
      Run an evaluation from the AI Analysis page and it will appear here for you to revisit anytime.
    </p>
  </div>
);

const NoResultsState: FC = () => (
  <div className="flex flex-col items-center justify-center rounded-xl border border-white/5 bg-card px-6 py-24 text-center">
    <Search className="mb-4 h-10 w-10 text-gray-500" />
    <h3 className="text-base font-medium text-white">No matching analyses</h3>
    <p className="mt-1 max-w-sm text-xs text-gray-400">Try a different search term or filter.</p>
  </div>
);

const ErrorState: FC<{ message: string; onRetry: () => void }> = ({ message, onRetry }) => (
  <div className="flex flex-col items-center justify-center rounded-xl border border-rose-500/20 bg-card px-6 py-24 text-center">
    <AlertCircle className="mb-4 h-10 w-10 text-rose-400" />
    <h3 className="text-base font-medium text-white">Couldn't load history</h3>
    <p className="mt-1 max-w-sm text-xs text-gray-400">{message}</p>
    <button
      onClick={onRetry}
      className="mt-4 flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500"
    >
      <RefreshCw className="h-4 w-4" /> Retry
    </button>
  </div>
);

export const History: FC = () => {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<HistoryFilter>('All');
  const [sort, setSort] = useState<HistorySort>('newest');

  // Report modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalReport, setModalReport] = useState<AnalysisReport | null>(null);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);

  // Confirmation state
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [confirmLoading, setConfirmLoading] = useState(false);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHistory();
      setRecords(data);
    } catch (err: any) {
      console.error(err);
      setError('Failed to load analysis history. Ensure the FastAPI backend is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = records.filter((r) => {
      const matchesSearch =
        !q || r.resume_filename.toLowerCase().includes(q) || r.jd_filename.toLowerCase().includes(q);
      const matchesFilter = filter === 'All' || classifyFit(r.overall_score) === filter;
      return matchesSearch && matchesFilter;
    });
    return [...filtered].sort((a, b) => {
      switch (sort) {
        case 'newest':
          return dateValue(b.created_at) - dateValue(a.created_at);
        case 'oldest':
          return dateValue(a.created_at) - dateValue(b.created_at);
        case 'highest':
          return b.overall_score - a.overall_score;
        case 'lowest':
          return a.overall_score - b.overall_score;
        default:
          return 0;
      }
    });
  }, [records, search, filter, sort]);

  const openReport = async (id: number) => {
    setModalOpen(true);
    setModalReport(null);
    setModalError(null);
    setModalLoading(true);
    try {
      const report = await fetchHistoryReport(id);
      setModalReport(report);
    } catch (err: any) {
      console.error(err);
      setModalError('Failed to load this report.');
    } finally {
      setModalLoading(false);
    }
  };

  const closeModal = () => {
    setModalOpen(false);
    setModalReport(null);
    setModalError(null);
  };

  const confirmAction = async () => {
    if (!pending) return;
    setConfirmLoading(true);
    setActionError(null);
    try {
      if (pending.kind === 'delete') {
        const { id } = pending.record;
        await deleteHistoryItem(id);
        setRecords((prev) => prev.filter((r) => r.id !== id));
      } else {
        await clearHistory();
        setRecords([]);
      }
      setPending(null);
    } catch (err: any) {
      console.error(err);
      setActionError(
        pending.kind === 'delete' ? 'Failed to delete this analysis.' : 'Failed to clear history.',
      );
      setPending(null);
    } finally {
      setConfirmLoading(false);
    }
  };

  const hasRecords = records.length > 0;

  return (
    <div className="space-y-8 animate-fadeIn">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Analysis History</h1>
          <p className="mt-2 text-sm text-gray-400">
            Every evaluation the platform has produced. Open any record to revisit its full explainable report.
          </p>
        </div>
        <button
          onClick={loadHistory}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-xs font-semibold text-gray-300 transition hover:bg-white/5 hover:text-white disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      {actionError && (
        <div className="flex items-start gap-2.5 rounded-lg border border-rose-500/20 bg-rose-500/10 p-3 text-xs text-rose-400">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{actionError}</span>
        </div>
      )}

      {!loading && !error && hasRecords && (
        <HistoryToolbar
          search={search}
          onSearchChange={setSearch}
          filter={filter}
          onFilterChange={setFilter}
          sort={sort}
          onSortChange={setSort}
          onClearAll={() => setPending({ kind: 'clear' })}
          clearDisabled={loading}
        />
      )}

      {loading ? (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          {[0, 1, 2, 3].map((i) => (
            <HistoryCardSkeleton key={i} />
          ))}
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={loadHistory} />
      ) : !hasRecords ? (
        <EmptyState />
      ) : visible.length === 0 ? (
        <NoResultsState />
      ) : (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          {visible.map((r) => (
            <HistoryCard
              key={r.id}
              record={r}
              onOpen={openReport}
              onDelete={(record) => setPending({ kind: 'delete', record })}
            />
          ))}
        </div>
      )}

      <ReportModal
        open={modalOpen}
        loading={modalLoading}
        error={modalError}
        report={modalReport}
        onClose={closeModal}
      />

      <ConfirmDialog
        open={pending !== null}
        title={pending?.kind === 'clear' ? 'Clear all history?' : 'Delete this analysis?'}
        message={
          pending?.kind === 'clear'
            ? 'This permanently removes every stored analysis report. This action cannot be undone.'
            : pending?.kind === 'delete'
              ? `This permanently removes the analysis for "${pending.record.resume_filename}". This action cannot be undone.`
              : ''
        }
        confirmLabel={pending?.kind === 'clear' ? 'Clear All' : 'Delete'}
        loading={confirmLoading}
        onConfirm={confirmAction}
        onCancel={() => setPending(null)}
      />
    </div>
  );
};
