import { useEffect, useMemo, useState, type FC } from 'react';
import { RefreshCw, Inbox, Search } from 'lucide-react';
import type { AnalysisReport, HistoryRecord } from '../../types';
import { fetchHistory, fetchHistoryReport, deleteHistoryItem, clearHistory } from '../../api/history';
import { classifyFit } from '../../components/analysis/scoreColors';
import { HistoryCard } from '../../components/history/HistoryCard';
import { HistoryToolbar } from '../../components/history/HistoryToolbar';
import type { HistoryFilter, HistorySort } from '../../components/history/HistoryToolbar';
import { ReportModal } from '../../components/history/ReportModal';
import { ConfirmDialog } from '../../components/common/ConfirmDialog';
import { Skeleton } from '../../components/common/Skeleton';
import { PageHeader } from '../../components/ui/PageHeader';
import { Button } from '../../components/ui/Button';
import { EmptyState } from '../../components/ui/EmptyState';
import { ErrorState } from '../../components/ui/ErrorState';
import { useToast } from '../../components/ui/toast-context';

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

export const History: FC = () => {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const toast = useToast();

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
    const action = pending;
    setConfirmLoading(true);
    try {
      if (action.kind === 'delete') {
        await deleteHistoryItem(action.record.id);
        setRecords((prev) => prev.filter((r) => r.id !== action.record.id));
        toast.success('Analysis deleted');
      } else {
        await clearHistory();
        setRecords([]);
        toast.success('History cleared');
      }
      setPending(null);
    } catch (err: any) {
      console.error(err);
      toast.error(action.kind === 'delete' ? 'Failed to delete this analysis.' : 'Failed to clear history.');
      setPending(null);
    } finally {
      setConfirmLoading(false);
    }
  };

  const hasRecords = records.length > 0;

  return (
    <div className="space-y-8 animate-fadeIn">
      <PageHeader
        title="Analysis History"
        description="Every evaluation the platform has produced. Open any record to revisit its full explainable report."
        actions={
          <Button
            variant="secondary"
            onClick={loadHistory}
            loading={loading}
            leftIcon={<RefreshCw className="h-4 w-4" />}
          >
            Refresh
          </Button>
        }
      />

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
        <ErrorState title="Couldn't load history" message={error} onRetry={loadHistory} />
      ) : !hasRecords ? (
        <EmptyState
          icon={<Inbox className="h-10 w-10" />}
          title="No analyses yet"
          description="Run an evaluation from the AI Analysis page and it will appear here for you to revisit anytime."
        />
      ) : visible.length === 0 ? (
        <EmptyState
          icon={<Search className="h-10 w-10" />}
          title="No matching analyses"
          description="Try a different search term or filter."
        />
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
