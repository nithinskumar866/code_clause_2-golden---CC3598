import { useCallback, useEffect, useMemo, useState, type FC } from 'react';
import { RefreshCw, Inbox, Search, Trash2 } from 'lucide-react';
import type { HistoryRecord, HistoryPageMeta } from '../../types';
import { fetchHistory, deleteHistoryItem, clearHistory } from '../../api/history';
import { useHistoryFilters } from '../../hooks/useHistoryFilters';
import { toQuery, activeChips, PAGE_SIZE } from '../../lib/historyFilters';
import { HistoryCard } from '../../components/history/HistoryCard';
import { HistoryFilters } from '../../components/history/HistoryFilters';
import { FilterChips } from '../../components/history/FilterChips';
import { ConfirmDialog } from '../../components/common/ConfirmDialog';
import { Skeleton } from '../../components/common/Skeleton';
import { PageHeader } from '../../components/ui/PageHeader';
import { Button } from '../../components/ui/Button';
import { EmptyState } from '../../components/ui/EmptyState';
import { ErrorState } from '../../components/ui/ErrorState';
import { Pagination } from '../../components/ui/Pagination';
import { useToast } from '../../components/ui/toast-context';

interface HistoryProps {
  onOpenCandidate: (record: HistoryRecord) => void;
}

type PendingAction = { kind: 'delete'; record: HistoryRecord } | { kind: 'clear' };

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

export const History: FC<HistoryProps> = ({ onOpenCandidate }) => {
  const { filters, setFilters, setPage, removeFilter, clearAll } = useHistoryFilters();

  const [items, setItems] = useState<HistoryRecord[]>([]);
  const [meta, setMeta] = useState<HistoryPageMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const toast = useToast();

  const [pending, setPending] = useState<PendingAction | null>(null);
  const [confirmLoading, setConfirmLoading] = useState(false);

  const query = useMemo(() => toQuery(filters), [filters]);
  const chips = useMemo(() => activeChips(filters), [filters]);
  const filtersActive = chips.length > 0;

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const page = await fetchHistory(query);
      setItems(page.items);
      setMeta(page.meta);
    } catch (err: any) {
      console.error(err);
      setError('Failed to load analysis history. Ensure the FastAPI backend is running.');
    } finally {
      setLoading(false);
    }
  }, [query]);

  useEffect(() => {
    load();
  }, [load]);

  // Clamp the page if it falls out of range (after deletes or filter changes).
  useEffect(() => {
    if (meta && meta.total_pages >= 1 && filters.page > meta.total_pages) {
      setPage(meta.total_pages);
    }
  }, [meta, filters.page, setPage]);

  const confirmAction = async () => {
    if (!pending) return;
    const action = pending;
    setConfirmLoading(true);
    try {
      if (action.kind === 'delete') {
        await deleteHistoryItem(action.record.id);
        toast.success('Analysis deleted');
      } else {
        await clearHistory();
        toast.success('History cleared');
      }
      setPending(null);
      await load();
    } catch (err: any) {
      console.error(err);
      toast.error(action.kind === 'delete' ? 'Failed to delete this analysis.' : 'Failed to clear history.');
      setPending(null);
    } finally {
      setConfirmLoading(false);
    }
  };

  const total = meta?.total_count ?? 0;
  const rangeStart = total === 0 ? 0 : (filters.page - 1) * PAGE_SIZE + 1;
  const rangeEnd = (filters.page - 1) * PAGE_SIZE + items.length;

  return (
    <div className="space-y-6 animate-fadeIn">
      <PageHeader
        title="Analysis History"
        description="Search, filter and sort every evaluation. Open any record to view the full candidate profile."
        actions={
          <>
            <Button
              variant="secondary"
              onClick={load}
              loading={loading}
              leftIcon={<RefreshCw className="h-4 w-4" />}
            >
              Refresh
            </Button>
            {total > 0 && (
              <Button
                variant="danger"
                onClick={() => setPending({ kind: 'clear' })}
                leftIcon={<Trash2 className="h-4 w-4" />}
              >
                Clear history
              </Button>
            )}
          </>
        }
      />

      <HistoryFilters filters={filters} onChange={setFilters} onClearAll={clearAll} hasActive={filtersActive} />

      <FilterChips chips={chips} onRemove={removeFilter} onClearAll={clearAll} />

      {!loading && !error && total > 0 && (
        <p className="text-xs text-gray-500" aria-live="polite">
          Showing {rangeStart}–{rangeEnd} of {total} {total === 1 ? 'analysis' : 'analyses'}
        </p>
      )}

      <div aria-busy={loading}>
        {loading ? (
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            {[0, 1, 2, 3].map((i) => (
              <HistoryCardSkeleton key={i} />
            ))}
          </div>
        ) : error ? (
          <ErrorState title="Couldn't load history" message={error} onRetry={load} />
        ) : items.length === 0 ? (
          filtersActive ? (
            <EmptyState
              icon={<Search className="h-10 w-10" />}
              title="No matching analyses"
              description="No analyses match your current filters."
              action={
                <Button variant="secondary" onClick={clearAll}>
                  Clear all filters
                </Button>
              }
            />
          ) : (
            <EmptyState
              icon={<Inbox className="h-10 w-10" />}
              title="No analyses yet"
              description="Run an evaluation from the AI Analysis page and it will appear here for you to revisit anytime."
            />
          )
        ) : (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
              {items.map((r) => (
                <HistoryCard
                  key={r.id}
                  record={r}
                  onOpen={onOpenCandidate}
                  onDelete={(record) => setPending({ kind: 'delete', record })}
                />
              ))}
            </div>
            {meta && (
              <Pagination page={filters.page} totalPages={meta.total_pages} onPage={setPage} />
            )}
          </div>
        )}
      </div>

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
