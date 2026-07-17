import type { FC } from 'react';
import { Search, Trash2 } from 'lucide-react';

export type HistoryFilter = 'All' | 'Selected' | 'Borderline' | 'Rejected';
export type HistorySort = 'newest' | 'oldest' | 'highest' | 'lowest';

interface HistoryToolbarProps {
  search: string;
  onSearchChange: (value: string) => void;
  filter: HistoryFilter;
  onFilterChange: (value: HistoryFilter) => void;
  sort: HistorySort;
  onSortChange: (value: HistorySort) => void;
  onClearAll: () => void;
  clearDisabled: boolean;
}

const selectClass =
  'rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500';

/** Search + outcome filter + sort + clear-all controls for the history list. */
export const HistoryToolbar: FC<HistoryToolbarProps> = ({
  search,
  onSearchChange,
  filter,
  onFilterChange,
  sort,
  onSortChange,
  onClearAll,
  clearDisabled,
}) => (
  <div className="flex flex-col gap-4 rounded-xl border border-white/5 bg-card p-4 lg:flex-row lg:items-center lg:justify-between">
    <div className="relative w-full max-w-sm">
      <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
      <input
        type="text"
        value={search}
        onChange={(e) => onSearchChange(e.target.value)}
        placeholder="Search by resume or JD filename..."
        className="w-full rounded-lg border border-white/10 bg-black/40 py-2 pl-9 pr-3 text-sm text-white placeholder:text-gray-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
      />
    </div>

    <div className="flex flex-wrap items-center gap-2">
      <select
        aria-label="Filter by outcome"
        value={filter}
        onChange={(e) => onFilterChange(e.target.value as HistoryFilter)}
        className={selectClass}
      >
        <option value="All">All outcomes</option>
        <option value="Selected">Selected</option>
        <option value="Borderline">Borderline</option>
        <option value="Rejected">Rejected</option>
      </select>

      <select
        aria-label="Sort order"
        value={sort}
        onChange={(e) => onSortChange(e.target.value as HistorySort)}
        className={selectClass}
      >
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="highest">Highest score</option>
        <option value="lowest">Lowest score</option>
      </select>

      <button
        onClick={onClearAll}
        disabled={clearDisabled}
        className="flex items-center gap-1.5 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs font-semibold text-rose-400 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Trash2 className="h-3.5 w-3.5" /> Clear All
      </button>
    </div>
  </div>
);
