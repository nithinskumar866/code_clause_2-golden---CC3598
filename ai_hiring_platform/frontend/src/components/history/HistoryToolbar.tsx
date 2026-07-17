import type { FC } from 'react';
import { Trash2 } from 'lucide-react';
import { SearchBar } from '../ui/SearchBar';
import { Select } from '../ui/Select';
import type { SelectOption } from '../ui/Select';

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

const FILTER_OPTIONS: SelectOption[] = [
  { value: 'All', label: 'All outcomes' },
  { value: 'Selected', label: 'Selected' },
  { value: 'Borderline', label: 'Borderline' },
  { value: 'Rejected', label: 'Rejected' },
];

const SORT_OPTIONS: SelectOption[] = [
  { value: 'newest', label: 'Newest first' },
  { value: 'oldest', label: 'Oldest first' },
  { value: 'highest', label: 'Highest score' },
  { value: 'lowest', label: 'Lowest score' },
];

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
    <SearchBar
      value={search}
      onChange={onSearchChange}
      placeholder="Search by resume or JD filename..."
      ariaLabel="Search analyses"
      className="w-full max-w-sm"
    />

    <div className="flex flex-wrap items-center gap-2">
      <Select
        label="Filter by outcome"
        srLabel
        options={FILTER_OPTIONS}
        value={filter}
        onChange={(e) => onFilterChange(e.target.value as HistoryFilter)}
      />
      <Select
        label="Sort order"
        srLabel
        options={SORT_OPTIONS}
        value={sort}
        onChange={(e) => onSortChange(e.target.value as HistorySort)}
      />
      <button
        onClick={onClearAll}
        disabled={clearDisabled}
        className="flex items-center gap-1.5 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs font-semibold text-rose-400 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
      >
        <Trash2 className="h-3.5 w-3.5" /> Clear All
      </button>
    </div>
  </div>
);
