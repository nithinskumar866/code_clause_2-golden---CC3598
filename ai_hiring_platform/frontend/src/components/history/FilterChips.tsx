import type { FC } from 'react';
import { X } from 'lucide-react';
import type { ChipKey, FilterChip } from '../../lib/historyFilters';

interface FilterChipsProps {
  chips: FilterChip[];
  onRemove: (key: ChipKey) => void;
  onClearAll: () => void;
}

/** Active-filter chips, each individually removable, plus Clear All. */
export const FilterChips: FC<FilterChipsProps> = ({ chips, onRemove, onClearAll }) => {
  if (chips.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2" aria-label="Active filters">
      {chips.map((chip) => (
        <span
          key={chip.key}
          className="inline-flex items-center gap-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/10 py-1 pl-3 pr-1.5 text-xs font-medium text-indigo-200"
        >
          {chip.label}
          <button
            type="button"
            onClick={() => onRemove(chip.key)}
            aria-label={`Remove filter ${chip.label}`}
            className="rounded-full p-0.5 text-indigo-300 transition hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}
      <button
        type="button"
        onClick={onClearAll}
        className="rounded-lg px-2 py-1 text-xs font-semibold text-gray-400 transition hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        Clear all
      </button>
    </div>
  );
};
