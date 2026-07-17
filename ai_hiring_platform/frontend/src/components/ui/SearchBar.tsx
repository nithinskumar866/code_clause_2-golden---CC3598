import type { FC } from 'react';
import { Search, X } from 'lucide-react';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  ariaLabel?: string;
  className?: string;
}

/** Reusable search input with icon and clear button. */
export const SearchBar: FC<SearchBarProps> = ({
  value,
  onChange,
  placeholder = 'Search...',
  ariaLabel = 'Search',
  className = '',
}) => (
  <div className={`relative ${className}`}>
    <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      aria-label={ariaLabel}
      className="w-full rounded-lg border border-white/10 bg-black/40 py-2 pl-9 pr-9 text-sm text-white placeholder:text-gray-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
    />
    {value && (
      <button
        type="button"
        onClick={() => onChange('')}
        aria-label="Clear search"
        className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-gray-500 transition hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    )}
  </div>
);
