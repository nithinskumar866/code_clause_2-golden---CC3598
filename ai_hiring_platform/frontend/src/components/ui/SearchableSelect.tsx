import { useEffect, useMemo, useRef, useState, type FC } from 'react';
import { Check, ChevronDown, Search, X } from 'lucide-react';

export interface SearchableOption {
  value: string;
  label: string;
  /** Secondary line — upload date, size, anything that separates similar names. */
  hint?: string;
}

interface SearchableSelectProps {
  label?: string;
  placeholder?: string;
  options: SearchableOption[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  emptyMessage?: string;
}

/**
 * A select you can type into.
 *
 * A native `<select>` is fine for five options and unusable for six hundred: picking a
 * resume meant scrolling a dropdown the length of the whole pool. Typing filters on
 * both the visible label and the hint, so a candidate is reachable by name or by file.
 *
 * Keyboard: type to filter, arrows to move, Enter to choose, Escape to close.
 */
export const SearchableSelect: FC<SearchableSelectProps> = ({
  label,
  placeholder = 'Search…',
  options,
  value,
  onChange,
  disabled = false,
  emptyMessage = 'Nothing matches that.',
}) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selected = options.find((o) => o.value === value) || null;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter(
      (o) => o.label.toLowerCase().includes(q) || (o.hint || '').toLowerCase().includes(q),
    );
  }, [options, query]);

  // Close on an outside click — the list overlays the page, so it must not linger.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  useEffect(() => {
    if (open) {
      setQuery('');
      setActive(0);
      // Focus after paint so the caret lands in the filter box, not the trigger.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const choose = (v: string) => {
    onChange(v);
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && filtered[active]) {
      e.preventDefault();
      choose(filtered[active].value);
    } else if (e.key === 'Escape') {
      setOpen(false);
    }
  };

  return (
    <div ref={rootRef} className="relative">
      {label && (
        <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-gray-400">
          {label}
        </span>
      )}

      <button
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-left text-sm text-white transition hover:border-white/20 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
      >
        <span className={`min-w-0 flex-1 truncate ${selected ? '' : 'text-gray-500'}`}>
          {selected ? selected.label : placeholder}
        </span>
        <span className="flex shrink-0 items-center gap-1">
          {selected && !disabled && (
            <span
              role="button"
              tabIndex={-1}
              aria-label="Clear selection"
              onClick={(e) => {
                e.stopPropagation();
                onChange('');
              }}
              className="rounded p-0.5 text-gray-500 hover:bg-white/10 hover:text-white"
            >
              <X className="h-3.5 w-3.5" />
            </span>
          )}
          <ChevronDown className="h-4 w-4 text-gray-500" />
        </span>
      </button>

      {open && (
        <div className="absolute z-30 mt-1 w-full overflow-hidden rounded-lg border border-white/10 bg-[#12131a] shadow-2xl shadow-black/60">
          <div className="flex items-center gap-2 border-b border-white/10 px-3 py-2">
            <Search className="h-3.5 w-3.5 shrink-0 text-gray-500" />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setActive(0);
              }}
              onKeyDown={onKeyDown}
              placeholder="Type to filter…"
              className="min-w-0 flex-1 bg-transparent text-sm text-white placeholder:text-gray-600 focus:outline-none"
            />
            <span className="shrink-0 text-[10px] text-gray-600">
              {filtered.length}/{options.length}
            </span>
          </div>

          <ul role="listbox" className="max-h-64 overflow-y-auto py-1">
            {filtered.length === 0 && (
              <li className="px-3 py-2 text-xs text-gray-500">{emptyMessage}</li>
            )}
            {filtered.map((o, i) => (
              <li key={o.value}>
                <button
                  type="button"
                  role="option"
                  aria-selected={o.value === value}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => choose(o.value)}
                  className={`flex w-full items-start gap-2 px-3 py-1.5 text-left text-sm transition ${
                    i === active ? 'bg-indigo-500/15 text-white' : 'text-gray-300'
                  }`}
                >
                  <Check
                    className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${
                      o.value === value ? 'text-indigo-400' : 'text-transparent'
                    }`}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate">{o.label}</span>
                    {o.hint && <span className="block truncate text-[11px] text-gray-500">{o.hint}</span>}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
