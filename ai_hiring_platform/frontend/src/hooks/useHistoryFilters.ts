import { useCallback, useEffect, useState } from 'react';
import { updateSearchParams } from '../lib/url';
import { DEFAULT_FILTERS, parseFilters, writeFilters } from '../lib/historyFilters';
import type { ChipKey, HistoryFilters } from '../lib/historyFilters';

export interface UseHistoryFilters {
  filters: HistoryFilters;
  /** Merge a partial update. Any change except `page` resets to page 1. */
  setFilters: (partial: Partial<HistoryFilters>) => void;
  setPage: (page: number) => void;
  removeFilter: (key: ChipKey) => void;
  clearAll: () => void;
}

/**
 * Owns the history filter state and keeps it synchronized with the URL query
 * string (so refresh and back/forward preserve state). Filtering itself is done
 * by the backend — this hook only manages the filter values.
 */
export function useHistoryFilters(): UseHistoryFilters {
  const [filters, setFiltersState] = useState<HistoryFilters>(() =>
    parseFilters(window.location.search),
  );

  // Reflect filter state into the URL (merging, so `view` etc. survive).
  useEffect(() => {
    updateSearchParams((params) => writeFilters(params, filters), 'replace');
  }, [filters]);

  // Re-read filters on browser back/forward.
  useEffect(() => {
    const onPop = () => setFiltersState(parseFilters(window.location.search));
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const setFilters = useCallback((partial: Partial<HistoryFilters>) => {
    setFiltersState((prev) => {
      const next = { ...prev, ...partial };
      // Changing any filter (other than the page itself) returns to page 1.
      if (!('page' in partial)) next.page = 1;
      return next;
    });
  }, []);

  const setPage = useCallback((page: number) => {
    setFiltersState((prev) => ({ ...prev, page }));
  }, []);

  const removeFilter = useCallback((key: ChipKey) => {
    setFiltersState((prev) => ({ ...prev, [key]: DEFAULT_FILTERS[key], page: 1 }));
  }, []);

  const clearAll = useCallback(() => {
    setFiltersState({ ...DEFAULT_FILTERS });
  }, []);

  return { filters, setFilters, setPage, removeFilter, clearAll };
}
