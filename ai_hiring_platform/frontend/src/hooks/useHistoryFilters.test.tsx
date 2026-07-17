import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useHistoryFilters } from './useHistoryFilters';

const setUrl = (url: string) => window.history.replaceState(null, '', url);

describe('useHistoryFilters URL synchronization', () => {
  beforeEach(() => setUrl('/'));

  it('initializes filter state from the URL', () => {
    setUrl('/?resume_filename=jane&page=2');
    const { result } = renderHook(() => useHistoryFilters());
    expect(result.current.filters.resume_filename).toBe('jane');
    expect(result.current.filters.page).toBe(2);
  });

  it('writes filter changes back to the URL', () => {
    const { result } = renderHook(() => useHistoryFilters());
    act(() => result.current.setFilters({ recommendation: 'Selected' }));
    expect(new URLSearchParams(window.location.search).get('recommendation')).toBe('Selected');
  });

  it('resets to page 1 when a non-page filter changes', () => {
    const { result } = renderHook(() => useHistoryFilters());
    act(() => result.current.setPage(3));
    expect(result.current.filters.page).toBe(3);
    act(() => result.current.setFilters({ recommendation: 'Selected' }));
    expect(result.current.filters.page).toBe(1);
    expect(new URLSearchParams(window.location.search).has('page')).toBe(false);
  });

  it('removes a single filter and clears all', () => {
    const { result } = renderHook(() => useHistoryFilters());
    act(() => result.current.setFilters({ recommendation: 'Selected', min_score: '70' }));
    act(() => result.current.removeFilter('recommendation'));
    expect(result.current.filters.recommendation).toBe('');
    expect(result.current.filters.min_score).toBe('70');
    act(() => result.current.clearAll());
    expect(result.current.filters.min_score).toBe('');
    expect(window.location.search).toBe('');
  });

  it('re-reads filters on popstate (back/forward)', () => {
    const { result } = renderHook(() => useHistoryFilters());
    act(() => {
      window.history.replaceState(null, '', '/?recommendation=Rejected');
      window.dispatchEvent(new PopStateEvent('popstate'));
    });
    expect(result.current.filters.recommendation).toBe('Rejected');
  });
});
