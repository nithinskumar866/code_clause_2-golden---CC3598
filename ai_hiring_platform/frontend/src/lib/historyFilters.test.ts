import { describe, it, expect } from 'vitest';
import {
  DEFAULT_FILTERS,
  PAGE_SIZE,
  parseFilters,
  writeFilters,
  toQuery,
  activeChips,
  hasActiveFilters,
} from './historyFilters';
import type { HistoryFilters } from './historyFilters';

const make = (over: Partial<HistoryFilters> = {}): HistoryFilters => ({ ...DEFAULT_FILTERS, ...over });

describe('historyFilters helpers', () => {
  it('parses filters from a query string', () => {
    const f = parseFilters('?resume_filename=jane&recommendation=Selected&min_score=70&sort=highest_score&page=3');
    expect(f.resume_filename).toBe('jane');
    expect(f.recommendation).toBe('Selected');
    expect(f.min_score).toBe('70');
    expect(f.sort).toBe('highest_score');
    expect(f.page).toBe(3);
  });

  it('falls back to defaults for invalid or absent params', () => {
    const f = parseFilters('?sort=bogus&recommendation=Nope&page=0');
    expect(f.sort).toBe('newest');
    expect(f.recommendation).toBe('');
    expect(f.page).toBe(1);
  });

  it('writes only active filters (defaults are omitted)', () => {
    const p = new URLSearchParams();
    writeFilters(p, make({ resume_filename: 'jane' }));
    expect(p.get('resume_filename')).toBe('jane');
    expect(p.has('sort')).toBe(false); // newest is the default
    expect(p.has('page')).toBe(false); // page 1 is the default
  });

  it('writes sort and page when non-default', () => {
    const p = new URLSearchParams();
    writeFilters(p, make({ sort: 'oldest', page: 2 }));
    expect(p.get('sort')).toBe('oldest');
    expect(p.get('page')).toBe('2');
  });

  it('round-trips parse <-> write', () => {
    const original = make({
      jd_filename: 'be.docx',
      recommendation: 'Rejected',
      max_score: '90',
      date_from: '2026-01-01',
      page: 4,
    });
    const p = new URLSearchParams();
    writeFilters(p, original);
    expect(parseFilters(`?${p.toString()}`)).toEqual(original);
  });

  it('maps filters to the backend query with numeric scores and page_size', () => {
    const q = toQuery(
      make({ resume_filename: 'jane', min_score: '70', max_score: '90', recommendation: 'Selected', page: 2 }),
    );
    expect(q).toEqual({
      sort: 'newest',
      page: 2,
      page_size: PAGE_SIZE,
      resume_filename: 'jane',
      min_score: 70,
      max_score: 90,
      recommendation: 'Selected',
    });
  });

  it('omits empty score strings from the query', () => {
    const q = toQuery(make({ min_score: '', max_score: '' }));
    expect(q.min_score).toBeUndefined();
    expect(q.max_score).toBeUndefined();
  });

  it('builds removable chips and detects active state', () => {
    const f = make({ resume_filename: 'jane', recommendation: 'Selected' });
    expect(activeChips(f).map((c) => c.key)).toEqual(['resume_filename', 'recommendation']);
    expect(hasActiveFilters(f)).toBe(true);
    expect(hasActiveFilters(DEFAULT_FILTERS)).toBe(false);
  });
});
