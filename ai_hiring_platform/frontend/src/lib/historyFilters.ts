import type { HistoryQuery, HistorySortValue, RecommendationValue } from '../types';

/**
 * UI/URL representation of the history filters. Keys mirror the backend query
 * parameter names so mapping to the API and the URL is direct and lossless.
 * Text/number fields are held as strings for controlled inputs; `page` is a
 * number. Empty string / default `sort` / `page === 1` mean "not applied".
 */
export interface HistoryFilters {
  resume_filename: string;
  jd_filename: string;
  recommendation: RecommendationValue | '';
  min_score: string;
  max_score: string;
  date_from: string;
  date_to: string;
  sort: HistorySortValue;
  page: number;
}

export const PAGE_SIZE = 8;

export const DEFAULT_FILTERS: HistoryFilters = {
  resume_filename: '',
  jd_filename: '',
  recommendation: '',
  min_score: '',
  max_score: '',
  date_from: '',
  date_to: '',
  sort: 'newest',
  page: 1,
};

const SORT_VALUES: HistorySortValue[] = ['newest', 'oldest', 'highest_score', 'lowest_score'];
const RECOMMENDATIONS: RecommendationValue[] = ['Selected', 'Borderline', 'Rejected'];

/** Keys that appear as removable active-filter chips. */
export type ChipKey =
  | 'resume_filename'
  | 'jd_filename'
  | 'recommendation'
  | 'min_score'
  | 'max_score'
  | 'date_from'
  | 'date_to';

export interface FilterChip {
  key: ChipKey;
  label: string;
}

/** Parse filters from a URL query string (e.g. window.location.search). */
export function parseFilters(search: string): HistoryFilters {
  const p = new URLSearchParams(search);
  const sortRaw = p.get('sort') as HistorySortValue | null;
  const recRaw = p.get('recommendation') as RecommendationValue | null;
  const pageRaw = Number(p.get('page'));
  return {
    resume_filename: p.get('resume_filename') ?? '',
    jd_filename: p.get('jd_filename') ?? '',
    recommendation: recRaw && RECOMMENDATIONS.includes(recRaw) ? recRaw : '',
    min_score: p.get('min_score') ?? '',
    max_score: p.get('max_score') ?? '',
    date_from: p.get('date_from') ?? '',
    date_to: p.get('date_to') ?? '',
    sort: sortRaw && SORT_VALUES.includes(sortRaw) ? sortRaw : 'newest',
    page: Number.isFinite(pageRaw) && pageRaw >= 1 ? Math.floor(pageRaw) : 1,
  };
}

/** Write the active (non-default) filters onto a URLSearchParams instance. */
export function writeFilters(params: URLSearchParams, f: HistoryFilters): void {
  const set = (key: string, value: string) => {
    if (value) params.set(key, value);
    else params.delete(key);
  };
  set('resume_filename', f.resume_filename.trim());
  set('jd_filename', f.jd_filename.trim());
  set('recommendation', f.recommendation);
  set('min_score', f.min_score);
  set('max_score', f.max_score);
  set('date_from', f.date_from);
  set('date_to', f.date_to);
  if (f.sort !== 'newest') params.set('sort', f.sort);
  else params.delete('sort');
  if (f.page > 1) params.set('page', String(f.page));
  else params.delete('page');
}

/** Map UI filters to the backend query object (page_size is always applied). */
export function toQuery(f: HistoryFilters): HistoryQuery {
  const query: HistoryQuery = { sort: f.sort, page: f.page, page_size: PAGE_SIZE };
  if (f.resume_filename.trim()) query.resume_filename = f.resume_filename.trim();
  if (f.jd_filename.trim()) query.jd_filename = f.jd_filename.trim();
  if (f.recommendation) query.recommendation = f.recommendation;
  const min = Number.parseInt(f.min_score, 10);
  if (!Number.isNaN(min)) query.min_score = min;
  const max = Number.parseInt(f.max_score, 10);
  if (!Number.isNaN(max)) query.max_score = max;
  if (f.date_from) query.date_from = f.date_from;
  if (f.date_to) query.date_to = f.date_to;
  return query;
}

/** Build the list of active filter chips for display/removal. */
export function activeChips(f: HistoryFilters): FilterChip[] {
  const chips: FilterChip[] = [];
  if (f.resume_filename.trim()) chips.push({ key: 'resume_filename', label: `Resume: ${f.resume_filename.trim()}` });
  if (f.jd_filename.trim()) chips.push({ key: 'jd_filename', label: `JD: ${f.jd_filename.trim()}` });
  if (f.recommendation) chips.push({ key: 'recommendation', label: `Recommendation: ${f.recommendation}` });
  if (f.min_score !== '') chips.push({ key: 'min_score', label: `Min score: ${f.min_score}` });
  if (f.max_score !== '') chips.push({ key: 'max_score', label: `Max score: ${f.max_score}` });
  if (f.date_from) chips.push({ key: 'date_from', label: `From: ${f.date_from}` });
  if (f.date_to) chips.push({ key: 'date_to', label: `To: ${f.date_to}` });
  return chips;
}

export function hasActiveFilters(f: HistoryFilters): boolean {
  return activeChips(f).length > 0;
}
