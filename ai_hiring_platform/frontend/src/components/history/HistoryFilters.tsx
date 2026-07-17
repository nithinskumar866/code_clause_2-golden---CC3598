import { useEffect, useState, type FC } from 'react';
import { SlidersHorizontal, X } from 'lucide-react';
import type { HistoryFilters as Filters } from '../../lib/historyFilters';
import type { RecommendationValue } from '../../types';
import { useDebouncedValue } from '../../hooks/useDebouncedValue';
import { SearchBar } from '../ui/SearchBar';
import { Select } from '../ui/Select';
import type { SelectOption } from '../ui/Select';
import { Input } from '../ui/Input';

interface HistoryFiltersProps {
  filters: Filters;
  onChange: (partial: Partial<Filters>) => void;
  onClearAll: () => void;
  hasActive: boolean;
}

const RECOMMENDATION_OPTIONS: SelectOption[] = [
  { value: '', label: 'All recommendations' },
  { value: 'Selected', label: 'Selected' },
  { value: 'Borderline', label: 'Borderline' },
  { value: 'Rejected', label: 'Rejected' },
];

const SORT_OPTIONS: SelectOption[] = [
  { value: 'newest', label: 'Newest first' },
  { value: 'oldest', label: 'Oldest first' },
  { value: 'highest_score', label: 'Highest score' },
  { value: 'lowest_score', label: 'Lowest score' },
];

/** Recruiter search & filter panel. Text search is debounced; every other
 *  control commits immediately. All values map directly to backend params. */
export const HistoryFilters: FC<HistoryFiltersProps> = ({ filters, onChange, onClearAll, hasActive }) => {
  // Local, responsive text state; committed to filters after a debounce.
  const [resumeLocal, setResumeLocal] = useState(filters.resume_filename);
  const [jdLocal, setJdLocal] = useState(filters.jd_filename);

  useEffect(() => setResumeLocal(filters.resume_filename), [filters.resume_filename]);
  useEffect(() => setJdLocal(filters.jd_filename), [filters.jd_filename]);

  const debouncedResume = useDebouncedValue(resumeLocal, 400);
  const debouncedJd = useDebouncedValue(jdLocal, 400);

  useEffect(() => {
    if (debouncedResume !== filters.resume_filename) onChange({ resume_filename: debouncedResume });
  }, [debouncedResume, filters.resume_filename, onChange]);

  useEffect(() => {
    if (debouncedJd !== filters.jd_filename) onChange({ jd_filename: debouncedJd });
  }, [debouncedJd, filters.jd_filename, onChange]);

  return (
    <section
      aria-label="Search and filter analyses"
      className="space-y-4 rounded-xl border border-white/5 bg-card p-4"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-white">
          <SlidersHorizontal className="h-4 w-4 text-indigo-400" /> Filters
        </div>
        {hasActive && (
          <button
            type="button"
            onClick={onClearAll}
            className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-semibold text-gray-400 transition hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <X className="h-3.5 w-3.5" /> Clear all filters
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <SearchBar
          id="filter-resume"
          label="Resume filename"
          value={resumeLocal}
          onChange={setResumeLocal}
          placeholder="e.g. jane_doe.pdf"
        />
        <SearchBar
          id="filter-jd"
          label="JD filename"
          value={jdLocal}
          onChange={setJdLocal}
          placeholder="e.g. backend_engineer.docx"
        />
        <Select
          label="Recommendation"
          options={RECOMMENDATION_OPTIONS}
          value={filters.recommendation}
          onChange={(e) => onChange({ recommendation: e.target.value as RecommendationValue | '' })}
          className="w-full"
        />
        <Select
          label="Sort by"
          options={SORT_OPTIONS}
          value={filters.sort}
          onChange={(e) => onChange({ sort: e.target.value as Filters['sort'] })}
          className="w-full"
        />
        <Input
          id="filter-min-score"
          label="Min score"
          type="number"
          min={0}
          max={100}
          inputMode="numeric"
          placeholder="0"
          value={filters.min_score}
          onChange={(e) => onChange({ min_score: e.target.value })}
        />
        <Input
          id="filter-max-score"
          label="Max score"
          type="number"
          min={0}
          max={100}
          inputMode="numeric"
          placeholder="100"
          value={filters.max_score}
          onChange={(e) => onChange({ max_score: e.target.value })}
        />
        <Input
          id="filter-date-from"
          label="Date from"
          type="date"
          value={filters.date_from}
          onChange={(e) => onChange({ date_from: e.target.value })}
        />
        <Input
          id="filter-date-to"
          label="Date to"
          type="date"
          value={filters.date_to}
          onChange={(e) => onChange({ date_to: e.target.value })}
        />
      </div>
    </section>
  );
};
