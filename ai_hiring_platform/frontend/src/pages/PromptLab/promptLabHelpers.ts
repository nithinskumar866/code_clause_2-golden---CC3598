import type { CaseExpectations, PromptCase } from '../../types';

/**
 * Shared, non-component pieces of the Prompt Lab.
 *
 * Split out from the components purely so each .tsx file exports components only —
 * a module that mixes the two breaks fast refresh for the whole file.
 */

/** Green above 90, amber above 70, red below — the same reading everywhere on the page. */
export const scoreTone = (score: number | null): 'success' | 'warning' | 'danger' | 'neutral' => {
  if (score === null) return 'neutral';
  if (score >= 90) return 'success';
  if (score >= 70) return 'warning';
  return 'danger';
};

/** A null score means no rule applied — shown as a dash, never as 0% or 100%. */
export const pct = (score: number | null) =>
  score === null ? '—' : `${score.toFixed(score % 1 === 0 ? 0 : 1)}%`;

export const EMPTY_EXPECTATIONS: CaseExpectations = {
  max_lines: 2,
  max_chars: null,
  link_allowed: false,
  link_required: false,
  years_known: false,
  expected_years: null,
  jobs_requested: false,
  list_allowed: false,
  skills_requested: false,
  must_contain: [],
  must_not_contain: [],
};

/**
 * The arrays are rebuilt rather than spread: a shallow copy would hand every case the
 * SAME `must_contain` array, so typing a required string into one case would silently
 * add it to all of them.
 */
export const blankCase = (index: number): PromptCase => ({
  id: `case-${index}`,
  name: `Case ${index}`,
  question: '',
  context: '',
  history: [],
  expectations: {
    ...EMPTY_EXPECTATIONS,
    must_contain: [],
    must_not_contain: [],
  },
});
