/**
 * Shared, deterministic score → presentation mappings for the hiring report.
 * Kept in one place so every card, bar and chart uses identical thresholds.
 *
 * The breakpoints mirror the backend's Match Score interpretation bands
 * (settings.MATCH_SCORE_BANDS): 85 Excellent · 70 Strong · 50 Moderate · below Weak.
 * They are duplicated here rather than fetched because they are presentation, but
 * they must not drift — a card coloured red while the report reads "Strong fit"
 * shows the recruiter two different verdicts on one candidate.
 */

/**
 * A score as it must be written everywhere: one decimal place.
 *
 * That precision is the agreed reporting contract, so it cannot vary by screen.
 * Rendering the raw number would print "80" on one card and "78.4" on the next,
 * and two people comparing candidates would not know whether "80" meant 80.0 or a
 * rounded 79.5.
 */
export const formatScore = (score: number): string => score.toFixed(1);

export const getScoreColor = (score: number): string => {
  if (score >= 70) return 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10';
  if (score >= 50) return 'text-amber-400 border-amber-500/20 bg-amber-500/10';
  return 'text-rose-400 border-rose-500/20 bg-rose-500/10';
};

export const getScoreBarBg = (score: number): string => {
  if (score >= 70) return 'bg-emerald-500';
  if (score >= 50) return 'bg-amber-500';
  return 'bg-rose-500';
};

/**
 * The band name for a score. Mirrors the backend's `MatchScore.band`, and is used
 * only where that field is unavailable (older reports, leaderboard rows).
 */
export const getScoreLabel = (score: number): string => {
  if (score >= 85) return 'Excellent fit';
  if (score >= 70) return 'Strong fit';
  if (score >= 50) return 'Moderate fit';
  return 'Weak fit';
};

/** Raw hex equivalents of the score thresholds, for SVG stroke/fill. */
export const scoreHex = (score: number): string => {
  if (score >= 70) return '#34d399'; // emerald-400
  if (score >= 50) return '#fbbf24'; // amber-400
  return '#fb7185'; // rose-400
};

/** Requirement-status palette (Matched / Partial / Missing). */
export const REQUIREMENT_STATUS_HEX: Record<string, string> = {
  Matched: '#34d399', // emerald-400
  Partial: '#fbbf24', // amber-400
  Missing: '#fb7185', // rose-400
};

/** Hiring outcome bucket derived from the overall score (same thresholds). */
export type FitCategory = 'Selected' | 'Borderline' | 'Rejected';

export const classifyFit = (score: number): FitCategory =>
  score >= 70 ? 'Selected' : score >= 50 ? 'Borderline' : 'Rejected';

/** Badge styling per outcome bucket. */
export const FIT_CATEGORY_STYLE: Record<FitCategory, string> = {
  Selected: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10',
  Borderline: 'text-amber-400 border-amber-500/20 bg-amber-500/10',
  Rejected: 'text-rose-400 border-rose-500/20 bg-rose-500/10',
};
