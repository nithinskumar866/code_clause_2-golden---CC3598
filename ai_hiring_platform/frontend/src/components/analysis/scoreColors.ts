/**
 * Shared, deterministic score → presentation mappings for the hiring report.
 * Kept in one place so every card, bar and chart uses identical thresholds.
 *
 * Thresholds: >= 75 strong (emerald) · >= 60 conditional (amber) · else weak (rose).
 */

export const getScoreColor = (score: number): string => {
  if (score >= 75) return 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10';
  if (score >= 60) return 'text-amber-400 border-amber-500/20 bg-amber-500/10';
  return 'text-rose-400 border-rose-500/20 bg-rose-500/10';
};

export const getScoreBarBg = (score: number): string => {
  if (score >= 75) return 'bg-emerald-500';
  if (score >= 60) return 'bg-amber-500';
  return 'bg-rose-500';
};

export const getScoreLabel = (score: number): string =>
  score >= 75 ? 'Recommended Fit' : score >= 60 ? 'Conditional Fit' : 'Not Recommended';

/** Raw hex equivalents of the score thresholds, for SVG stroke/fill. */
export const scoreHex = (score: number): string => {
  if (score >= 75) return '#34d399'; // emerald-400
  if (score >= 60) return '#fbbf24'; // amber-400
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
  score >= 75 ? 'Selected' : score >= 60 ? 'Borderline' : 'Rejected';

/** Badge styling per outcome bucket. */
export const FIT_CATEGORY_STYLE: Record<FitCategory, string> = {
  Selected: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10',
  Borderline: 'text-amber-400 border-amber-500/20 bg-amber-500/10',
  Rejected: 'text-rose-400 border-rose-500/20 bg-rose-500/10',
};
