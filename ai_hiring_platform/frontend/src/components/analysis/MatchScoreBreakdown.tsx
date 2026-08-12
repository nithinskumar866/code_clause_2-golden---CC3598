import type { FC } from 'react';
import { Info } from 'lucide-react';
import type { MatchScore } from '../../types';
import { getScoreBarBg } from './scoreColors';

interface MatchScoreBreakdownProps {
  match: MatchScore;
}

/**
 * The nine parameters behind a Match Score.
 *
 * The number itself is arithmetic, so the only thing worth showing is how it was
 * reached. Every row carries three facts a recruiter can act on: what the parameter
 * scored, how much of the final score it controls, and the sentence explaining the
 * result. That last part is why the rows are not a chart — "83.3%" tells nobody
 * which skill was missing.
 *
 * Parameters the job description never stated are marked rather than hidden. A
 * neutral 50 that looks like a measurement is misleading; a neutral 50 labelled
 * "not stated" tells the recruiter their JD is what is vague, not the candidate.
 */
export const MatchScoreBreakdown: FC<MatchScoreBreakdownProps> = ({ match }) => {
  const neutralCount = match.parameters.filter((p) => p.neutral).length;
  const neutralWeight = match.parameters
    .filter((p) => p.neutral)
    .reduce((sum, p) => sum + p.weight, 0);

  return (
    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3.5">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-white/5 pb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-white">
          Match Score Breakdown
        </h3>
        <span className="text-[10px] text-gray-500">
          9 weighted parameters · deterministic
        </span>
      </div>

      <ul className="space-y-3">
        {match.parameters.map((p) => (
          <li key={p.key} className="space-y-1">
            <div className="flex items-baseline justify-between gap-2 text-xs">
              <span className="font-medium text-gray-200">
                {p.label}
                <span className="ml-1.5 text-[10px] font-normal text-gray-500">
                  {Math.round(p.weight * 100)}%
                </span>
                {p.neutral && (
                  <span className="ml-1.5 rounded border border-white/10 px-1 py-px text-[9px] uppercase tracking-wide text-gray-500">
                    not stated
                  </span>
                )}
              </span>
              <span className="shrink-0 tabular-nums text-gray-400">
                <span className="font-semibold text-white">{p.score.toFixed(1)}</span>
                <span className="text-[10px]"> → +{p.contribution.toFixed(1)}</span>
              </span>
            </div>

            <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
              <div
                className={`h-full ${p.neutral ? 'bg-gray-600' : getScoreBarBg(p.score)}`}
                style={{ width: `${Math.max(0, Math.min(100, p.score))}%` }}
              />
            </div>

            <p className="text-[11px] leading-relaxed text-gray-500">{p.basis}</p>
          </li>
        ))}
      </ul>

      <div className="flex items-baseline justify-between border-t border-white/5 pt-2.5 text-xs">
        <span className="font-semibold uppercase tracking-wide text-gray-400">
          Match Score
        </span>
        <span className="tabular-nums text-base font-bold text-white">
          {match.score.toFixed(1)}
          <span className="ml-1.5 text-[11px] font-medium text-gray-400">{match.band}</span>
        </span>
      </div>

      {/* When a JD is vague, say so here rather than letting the recruiter read a
          compressed score as a weak candidate. */}
      {neutralCount > 0 && (
        <p className="flex gap-1.5 rounded-lg border border-amber-500/20 bg-amber-500/5 px-2.5 py-2 text-[11px] leading-relaxed text-amber-300/90">
          <Info className="mt-px h-3 w-3 shrink-0" />
          <span>
            {neutralCount} of 9 parameters ({Math.round(neutralWeight * 100)}% of the score) scored
            neutrally because the job description did not state them. Adding those details to the JD
            will separate candidates more sharply.
          </span>
        </p>
      )}
    </div>
  );
};
