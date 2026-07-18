import type { FC } from 'react';
import { AlertTriangle } from 'lucide-react';
import type { RequirementFit } from '../../types';
import { StatusBadge } from './StatusBadge';

interface RequirementListProps {
  requirements: RequirementFit[];
  /** Skills the authenticity check flagged as listed-but-not-demonstrated. */
  overClaimedSkills?: string[];
}

/** Must-have vs nice-to-have pill (F2). Absent importance renders nothing. */
const ImportanceBadge: FC<{ importance?: 'must' | 'nice' | null; weight?: number | null }> = ({
  importance,
  weight,
}) => {
  if (!importance) return null;
  const isMust = importance === 'must';
  const style = isMust
    ? 'text-indigo-300 border-indigo-500/30 bg-indigo-500/10'
    : 'text-gray-400 border-white/10 bg-white/5';
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${style}`}
      title={weight != null ? `Scoring weight ${weight}` : undefined}
    >
      {isMust ? 'Must-have' : 'Nice-to-have'}
    </span>
  );
};

/**
 * Requirement-by-requirement evidence cards (Requirement → Evidence → Reasoning →
 * Confidence). Must-haves sort first (they drive the importance-weighted coverage
 * score); over-claimed skills are flagged inline.
 */
export const RequirementList: FC<RequirementListProps> = ({ requirements, overClaimedSkills = [] }) => {
  const overClaimed = new Set(overClaimedSkills.map((s) => s.toLowerCase()));
  // Stable sort: must-haves before nice-to-haves, otherwise preserve original order.
  const ordered = requirements
    .map((req, idx) => ({ req, idx }))
    .sort((a, b) => {
      const rank = (r: RequirementFit) => (r.importance === 'nice' ? 1 : 0);
      return rank(a.req) - rank(b.req) || a.idx - b.idx;
    })
    .map((x) => x.req);

  return (
    <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
      <h3 className="text-sm font-semibold text-white border-b border-white/5 pb-2">Skill Evidence Analysis</h3>
      <div className="divide-y divide-white/5">
        {ordered.map((req, idx) => {
          const isOverClaimed = overClaimed.has(req.requirement.toLowerCase());
          return (
            <div key={idx} className="py-4 first:pt-0 last:pb-0 space-y-2.5">
              <div className="flex justify-between items-start gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-sm text-gray-100 uppercase tracking-wide">
                    {req.requirement}
                  </span>
                  <span className="inline-flex items-center rounded bg-white/5 border border-white/10 px-1.5 py-0.5 text-[10px] font-medium text-gray-400">
                    {req.category}
                  </span>
                  <ImportanceBadge importance={req.importance} weight={req.weight} />
                  {isOverClaimed && (
                    <span
                      className="inline-flex items-center gap-1 rounded border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-rose-300"
                      title="Listed but not demonstrated in Experience/Projects"
                    >
                      <AlertTriangle className="h-3 w-3" /> Listed, not demonstrated
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-[10px] text-gray-400 font-medium">Confidence: {req.confidence}%</span>
                  <StatusBadge status={req.status} />
                </div>
              </div>

              <div className="space-y-1">
                <p className="text-xs text-gray-300">
                  <span className="text-indigo-400 font-semibold">Relevance:</span> {req.explanation}
                </p>
                {req.limitations && req.limitations !== 'None' && (
                  <p className="text-xs text-gray-400">
                    <span className="text-rose-400 font-semibold">Limitations:</span> {req.limitations}
                  </p>
                )}
              </div>

              {req.matched_evidence && (
                <div className="rounded-lg bg-black/40 border border-white/5 p-3 text-xs font-mono text-indigo-300 leading-relaxed">
                  "{req.matched_evidence}"
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
