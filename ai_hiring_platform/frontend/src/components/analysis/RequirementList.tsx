import type { FC } from 'react';
import type { RequirementFit } from '../../types';
import { StatusBadge } from './StatusBadge';

interface RequirementListProps {
  requirements: RequirementFit[];
}

/** Requirement-by-requirement evidence cards (Requirement → Evidence → Reasoning → Confidence). */
export const RequirementList: FC<RequirementListProps> = ({ requirements }) => (
  <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
    <h3 className="text-sm font-semibold text-white border-b border-white/5 pb-2">Skill Evidence Analysis</h3>
    <div className="divide-y divide-white/5">
      {requirements.map((req, idx) => (
        <div key={idx} className="py-4 first:pt-0 last:pb-0 space-y-2.5">
          <div className="flex justify-between items-start">
            <div>
              <span className="font-semibold text-sm text-gray-100 uppercase tracking-wide">{req.requirement}</span>
              <span className="ml-2.5 inline-flex items-center rounded bg-white/5 border border-white/10 px-1.5 py-0.5 text-[10px] font-medium text-gray-400">
                {req.category}
              </span>
            </div>

            <div className="flex items-center gap-2">
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
      ))}
    </div>
  </div>
);
