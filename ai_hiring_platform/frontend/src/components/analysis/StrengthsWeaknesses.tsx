import type { FC } from 'react';

interface StrengthsWeaknessesProps {
  strengths: string[];
  weaknesses: string[];
}

/** Two-column strengths vs. weaknesses/gaps lists. */
export const StrengthsWeaknesses: FC<StrengthsWeaknessesProps> = ({ strengths, weaknesses }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3">
      <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider border-b border-white/5 pb-1">
        Candidate Strengths
      </h3>
      <ul className="space-y-2 text-xs text-gray-300 list-disc list-inside">
        {strengths.map((str, idx) => (
          <li key={idx} className="leading-relaxed">
            {str}
          </li>
        ))}
      </ul>
    </div>

    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3">
      <h3 className="text-xs font-semibold text-rose-400 uppercase tracking-wider border-b border-white/5 pb-1">
        Candidate Weaknesses / Gaps
      </h3>
      <ul className="space-y-2 text-xs text-gray-300 list-disc list-inside">
        {weaknesses.map((weak, idx) => (
          <li key={idx} className="leading-relaxed">
            {weak}
          </li>
        ))}
      </ul>
    </div>
  </div>
);
