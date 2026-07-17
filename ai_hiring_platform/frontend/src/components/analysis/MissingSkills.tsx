import type { FC } from 'react';

interface MissingSkillsProps {
  skills: string[];
}

/** Chips of technologies with no structural evidence in the resume. */
export const MissingSkills: FC<MissingSkillsProps> = ({ skills }) => (
  <div className="rounded-xl border border-white/5 bg-card p-5 space-y-4">
    <h3 className="text-sm font-semibold text-white border-b border-white/5 pb-2">Missing Skills</h3>
    {skills.length === 0 ? (
      <p className="text-xs text-emerald-400 bg-emerald-500/5 border border-emerald-500/10 p-3 rounded-lg">
        Excellent! Candidate covers all required technologies structurally.
      </p>
    ) : (
      <div className="flex flex-wrap gap-2">
        {skills.map((skill, idx) => (
          <span
            key={idx}
            className="rounded px-2.5 py-1 text-xs font-semibold bg-rose-500/15 border border-rose-500/30 text-rose-400 uppercase"
          >
            {skill}
          </span>
        ))}
      </div>
    )}
  </div>
);
