import type { FC } from 'react';
import { Brain } from 'lucide-react';

interface SkillRelationshipsProps {
  relationships: string[];
}

/** Transferable-skill relationships surfaced by the semantics service. */
export const SkillRelationships: FC<SkillRelationshipsProps> = ({ relationships }) => (
  <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3">
    <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-400 uppercase tracking-wider">
      <Brain className="h-4 w-4" /> Transferable Skill Relationships
    </div>
    <div className="space-y-2.5">
      {relationships.map((rel, idx) => (
        <p
          key={idx}
          className="text-xs text-gray-300 bg-black/20 p-2.5 rounded-lg border border-white/5 leading-relaxed"
        >
          {rel}
        </p>
      ))}
      {relationships.length === 0 && (
        <p className="text-xs text-gray-500 italic">No transferable skill dependencies detected.</p>
      )}
    </div>
  </div>
);
