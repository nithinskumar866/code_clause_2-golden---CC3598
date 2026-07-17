import type { FC } from 'react';
import { BookOpen } from 'lucide-react';
import type { LearningRoadmapItem } from '../../types';

interface LearningRoadmapProps {
  items: LearningRoadmapItem[];
}

/** Personalized upskilling roadmap estimated from missing concepts. */
export const LearningRoadmap: FC<LearningRoadmapProps> = ({ items }) => (
  <div className="rounded-xl border border-white/5 bg-card p-5 space-y-4">
    <div className="flex items-center gap-1.5 text-sm font-semibold text-indigo-400">
      <BookOpen className="h-4 w-4" /> Personalized Learning Roadmap
    </div>
    <p className="text-xs text-gray-400">
      Dynamic upskilling roadmaps estimated by analyzing technology overlap and missing concepts:
    </p>

    <div className="space-y-3">
      {items.map((item, idx) => (
        <div key={idx} className="border-l-2 border-indigo-500/40 pl-3 space-y-0.5">
          <div className="flex justify-between items-center text-xs">
            <span className="font-semibold text-gray-200 uppercase">{item.skill}</span>
            <span className="text-[10px] bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 rounded px-1.5 py-0.5 font-semibold">
              {item.estimated_time}
            </span>
          </div>
          <p className="text-[11px] text-gray-400">{item.reason}</p>
        </div>
      ))}
      {items.length === 0 && <p className="text-xs text-gray-500 italic">No upskilling items required.</p>}
    </div>
  </div>
);
