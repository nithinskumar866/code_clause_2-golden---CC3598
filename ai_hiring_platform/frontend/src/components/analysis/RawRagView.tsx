import type { FC } from 'react';
import type { RetrievalResult } from '../../types';

interface RawRagViewProps {
  results: RetrievalResult[];
}

/** Raw semantic retrieval matches per requirement (the evidence behind the report). */
export const RawRagView: FC<RawRagViewProps> = ({ results }) => (
  <div className="space-y-5">
    <p className="text-xs text-gray-400">
      Here are the semantic chunks matching the Job Description requirements extracted directly from the FAISS database
      index:
    </p>
    {results.map((result, idx) => (
      <div key={idx} className="rounded-xl border border-white/5 bg-card p-5 space-y-4 hover:border-white/10 transition">
        <div className="flex items-center justify-between">
          <span className="inline-flex items-center rounded-lg bg-indigo-500/10 px-2.5 py-1 text-xs font-semibold text-indigo-400 border border-indigo-500/20">
            {result.requirement}
          </span>
          <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider">
            {result.matches.length} matches
          </span>
        </div>

        {result.matches.length === 0 ? (
          <div className="text-xs text-gray-500 bg-black/25 rounded-lg p-3 italic">
            No direct semantic match found in resume sections.
          </div>
        ) : (
          <div className="space-y-3">
            {result.matches.map((match, mIdx) => (
              <div key={mIdx} className="rounded-lg bg-black/30 p-3.5 border border-white/5 space-y-2 text-xs">
                <div className="flex justify-between items-center text-[10px] font-semibold">
                  <div className="flex gap-2">
                    <span className="text-indigo-300">Section: {match.section}</span>
                    <span className="text-gray-500">•</span>
                    <span className="text-cyan-400">Page {match.page}</span>
                  </div>
                  <span className="px-2 py-0.5 rounded border text-gray-400 border-white/10 bg-white/5">
                    Score: {match.score}
                  </span>
                </div>
                <p className="text-gray-300 leading-relaxed font-sans font-normal whitespace-pre-wrap">
                  "{match.chunk}"
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    ))}
  </div>
);
