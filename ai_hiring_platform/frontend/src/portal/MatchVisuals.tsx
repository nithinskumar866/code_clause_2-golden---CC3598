import type { FC, ReactNode } from 'react';
import { scoreTone } from './match-tone';
import type { ScoringMode } from './types';

/**
 * The pieces both match cards are drawn from.
 *
 * They live here rather than in either card because the computed score and the
 * reasoned one must LOOK like the same kind of claim — same ring, same bands,
 * same colours at the same numbers. Two cards with their own palettes would make
 * a 62 read as better or worse depending only on which scorer produced it, which
 * is exactly the comparison the mode switch exists to let someone make honestly.
 */

interface ScoreRingProps {
  score: number;
  /** Diameter in pixels. */
  size?: number;
  label?: string;
}

/**
 * The fit percentage as a ring.
 *
 * A ring rather than a bare number because a percentage on its own gives no sense
 * of where it sits — 46% reads as a passing grade to anyone who has taken an exam,
 * and it is not one. The arc shows how much of the whole it actually is.
 */
export const ScoreRing: FC<ScoreRingProps> = ({ score, size = 56, label }) => {
  const tone = scoreTone(score);
  const bounded = Math.min(100, Math.max(0, score));
  const stroke = size >= 56 ? 5 : 4;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden="true">
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" strokeWidth={stroke} className="stroke-white/10"
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" strokeWidth={stroke} strokeLinecap="round"
          className={`${tone.ring} transition-[stroke-dashoffset] duration-700 ease-out`}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - bounded / 100)}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`font-bold leading-none ${tone.text} ${size >= 56 ? 'text-base' : 'text-sm'}`}>
          {Math.round(bounded)}
        </span>
        {label && <span className="text-[8px] uppercase tracking-wide text-gray-500">{label}</span>}
      </div>
    </div>
  );
};

/**
 * The badge that says which scorer produced the number beside it.
 *
 * Never optional. The two modes disagree by design, and a card that showed only
 * the percentage would let a candidate compare a reasoned 71 against a computed
 * 43 as though one role were better than the other.
 */
export const ModeBadge: FC<{ mode: ScoringMode }> = ({ mode }) => {
  const badge = {
    reasoned: {
      label: 'AI reasoned',
      title: 'Judged by the model against this posting’s stated requirements',
      tone: 'border-violet-500/25 bg-violet-500/10 text-violet-300',
    },
    rag: {
      label: 'RAG',
      title: 'Retrieved the CV passages bearing on each requirement, then judged only those',
      tone: 'border-teal-500/25 bg-teal-500/10 text-teal-300',
    },
    computed: {
      label: 'Computed',
      title: 'Computed from measured similarity, skill overlap, title and experience',
      tone: 'border-white/10 bg-white/5 text-gray-400',
    },
  }[mode] ?? {
    label: mode,
    title: '',
    tone: 'border-white/10 bg-white/5 text-gray-400',
  };

  return (
    <span
      title={badge.title}
      className={`inline-flex shrink-0 items-center gap-1 rounded-md border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide ${badge.tone}`}
    >
      {badge.label}
    </span>
  );
};

/** A titled block inside a card, with a hairline above it. */
export const CardSection: FC<{ title: string; aside?: ReactNode; children: ReactNode }> = ({
  title, aside, children,
}) => (
  <section className="border-t border-white/5 pt-3">
    <div className="mb-2 flex items-center justify-between gap-2">
      <h4 className="text-[10px] font-semibold uppercase tracking-wider text-gray-500">{title}</h4>
      {aside}
    </div>
    {children}
  </section>
);
