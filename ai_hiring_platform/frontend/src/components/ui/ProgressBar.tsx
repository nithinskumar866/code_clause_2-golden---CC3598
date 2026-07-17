import type { FC } from 'react';

interface ProgressBarProps {
  /** 0–100. */
  value: number;
  /** Fill color utility, e.g. "bg-emerald-500". */
  barClass?: string;
  /** Accessible label describing what the bar measures. */
  label?: string;
  className?: string;
}

/** Accessible horizontal progress/meter bar. */
export const ProgressBar: FC<ProgressBarProps> = ({
  value,
  barClass = 'bg-indigo-500',
  label,
  className = 'h-1.5',
}) => {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div
      className={`w-full overflow-hidden rounded-full bg-white/5 ${className}`}
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div className={`h-full rounded-full ${barClass}`} style={{ width: `${pct}%` }} />
    </div>
  );
};
