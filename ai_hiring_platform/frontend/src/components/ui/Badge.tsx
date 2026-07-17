import type { FC, ReactNode } from 'react';

export type BadgeTone = 'neutral' | 'success' | 'warning' | 'danger' | 'info';

interface BadgeProps {
  tone?: BadgeTone;
  children: ReactNode;
  className?: string;
}

const TONES: Record<BadgeTone, string> = {
  neutral: 'text-gray-300 border-white/10 bg-white/5',
  success: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10',
  warning: 'text-amber-400 border-amber-500/20 bg-amber-500/10',
  danger: 'text-rose-400 border-rose-500/20 bg-rose-500/10',
  info: 'text-indigo-300 border-indigo-500/20 bg-indigo-500/10',
};

/** Small status pill. */
export const Badge: FC<BadgeProps> = ({ tone = 'neutral', children, className = '' }) => (
  <span
    className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${TONES[tone]} ${className}`}
  >
    {children}
  </span>
);
