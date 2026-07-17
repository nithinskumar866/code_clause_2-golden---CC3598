import type { FC, ReactNode } from 'react';

interface StatCardProps {
  icon: ReactNode;
  /** Icon chip color classes, e.g. "bg-cyan-500/10 text-cyan-400". */
  iconClass?: string;
  label: string;
  value: string;
  badge?: ReactNode;
}

/** Compact metric / status tile for dashboards. */
export const StatCard: FC<StatCardProps> = ({
  icon,
  iconClass = 'bg-indigo-500/10 text-indigo-400',
  label,
  value,
  badge,
}) => (
  <div className="rounded-xl border border-white/5 bg-card p-5 transition hover:border-white/10">
    <div className="flex items-center justify-between">
      <div className={`rounded-lg p-2.5 ${iconClass}`}>{icon}</div>
      {badge}
    </div>
    <div className="mt-4">
      <h3 className="text-sm font-medium text-gray-400">{label}</h3>
      <p className="mt-1 text-xl font-semibold text-white">{value}</p>
    </div>
  </div>
);
