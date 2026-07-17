import type { FC, ReactNode } from 'react';

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

/** Reusable empty-state panel (no data / no results). */
export const EmptyState: FC<EmptyStateProps> = ({ icon, title, description, action }) => (
  <div className="flex flex-col items-center justify-center rounded-xl border border-white/5 bg-card px-6 py-20 text-center">
    {icon && <div className="mb-4 text-gray-500">{icon}</div>}
    <h3 className="text-base font-medium text-white">{title}</h3>
    {description && <p className="mt-1 max-w-sm text-sm text-gray-400">{description}</p>}
    {action && <div className="mt-5">{action}</div>}
  </div>
);
