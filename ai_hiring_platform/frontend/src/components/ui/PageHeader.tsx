import type { FC, ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  description?: string;
  /** Primary/secondary actions rendered on the right (desktop) / below (mobile). */
  actions?: ReactNode;
  icon?: ReactNode;
}

/** Consistent page heading used by every route. Renders the page's single h1. */
export const PageHeader: FC<PageHeaderProps> = ({ title, description, actions, icon }) => (
  <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
    <div className="flex items-start gap-3">
      {icon && <div className="mt-0.5 rounded-lg bg-indigo-500/10 p-2 text-indigo-400">{icon}</div>}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">{title}</h1>
        {description && <p className="mt-1.5 max-w-2xl text-sm text-gray-400">{description}</p>}
      </div>
    </div>
    {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
  </div>
);
