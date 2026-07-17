import type { FC } from 'react';
import { CheckCircle2, HelpCircle, AlertCircle } from 'lucide-react';

interface StatusBadgeProps {
  status: string; // "Matched" | "Partial" | "Missing"
}

/** Coloured pill for a requirement's fit status. */
export const StatusBadge: FC<StatusBadgeProps> = ({ status }) => {
  switch (status) {
    case 'Matched':
      return (
        <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 text-xs font-semibold text-emerald-400">
          <CheckCircle2 className="h-3 w-3" /> Matched
        </span>
      );
    case 'Partial':
      return (
        <span className="inline-flex items-center gap-1 rounded-md bg-amber-500/10 border border-amber-500/20 px-2.5 py-0.5 text-xs font-semibold text-amber-400">
          <HelpCircle className="h-3 w-3" /> Partial
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 rounded-md bg-rose-500/10 border border-rose-500/20 px-2.5 py-0.5 text-xs font-semibold text-rose-400">
          <AlertCircle className="h-3 w-3" /> Missing
        </span>
      );
  }
};
