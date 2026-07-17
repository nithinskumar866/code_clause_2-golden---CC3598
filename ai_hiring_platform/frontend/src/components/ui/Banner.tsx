import type { FC, ReactNode } from 'react';
import { CheckCircle2, AlertCircle, AlertTriangle, Info, X } from 'lucide-react';

export type BannerVariant = 'success' | 'error' | 'warning' | 'info';

interface BannerProps {
  variant?: BannerVariant;
  title?: string;
  children: ReactNode;
  onDismiss?: () => void;
  className?: string;
}

const STYLES: Record<BannerVariant, { icon: typeof Info; box: string; icon_: string }> = {
  success: { icon: CheckCircle2, box: 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300', icon_: 'text-emerald-400' },
  error: { icon: AlertCircle, box: 'border-rose-500/20 bg-rose-500/10 text-rose-300', icon_: 'text-rose-400' },
  warning: { icon: AlertTriangle, box: 'border-amber-500/20 bg-amber-500/10 text-amber-300', icon_: 'text-amber-400' },
  info: { icon: Info, box: 'border-indigo-500/20 bg-indigo-500/10 text-indigo-300', icon_: 'text-indigo-400' },
};

/** Inline, persistent status message. Use toasts for transient feedback. */
export const Banner: FC<BannerProps> = ({ variant = 'info', title, children, onDismiss, className = '' }) => {
  const { icon: Icon, box, icon_ } = STYLES[variant];
  return (
    <div
      role={variant === 'error' ? 'alert' : 'status'}
      className={`flex items-start gap-3 rounded-lg border p-3 text-sm ${box} ${className}`}
    >
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${icon_}`} />
      <div className="min-w-0 flex-1">
        {title && <p className="font-semibold">{title}</p>}
        <div className="text-xs leading-relaxed opacity-90">{children}</div>
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className="rounded p-0.5 opacity-70 transition hover:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
};
