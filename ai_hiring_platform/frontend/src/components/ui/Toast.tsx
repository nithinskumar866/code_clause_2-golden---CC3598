import { useCallback, useMemo, useRef, useState, type FC, type ReactNode } from 'react';
import { CheckCircle2, AlertCircle, AlertTriangle, Info, X } from 'lucide-react';
import { ToastContext } from './toast-context';
import type { ToastApi, ToastOptions, ToastVariant } from './toast-context';

interface ToastItem {
  id: number;
  message: string;
  title?: string;
  variant: ToastVariant;
  duration: number;
}

const ICONS: Record<ToastVariant, typeof Info> = {
  success: CheckCircle2,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

const ACCENT: Record<ToastVariant, string> = {
  success: 'text-emerald-400',
  error: 'text-rose-400',
  warning: 'text-amber-400',
  info: 'text-indigo-400',
};

const ToastCard: FC<{ item: ToastItem; onDismiss: (id: number) => void }> = ({ item, onDismiss }) => {
  const Icon = ICONS[item.variant];
  return (
    <div
      role={item.variant === 'error' ? 'alert' : 'status'}
      className="pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-xl border border-white/10 bg-card p-4 shadow-2xl shadow-black/40 animate-fadeIn"
    >
      <Icon className={`mt-0.5 h-5 w-5 shrink-0 ${ACCENT[item.variant]}`} />
      <div className="min-w-0 flex-1">
        {item.title && <p className="text-sm font-semibold text-white">{item.title}</p>}
        <p className="text-xs leading-relaxed text-gray-400">{item.message}</p>
      </div>
      <button
        type="button"
        onClick={() => onDismiss(item.id)}
        aria-label="Dismiss notification"
        className="rounded p-0.5 text-gray-500 transition hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
};

const ToastViewport: FC<{ items: ToastItem[]; onDismiss: (id: number) => void }> = ({ items, onDismiss }) => (
  <div
    className="pointer-events-none fixed inset-x-0 top-0 z-[60] flex flex-col items-center gap-2 p-4 sm:inset-x-auto sm:right-0 sm:items-end sm:p-6"
    aria-live="polite"
  >
    {items.map((item) => (
      <ToastCard key={item.id} item={item} onDismiss={onDismiss} />
    ))}
  </div>
);

/** App-wide toast provider. Wrap the application once, near the root. */
export const ToastProvider: FC<{ children: ReactNode }> = ({ children }) => {
  const [items, setItems] = useState<ToastItem[]>([]);
  const idRef = useRef(0);

  const remove = useCallback((id: number) => {
    setItems((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (opts: ToastOptions) => {
      idRef.current += 1;
      const id = idRef.current;
      const duration = opts.duration ?? 4000;
      setItems((prev) => [
        ...prev,
        { id, message: opts.message, title: opts.title, variant: opts.variant ?? 'info', duration },
      ]);
      if (duration > 0) {
        window.setTimeout(() => remove(id), duration);
      }
    },
    [remove],
  );

  const api = useMemo<ToastApi>(
    () => ({
      toast: push,
      success: (message, title) => push({ message, title, variant: 'success' }),
      error: (message, title) => push({ message, title, variant: 'error' }),
      info: (message, title) => push({ message, title, variant: 'info' }),
      warning: (message, title) => push({ message, title, variant: 'warning' }),
    }),
    [push],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      <ToastViewport items={items} onDismiss={remove} />
    </ToastContext.Provider>
  );
};
