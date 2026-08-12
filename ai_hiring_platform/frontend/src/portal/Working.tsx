import { useEffect, useState, type FC } from 'react';

/**
 * The waiting state, for work that takes long enough to need one.
 *
 * Two rules, both of which come from the platform's narration principle — the
 * visible reasoning reports what the pipeline actually did, and "a fixed script
 * on a timer would be theatre":
 *
 *  1. The label is passed in by whoever knows what is really happening. This
 *     component never invents stages, and never advances through a made-up list
 *     to look busy.
 *  2. The elapsed counter is real. It appears only once the wait has gone past
 *     the point where a person starts wondering whether anything is happening,
 *     and it is the honest answer to "is this stuck?".
 */
interface WorkingProps {
  label: string;
  /** The one true detail, when there is one: "3 of 5 letters written". */
  detail?: string;
  /** Shows a seconds counter once the wait passes this. */
  elapsedAfterMs?: number;
  className?: string;
}

export const Working: FC<WorkingProps> = ({
  label, detail, elapsedAfterMs = 4000, className = '',
}) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Date.now() - started), 500);
    return () => clearInterval(timer);
  }, []);

  const showElapsed = elapsed >= elapsedAfterMs;

  return (
    <div
      role="status"
      aria-live="polite"
      className={`flex items-start gap-3 rounded-xl border border-white/5 bg-white/5 px-3 py-2.5 ${className}`}
    >
      {/* Three dots on a stagger. Motion carries "still alive" without claiming
          progress the app cannot measure — a progress BAR here would be a lie,
          because none of this work reports a percentage. */}
      <span className="mt-1.5 flex shrink-0 gap-1" aria-hidden="true">
        {[0, 150, 300].map(delay => (
          <span
            key={delay}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-indigo-400"
            style={{ animationDelay: `${delay}ms`, animationDuration: '900ms' }}
          />
        ))}
      </span>

      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-200">{label}</p>
        {detail && <p className="mt-0.5 text-[11px] text-gray-500">{detail}</p>}
        {showElapsed && (
          <p className="mt-0.5 text-[11px] tabular-nums text-gray-600">
            {Math.round(elapsed / 1000)}s elapsed
          </p>
        )}
      </div>
    </div>
  );
};

/**
 * Placeholder rows for content whose shape is known before its content is.
 *
 * Used where a real list is coming, so the panel does not jump when it lands.
 */
export const SkeletonRows: FC<{ rows?: number; className?: string }> = ({
  rows = 3, className = '',
}) => (
  <div className={`space-y-2 ${className}`} aria-hidden="true">
    {Array.from({ length: rows }, (_, index) => (
      <div key={index} className="rounded-xl border border-white/5 bg-card p-4">
        <div className="mb-2 h-3 w-1/2 animate-pulse rounded bg-white/10" />
        <div className="mb-3 h-2 w-1/3 animate-pulse rounded bg-white/5" />
        <div className="h-16 animate-pulse rounded-lg bg-black/20" />
      </div>
    ))}
  </div>
);
