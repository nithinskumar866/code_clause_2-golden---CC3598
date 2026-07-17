import type { FC } from 'react';

interface SpinnerProps {
  /** Sizing utilities, e.g. "h-4 w-4". */
  className?: string;
  label?: string;
}

/** Dependency-free CSS spinner. Announces itself for assistive tech. */
export const Spinner: FC<SpinnerProps> = ({ className = 'h-4 w-4', label = 'Loading' }) => (
  <span
    role="status"
    aria-label={label}
    className={`inline-block animate-spin rounded-full border-2 border-current border-r-transparent ${className}`}
  />
);
