import type { FC } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from './Button';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  retrying?: boolean;
}

/** Reusable error panel with an optional retry action. */
export const ErrorState: FC<ErrorStateProps> = ({
  title = 'Something went wrong',
  message,
  onRetry,
  retrying = false,
}) => (
  <div
    className="flex flex-col items-center justify-center rounded-xl border border-rose-500/20 bg-card px-6 py-20 text-center"
    role="alert"
  >
    <AlertTriangle className="mb-4 h-9 w-9 text-rose-400" />
    <h3 className="text-base font-medium text-white">{title}</h3>
    <p className="mt-1 max-w-sm text-sm text-gray-400">{message}</p>
    {onRetry && (
      <Button
        className="mt-5"
        onClick={onRetry}
        loading={retrying}
        leftIcon={<RefreshCw className="h-4 w-4" />}
      >
        Retry
      </Button>
    )}
  </div>
);
