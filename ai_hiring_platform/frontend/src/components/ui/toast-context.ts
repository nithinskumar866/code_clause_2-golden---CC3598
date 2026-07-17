import { createContext, useContext } from 'react';

export type ToastVariant = 'success' | 'error' | 'info' | 'warning';

export interface ToastOptions {
  message: string;
  title?: string;
  variant?: ToastVariant;
  /** Auto-dismiss after ms. 0 keeps it until dismissed. Default 4000. */
  duration?: number;
}

export interface ToastApi {
  toast: (opts: ToastOptions) => void;
  success: (message: string, title?: string) => void;
  error: (message: string, title?: string) => void;
  info: (message: string, title?: string) => void;
  warning: (message: string, title?: string) => void;
}

export const ToastContext = createContext<ToastApi | null>(null);

/** Access the toast API. Must be called within a ToastProvider. */
export const useToast = (): ToastApi => {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return ctx;
};
