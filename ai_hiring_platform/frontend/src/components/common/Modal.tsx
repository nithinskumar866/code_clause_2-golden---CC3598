import { useEffect, type FC, type ReactNode } from 'react';
import { X } from 'lucide-react';

interface ModalProps {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  /** id of the element that labels the dialog, for a11y. */
  labelledBy?: string;
  /** Tailwind max-width utility for the panel (e.g. "max-w-md", "max-w-5xl"). */
  widthClass?: string;
}

/**
 * Generic centered modal: dimmed backdrop, Escape / backdrop-click to close,
 * scrollable panel. Base for ConfirmDialog and the History report modal.
 */
export const Modal: FC<ModalProps> = ({ open, onClose, children, labelledBy, widthClass = 'max-w-md' }) => {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby={labelledBy}
    >
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div
        className={`relative z-10 w-full ${widthClass} max-h-[90vh] overflow-y-auto rounded-xl border border-white/10 bg-card shadow-2xl shadow-black/40`}
      >
        <button
          onClick={onClose}
          className="absolute right-3 top-3 z-20 rounded-lg p-1.5 text-gray-400 hover:bg-white/5 hover:text-white transition"
          aria-label="Close dialog"
        >
          <X className="h-4 w-4" />
        </button>
        {children}
      </div>
    </div>
  );
};
