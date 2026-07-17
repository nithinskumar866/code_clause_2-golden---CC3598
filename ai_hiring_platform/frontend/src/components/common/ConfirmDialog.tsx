import type { FC } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Modal } from './Modal';

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  loading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Destructive-action confirmation modal (delete / clear-all). */
export const ConfirmDialog: FC<ConfirmDialogProps> = ({
  open,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  loading = false,
  onConfirm,
  onCancel,
}) => (
  <Modal open={open} onClose={onCancel} widthClass="max-w-md" labelledBy="confirm-dialog-title">
    <div className="p-6 space-y-4">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400">
          <AlertTriangle className="h-5 w-5" />
        </div>
        <h3 id="confirm-dialog-title" className="text-base font-semibold text-white">
          {title}
        </h3>
      </div>
      <p className="text-sm leading-relaxed text-gray-400">{message}</p>
      <div className="flex justify-end gap-3 pt-2">
        <button
          onClick={onCancel}
          disabled={loading}
          className="rounded-lg border border-white/10 px-4 py-2 text-sm font-semibold text-gray-300 hover:bg-white/5 disabled:opacity-50 transition"
        >
          {cancelLabel}
        </button>
        <button
          onClick={onConfirm}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-500 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <AlertTriangle className="h-4 w-4" />}
          {confirmLabel}
        </button>
      </div>
    </div>
  </Modal>
);
