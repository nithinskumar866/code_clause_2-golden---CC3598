import type { FC } from 'react';
import { AlertCircle } from 'lucide-react';
import type { AnalysisReport } from '../../types';
import { Modal } from '../common/Modal';
import { Skeleton } from '../common/Skeleton';
import { ReportViewer } from '../analysis/ReportViewer';

interface ReportModalProps {
  open: boolean;
  loading: boolean;
  error: string | null;
  report: AnalysisReport | null;
  onClose: () => void;
}

/** Modal wrapper that renders a stored report via the shared ReportViewer. */
export const ReportModal: FC<ReportModalProps> = ({ open, loading, error, report, onClose }) => (
  <Modal open={open} onClose={onClose} widthClass="max-w-5xl" labelledBy="report-modal-title">
    <div className="p-6">
      <h2 id="report-modal-title" className="sr-only">
        Analysis report
      </h2>

      {loading ? (
        <div className="space-y-6">
          <Skeleton className="h-8 w-64" />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full md:col-span-2" />
          </div>
          <Skeleton className="h-48 w-full" />
        </div>
      ) : error ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <AlertCircle className="mb-4 h-10 w-10 text-rose-400" />
          <p className="text-sm text-gray-300">{error}</p>
        </div>
      ) : report ? (
        <ReportViewer report={report} />
      ) : null}
    </div>
  </Modal>
);
