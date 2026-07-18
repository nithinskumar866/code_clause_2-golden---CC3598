import { useCallback, useEffect, useState, type FC } from 'react';
import { GitBranch, Check } from 'lucide-react';
import { WORKFLOW_STAGES, type WorkflowStatus } from '../../types';
import { getWorkflowStatus, updateWorkflowStatus } from '../../api/workflow';
import { Card } from '../ui/Card';
import { Select } from '../ui/Select';
import { Skeleton } from '../common/Skeleton';
import { ErrorState } from '../ui/ErrorState';
import { useToast } from '../ui/toast-context';

interface WorkflowStatusControlProps {
  analysisId: number;
  /** Called after a successful status change (e.g. to refresh a history list). */
  onChange?: (status: WorkflowStatus) => void;
}

/**
 * Editable, persistent recruiter-workflow stage for one analysis. Reads the
 * current stage on mount and PATCHes changes; renders a pipeline stepper so the
 * recruiter sees where the candidate sits.
 */
export const WorkflowStatusControl: FC<WorkflowStatusControlProps> = ({ analysisId, onChange }) => {
  const [status, setStatus] = useState<WorkflowStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setStatus(await getWorkflowStatus(analysisId));
    } catch {
      setError('Failed to load workflow status.');
    } finally {
      setLoading(false);
    }
  }, [analysisId]);

  useEffect(() => {
    load();
  }, [load]);

  const change = async (next: WorkflowStatus) => {
    const prev = status;
    setStatus(next); // optimistic
    setSaving(true);
    try {
      const saved = await updateWorkflowStatus(analysisId, next);
      setStatus(saved);
      onChange?.(saved);
      toast.success('Workflow updated', `Stage set to “${saved}”.`);
    } catch (err: any) {
      setStatus(prev); // revert
      toast.error(err?.message || 'Failed to update workflow status');
    } finally {
      setSaving(false);
    }
  };

  const activeIndex = status ? WORKFLOW_STAGES.indexOf(status) : -1;
  const isRejected = status === 'Rejected';

  return (
    <Card className="p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-white/5 pb-2">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
          <GitBranch className="h-4 w-4 text-indigo-400" /> Recruiter Workflow
        </h3>
        {!loading && !error && status && (
          <Select
            label="Workflow stage"
            srLabel
            value={status}
            disabled={saving}
            onChange={(e) => change(e.target.value as WorkflowStatus)}
            options={WORKFLOW_STAGES.map((s) => ({ value: s, label: s }))}
          />
        )}
      </div>

      {loading ? (
        <Skeleton className="h-10 w-full" />
      ) : error ? (
        <ErrorState message={error} onRetry={load} />
      ) : (
        <ol className="flex flex-wrap items-center gap-1.5">
          {WORKFLOW_STAGES.map((stage, i) => {
            const done = !isRejected && i < activeIndex;
            const current = i === activeIndex;
            const rejectedStage = isRejected && stage === 'Rejected';
            return (
              <li key={stage} className="flex items-center gap-1.5">
                <span
                  className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide transition ${
                    rejectedStage
                      ? 'border-rose-500/40 bg-rose-500/15 text-rose-300'
                      : current
                        ? 'border-indigo-500/50 bg-indigo-500/20 text-indigo-200'
                        : done
                          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                          : 'border-white/10 bg-white/5 text-gray-500'
                  }`}
                >
                  {done && <Check className="h-3 w-3" />}
                  {stage}
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </Card>
  );
};
