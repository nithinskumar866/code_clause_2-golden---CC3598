import { api } from './client';
import type { WorkflowStatus } from '../types';

/** Get the current recruiter-workflow stage for an analysis. */
export async function getWorkflowStatus(analysisId: number): Promise<WorkflowStatus> {
  const res = await api.get(`/analysis/${analysisId}/status`);
  if (res.data?.success) return res.data.data.workflow_status as WorkflowStatus;
  throw new Error(res.data?.message || 'Failed to load workflow status');
}

/** Update (and persist) the recruiter-workflow stage for an analysis. */
export async function updateWorkflowStatus(
  analysisId: number,
  status: WorkflowStatus,
): Promise<WorkflowStatus> {
  const res = await api.patch(`/analysis/${analysisId}/status`, { workflow_status: status });
  if (res.data?.success) return res.data.data.workflow_status as WorkflowStatus;
  throw new Error(res.data?.message || 'Failed to update workflow status');
}
