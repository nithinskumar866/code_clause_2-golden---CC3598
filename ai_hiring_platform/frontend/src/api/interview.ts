import { api } from './client';
import type { InterviewSimulation } from '../types';

/**
 * Run the AI Recruiter interview simulation for a completed analysis. The backend
 * reasons over the retrieved resume evidence (LLM if configured, else deterministic)
 * and returns, per question, an ideal answer + evidence + confidence + verdict.
 * This is an on-demand, potentially slow (LLM) call.
 */
export async function simulateInterview(analysisId: number): Promise<InterviewSimulation> {
  const res = await api.post(`/analysis/${analysisId}/interview`);
  if (res.data && res.data.success) return res.data.data as InterviewSimulation;
  throw new Error(res.data?.message || 'Failed to simulate interview');
}
