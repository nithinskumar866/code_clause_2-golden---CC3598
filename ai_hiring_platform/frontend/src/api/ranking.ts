import { api } from './client';
import type { RankingResponse } from '../types';

/**
 * Rank several resumes against one job description. Runs the full evaluation
 * pipeline per resume on the backend and returns a leaderboard (best first).
 */
export async function rankCandidates(jd_id: number, resume_ids: number[]): Promise<RankingResponse> {
  const res = await api.post('/analysis/rank', { jd_id, resume_ids });
  if (res.data && res.data.success) {
    return res.data.data as RankingResponse;
  }
  throw new Error(res.data?.message || 'Ranking failed');
}
