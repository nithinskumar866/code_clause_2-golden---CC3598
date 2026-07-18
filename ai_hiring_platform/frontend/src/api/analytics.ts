import { api } from './client';
import type {
  OverallStatistics,
  ScoreDistribution,
  RecommendationDistribution,
  TrendData,
  TopItem,
  SkillFrequency,
} from '../types';

/** Unwrap the standard `{success, data}` envelope or throw. */
async function get<T>(path: string): Promise<T> {
  const res = await api.get(path);
  if (res.data && res.data.success) return res.data.data as T;
  throw new Error(res.data?.message || `Request failed: ${path}`);
}

export const fetchOverview = () => get<OverallStatistics>('/analytics/overview');
export const fetchScoreDistribution = () => get<ScoreDistribution>('/analytics/score-distribution');
export const fetchRecommendationDistribution = () =>
  get<RecommendationDistribution>('/analytics/recommendation-distribution');
export const fetchTrends = () => get<TrendData>('/analytics/trends');
export const fetchTopResumes = (limit = 5) => get<TopItem[]>(`/analytics/top-resumes?limit=${limit}`);
export const fetchTopJobs = (limit = 5) => get<TopItem[]>(`/analytics/top-jobs?limit=${limit}`);
export const fetchSkillFrequency = (limit = 8) => get<SkillFrequency>(`/analytics/skill-frequency?limit=${limit}`);
