import { api } from './client';
import type { DashboardOverview } from '../types';

/** Aggregate hiring stats for the dashboard header (completed analyses only). */
export async function fetchDashboardOverview(): Promise<DashboardOverview> {
  const res = await api.get('/dashboard/overview');
  if (res.data?.success) return res.data.data as DashboardOverview;
  throw new Error(res.data?.message || 'Failed to load dashboard overview');
}
