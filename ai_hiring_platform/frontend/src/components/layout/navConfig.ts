import type { ComponentType } from 'react';
import { LayoutDashboard, FileUp, Briefcase, Sparkles, Clock, Settings2 } from 'lucide-react';

export type PageId = 'dashboard' | 'resume' | 'job' | 'analysis' | 'history' | 'profile' | 'status';

export interface NavItem {
  id: PageId;
  name: string;
  icon: ComponentType<{ className?: string }>;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

/** Grouped primary navigation — the single source of truth for routes + labels. */
export const NAV_GROUPS: NavGroup[] = [
  { label: 'Overview', items: [{ id: 'dashboard', name: 'Dashboard', icon: LayoutDashboard }] },
  {
    label: 'Documents',
    items: [
      { id: 'resume', name: 'Resumes', icon: FileUp },
      { id: 'job', name: 'Job Descriptions', icon: Briefcase },
    ],
  },
  {
    label: 'Evaluation',
    items: [
      { id: 'analysis', name: 'AI Analysis', icon: Sparkles },
      { id: 'history', name: 'Analysis History', icon: Clock },
    ],
  },
  { label: 'System', items: [{ id: 'status', name: 'System Status', icon: Settings2 }] },
];

/** Short title shown in the top bar per page. */
export const PAGE_TITLES: Record<PageId, string> = {
  dashboard: 'Dashboard',
  resume: 'Resumes',
  job: 'Job Descriptions',
  analysis: 'AI Analysis',
  history: 'Analysis History',
  profile: 'Candidate Profile',
  status: 'System Status',
};
