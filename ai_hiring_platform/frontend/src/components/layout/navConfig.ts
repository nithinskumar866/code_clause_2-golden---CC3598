import type { ComponentType } from 'react';
import {
  LayoutDashboard,
  FileUp,
  Briefcase,
  Sparkles,
  Clock,
  Settings2,
  Trophy,
  BarChart3,
  MessagesSquare,
  FlaskConical,
  FolderCog,
  Building2,
  Megaphone,
  Inbox,
} from 'lucide-react';

export type PageId =
  | 'dashboard'
  | 'resume'
  | 'job'
  | 'documents'
  | 'analysis'
  | 'chat'
  | 'ranking'
  | 'history'
  | 'profile'
  | 'analytics'
  | 'modellab'
  | 'jobboard'
  | 'postjob'
  | 'applicants'
  | 'status';

export interface NavItem {
  id: PageId;
  name: string;
  icon: ComponentType<{ className?: string }>;
  /**
   * One sentence describing what the page is for. Shown when the assistant is
   * asked what a section does, and when it offers a choice between sections.
   */
  description: string;
  /**
   * Words a recruiter might use for this page other than its label.
   *
   * Kept here, on the route itself, rather than in a lookup table beside the
   * assistant: a page and the words that reach it are one fact, and splitting
   * them is how a renamed page keeps answering to nothing. Adding a route
   * without vocabulary makes it unreachable by name, which `navConfig.test.ts`
   * fails on.
   */
  keywords: string[];
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

/** Grouped primary navigation — the single source of truth for routes + labels. */
export const NAV_GROUPS: NavGroup[] = [
  {
    label: 'Overview',
    items: [{
      id: 'dashboard',
      name: 'Dashboard',
      icon: LayoutDashboard,
      description: 'The landing overview: recent activity and headline numbers across the platform.',
      keywords: ['home', 'overview', 'main', 'start', 'landing', 'front'],
    }],
  },
  {
    label: 'Documents',
    items: [
      {
        id: 'resume',
        name: 'Resumes',
        icon: FileUp,
        description: 'Upload candidate resumes (PDF or DOCX) and see the ones already registered.',
        keywords: ['resume', 'cv', 'upload', 'update', 'add', 'candidate file', 'applicant file'],
      },
      {
        id: 'job',
        name: 'Job Descriptions',
        icon: Briefcase,
        description: 'Upload job descriptions and review the requirements extracted from them.',
        keywords: ['jd', 'job description', 'posting', 'requirement', 'spec', 'role description', 'upload'],
      },
      {
        id: 'documents',
        name: 'Manage & Index',
        icon: FolderCog,
        description: 'The working set: choose which resumes are indexed per embedding model, or remove them.',
        keywords: ['manage', 'index', 'indexing', 'embed', 'embedding', 'working set', 'library', 'documents', 'files', 'storage', 'coverage'],
      },
    ],
  },
  {
    label: 'Evaluation',
    items: [
      {
        id: 'analysis',
        name: 'AI Analysis',
        icon: Sparkles,
        description: 'Evaluate one resume against one job description and get the full explainable hiring report.',
        keywords: ['analyse', 'analyze', 'evaluate', 'evaluation', 'match', 'score', 'report', 'assess', 'fit'],
      },
      {
        id: 'chat',
        name: 'Recruiter Assistant',
        icon: MessagesSquare,
        description: 'Ask questions across the whole indexed resume pool and get evidence-backed answers.',
        keywords: ['assistant', 'chatbot', 'ask', 'talent pool', 'search candidates', 'query', 'conversation'],
      },
      {
        id: 'ranking',
        name: 'Candidate Ranking',
        icon: Trophy,
        description: 'Score many resumes against one job description and see them on a leaderboard.',
        keywords: ['rank', 'ranking', 'leaderboard', 'shortlist', 'top candidates', 'compare candidates', 'best'],
      },
      {
        id: 'history',
        name: 'Analysis History',
        icon: Clock,
        description: 'Every past analysis: search, filter, reopen a candidate, or export a report.',
        // "analyses" is listed because the crude singulariser turns it into
        // "analyse", which never meets the page name's "analysis".
        keywords: ['history', 'past', 'previous', 'earlier', 'records', 'archive', 'log', 'export', 'analyses'],
      },
    ],
  },
  {
    label: 'Insights',
    items: [{
      id: 'analytics',
      name: 'Analytics',
      icon: BarChart3,
      description: 'Totals, score distribution, recommendation split, trends and the most common skill gaps.',
      keywords: ['statistics', 'stats', 'charts', 'graphs', 'trends', 'insights', 'metrics', 'distribution', 'numbers'],
    }],
  },
  {
    // The candidate-facing job portal, served by the separate .NET API. Grouped
    // apart from the recruiter sections because it is a different audience and a
    // different backend — not because it is a different application.
    label: 'Job Portal',
    items: [
      {
        id: 'jobboard',
        name: 'Job Board',
        icon: Building2,
        description: 'Browse the open roles on the candidate-facing portal and filter them.',
        keywords: ['board', 'roles', 'openings', 'vacancies', 'listings', 'browse', 'open roles', 'careers'],
      },
      {
        id: 'postjob',
        name: 'Post a Job',
        icon: Megaphone,
        description: 'Publish a role to the candidate-facing board, by upload or by form.',
        keywords: ['post', 'publish', 'advertise', 'new job', 'create job', 'add role', 'vacancy'],
      },
      {
        id: 'applicants',
        name: 'Applicants',
        icon: Inbox,
        description: 'People who applied through the board, with the fit score captured when they applied.',
        keywords: ['applicant', 'applied', 'application', 'candidates who applied', 'inbox', 'submissions'],
      },
    ],
  },
  {
    label: 'Evaluation quality',
    items: [{
      id: 'modellab',
      name: 'Model Lab',
      icon: FlaskConical,
      description: 'Compare the embedding models against each other on the resumes they have all indexed.',
      keywords: ['model', 'models', 'lab', 'compare models', 'benchmark', 'embedding model', 'quality', 'experiment'],
    }],
  },
  {
    label: 'System',
    items: [{
      id: 'status',
      name: 'System Status',
      icon: Settings2,
      description: 'What is running: backend health, configured models, index state and versions.',
      keywords: ['health', 'system', 'diagnostics', 'config', 'configuration', 'settings', 'version', 'uptime'],
    }],
  },
];

/** Short title shown in the top bar per page. */
export const PAGE_TITLES: Record<PageId, string> = {
  dashboard: 'Dashboard',
  resume: 'Resumes',
  job: 'Job Descriptions',
  documents: 'Documents',
  analysis: 'AI Analysis',
  chat: 'Recruiter Assistant',
  ranking: 'Candidate Ranking',
  history: 'Analysis History',
  profile: 'Candidate Profile',
  analytics: 'Analytics',
  modellab: 'Model Lab',
  jobboard: 'Job Board',
  postjob: 'Post a Job',
  applicants: 'Applicants',
  status: 'System Status',
};
