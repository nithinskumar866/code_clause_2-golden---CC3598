import { useState, useEffect, type FC } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Topbar } from './components/layout/Topbar';
import { PAGE_TITLES } from './components/layout/navConfig';
import type { PageId } from './components/layout/navConfig';
import type { HistoryRecord } from './types';
import { updateSearchParams } from './lib/url';
import { Dashboard } from './pages/Dashboard/Dashboard';
import { ResumeUpload } from './pages/Resume/ResumeUpload';
import { JobUpload } from './pages/Job/JobUpload';
import { Analysis } from './pages/Analysis/Analysis';
import { History } from './pages/History/History';
import { CandidateProfile } from './pages/CandidateProfile/CandidateProfile';
import { SystemStatus } from './pages/SystemStatus/SystemStatus';

// Pages that can be restored from the URL `view` param (profile needs a
// selected record, so it is not URL-restorable and is omitted here).
const VIEW_PAGES: PageId[] = ['dashboard', 'resume', 'job', 'analysis', 'history', 'status'];

const readView = (): PageId => {
  const v = new URLSearchParams(window.location.search).get('view') as PageId | null;
  return v && VIEW_PAGES.includes(v) ? v : 'dashboard';
};

const App: FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>(() => readView());
  const [selectedCandidate, setSelectedCandidate] = useState<HistoryRecord | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Escape closes the mobile navigation drawer.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSidebarOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Restore the active page on browser back/forward.
  useEffect(() => {
    const onPop = () => setCurrentPage(readView());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const navigate = (id: PageId) => {
    setCurrentPage(id);
    setSidebarOpen(false);
    updateSearchParams((params) => {
      if (id === 'dashboard') params.delete('view');
      else params.set('view', id);
    }, 'push');
  };

  const openCandidate = (record: HistoryRecord) => {
    setSelectedCandidate(record);
    setCurrentPage('profile');
    setSidebarOpen(false);
  };

  // Keep History highlighted in the sidebar while viewing a candidate profile.
  const navHighlight: PageId = currentPage === 'profile' ? 'history' : currentPage;

  const renderContent = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard onNavigate={navigate} />;
      case 'resume':
        return <ResumeUpload />;
      case 'job':
        return <JobUpload />;
      case 'analysis':
        return <Analysis />;
      case 'history':
        return <History onOpenCandidate={openCandidate} />;
      case 'profile':
        return selectedCandidate ? (
          <CandidateProfile record={selectedCandidate} onBack={() => navigate('history')} />
        ) : (
          <History onOpenCandidate={openCandidate} />
        );
      case 'status':
        return <SystemStatus />;
      default:
        return <Dashboard onNavigate={navigate} />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background text-white">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-indigo-600 focus:px-4 focus:py-2 focus:text-sm focus:font-semibold focus:text-white"
      >
        Skip to content
      </a>

      {/* Mobile drawer backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <Sidebar
        current={navHighlight}
        onNavigate={navigate}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar title={PAGE_TITLES[currentPage]} onMenu={() => setSidebarOpen(true)} />
        <main id="main-content" className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">{renderContent()}</div>
        </main>
      </div>
    </div>
  );
};

export default App;
