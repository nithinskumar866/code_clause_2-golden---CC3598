import { useState, useEffect, type FC } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Topbar } from './components/layout/Topbar';
import { PAGE_TITLES } from './components/layout/navConfig';
import type { PageId } from './components/layout/navConfig';
import { Dashboard } from './pages/Dashboard/Dashboard';
import { ResumeUpload } from './pages/Resume/ResumeUpload';
import { JobUpload } from './pages/Job/JobUpload';
import { Analysis } from './pages/Analysis/Analysis';
import { History } from './pages/History/History';
import { SystemStatus } from './pages/SystemStatus/SystemStatus';

const App: FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Escape closes the mobile navigation drawer.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSidebarOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const navigate = (id: PageId) => {
    setCurrentPage(id);
    setSidebarOpen(false);
  };

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
        return <History />;
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
        current={currentPage}
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
