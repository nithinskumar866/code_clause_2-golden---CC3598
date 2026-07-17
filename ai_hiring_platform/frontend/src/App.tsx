import { useState } from 'react';
import { LayoutDashboard, FileUp, Briefcase, Sparkles, Clock, Settings2, BrainCircuit } from 'lucide-react';
import { Dashboard } from './pages/Dashboard/Dashboard';
import { ResumeUpload } from './pages/Resume/ResumeUpload';
import { JobUpload } from './pages/Job/JobUpload';
import { Analysis } from './pages/Analysis/Analysis';
import { History } from './pages/History/History';
import { SystemStatus } from './pages/SystemStatus/SystemStatus';

type Page = 'dashboard' | 'resume' | 'job' | 'analysis' | 'history' | 'status';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('dashboard');

  const navigation = [
    { name: 'Dashboard', id: 'dashboard' as Page, icon: LayoutDashboard },
    { name: 'Upload Resumes', id: 'resume' as Page, icon: FileUp },
    { name: 'Upload Job Description', id: 'job' as Page, icon: Briefcase },
    { name: 'AI Analysis', id: 'analysis' as Page, icon: Sparkles },
    { name: 'Analysis History', id: 'history' as Page, icon: Clock },
    { name: 'System Status', id: 'status' as Page, icon: Settings2 },
  ];

  const renderContent = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
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
        return <Dashboard />;
    }
  };

  return (
    <div className="flex h-screen bg-background text-white font-sans overflow-hidden">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r border-border bg-card flex flex-col justify-between shrink-0">
        <div>
          {/* Logo / Header */}
          <div className="flex items-center gap-2.5 px-6 py-6 border-b border-border">
            <div className="p-2 rounded-lg bg-indigo-600 text-white shadow-lg shadow-indigo-600/20">
              <BrainCircuit className="h-6 w-6" />
            </div>
            <div>
              <h2 className="font-semibold text-sm leading-tight text-white">AI Hiring Intel</h2>
              <span className="text-[10px] text-indigo-400 font-semibold tracking-wider uppercase">Copilot Network</span>
            </div>
          </div>

          {/* Nav Links */}
          <nav className="px-3 py-6 space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = currentPage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentPage(item.id)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/10'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className="h-5 w-5 shrink-0" />
                  <span>{item.name}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Footer info */}
        <div className="p-6 border-t border-border bg-black/20 text-center">
          <p className="text-[10px] text-gray-500 font-semibold uppercase tracking-widest">
            Sprint 1 Scaffolding
          </p>
          <p className="text-[9px] text-gray-600 mt-1">
            Modular Monolith Foundation
          </p>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-y-auto bg-background/50">
        <header className="h-16 border-b border-border shrink-0 flex items-center justify-end px-8">
          <div className="flex items-center gap-4 text-xs font-semibold text-gray-400">
            <span className="flex h-2 w-2 rounded-full bg-emerald-500" />
            <span>Sprint 1 Environment Active</span>
          </div>
        </header>
        <div className="flex-1 p-8 max-w-5xl mx-auto w-full">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}
