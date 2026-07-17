import type { FC } from 'react';
import { BrainCircuit, X } from 'lucide-react';
import { NAV_GROUPS } from './navConfig';
import type { PageId } from './navConfig';

interface SidebarProps {
  current: PageId;
  onNavigate: (id: PageId) => void;
  /** Drawer open state (mobile only; sidebar is always visible on lg+). */
  open: boolean;
  onClose: () => void;
}

/** Responsive primary navigation: persistent rail on desktop, drawer on mobile. */
export const Sidebar: FC<SidebarProps> = ({ current, onNavigate, open, onClose }) => (
  <aside
    aria-label="Sidebar"
    className={`fixed inset-y-0 left-0 z-40 flex w-64 shrink-0 flex-col border-r border-border bg-card transition-transform duration-200 ease-out lg:static lg:translate-x-0 ${
      open ? 'translate-x-0' : '-translate-x-full'
    }`}
  >
    <div className="flex items-center justify-between gap-2.5 border-b border-border px-5 py-5">
      <div className="flex items-center gap-2.5">
        <div className="rounded-lg bg-indigo-600 p-2 text-white shadow-lg shadow-indigo-600/20">
          <BrainCircuit className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-sm font-semibold leading-tight text-white">Hiring Intelligence</h2>
          <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-400">
            Recruiter Console
          </span>
        </div>
      </div>
      <button
        onClick={onClose}
        className="rounded-lg p-1.5 text-gray-400 transition hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 lg:hidden"
        aria-label="Close navigation"
      >
        <X className="h-4 w-4" />
      </button>
    </div>

    <nav aria-label="Primary" className="flex-1 space-y-6 overflow-y-auto px-3 py-6">
      {NAV_GROUPS.map((group) => (
        <div key={group.label}>
          <p className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-widest text-gray-600">
            {group.label}
          </p>
          <ul className="space-y-1">
            {group.items.map((item) => {
              const Icon = item.icon;
              const active = current === item.id;
              return (
                <li key={item.id}>
                  <button
                    onClick={() => onNavigate(item.id)}
                    aria-current={active ? 'page' : undefined}
                    className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                      active
                        ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/10'
                        : 'text-gray-400 hover:bg-white/5 hover:text-white'
                    }`}
                  >
                    <Icon className="h-5 w-5 shrink-0" />
                    <span>{item.name}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>

    <div className="border-t border-border px-5 py-4">
      <p className="text-[10px] font-medium text-gray-600">Agentic RAG · Local-first</p>
    </div>
  </aside>
);
