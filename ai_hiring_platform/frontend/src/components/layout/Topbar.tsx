import type { FC } from 'react';
import { Menu } from 'lucide-react';

interface TopbarProps {
  title: string;
  onMenu: () => void;
}

/** Sticky top bar: mobile nav trigger + current-section wayfinding. */
export const Topbar: FC<TopbarProps> = ({ title, onMenu }) => (
  <header className="sticky top-0 z-20 flex h-16 shrink-0 items-center gap-3 border-b border-border bg-background/80 px-4 backdrop-blur sm:px-6 lg:px-8">
    <button
      onClick={onMenu}
      className="rounded-lg p-2 text-gray-400 transition hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 lg:hidden"
      aria-label="Open navigation"
    >
      <Menu className="h-5 w-5" />
    </button>
    <span className="text-sm font-medium text-gray-300">{title}</span>
    <span className="ml-auto hidden text-[11px] font-medium text-gray-500 sm:inline">
      AI Hiring Intelligence Platform
    </span>
  </header>
);
