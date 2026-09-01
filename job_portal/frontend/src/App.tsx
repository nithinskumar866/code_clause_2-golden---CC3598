import { useEffect, useState } from 'react';
import { BrowserRouter, NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { api } from './api/client';
import { ChatPage } from './pages/ChatPage';
import { JobBoardPage } from './pages/JobBoardPage';
import { PostJobPage } from './pages/PostJobPage';
import { PortalStateProvider } from './state/PortalState';
import type { Health } from './types';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    const poll = () => api.health()
      .then(result => { if (!cancelled) { setHealth(result); setReachable(true); } })
      .catch(() => { if (!cancelled) setReachable(false); });

    void poll();
    // Job and index counts change as documents are posted, and the matching mode
    // changes if the embedding endpoint comes back. Slow enough to be invisible.
    const timer = setInterval(poll, 20_000);
    return () => { cancelled = false; clearInterval(timer); };
  }, []);

  const statusClass = reachable === false ? 'bad' : health?.semanticMatching ? 'ok' : 'warn';
  const statusText = reachable === false
    ? 'API unreachable'
    : health
      ? `${health.semanticMatching ? 'semantic' : 'lexical'} · ${health.jobCount} jobs`
      : 'connecting…';

  return (
    <PortalStateProvider>
      <BrowserRouter>
      <div className="app">
        <header className="topbar">
          <span className="brand">Job Portal</span>
          <nav>
            <NavLink to="/chat" className={({ isActive }) => (isActive ? 'active' : '')}>Chat</NavLink>
            <NavLink to="/jobs" className={({ isActive }) => (isActive ? 'active' : '')}>Board</NavLink>
            <NavLink to="/post" className={({ isActive }) => (isActive ? 'active' : '')}>Post a job</NavLink>
          </nav>
          <div className="status-pill">
            <span className={`dot ${statusClass}`} />
            {statusText}
          </div>
        </header>

        <Routes>
          <Route path="/chat" element={<ChatPage health={health} />} />
          <Route path="/jobs" element={<JobBoardPage />} />
          <Route path="/post" element={<PostJobPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
    </PortalStateProvider>
  );
}
