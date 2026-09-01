import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { JobFilters, UiAction } from '../types';

interface PortalState {
  /** Filters the board grid is currently showing. Owned here, not by the board,
   *  because the chat is allowed to change them. */
  filters: JobFilters;
  setFilters: (filters: JobFilters) => void;

  /** Job ids the bot last surfaced. When set, the board shows exactly these, in
   *  this order — the bot's ranking, not the board's default recency order. */
  spotlight: number[] | null;
  clearSpotlight: () => void;

  /** Set when the bot changed the board while the user was on another page, so
   *  the board can say why it looks different when they arrive. */
  lastCommand: string | null;

  /** Applies a command from the backend. Unknown actions are ignored. */
  applyAction: (action: UiAction) => void;
}

const Context = createContext<PortalState | null>(null);

/**
 * Copilot Mode.
 *
 * Board filters and the bot's shortlist live here rather than inside the board
 * page, because the whole point is that the chat can drive them. State owned by
 * the page it renders could only ever be changed by that page, so "show me remote
 * roles over $120k" could not do anything but talk.
 */
export function PortalStateProvider({ children }: { children: ReactNode }) {
  const [filters, setFilters] = useState<JobFilters>({});
  const [spotlight, setSpotlight] = useState<number[] | null>(null);
  const [lastCommand, setLastCommand] = useState<string | null>(null);

  const applyAction = useCallback((action: UiAction) => {
    switch (action.action) {
      case 'update_filters': {
        const next = (action.payload ?? {}) as JobFilters;
        setFilters(next);
        setLastCommand(describe(next));
        break;
      }
      case 'show_jobs': {
        const payload = action.payload as { jobIds?: number[] } | null;
        // An empty shortlist must not become an empty board. "No matches" is the
        // chat's message to deliver; silently blanking the grid would look broken.
        if (payload?.jobIds?.length) setSpotlight(payload.jobIds);
        break;
      }
      default:
        // Forward compatible: a backend that learns a new action must not break
        // a client that has not learned it yet.
        break;
    }
  }, []);

  const clearSpotlight = useCallback(() => {
    setSpotlight(null);
    setLastCommand(null);
  }, []);

  const value = useMemo(
    () => ({ filters, setFilters, spotlight, clearSpotlight, lastCommand, applyAction }),
    [filters, spotlight, lastCommand, applyAction, clearSpotlight],
  );

  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function usePortalState(): PortalState {
  const state = useContext(Context);
  if (!state) throw new Error('usePortalState must be used inside PortalStateProvider');
  return state;
}

function describe(filters: JobFilters): string | null {
  const parts: string[] = [];
  if (filters.workMode) parts.push(filters.workMode.toLowerCase());
  if (filters.location) parts.push(`in ${filters.location}`);
  if (filters.seniorityLevel) parts.push(`${filters.seniorityLevel.toLowerCase()} level`);
  if (filters.employmentType) parts.push(filters.employmentType.toLowerCase());
  if (filters.minSalary) parts.push(`paying at least ${filters.minSalary.toLocaleString()}`);
  if (filters.keywords?.length) parts.push(`involving ${filters.keywords.join(' and ')}`);
  return parts.length ? parts.join(', ') : null;
}
