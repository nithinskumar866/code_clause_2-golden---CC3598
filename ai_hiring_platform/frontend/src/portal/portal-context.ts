import { createContext, useContext } from 'react';
import type { ChatTurn, ConnectionState } from './useChatHub';
import type { NavOption } from './navigator';
import type { Health, JobFilters, ResumeProfile, ScoringMode } from './types';

/**
 * The job portal's shared state, split from its provider so the provider module
 * exports only a component (Fast Refresh cannot handle a file that exports
 * both). Mirrors `components/ui/toast-context.ts` in the recruiter app.
 */
export interface PortalState {
  // -- the chat popup -------------------------------------------------------
  /** Whether the popup panel is on screen. Its state is kept either way, so a
   *  reply that arrives while it is shut is still there when it is reopened. */
  chatOpen: boolean;
  openChat: () => void;
  closeChat: () => void;
  /** Assistant replies that landed while the panel was closed. */
  unread: number;

  connection: ConnectionState;
  turns: ChatTurn[];
  busy: boolean;
  send: (message: string) => Promise<void>;
  /** Opens a page the assistant offered as a button. */
  openPage: (option: NavOption, lead?: string) => void;

  /**
   * Which scorer the assistant uses — the measured arithmetic, or a model
   * reasoning over the posting's requirements. Switchable mid-conversation:
   * answers already on screen keep the mode they were produced under, and each
   * card says which that was.
   */
  scoringMode: ScoringMode;
  setScoringMode: (mode: ScoringMode) => void;

  /** Clears the conversation and its narrowed filters. The CV stays attached. */
  newConversation: () => Promise<void>;
  resetting: boolean;

  resume: ResumeProfile | null;
  uploadResume: (file: File) => Promise<void>;
  uploading: boolean;
  chatError: string | null;
  dismissChatError: () => void;

  // -- the board ------------------------------------------------------------
  /** Filters the board is currently showing. Owned here, not by the board page,
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

  // -- portal API status ----------------------------------------------------
  health: Health | null;
  /** null until the first probe answers. */
  reachable: boolean | null;
  /** Called by the portal sections on mount: nothing talks to the .NET API
   *  until some part of the app that needs it is actually in use. */
  engage: () => void;
}

export const PortalContext = createContext<PortalState | null>(null);

export function usePortalState(): PortalState {
  const state = useContext(PortalContext);
  if (!state) throw new Error('usePortalState must be used inside PortalStateProvider');
  return state;
}
