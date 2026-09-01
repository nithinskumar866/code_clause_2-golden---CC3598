import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { portalApi, resetConversation } from './api';
import { useChatHub } from './useChatHub';
import { PortalContext } from './portal-context';
import type { PortalState } from './portal-context';
import type { AssistantNavigator, NavOption, NavOutcome } from './navigator';
import type { JobFilters, Health, ResumeProfile, UiAction } from './types';

/**
 * The session id lives in localStorage so a refresh resumes the same
 * conversation rather than silently starting a new one and losing the resume the
 * candidate already uploaded.
 */
function readSessionId(): string {
  const existing = localStorage.getItem('portal.sessionId');
  if (existing) return existing;
  const created = crypto.randomUUID();
  localStorage.setItem('portal.sessionId', created);
  return created;
}

interface ProviderProps {
  children: ReactNode;
  /** Called when the assistant surfaces a shortlist, so the app can show the
   *  board it just filtered. Passed in rather than imported to keep this file
   *  independent of how the host app navigates. */
  onShowBoard?: () => void;
  /** Lets the assistant move around the application. Injected for the same
   *  reason: the popup knows a page can be opened, not which pages exist. */
  navigator?: AssistantNavigator;
}

/**
 * Copilot Mode, application-wide.
 *
 * Board filters, the bot's shortlist and the conversation all live above the
 * page switch rather than inside the chat or the board, for two reasons:
 *
 *  - The popup has to survive navigation. State owned by a page dies when that
 *    page unmounts, which would drop the socket and the transcript every time
 *    the recruiter clicked a different section.
 *  - The chat has to be able to drive the board. State owned by the board could
 *    only ever be changed by the board, so "show me remote roles over $120k"
 *    could not do anything but talk.
 */
export function PortalStateProvider({ children, onShowBoard, navigator }: ProviderProps) {
  const sessionId = useMemo(readSessionId, []);

  const [chatOpen, setChatOpen] = useState(false);
  // Separate from `chatOpen`: the connection is made the first time the chat is
  // opened and then kept, so closing the panel does not drop a reply in flight.
  const [chatEverOpened, setChatEverOpened] = useState(false);
  const [engaged, setEngaged] = useState(false);

  const [filters, setFilters] = useState<JobFilters>({});
  const [spotlight, setSpotlight] = useState<number[] | null>(null);
  const [lastCommand, setLastCommand] = useState<string | null>(null);

  const [resume, setResume] = useState<ResumeProfile | null>(null);
  const [uploading, setUploading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const [health, setHealth] = useState<Health | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  const showBoardRef = useRef(onShowBoard);
  showBoardRef.current = onShowBoard;

  const navigatorRef = useRef(navigator);
  navigatorRef.current = navigator;

  const engage = useCallback(() => setEngaged(true), []);

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
        if (payload?.jobIds?.length) {
          setSpotlight(payload.jobIds);
          // Now that the board is a section of this app rather than a separate
          // site, the action can finish what it started and take the user there.
          showBoardRef.current?.();
        }
        break;
      }
      default:
        // Forward compatible: a backend that learns a new action must not break
        // a client that has not learned it yet.
        break;
    }
  }, []);

  const {
    state: connection, turns, busy, send: hubSend, analyzeResume, appendLocal, clearTurns,
    scoringMode, setScoringMode,
  } = useChatHub(sessionId, chatEverOpened, applyAction);

  /**
   * Starts a fresh conversation.
   *
   * The transcript clears immediately rather than waiting on the server: the
   * point of the button is that the screen is empty when it is pressed, and a
   * portal API that is down should not leave the user staring at old messages
   * they asked to be rid of. The server call clears the stored history behind it.
   */
  const [resetting, setResetting] = useState(false);
  const newConversation = useCallback(async () => {
    setResetting(true);
    clearTurns();
    setFilters({});
    setSpotlight(null);
    setLastCommand(null);
    setChatError(null);
    // The CV goes too. Leaving it meant a "new" chat still answered as the last
    // person, and the uploader offered only Replace on a document the candidate
    // believed they had already cleared.
    setResume(null);
    try {
      await resetConversation(sessionId);
    } catch {
      // The visible transcript is already gone; a failed server clear only means
      // the old messages would come back on a reload, which is not worth an alarm.
    } finally {
      setResetting(false);
    }
  }, [clearTurns, sessionId]);

  /** Opens a page and says so, so the transcript records where it went. */
  const openPage = useCallback((option: NavOption, lead = 'Opening') => {
    navigatorRef.current?.go(option.pageId);
    appendLocal({
      assistant: `${lead} **${option.label}**.${option.description ? ` ${option.description}` : ''}`,
    });
  }, [appendLocal]);

  const answerNavigation = useCallback((message: string, outcome: NavOutcome) => {
    switch (outcome.kind) {
      case 'navigate':
        appendLocal({
          user: message,
          assistant: `Opening **${outcome.option.label}** for you. ${outcome.option.description ?? ''}`.trim(),
        });
        navigatorRef.current?.go(outcome.option.pageId);
        break;

      case 'describe':
        // Answered, not obeyed — plus a way to act on it without retyping.
        appendLocal({
          user: message,
          assistant: `**${outcome.option.label}** — ${outcome.option.description ?? 'a section of this application.'}`,
          options: [outcome.option],
        });
        break;

      case 'ambiguous':
        // Guessing wrong is worse than asking: the user cannot tell a wrong
        // guess from the application simply not having what they wanted.
        appendLocal({
          user: message,
          assistant: outcome.options.length === 1
            ? `Did you mean **${outcome.options[0].label}**?`
            : `I can take you to a few places that fit${outcome.term ? ` "${outcome.term}"` : ''}. Which did you mean?`,
          options: outcome.options,
        });
        break;

      case 'unknown':
        appendLocal({
          user: message,
          assistant:
            `Sorry — there is no${outcome.term ? ` **${outcome.term}**` : ' such'} page in this application. ` +
            'Here is everything you can go to:',
          options: outcome.options,
        });
        break;
    }
  }, [appendLocal]);

  /**
   * Navigation is resolved before the hub is involved. A request to move around
   * the app is not a question for the job-portal bot, and answering it locally
   * means it still works when that API is down.
   */
  const send = useCallback(async (message: string) => {
    const trimmed = message.trim();
    if (!trimmed || busy) return;

    const outcome = navigatorRef.current?.resolve(trimmed) ?? null;
    if (outcome) {
      answerNavigation(trimmed, outcome);
      return;
    }
    await hubSend(trimmed);
  }, [answerNavigation, busy, hubSend]);

  // -- unread ---------------------------------------------------------------
  // Counted from the transcript length rather than incremented in the socket
  // handlers, so it cannot drift out of step with what is actually on screen.
  const seenRef = useRef(0);
  const [unread, setUnread] = useState(0);

  useEffect(() => {
    if (chatOpen) {
      seenRef.current = turns.length;
      setUnread(0);
      return;
    }
    const assistantSinceSeen = turns
      .slice(seenRef.current)
      .filter(turn => turn.role === 'assistant' && !turn.streaming).length;
    setUnread(assistantSinceSeen);
  }, [turns, chatOpen]);

  const openChat = useCallback(() => {
    setChatOpen(true);
    setChatEverOpened(true);
    setEngaged(true);
  }, []);
  const closeChat = useCallback(() => setChatOpen(false), []);

  // -- health ---------------------------------------------------------------

  useEffect(() => {
    // Nothing is probed until a portal section is in use: a recruiter who never
    // opens the chat or the board should not see failed requests to a service
    // they are not running.
    if (!engaged) return;

    let cancelled = false;
    const poll = () => portalApi.health()
      .then(result => { if (!cancelled) { setHealth(result); setReachable(true); } })
      .catch(() => { if (!cancelled) setReachable(false); });

    void poll();
    // Job and index counts change as documents are posted, and the matching mode
    // changes if the embedding endpoint comes back. Slow enough to be invisible.
    const timer = setInterval(poll, 20_000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [engaged]);

  // Restore whatever the session already knows, so a refresh does not present an
  // empty chat to someone who has already uploaded a resume.
  useEffect(() => {
    if (!chatEverOpened) return;
    portalApi.getSession(sessionId)
      .then(session => setResume(previous => previous ?? session.resume))
      .catch(() => { /* A fresh session simply has nothing to restore. */ });
  }, [sessionId, chatEverOpened]);

  const uploadResume = useCallback(async (file: File) => {
    setUploading(true);
    setChatError(null);
    try {
      const profile = await portalApi.uploadResume(file);
      setResume(profile);
      // Upload over REST, narrate over the hub: the file does not belong in a
      // WebSocket frame, but the analysis is what the candidate watches.
      await analyzeResume(profile.id);
    } catch (error) {
      setChatError(error instanceof Error ? error.message : 'The resume could not be read.');
    } finally {
      setUploading(false);
    }
  }, [analyzeResume]);

  const clearSpotlight = useCallback(() => {
    setSpotlight(null);
    setLastCommand(null);
  }, []);

  const dismissChatError = useCallback(() => setChatError(null), []);

  const value = useMemo<PortalState>(() => ({
    chatOpen, openChat, closeChat, unread,
    connection, turns, busy, send, openPage, scoringMode, setScoringMode,
    resume, uploadResume, uploading, chatError, dismissChatError,
    filters, setFilters, spotlight, clearSpotlight, lastCommand,
    health, reachable, engage,
    newConversation, resetting,
  }), [
    chatOpen, openChat, closeChat, unread,
    connection, turns, busy, send, openPage, scoringMode, setScoringMode,
    resume, uploadResume, uploading, chatError, dismissChatError,
    filters, spotlight, clearSpotlight, lastCommand,
    health, reachable, engage,
    newConversation, resetting,
  ]);

  return <PortalContext.Provider value={value}>{children}</PortalContext.Provider>;
}

/** Turns a filter set into the phrase the board shows to explain itself. */
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
