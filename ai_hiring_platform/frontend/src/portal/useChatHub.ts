import { useCallback, useEffect, useRef, useState } from 'react';
import {
  HubConnection, HubConnectionBuilder, HubConnectionState, LogLevel,
} from '@microsoft/signalr';
import { PORTAL_API_BASE } from './api';
import type { JobSuggestionResult, MatchResult, Thought, UiAction } from './types';
import type { NavOption } from './navigator';

export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  /** Grows token by token while the reply streams. */
  content: string;
  /** The visible reasoning for this turn. Assistant turns only. */
  thoughts: Thought[];
  matches: MatchResult | null;
  /** Postings offered when there is no CV yet. Carries no fit score. */
  suggestions: JobSuggestionResult | null;
  streaming: boolean;
  error?: string;
  /** Pages offered as buttons — set on turns the app answered by itself. */
  options?: NavOption[];
}

/** An exchange produced locally, without the hub. */
export interface LocalExchange {
  /** Omitted when only the assistant speaks, e.g. after an option is clicked. */
  user?: string;
  assistant: string;
  options?: NavOption[];
}

export interface UseChatHub {
  state: ConnectionState;
  turns: ChatTurn[];
  busy: boolean;
  send: (message: string) => Promise<void>;
  analyzeResume: (resumeId: number) => Promise<void>;
  /**
   * Adds an exchange the app answered itself, into the same transcript as the
   * hub's replies. Navigation is answered here rather than over the socket, so
   * it keeps working when the portal API is down.
   */
  appendLocal: (exchange: LocalExchange) => void;
  /** Empties the transcript. The socket stays up; only the history goes. */
  clearTurns: () => void;
}

/**
 * Board filters are deliberately NOT state of this hook. They are owned by
 * `PortalState`, because the job board has to obey them whether they came from
 * the chat or from the board's own controls — a copy living in the socket layer
 * would be a second answer to "what is the board filtered by".
 */

/**
 * Owns the WebSocket to the .NET hub and turns its events into render state.
 *
 * The streaming events land in a ref-backed "current turn" rather than being
 * pushed straight into React state one token at a time: a token event arriving
 * every few milliseconds would otherwise queue a render per token. Tokens
 * accumulate and are flushed on an animation frame, which keeps the text
 * animating smoothly without re-rendering the whole transcript hundreds of times.
 *
 * `enabled` exists because this hook now lives in a provider mounted for the
 * whole app rather than on a chat page. Opening a socket to the portal API on
 * every page load would fail loudly for anyone using only the recruiter
 * sections, so the connection is made the first time the chat is actually
 * opened and kept alive from then on.
 */
export function useChatHub(
  sessionId: string,
  enabled: boolean,
  onAction?: (action: UiAction) => void,
): UseChatHub {
  const [state, setState] = useState<ConnectionState>('idle');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);

  const connectionRef = useRef<HubConnection | null>(null);
  const pendingRef = useRef<string>('');
  const frameRef = useRef<number | null>(null);
  const turnIdRef = useRef<string | null>(null);

  // Held in a ref so a caller passing an inline arrow does not tear down and
  // rebuild the whole hub connection on every render.
  const actionRef = useRef(onAction);
  actionRef.current = onAction;

  /** Applies an update to the turn currently streaming, if there is one. */
  const patchCurrent = useCallback((patch: (turn: ChatTurn) => ChatTurn) => {
    const id = turnIdRef.current;
    if (!id) return;
    setTurns(previous => previous.map(turn => (turn.id === id ? patch(turn) : turn)));
  }, []);

  const flush = useCallback(() => {
    frameRef.current = null;
    const text = pendingRef.current;
    if (!text) return;
    pendingRef.current = '';
    patchCurrent(turn => ({ ...turn, content: turn.content + text }));
  }, [patchCurrent]);

  const scheduleFlush = useCallback(() => {
    if (frameRef.current !== null) return;
    frameRef.current = requestAnimationFrame(flush);
  }, [flush]);

  useEffect(() => {
    if (!enabled) return;

    // StrictMode runs this effect twice on the same component instance, so a
    // second connection is built while the first is still negotiating. Two
    // things follow from that, and both were breaking the chat:
    //
    //  1. Tearing down synchronously calls stop() mid-negotiate, which rejects
    //     the in-flight start with "The connection was stopped during
    //     negotiation" and makes the server log an aborted connection. The
    //     cleanup below waits for start() to settle before stopping.
    //  2. Both effect runs close over the SAME setState. The discarded
    //     connection's onclose would otherwise fire after the live one reported
    //     'connected' and overwrite it with 'disconnected' — a working socket
    //     showing as offline. `disposed` makes a torn-down connection unable to
    //     report anything.
    let disposed = false;

    const connection = new HubConnectionBuilder()
      .withUrl(`${PORTAL_API_BASE}/hubs/chat`)
      .withAutomaticReconnect()
      .configureLogging(LogLevel.Warning)
      .build();

    connection.on('Thought', (thought: Thought) => {
      patchCurrent(turn => ({ ...turn, thoughts: [...turn.thoughts, thought] }));
    });

    connection.on('Token', (token: string) => {
      pendingRef.current += token;
      scheduleFlush();
    });

    connection.on('Matches', (matches: MatchResult) => {
      patchCurrent(turn => ({ ...turn, matches }));
    });

    // Postings offered when there is no CV. A separate event from Matches so the
    // UI cannot accidentally render a fit score for someone it knows nothing about.
    connection.on('Suggestions', (suggestions: JobSuggestionResult) => {
      patchCurrent(turn => ({ ...turn, suggestions }));
    });

    connection.on('UiAction', (action: UiAction) => {
      // Copilot Mode. Handed straight to the app, so a command like "remote
      // roles over 120k" changes the job board rather than only the wording of
      // the reply.
      actionRef.current?.(action);
    });

    connection.on('Error', (message: string) => {
      patchCurrent(turn => ({ ...turn, error: message, streaming: false }));
    });

    connection.on('Complete', () => {
      flush();
    });

    connection.on('End', () => {
      // Always fires, however the turn ended. Without it a dropped connection
      // would leave the typing indicator up forever and read as a hang.
      flush();
      patchCurrent(turn => ({ ...turn, streaming: false }));
      turnIdRef.current = null;
      setBusy(false);
    });

    connection.onreconnecting(() => { if (!disposed) setState('reconnecting'); });
    connection.onreconnected(() => { if (!disposed) setState('connected'); });
    connection.onclose(() => { if (!disposed) setState('disconnected'); });

    connectionRef.current = connection;
    setState('connecting');

    const started = connection.start()
      .then(() => { if (!disposed) setState('connected'); })
      .catch(() => { if (!disposed) setState('disconnected'); });

    return () => {
      disposed = true;
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      // Stop only once negotiation has settled, so the socket is closed rather
      // than aborted half-open.
      void started.finally(() => connection.stop());
      if (connectionRef.current === connection) connectionRef.current = null;
    };
  }, [enabled, flush, patchCurrent, scheduleFlush]);

  /** Opens an assistant turn for the reply that is about to stream in. */
  const beginAssistantTurn = useCallback(() => {
    const id = `a-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    turnIdRef.current = id;
    pendingRef.current = '';
    setTurns(previous => [...previous, {
      id, role: 'assistant', content: '', thoughts: [], matches: null, suggestions: null, streaming: true,
    }]);
    setBusy(true);
  }, []);

  const failCurrent = useCallback((message: string) => {
    patchCurrent(turn => ({ ...turn, error: message, streaming: false }));
    turnIdRef.current = null;
    setBusy(false);
  }, [patchCurrent]);

  const send = useCallback(async (message: string) => {
    const connection = connectionRef.current;
    const trimmed = message.trim();
    if (!trimmed || busy) return;

    setTurns(previous => [...previous, {
      id: `u-${Date.now()}`, role: 'user', content: trimmed,
      thoughts: [], matches: null, suggestions: null, streaming: false,
    }]);

    beginAssistantTurn();

    if (connection?.state !== HubConnectionState.Connected) {
      failCurrent('Not connected to the job portal API. Check that it is running on ' +
                  `${PORTAL_API_BASE}.`);
      return;
    }

    try {
      await connection.invoke('SendMessage', sessionId, trimmed);
    } catch (error) {
      failCurrent(error instanceof Error ? error.message : 'The message could not be sent.');
    }
  }, [beginAssistantTurn, busy, failCurrent, sessionId]);

  const analyzeResume = useCallback(async (resumeId: number) => {
    const connection = connectionRef.current;
    beginAssistantTurn();

    if (connection?.state !== HubConnectionState.Connected) {
      failCurrent('Not connected to the job portal API. Check that it is running on ' +
                  `${PORTAL_API_BASE}.`);
      return;
    }

    try {
      await connection.invoke('AnalyzeResume', sessionId, resumeId);
    } catch (error) {
      failCurrent(error instanceof Error ? error.message : 'The resume could not be analysed.');
    }
  }, [beginAssistantTurn, failCurrent, sessionId]);

  const appendLocal = useCallback((exchange: LocalExchange) => {
    const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setTurns(previous => [
      ...previous,
      ...(exchange.user
        ? [{
            id: `u-${stamp}`, role: 'user' as const, content: exchange.user,
            thoughts: [], matches: null, suggestions: null, streaming: false,
          }]
        : []),
      {
        id: `l-${stamp}`, role: 'assistant' as const, content: exchange.assistant,
        thoughts: [], matches: null, suggestions: null, streaming: false, options: exchange.options,
      },
    ]);
  }, []);

  const clearTurns = useCallback(() => {
    // Any in-flight turn is abandoned along with the transcript, so a reply that
    // lands after a reset cannot patch a turn that no longer exists.
    turnIdRef.current = null;
    pendingRef.current = '';
    setTurns([]);
    setBusy(false);
  }, []);

  return { state, turns, busy, send, analyzeResume, appendLocal, clearTurns };
}
