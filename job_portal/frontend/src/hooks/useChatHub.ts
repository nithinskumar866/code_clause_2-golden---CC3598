import { useCallback, useEffect, useRef, useState } from 'react';
import {
  HubConnection, HubConnectionBuilder, HubConnectionState, LogLevel,
} from '@microsoft/signalr';
import { API_BASE } from '../api/client';
import type { JobFilters, MatchResult, Thought, UiAction, CopilotCommand, UpdateFiltersPayload } from '../types';
import { ALLOWED_ROUTES } from '../types';

export type ConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  /** Grows token by token while the reply streams. */
  content: string;
  /** The visible reasoning for this turn. Assistant turns only. */
  thoughts: Thought[];
  matches: MatchResult | null;
  streaming: boolean;
  error?: string;
}

export interface UseChatHub {
  state: ConnectionState;
  turns: ChatTurn[];
  filters: JobFilters;
  busy: boolean;
  send: (message: string) => Promise<void>;
  analyzeResume: (resumeId: number) => Promise<void>;
  setFilters: (filters: JobFilters) => void;
  /** Messages queued while reconnecting, flushed on reconnect */
  queuedMessages: string[];
  /** Whether there are unsent messages in the queue */
  hasQueuedMessages: boolean;
}

/**
 * LRU cache for command deduplication.
 * Prevents duplicate command execution on SignalR reconnect replay.
 */
class CommandDeduplicator {
  private readonly maxSize: number;
  private readonly cache: Map<string, number> = new Map();
  private readonly order: string[] = [];

  constructor(maxSize = 50) {
    this.maxSize = maxSize;
  }

  /** Returns true if this is a new command, false if duplicate. */
  tryAdd(commandId: string): boolean {
    if (this.cache.has(commandId)) {
      return false; // Duplicate
    }
    this.cache.set(commandId, Date.now());
    this.order.push(commandId);
    if (this.cache.size > this.maxSize) {
      const oldest = this.order.shift();
      if (oldest) this.cache.delete(oldest);
    }
    return true;
  }
}

/**
 * Validates a UiAction against the strict CopilotCommand schema.
 * Returns the validated command or null if invalid.
 */
function validateCommand(action: UiAction): CopilotCommand | null {
  if (!action.commandId) {
    console.warn('[CommandValidation] Rejected: missing commandId', action);
    return null;
  }

  // Check allowed actions
  const allowedActions = ['update_filters', 'show_jobs', 'navigate', 'resume_ready'];
  if (!allowedActions.includes(action.action)) {
    console.warn('[CommandValidation] Rejected: unknown action', action);
    return null;
  }

  try {
    switch (action.action) {
      case 'update_filters': {
        const payload = action.payload as Record<string, unknown> | null;
        if (!payload) return null;
        const validated: UpdateFiltersPayload = {};
        if (typeof payload.remote === 'boolean') validated.remote = payload.remote;
        if (typeof payload.minSalary === 'number') validated.minSalary = payload.minSalary;
        if (typeof payload.maxSalary === 'number') validated.maxSalary = payload.maxSalary;
        if (typeof payload.location === 'string') validated.location = payload.location;
        if (typeof payload.jobType === 'string') validated.jobType = payload.jobType;
        // Strip unknown keys - only allow known properties
        return { type: 'update_filters', payload: validated, commandId: action.commandId };
      }
      case 'show_jobs': {
        const payload = action.payload as Record<string, unknown> | null;
        if (!payload || !Array.isArray(payload.jobIds)) {
          console.warn('[CommandValidation] Rejected show_jobs: jobIds array missing', action);
          return null;
        }
        const jobIds = payload.jobIds.filter((id): id is number => typeof id === 'number');
        return { type: 'show_jobs', payload: { jobIds }, commandId: action.commandId };
      }
      case 'navigate': {
        const payload = action.payload as Record<string, unknown> | null;
        if (!payload || typeof payload.route !== 'string') {
          console.warn('[CommandValidation] Rejected navigate: route string missing', action);
          return null;
        }
        const route = payload.route;
        if (!ALLOWED_ROUTES.includes(route as typeof ALLOWED_ROUTES[number])) {
          console.warn('[CommandValidation] Rejected navigate: route not in allowlist', route);
          return null;
        }
        return { type: 'navigate', payload: { route }, commandId: action.commandId };
      }
      case 'resume_ready':
        return { type: 'resume_ready', payload: {}, commandId: action.commandId };
      default:
        return null;
    }
  } catch (error) {
    console.warn('[CommandValidation] Rejected: validation error', error, action);
    return null;
  }
}

/**
 * Owns the WebSocket to the .NET hub and turns its events into render state.
 *
 * The streaming events land in a ref-backed "current turn" rather than being
 * pushed straight into React state one token at a time: a token event arriving
 * every few milliseconds would otherwise queue a render per token. Tokens
 * accumulate and are flushed on an animation frame, which keeps the text
 * animating smoothly without re-rendering the whole transcript hundreds of times.
 */
export function useChatHub(sessionId: string, onAction?: (action: UiAction) => void): UseChatHub {
  const [state, setState] = useState<ConnectionState>('connecting');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [filters, setFilters] = useState<JobFilters>({});
  const [busy, setBusy] = useState(false);
  const [queuedMessages, setQueuedMessages] = useState<string[]>([]);

  const connectionRef = useRef<HubConnection | null>(null);
  const pendingRef = useRef<string>('');
  const frameRef = useRef<number | null>(null);
  const turnIdRef = useRef<string | null>(null);
  const deduplicatorRef = useRef(new CommandDeduplicator());
  const messageQueueRef = useRef<string[]>([]);

  // Held in a ref so a caller passing an inline arrow does not tear down and
  // rebuild the whole hub connection on every render.
  const actionRef = useRef(onAction);
  actionRef.current = onAction;

  // Flush queued messages when connection is restored
  const flushMessageQueue = useCallback(() => {
    const connection = connectionRef.current;
    if (!connection || connection.state !== HubConnectionState.Connected) return;
    
    const queue = messageQueueRef.current;
    if (queue.length === 0) return;
    
    messageQueueRef.current = [];
    setQueuedMessages([]);
    
    queue.forEach(msg => {
      connection.invoke('SendMessage', sessionId, msg).catch(err => {
        console.error('[MessageQueue] Failed to send queued message:', err);
      });
    });
  }, [sessionId]);

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
    const connection = new HubConnectionBuilder()
      .withUrl(`${API_BASE}/hubs/chat`)
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

    connection.on('UiAction', (action: UiAction) => {
      // Client-side validation as defense in depth
      const validated = validateCommand(action);
      if (!validated) {
        return; // Silently drop invalid commands
      }

      // Deduplication: ignore duplicate commandIds (e.g., from SignalR reconnect replay)
      if (!deduplicatorRef.current.tryAdd(validated.commandId)) {
        console.debug('[CommandValidation] Duplicate commandId ignored', validated.commandId);
        return;
      }

      // Copilot Mode. The chat sidebar tracks filters for display; the handler
      // passed in is what actually drives the rest of the app, so a command like
      // "remote roles over 120k" changes the board grid and not just the wording
      // of the reply.
      if (validated.type === 'update_filters') {
        setFilters(validated.payload as JobFilters);
      }
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

    connection.onreconnecting(() => setState('reconnecting'));
    connection.onreconnected(() => setState('connected'));
    connection.onclose(() => setState('disconnected'));

    connectionRef.current = connection;

    connection.start()
      .then(() => setState('connected'))
      .catch(() => setState('disconnected'));

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      void connection.stop();
      connectionRef.current = null;
    };
  }, [flush, patchCurrent, scheduleFlush]);

  /** Opens an assistant turn for the reply that is about to stream in. */
  const beginAssistantTurn = useCallback(() => {
    const id = `a-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    turnIdRef.current = id;
    pendingRef.current = '';
    setTurns(previous => [...previous, {
      id, role: 'assistant', content: '', thoughts: [], matches: null, streaming: true,
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
      thoughts: [], matches: null, streaming: false,
    }]);

    beginAssistantTurn();

    if (connection?.state !== HubConnectionState.Connected) {
      // Queue the message for later delivery when reconnected
      messageQueueRef.current.push(trimmed);
      setQueuedMessages([...messageQueueRef.current]);
      failCurrent('Connection lost. Your message has been queued and will be sent when reconnected.');
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
      failCurrent('Not connected to the server. Check that the API is running.');
      return;
    }

    try {
      await connection.invoke('AnalyzeResume', sessionId, resumeId);
    } catch (error) {
      failCurrent(error instanceof Error ? error.message : 'The resume could not be analysed.');
    }
  }, [beginAssistantTurn, failCurrent, sessionId]);

  // Update queuedMessages when messageQueueRef changes
  useEffect(() => {
    setQueuedMessages([...messageQueueRef.current]);
  }, [state]); // Re-run when connection state changes

  // Flush queue when reconnected
  useEffect(() => {
    if (state === 'connected') {
      flushMessageQueue();
    }
  }, [state, flushMessageQueue]);

  return { 
    state,
    turns,
    filters,
    busy,
    send,
    analyzeResume,
    setFilters,
    queuedMessages,
    hasQueuedMessages: queuedMessages.length > 0,
  };
}
