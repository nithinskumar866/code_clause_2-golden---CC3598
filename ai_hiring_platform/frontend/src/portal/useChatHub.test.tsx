import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StrictMode } from 'react';
import { render, screen, act, cleanup } from '@testing-library/react';
import { useChatHub } from './useChatHub';

/**
 * Regression tests for the failure the chat shipped with:
 *
 *   Failed to start the connection: The connection was stopped during negotiation.
 *   Connection disconnected with error 'Server returned an error on close'.
 *
 * StrictMode double-invokes the effect on the same component instance, so a
 * second connection is built while the first is still negotiating. Both bugs
 * below follow from that, and both made a working chat look offline.
 */

const { connections } = vi.hoisted(() => ({ connections: [] as FakeConnection[] }));

class FakeConnection {
  state = 'Disconnected';
  stopped = false;
  private settleStart: (() => void) | null = null;
  private closeCallbacks: Array<() => void> = [];

  on() { /* event handlers are not what these tests exercise */ }
  onreconnecting() {}
  onreconnected() {}
  onclose(callback: () => void) { this.closeCallbacks.push(callback); }
  invoke() { return Promise.resolve(); }

  /** Negotiation stays pending until the test says otherwise. */
  start() {
    return new Promise<void>(resolve => {
      this.settleStart = () => { this.state = 'Connected'; resolve(); };
    });
  }

  stop() {
    this.stopped = true;
    // Real SignalR notifies close handlers when a connection is stopped.
    this.closeCallbacks.forEach(callback => callback());
    return Promise.resolve();
  }

  /** Completes this connection's negotiation. */
  settle() { this.settleStart?.(); }
}

vi.mock('@microsoft/signalr', () => ({
  HubConnectionBuilder: class {
    withUrl() { return this; }
    withAutomaticReconnect() { return this; }
    configureLogging() { return this; }
    build() {
      const connection = new FakeConnection();
      connections.push(connection);
      return connection;
    }
  },
  HubConnectionState: { Connected: 'Connected', Disconnected: 'Disconnected' },
  LogLevel: { Warning: 3 },
}));

function Harness() {
  const { state } = useChatHub('session-1', true);
  return <span data-testid="state">{state}</span>;
}

const stateText = () => screen.getByTestId('state').textContent;

beforeEach(() => {
  connections.length = 0;
});

afterEach(() => cleanup());

describe('useChatHub under StrictMode', () => {
  it('does not stop a connection that is still negotiating', async () => {
    render(<StrictMode><Harness /></StrictMode>);

    // StrictMode built a second connection and tore the first one down.
    expect(connections.length).toBe(2);

    // The torn-down connection must still be negotiating, not aborted: calling
    // stop() here is what produced "stopped during negotiation" and made the
    // server log a connection closed with an error.
    expect(connections[0].stopped).toBe(false);

    // Once negotiation settles, the discarded connection is closed properly.
    await act(async () => { connections[0].settle(); });
    expect(connections[0].stopped).toBe(true);
  });

  it('does not let a discarded connection report the live one as disconnected', async () => {
    render(<StrictMode><Harness /></StrictMode>);

    // The connection that survived StrictMode comes up.
    await act(async () => { connections[1].settle(); });
    expect(stateText()).toBe('connected');

    // The discarded one now finishes negotiating and is stopped, firing its
    // close handler. Both effect runs close over the same setState, so before
    // the `disposed` guard this overwrote a live connection with 'disconnected'
    // — the UI showed offline while the socket was working.
    await act(async () => { connections[0].settle(); });
    expect(stateText()).toBe('connected');
  });

  it('reports disconnected when the surviving connection genuinely fails', async () => {
    render(<StrictMode><Harness /></StrictMode>);

    await act(async () => { connections[1].settle(); });
    expect(stateText()).toBe('connected');

    await act(async () => { connections[1].stop(); });
    expect(stateText()).toBe('disconnected');
  });
});

describe('useChatHub when disabled', () => {
  it('opens no connection until the chat is enabled', () => {
    function Disabled() {
      const { state } = useChatHub('session-1', false);
      return <span data-testid="state">{state}</span>;
    }
    render(<StrictMode><Disabled /></StrictMode>);

    // A recruiter who never opens the assistant should not have a socket opened
    // to the portal API on their behalf.
    expect(connections.length).toBe(0);
    expect(stateText()).toBe('idle');
  });
});
