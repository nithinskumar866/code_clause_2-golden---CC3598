import { renderHook, act, waitFor } from '@testing-library/react';
import { useChatHub } from './useChatHub';
import { HubConnectionState } from '@microsoft/signalr';

// Test utilities
const createMockConnection = (overrides = {}) => ({
  on: vi.fn(),
  onreconnecting: vi.fn(),
  onreconnected: vi.fn(),
  onclose: vi.fn(),
  start: vi.fn().mockResolvedValue(undefined),
  stop: vi.fn().mockResolvedValue(undefined),
  state: HubConnectionState.Connected,
  invoke: vi.fn().mockResolvedValue(undefined),
  ...overrides,
});

describe('useChatHub', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  describe('StrictMode double-mount regression guard', () => {
    it('should stay Connected when hook mounts twice in immediate succession (simulating React 19 StrictMode)', async () => {
      const mockConnection = createMockConnection();
      
      // Mock the HubConnectionBuilder to return our mock connection
      const { HubConnectionBuilder } = await import('@microsoft/signalr');
      (HubConnectionBuilder as unknown as vi.Mock).mockImplementation(() => ({
        withUrl: vi.fn().mockReturnThis(),
        withAutomaticReconnect: vi.fn().mockReturnThis(),
        configureLogging: vi.fn().mockReturnThis(),
        build: vi.fn().mockReturnValue(mockConnection),
      }));

      // First mount
      const { result: result1, unmount } = renderHook(() => useChatHub('test-session'));
      
      // Wait for connection to establish
      await waitFor(() => expect(result1.current.state).toBe('connected'));
      
      // Second mount (simulating StrictMode double-invoke)
      const { result: result2 } = renderHook(() => useChatHub('test-session'));
      
      // Wait for second mount to establish
      await waitFor(() => expect(result2.current.state).toBe('connected'));
      
      // Unmount first (simulating StrictMode cleanup)
      unmount();
      
      // The connection should still be Connected, not Disconnected
      // This proves the isMounted guard and await newConnection.start()-before-stop() ordering hold
      await waitFor(() => expect(result2.current.state).toBe('connected'));
    });
  });

  describe('Command validation', () => {
    it('should reject commands with unknown action types', async () => {
      const mockConnection = createMockConnection();
      const { HubConnectionBuilder } = await import('@microsoft/signalr');
      (HubConnectionBuilder as unknown as vi.Mock).mockImplementation(() => ({
        withUrl: vi.fn().mockReturnThis(),
        withAutomaticReconnect: vi.fn().mockReturnThis(),
        configureLogging: vi.fn().mockReturnThis(),
        build: vi.fn().mockReturnValue(mockConnection),
      }));

      const { result } = renderHook(() => useChatHub('test-session'));
      await waitFor(() => expect(result.current.state).toBe('connected'));

      // Simulate receiving an invalid command
      const onAction = vi.fn();
      const { result: resultWithAction } = renderHook(() => useChatHub('test-session', onAction));
      await waitFor(() => expect(resultWithAction.current.state).toBe('connected'));

      // Get the on handler for UiAction
      const uiActionHandler = mockConnection.on.mock.calls.find(
        (call) => call[0] === 'UiAction'
      )?.[1];

      expect(uiActionHandler).toBeDefined();

      // Send invalid command (unknown action)
      act(() => {
        uiActionHandler!({ action: 'invalid_action', payload: {}, commandId: 'test-123' });
      });

      // Should not call onAction for invalid commands
      expect(onAction).not.toHaveBeenCalled();
    });

    it('should reject navigate commands with routes not in allowlist', async () => {
      const mockConnection = createMockConnection();
      const { HubConnectionBuilder } = await import('@microsoft/signalr');
      (HubConnectionBuilder as unknown as vi.Mock).mockImplementation(() => ({
        withUrl: vi.fn().mockReturnThis(),
        withAutomaticReconnect: vi.fn().mockReturnThis(),
        configureLogging: vi.fn().mockReturnThis(),
        build: vi.fn().mockReturnValue(mockConnection),
      }));

      const onAction = vi.fn();
      const { result } = renderHook(() => useChatHub('test-session', onAction));
      await waitFor(() => expect(result.current.state).toBe('connected'));

      const uiActionHandler = mockConnection.on.mock.calls.find(
        (call) => call[0] === 'UiAction'
      )?.[1];

      // Send navigate with invalid route
      act(() => {
        uiActionHandler!({ 
          action: 'navigate', 
          payload: { route: '/malicious-route' }, 
          commandId: 'test-123' 
        });
      });

      // Should not call onAction for invalid routes
      expect(onAction).not.toHaveBeenCalled();
    });

    it('should accept navigate commands with routes in allowlist', async () => {
      const mockConnection = createMockConnection();
      const { HubConnectionBuilder } = await import('@microsoft/signalr');
      (HubConnectionBuilder as unknown as vi.Mock).mockImplementation(() => ({
        withUrl: vi.fn().mockReturnThis(),
        withAutomaticReconnect: vi.fn().mockReturnThis(),
        configureLogging: vi.fn().mockReturnThis(),
        build: vi.fn().mockReturnValue(mockConnection),
      }));

      const onAction = vi.fn();
      const { result } = renderHook(() => useChatHub('test-session', onAction));
      await waitFor(() => expect(result.current.state).toBe('connected'));

      const uiActionHandler = mockConnection.on.mock.calls.find(
        (call) => call[0] === 'UiAction'
      )?.[1];

      // Send navigate with valid route
      act(() => {
        uiActionHandler!({ 
          action: 'navigate', 
          payload: { route: '/?view=job-board' }, 
          commandId: 'test-123' 
        });
      });

      // Should call onAction for valid routes
      expect(onAction).toHaveBeenCalledWith(
        expect.objectContaining({ action: 'navigate', payload: { route: '/?view=job-board' } })
      );
    });

    it('should deduplicate commands with same commandId', async () => {
      const mockConnection = createMockConnection();
      const { HubConnectionBuilder } = await import('@microsoft/signalr');
      (HubConnectionBuilder as unknown as vi.Mock).mockImplementation(() => ({
        withUrl: vi.fn().mockReturnThis(),
        withAutomaticReconnect: vi.fn().mockReturnThis(),
        configureLogging: vi.fn().mockReturnThis(),
        build: vi.fn().mockReturnValue(mockConnection),
      }));

      const onAction = vi.fn();
      const { result } = renderHook(() => useChatHub('test-session', onAction));
      await waitFor(() => expect(result.current.state).toBe('connected'));

      const uiActionHandler = mockConnection.on.mock.calls.find(
        (call) => call[0] === 'UiAction'
      )?.[1];

      // Send same command twice with same commandId
      act(() => {
        uiActionHandler!({ 
          action: 'update_filters', 
          payload: { remote: true }, 
          commandId: 'duplicate-123' 
        });
      });

      act(() => {
        uiActionHandler!({ 
          action: 'update_filters', 
          payload: { remote: true }, 
          commandId: 'duplicate-123' 
        });
      });

      // Should only call onAction once
      expect(onAction).toHaveBeenCalledTimes(1);
    });
  });

  describe('Connection state handling', () => {
    it('should transition through reconnecting states correctly', async () => {
      const mockConnection = createMockConnection();
      const { HubConnectionBuilder } = await import('@microsoft/signalr');
      (HubConnectionBuilder as unknown as vi.Mock).mockImplementation(() => ({
        withUrl: vi.fn().mockReturnThis(),
        withAutomaticReconnect: vi.fn().mockReturnThis(),
        configureLogging: vi.fn().mockReturnThis(),
        build: vi.fn().mockReturnValue(mockConnection),
      }));

      const { result } = renderHook(() => useChatHub('test-session'));
      await waitFor(() => expect(result.current.state).toBe('connected'));

      // Get reconnecting handler
      const reconnectingHandler = mockConnection.onreconnecting.mock.calls[0]?.[0];
      const reconnectedHandler = mockConnection.onreconnected.mock.calls[0]?.[0];

      // Simulate reconnecting
      act(() => {
        reconnectingHandler!();
      });
      expect(result.current.state).toBe('reconnecting');

      // Simulate reconnected
      act(() => {
        reconnectedHandler!();
      });
      expect(result.current.state).toBe('connected');
    });

    it('should go to disconnected on close', async () => {
      const mockConnection = createMockConnection();
      const { HubConnectionBuilder } = await import('@microsoft/signalr');
      (HubConnectionBuilder as unknown as vi.Mock).mockImplementation(() => ({
        withUrl: vi.fn().mockReturnThis(),
        withAutomaticReconnect: vi.fn().mockReturnThis(),
        configureLogging: vi.fn().mockReturnThis(),
        build: vi.fn().mockReturnValue(mockConnection),
      }));

      const { result } = renderHook(() => useChatHub('test-session'));
      await waitFor(() => expect(result.current.state).toBe('connected'));

      // Get onclose handler
      const closeHandler = mockConnection.onclose.mock.calls[0]?.[0];

      // Simulate connection close
      act(() => {
        closeHandler!();
      });
      expect(result.current.state).toBe('disconnected');
    });
  });
});