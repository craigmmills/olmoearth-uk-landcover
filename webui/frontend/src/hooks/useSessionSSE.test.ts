import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSessionSSE } from './useSessionSSE';

// Mock EventSource
class MockEventSource {
  static instances: MockEventSource[] = [];
  url: string;
  withCredentials: boolean;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  listeners: Record<string, ((e: MessageEvent) => void)[]> = {};
  readyState = 0;

  constructor(url: string, init?: EventSourceInit) {
    this.url = url;
    this.withCredentials = init?.withCredentials ?? false;
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, fn: (e: MessageEvent) => void) {
    (this.listeners[type] ??= []).push(fn);
  }

  removeEventListener = vi.fn();
  close = vi.fn();

  // Test helpers
  simulateOpen() {
    this.readyState = 1;
    this.onopen?.();
  }

  simulateEvent(type: string, data: unknown) {
    const event = new MessageEvent(type, { data: JSON.stringify(data) });
    this.listeners[type]?.forEach((fn) => fn(event));
  }

  simulateError() {
    this.onerror?.();
  }
}

beforeEach(() => {
  MockEventSource.instances = [];
  vi.stubGlobal('EventSource', MockEventSource);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useSessionSSE', () => {
  it('connects to /api/events on mount', () => {
    renderHook(() => useSessionSSE());
    expect(MockEventSource.instances.length).toBeGreaterThanOrEqual(1);
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    expect(last.url).toContain('/api/events');
  });

  it('sets withCredentials to true', () => {
    renderHook(() => useSessionSSE());
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    expect(last.withCredentials).toBe(true);
  });

  it('returns connected=false initially', () => {
    const { result } = renderHook(() => useSessionSSE());
    expect(result.current.connected).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('sets connected=true on open', () => {
    const { result } = renderHook(() => useSessionSSE());
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    act(() => last.simulateOpen());
    expect(result.current.connected).toBe(true);
    expect(result.current.error).toBeNull();
  });

  it('calls onNewIteration callback when event arrives', () => {
    const onNewIteration = vi.fn();
    renderHook(() => useSessionSSE({ onNewIteration }));
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    act(() => {
      last.simulateOpen();
      last.simulateEvent('new_iteration', {
        session_id: 'session_20260404_093509',
        iteration: 3,
        timestamp: '2026-04-04T10:00:00',
      });
    });
    expect(onNewIteration).toHaveBeenCalledWith({
      session_id: 'session_20260404_093509',
      iteration: 3,
      timestamp: '2026-04-04T10:00:00',
    });
  });

  it('calls onSessionComplete callback when event arrives', () => {
    const onSessionComplete = vi.fn();
    renderHook(() => useSessionSSE({ onSessionComplete }));
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    act(() => {
      last.simulateOpen();
      last.simulateEvent('session_complete', {
        session_id: 'session_20260404_093509',
        end_time: '2026-04-04T11:00:00',
        stop_reason: 'target_reached',
        best_iteration: 5,
        final_score: 0.87,
        n_iterations: 5,
      });
    });
    expect(onSessionComplete).toHaveBeenCalledWith(
      expect.objectContaining({ stop_reason: 'target_reached', final_score: 0.87 }),
    );
  });

  it('ignores malformed event data', () => {
    const onNewIteration = vi.fn();
    renderHook(() => useSessionSSE({ onNewIteration }));
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    act(() => {
      last.simulateOpen();
      // Send non-JSON data
      const event = new MessageEvent('new_iteration', { data: 'NOT JSON{{{' });
      last.listeners['new_iteration']?.forEach((fn) => fn(event));
    });
    expect(onNewIteration).not.toHaveBeenCalled();
  });

  it('reconnects with backoff on error', async () => {
    vi.useFakeTimers();
    renderHook(() => useSessionSSE());
    const first = MockEventSource.instances[MockEventSource.instances.length - 1];

    // Simulate error
    act(() => first.simulateError());
    expect(first.close).toHaveBeenCalled();

    // After MIN_RETRY_MS (1000ms), should create a new EventSource
    const countBefore = MockEventSource.instances.length;
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(MockEventSource.instances.length).toBe(countBefore + 1);

    // Simulate second error -- backoff should be 2000ms now
    const second = MockEventSource.instances[MockEventSource.instances.length - 1];
    act(() => second.simulateError());

    // After 1000ms, should NOT have reconnected yet (backoff is 2000ms)
    const countBefore2 = MockEventSource.instances.length;
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(MockEventSource.instances.length).toBe(countBefore2);

    // After another 1000ms (total 2000ms), should reconnect
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(MockEventSource.instances.length).toBe(countBefore2 + 1);

    vi.useRealTimers();
  });

  it('resets backoff on successful reconnect', async () => {
    vi.useFakeTimers();
    renderHook(() => useSessionSSE());
    const first = MockEventSource.instances[MockEventSource.instances.length - 1];

    // Error -> reconnect after 1s
    act(() => first.simulateError());
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    // Successful reconnect
    const second = MockEventSource.instances[MockEventSource.instances.length - 1];
    act(() => second.simulateOpen());

    // Error again -> should wait 1s (backoff reset), not 2s
    act(() => second.simulateError());
    const countBefore = MockEventSource.instances.length;
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(MockEventSource.instances.length).toBe(countBefore + 1);

    vi.useRealTimers();
  });

  it('sets error state on connection error', () => {
    const { result } = renderHook(() => useSessionSSE());
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    act(() => last.simulateError());
    expect(result.current.connected).toBe(false);
    expect(result.current.error).toBe('SSE connection lost. Reconnecting...');
  });

  it('closes EventSource on unmount', () => {
    const { unmount } = renderHook(() => useSessionSSE());
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    unmount();
    expect(last.close).toHaveBeenCalled();
  });
});
