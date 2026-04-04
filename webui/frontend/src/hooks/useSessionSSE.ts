import { useState, useEffect, useRef } from 'react';
import { API_BASE_URL } from '@/constants';
import type { SSENewIterationEvent, SSESessionCompleteEvent } from '@/types';

export interface UseSessionSSEOptions {
  /** Called when a new_iteration event arrives */
  onNewIteration?: (event: SSENewIterationEvent) => void;
  /** Called when a session_complete event arrives */
  onSessionComplete?: (event: SSESessionCompleteEvent) => void;
}

export interface UseSessionSSEState {
  connected: boolean;
  error: string | null;
}

const MIN_RETRY_MS = 1_000;
const MAX_RETRY_MS = 30_000;
const BACKOFF_FACTOR = 2;

/**
 * Subscribe to backend SSE events for live iteration and session updates.
 * Manages EventSource lifecycle with exponential backoff reconnection.
 *
 * Callbacks are stored in refs so the EventSource is NOT torn down
 * on every re-render -- the effect dependency array stays empty.
 */
export function useSessionSSE(
  options: UseSessionSSEOptions = {},
): UseSessionSSEState {
  const [state, setState] = useState<UseSessionSSEState>({
    connected: false,
    error: null,
  });

  // Store callbacks in refs so reconnection logic always sees latest
  const onNewIterationRef = useRef(options.onNewIteration);
  const onSessionCompleteRef = useRef(options.onSessionComplete);
  onNewIterationRef.current = options.onNewIteration;
  onSessionCompleteRef.current = options.onSessionComplete;

  useEffect(() => {
    let cancelled = false;
    let retryMs = MIN_RETRY_MS;
    let retryTimeout: ReturnType<typeof setTimeout> | null = null;
    let es: EventSource | null = null;

    function connect() {
      if (cancelled) return;

      es = new EventSource(`${API_BASE_URL}/api/events`, {
        withCredentials: true,
      });

      es.onopen = () => {
        if (!cancelled) {
          retryMs = MIN_RETRY_MS; // reset backoff on successful connect
          setState({ connected: true, error: null });
        }
      };

      es.addEventListener('new_iteration', (e: MessageEvent) => {
        if (cancelled) return;
        try {
          const data: SSENewIterationEvent = JSON.parse(e.data);
          onNewIterationRef.current?.(data);
        } catch {
          // Malformed event data -- ignore silently
        }
      });

      es.addEventListener('session_complete', (e: MessageEvent) => {
        if (cancelled) return;
        try {
          const data: SSESessionCompleteEvent = JSON.parse(e.data);
          onSessionCompleteRef.current?.(data);
        } catch {
          // Malformed event data -- ignore silently
        }
      });

      es.onerror = () => {
        if (cancelled) return;
        es?.close();
        setState({ connected: false, error: 'SSE connection lost. Reconnecting...' });
        const delay = retryMs;
        retryMs = Math.min(retryMs * BACKOFF_FACTOR, MAX_RETRY_MS);
        retryTimeout = setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      cancelled = true;
      es?.close();
      if (retryTimeout) clearTimeout(retryTimeout);
    };
  }, []);

  return state;
}
