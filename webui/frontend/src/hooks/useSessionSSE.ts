import { useEffect, useRef } from 'react';
import { API_BASE_URL } from '@/constants';

interface SSENewIterationEvent {
  session_id: string;
  iteration: number;
  timestamp: string;
}

/**
 * Subscribe to backend SSE events for live iteration updates.
 * Calls onNewIteration when a new_iteration event is received.
 * EventSource auto-reconnects on failure -- no user-facing errors needed.
 */
export function useSessionSSE(
  onNewIteration: (event: SSENewIterationEvent) => void,
): void {
  const callbackRef = useRef(onNewIteration);
  callbackRef.current = onNewIteration;

  useEffect(() => {
    const es = new EventSource(`${API_BASE_URL}/api/events`);

    es.addEventListener('new_iteration', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data) as SSENewIterationEvent;
        callbackRef.current(data);
      } catch {
        // Ignore malformed events
      }
    });

    es.onerror = () => {
      // EventSource auto-reconnects; no action needed
    };

    return () => es.close();
  }, []);
}
