import { useState, useEffect } from 'react';
import { checkHealth } from '@/api/client';

/**
 * Polls backend health every `intervalMs` milliseconds.
 * Returns { healthy: boolean | null } -- null means "checking".
 */
export function useBackendHealth(intervalMs = 10_000) {
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      const ok = await checkHealth();
      if (!cancelled) setHealthy(ok);
    }

    check();
    const id = setInterval(check, intervalMs);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { healthy };
}
