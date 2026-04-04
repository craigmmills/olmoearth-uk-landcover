import { useState, useEffect } from 'react';
import { listSessions } from '@/api/client';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import type { SessionSummary } from '@/types';

interface SessionSelectorProps {
  selectedSessionId: string | null;
  onSessionChange: (sessionId: string) => void;
}

function formatSessionLabel(session: SessionSummary): string {
  // session_legacy is a special case
  if (session.session_id === 'session_legacy') {
    return `Legacy Session (${session.n_iterations} iterations)`;
  }

  try {
    const date = new Date(session.start_time);
    if (isNaN(date.getTime())) {
      return `${session.session_id} (${session.n_iterations} iters)`;
    }
    const dateStr = date.toLocaleDateString('en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    });
    const timeStr = date.toLocaleTimeString('en-GB', {
      hour: '2-digit',
      minute: '2-digit',
    });
    return `${dateStr} ${timeStr} (${session.n_iterations} iters)`;
  } catch {
    return `${session.session_id} (${session.n_iterations} iters)`;
  }
}

export function SessionSelector({ selectedSessionId, onSessionChange }: SessionSelectorProps) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    listSessions()
      .then((data) => {
        if (cancelled) return;
        // Sort by start_time descending (most recent first)
        const sorted = [...data].sort((a, b) => b.start_time.localeCompare(a.start_time));
        setSessions(sorted);
        setLoading(false);

        // Auto-select the most recent session if none selected
        if (!selectedSessionId && sorted.length > 0) {
          onSessionChange(sorted[0].session_id);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load sessions');
        setLoading(false);
      });

    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) {
    return <Skeleton className="h-10 w-full" />;
  }

  if (error) {
    return (
      <div className="rounded-md bg-destructive/10 p-2 text-xs text-destructive">
        {error}
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">No sessions found.</div>
    );
  }

  return (
    <Select value={selectedSessionId ?? undefined} onValueChange={onSessionChange}>
      <SelectTrigger className="w-full">
        <SelectValue placeholder="Select a session..." />
      </SelectTrigger>
      <SelectContent>
        {sessions.map((session) => (
          <SelectItem key={session.session_id} value={session.session_id}>
            {formatSessionLabel(session)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
