import { Loader2, CheckCircle2, AlertCircle, Wifi } from 'lucide-react';
import type { LoopStatus as LoopStatusType } from '@/types';

interface LoopStatusProps {
  status: LoopStatusType;
}

export default function LoopStatus({ status }: LoopStatusProps) {
  switch (status.state) {
    case 'connecting':
      return (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Wifi className="h-3 w-3 animate-pulse" />
          Connecting to event stream...
        </div>
      );
    case 'running':
      return (
        <div
          key={status.currentIteration}
          className="flex items-center gap-2 text-xs text-blue-600 animate-fade-in-up"
        >
          <Loader2 className="h-3 w-3 animate-spin" />
          {status.currentIteration === 0
            ? 'Starting up...'
            : `Iteration ${status.currentIteration} running...`}
        </div>
      );
    case 'complete':
      return (
        <div className="flex items-center gap-2 text-xs text-green-600">
          <CheckCircle2 className="h-3 w-3" />
          Loop complete &mdash; {status.nIterations} iterations
          {status.bestScore !== null && `, best score ${status.bestScore.toFixed(2)}`}
        </div>
      );
    case 'idle':
      return (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="inline-block w-2 h-2 rounded-full bg-gray-400" />
          No active loop
        </div>
      );
    case 'error':
      return (
        <div className="flex items-center gap-2 text-xs text-destructive">
          <AlertCircle className="h-3 w-3" />
          {status.message}
        </div>
      );
  }
}
