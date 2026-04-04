import type { IterationSummary } from '@/types';

interface SummaryStatsProps {
  iterations: IterationSummary[];
  bestIteration: number | null;
}

export function SummaryStats({ iterations, bestIteration }: SummaryStatsProps) {
  if (iterations.length === 0) return null;

  const total = iterations.length;
  const accepted = iterations.filter((it) => it.status === 'accepted').length;
  const reverted = iterations.filter((it) => it.status === 'reverted').length;

  // Build accuracy trajectory
  const sorted = [...iterations].sort((a, b) => a.iteration - b.iteration);
  const trajectory = sorted
    .filter((it) => it.overall_accuracy != null)
    .map((it) => (it.overall_accuracy! * 100).toFixed(1) + '%');

  // Best score
  const bestScore = bestIteration
    ? iterations.find((it) => it.iteration === bestIteration)?.overall_accuracy
    : null;

  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="rounded-md border p-2 text-center">
        <div className="text-lg font-bold">{total}</div>
        <div className="text-xs text-muted-foreground">Iterations</div>
      </div>
      <div className="rounded-md border p-2 text-center">
        <div className="text-lg font-bold text-green-600">{accepted}</div>
        <div className="text-xs text-muted-foreground">Accepted</div>
      </div>
      <div className="rounded-md border p-2 text-center">
        <div className="text-lg font-bold text-red-600">{reverted}</div>
        <div className="text-xs text-muted-foreground">Reverted</div>
      </div>
      <div className="rounded-md border p-2 text-center">
        <div className="text-lg font-bold text-amber-600">
          {bestScore != null ? `${(bestScore * 100).toFixed(1)}%` : '--'}
        </div>
        <div className="text-xs text-muted-foreground">Best Score</div>
      </div>

      {/* Score trajectory */}
      {trajectory.length > 1 && (
        <div className="col-span-2 rounded-md border p-2">
          <div className="text-xs text-muted-foreground mb-1">Score Trajectory</div>
          <div className="text-xs font-mono truncate">
            {trajectory.join(' → ')}
          </div>
        </div>
      )}
    </div>
  );
}
