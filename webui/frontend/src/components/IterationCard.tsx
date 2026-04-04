import { useState, useEffect, useRef } from 'react';
import { getIterationDetail } from '@/api/client';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { F1Chart } from '@/components/F1Chart';
import { ComparisonImages } from '@/components/ComparisonImages';
import type {
  IterationSummary,
  IterationDetail,
  Metrics,
  Hypothesis,
  Evaluation,
  ConfigDiffEntry,
} from '@/types';

interface IterationCardProps {
  sessionId: string;
  iteration: IterationSummary;
  isBest: boolean;
  isExpanded: boolean;
}

// --- Sub-components ---

/** Status badge: accepted (green), reverted (red), baseline (blue) */
function StatusBadge({ status, isFirst }: { status: string; isFirst: boolean }) {
  const displayStatus = isFirst ? 'baseline' : status;

  const variantMap: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
    accepted: 'default',
    baseline: 'secondary',
    reverted: 'destructive',
  };

  const colorMap: Record<string, string> = {
    accepted: 'bg-green-600 hover:bg-green-600/80 text-white',
    baseline: 'bg-blue-600 hover:bg-blue-600/80 text-white',
    reverted: 'bg-red-600 hover:bg-red-600/80 text-white',
  };

  return (
    <Badge
      variant={variantMap[displayStatus] ?? 'outline'}
      className={colorMap[displayStatus] ?? ''}
    >
      {displayStatus}
    </Badge>
  );
}

/** Score with delta indicator */
function DeltaIndicator({ accuracy, prevAccuracy }: { accuracy: number | null; prevAccuracy?: number | null }) {
  if (accuracy == null) return <span className="text-sm text-muted-foreground">--</span>;

  const pct = (accuracy * 100).toFixed(2);

  if (prevAccuracy == null) {
    return <span className="text-sm font-mono font-semibold">{pct}%</span>;
  }

  const delta = accuracy - prevAccuracy;
  const deltaPct = (delta * 100).toFixed(2);
  const isPositive = delta > 0;
  const isNeutral = delta === 0;

  return (
    <span className="text-sm font-mono font-semibold">
      {pct}%
      {!isNeutral && (
        <span className={isPositive ? 'text-green-600 ml-1' : 'text-red-600 ml-1'}>
          {isPositive ? '+' : ''}{deltaPct}%
        </span>
      )}
    </span>
  );
}

/** Hypothesis callout with component, reasoning, expected impact */
function HypothesisCallout({ hypothesis }: { hypothesis: Hypothesis }) {
  return (
    <div className="rounded-md border border-blue-200 bg-blue-50 p-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold uppercase text-blue-700">Hypothesis</span>
        <Badge variant="outline" className="text-xs">
          {hypothesis.component}
        </Badge>
      </div>
      <p className="text-sm text-blue-900">{hypothesis.hypothesis}</p>
      {hypothesis.reasoning && (
        <p className="text-xs text-blue-700">
          <span className="font-medium">Reasoning:</span> {hypothesis.reasoning}
        </p>
      )}
      {hypothesis.expected_impact && (
        <p className="text-xs text-blue-700">
          <span className="font-medium">Expected Impact:</span> {hypothesis.expected_impact}
        </p>
      )}
    </div>
  );
}

/** Config diff display: param: old -> new */
function ConfigDiffDisplay({ configDiff }: { configDiff: Record<string, ConfigDiffEntry> }) {
  const entries = Object.entries(configDiff);
  if (entries.length === 0) return null;

  return (
    <div>
      <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
        Config Changes
      </h4>
      <div className="space-y-1">
        {entries.map(([param, diff]) => (
          <div key={param} className="text-xs font-mono">
            <span className="font-medium">{param}:</span>{' '}
            <span className="text-red-600 line-through">{String(diff.a)}</span>
            {' → '}
            <span className="text-green-600">{String(diff.b)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Gemini feedback section */
function GeminiFeedback({ evaluations }: { evaluations: Record<string, Evaluation | null> }) {
  const years = Object.entries(evaluations).filter(
    (entry): entry is [string, Evaluation] => entry[1] != null && entry[1].evaluation != null,
  );

  if (years.length === 0) return null;

  return (
    <div className="space-y-3">
      <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
        Gemini Evaluation
      </h4>
      {years.map(([year, evalData]) => {
        const ev = evalData.evaluation;
        return (
          <div key={year} className="rounded-md border p-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">{year}</span>
              <Badge variant="outline">Score: {ev.overall_score}/10</Badge>
            </div>

            {/* Per-class notes */}
            {ev.per_class && ev.per_class.length > 0 && (
              <div>
                <span className="text-xs font-medium text-muted-foreground">Per-class notes:</span>
                <div className="space-y-1 mt-1">
                  {ev.per_class.map((pc) => (
                    <div key={pc.class_name} className="text-xs">
                      <span className="font-medium">{pc.class_name}</span>
                      <span className="text-muted-foreground ml-1">({pc.score}/10)</span>
                      {pc.notes && <span className="ml-1">{pc.notes}</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Error regions */}
            {ev.error_regions && ev.error_regions.length > 0 && (
              <div>
                <span className="text-xs font-medium text-muted-foreground">Error regions:</span>
                <div className="space-y-1 mt-1">
                  {ev.error_regions.map((er, i) => (
                    <div key={i} className="text-xs">
                      <Badge
                        variant={er.severity === 'high' ? 'destructive' : 'outline'}
                        className="text-[10px] mr-1"
                      >
                        {er.severity}
                      </Badge>
                      {er.location}: expected {er.expected}, got {er.predicted}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {ev.recommendations && ev.recommendations.length > 0 && (
              <div>
                <span className="text-xs font-medium text-muted-foreground">Recommendations:</span>
                <ul className="list-disc list-inside mt-1">
                  {ev.recommendations.map((rec, i) => (
                    <li key={i} className="text-xs">{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Loading skeleton for iteration detail */
function DetailSkeleton() {
  return (
    <div className="space-y-3">
      <Skeleton className="h-20 w-full" />
      <Skeleton className="h-[200px] w-full" />
      <Skeleton className="h-16 w-full" />
      <Skeleton className="h-40 w-full" />
    </div>
  );
}

// --- Main IterationCard ---

export function IterationCard({ sessionId, iteration, isBest, isExpanded }: IterationCardProps) {
  const [detail, setDetail] = useState<IterationDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const fetchedRef = useRef(false);

  // Fetch detail on first expand
  useEffect(() => {
    if (!isExpanded || fetchedRef.current) return;
    fetchedRef.current = true;

    const controller = new AbortController();
    setDetailLoading(true);
    setDetailError(null);

    getIterationDetail(sessionId, iteration.iteration, controller.signal)
      .then((data) => {
        setDetail(data);
        setDetailLoading(false);
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setDetailError(err instanceof Error ? err.message : 'Failed to load details');
        setDetailLoading(false);
      });

    return () => controller.abort();
  }, [isExpanded, sessionId, iteration.iteration]);

  const isFirst = iteration.iteration === 1;

  // Summary line (always visible as accordion trigger content)
  const summaryContent = (
    <div className="flex items-center gap-3 flex-1 min-w-0">
      <span className="text-sm font-semibold whitespace-nowrap">
        #{iteration.iteration}
      </span>
      <StatusBadge status={iteration.status} isFirst={isFirst} />
      <DeltaIndicator accuracy={iteration.overall_accuracy} />
      {isBest && (
        <Badge className="bg-amber-500 hover:bg-amber-500/80 text-white text-[10px]">
          BEST
        </Badge>
      )}
    </div>
  );

  // Detail content (shown when expanded)
  const detailContent = (
    <div className="space-y-4">
      {detailLoading && <DetailSkeleton />}

      {detailError && (
        <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          {detailError}
        </div>
      )}

      {detail && (
        <>
          {/* Hypothesis callout */}
          {detail.hypothesis && (
            <HypothesisCallout hypothesis={detail.hypothesis as unknown as Hypothesis} />
          )}

          {/* Config diff */}
          {detail.config_diff && Object.keys(detail.config_diff).length > 0 && (
            <ConfigDiffDisplay
              configDiff={detail.config_diff as unknown as Record<string, ConfigDiffEntry>}
            />
          )}

          {/* Per-class F1 chart */}
          {detail.metrics && (detail.metrics as unknown as Metrics).per_class && (
            <F1Chart metrics={detail.metrics as unknown as Metrics} />
          )}

          {/* Gemini feedback */}
          {detail.evaluations && Object.keys(detail.evaluations).length > 0 && (
            <GeminiFeedback
              evaluations={detail.evaluations as unknown as Record<string, Evaluation | null>}
            />
          )}

          {/* Comparison images */}
          {detail.images && Object.keys(detail.images).length > 0 && (
            <ComparisonImages images={detail.images} />
          )}
        </>
      )}
    </div>
  );

  return { summaryContent, detailContent };
}
