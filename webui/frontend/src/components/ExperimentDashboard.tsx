import { useState, useEffect, useCallback, useRef } from 'react';
import { ChevronRight, ChevronLeft, Download } from 'lucide-react';
import { listIterations } from '@/api/client';
import { SessionSelector } from '@/components/SessionSelector';
import { SummaryStats } from '@/components/SummaryStats';
import { IterationCard } from '@/components/IterationCard';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Accordion,
  AccordionItem,
} from '@/components/ui/accordion';
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from '@/components/ui/collapsible';
import { useSessionSSE } from '@/hooks/useSessionSSE';
import type { IterationSummary } from '@/types';

/** Skeleton for the dashboard while loading iterations */
function DashboardSkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="rounded-md border p-3 space-y-2">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
          <Skeleton className="h-20 w-full" />
        </div>
      ))}
    </div>
  );
}

interface ExperimentDashboardProps {
  panelOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  onSessionChange?: (sessionId: string) => void;
}

export function ExperimentDashboard({
  panelOpen: controlledPanelOpen,
  onOpenChange: controlledOnOpenChange,
  onSessionChange,
}: ExperimentDashboardProps) {
  // Internal state as fallback when not controlled
  const [internalPanelOpen, setInternalPanelOpen] = useState(true);
  const panelOpen = controlledPanelOpen ?? internalPanelOpen;
  const onOpenChange = controlledOnOpenChange ?? setInternalPanelOpen;

  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [iterations, setIterations] = useState<IterationSummary[]>([]);
  const [bestIteration, setBestIteration] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedItems, setExpandedItems] = useState<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  // Fetch iterations when session changes
  useEffect(() => {
    if (!selectedSessionId) {
      setIterations([]);
      setBestIteration(null);
      return;
    }

    // Abort any in-flight request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);

    listIterations(selectedSessionId, controller.signal)
      .then((data) => {
        const sorted = [...data].sort((a, b) => a.iteration - b.iteration);
        setIterations(sorted);

        // Determine best iteration (highest overall_accuracy)
        const withAccuracy = sorted.filter((it) => it.overall_accuracy != null);
        if (withAccuracy.length > 0) {
          const best = withAccuracy.reduce((a, b) =>
            (a.overall_accuracy ?? 0) > (b.overall_accuracy ?? 0) ? a : b,
          );
          setBestIteration(best.iteration);
          // Auto-expand best iteration
          setExpandedItems([`iteration-${best.iteration}`]);
        } else {
          setBestIteration(null);
          setExpandedItems([]);
        }

        setLoading(false);
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setError(err instanceof Error ? err.message : 'Failed to load iterations');
        setLoading(false);
      });

    return () => controller.abort();
  }, [selectedSessionId]);

  const endOfListRef = useRef<HTMLDivElement>(null);

  // SSE: auto-refresh iterations when a new one arrives for the current session
  const handleNewIteration = useCallback(
    (event: { session_id: string; iteration: number }) => {
      if (event.session_id === selectedSessionId) {
        // Re-fetch iterations
        listIterations(selectedSessionId)
          .then((data) => {
            const sorted = [...data].sort((a, b) => a.iteration - b.iteration);
            setIterations(sorted);

            // Update best
            const withAccuracy = sorted.filter((it) => it.overall_accuracy != null);
            if (withAccuracy.length > 0) {
              const best = withAccuracy.reduce((a, b) =>
                (a.overall_accuracy ?? 0) > (b.overall_accuracy ?? 0) ? a : b,
              );
              setBestIteration(best.iteration);
            }

            // Scroll to latest iteration after React renders
            requestAnimationFrame(() => {
              endOfListRef.current?.scrollIntoView({ behavior: 'smooth' });
            });
          })
          .catch(() => {
            // Silently ignore SSE refresh errors
          });
      }
    },
    [selectedSessionId],
  );

  useSessionSSE({ onNewIteration: handleNewIteration });

  const handleSessionChange = useCallback((sessionId: string) => {
    setSelectedSessionId(sessionId);
    onSessionChange?.(sessionId);
  }, [onSessionChange]);

  // CSV export handler
  const handleExportCSV = useCallback(() => {
    if (iterations.length === 0) return;

    const header = 'iteration,timestamp,status,overall_accuracy\n';
    const rows = iterations
      .map(it => `${it.iteration},${it.timestamp},${it.status},${it.overall_accuracy ?? ''}`)
      .join('\n');

    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `iterations-${selectedSessionId ?? 'data'}.csv`;
    link.href = url;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [iterations, selectedSessionId]);

  return (
    <Collapsible open={panelOpen} onOpenChange={onOpenChange}>
      {/* Collapse/expand toggle bar */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Experiment Dashboard</h2>
        <div className="flex items-center gap-1">
          {iterations.length > 0 && (
            <button
              onClick={handleExportCSV}
              className="rounded-md p-1 hover:bg-accent"
              aria-label="Export iterations as CSV"
              title="Export CSV"
            >
              <Download className="h-4 w-4" />
            </button>
          )}
          <CollapsibleTrigger asChild>
            <button
              className="rounded-md p-1 hover:bg-accent"
              aria-label={panelOpen ? 'Collapse dashboard' : 'Expand dashboard'}
            >
              {panelOpen ? (
                <ChevronLeft className="h-5 w-5" />
              ) : (
                <ChevronRight className="h-5 w-5" />
              )}
            </button>
          </CollapsibleTrigger>
        </div>
      </div>

      <CollapsibleContent>
        <div className="space-y-4">
          {/* Session selector */}
          <SessionSelector
            selectedSessionId={selectedSessionId}
            onSessionChange={handleSessionChange}
          />

          {/* Error state */}
          {error && (
            <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              {error}
            </div>
          )}

          {/* Loading state */}
          {loading && <DashboardSkeleton />}

          {/* Summary stats */}
          {!loading && iterations.length > 0 && (
            <SummaryStats iterations={iterations} bestIteration={bestIteration} />
          )}

          {/* Iteration cards */}
          {!loading && iterations.length > 0 && (
            <Accordion
              type="multiple"
              value={expandedItems}
              onValueChange={setExpandedItems}
            >
              {iterations.map((iteration, index) => {
                const itemId = `iteration-${iteration.iteration}`;
                const isBest = iteration.iteration === bestIteration;
                const isExpanded = expandedItems.includes(itemId);
                const isNewest = index === iterations.length - 1;

                return (
                  <AccordionItem
                    key={itemId}
                    value={itemId}
                    className={`${isBest ? 'border-amber-400 bg-amber-50/50' : ''} ${isNewest ? 'animate-fade-in-up' : ''}`}
                  >
                    <IterationCard
                      sessionId={selectedSessionId!}
                      iteration={iteration}
                      isBest={isBest}
                      isExpanded={isExpanded}
                    />
                  </AccordionItem>
                );
              })}
            </Accordion>
          )}

          {/* Scroll sentinel for auto-scroll on new iterations */}
          <div ref={endOfListRef} />

          {/* Empty state */}
          {!loading && !error && selectedSessionId && iterations.length === 0 && (
            <div className="text-sm text-muted-foreground text-center py-4">
              No iterations found for this session.
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
