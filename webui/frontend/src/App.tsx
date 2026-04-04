import { useState, useEffect, useCallback } from 'react';
import MapView from '@/components/MapView';
import ControlPanel from '@/components/ControlPanel';
import { ExperimentDashboard } from '@/components/ExperimentDashboard';
import { useLatestSession } from '@/hooks/useLatestSession';
import { useBackendHealth } from '@/hooks/useBackendHealth';
import { useSessionSSE } from '@/hooks/useSessionSSE';
import { getSession } from '@/api/client';
import type {
  LayerState,
  BasemapType,
  LoopStatus,
  SSENewIterationEvent,
  SSESessionCompleteEvent,
} from '@/types';

export default function App() {
  const {
    loading,
    error,
    sessionId,
    layers: resolvedLayers,
    refreshIteration,
  } = useLatestSession();
  const { healthy } = useBackendHealth();

  const [loopStatus, setLoopStatus] = useState<LoopStatus>({ state: 'connecting' });

  const handleNewIteration = useCallback(
    (event: SSENewIterationEvent) => {
      // Only update if this event is for our current session (or if we have no session yet)
      if (sessionId && event.session_id !== sessionId) return;

      setLoopStatus({
        state: 'running',
        sessionId: event.session_id,
        currentIteration: event.iteration,
      });

      // Update tile URLs to show the latest iteration
      refreshIteration(event.session_id, event.iteration);
    },
    [sessionId, refreshIteration],
  );

  const handleSessionComplete = useCallback(
    (event: SSESessionCompleteEvent) => {
      if (sessionId && event.session_id !== sessionId) return;

      setLoopStatus({
        state: 'complete',
        sessionId: event.session_id,
        nIterations: event.n_iterations,
        bestScore: event.final_score,
        stopReason: event.stop_reason,
      });

      // Update tiles to show the best iteration if available
      if (event.best_iteration !== null) {
        refreshIteration(event.session_id, event.best_iteration);
      }
    },
    [sessionId, refreshIteration],
  );

  const { connected } = useSessionSSE({
    onNewIteration: handleNewIteration,
    onSessionComplete: handleSessionComplete,
  });

  // Determine initial loop status from session data
  useEffect(() => {
    if (loading) return;
    if (!sessionId) {
      // No sessions exist
      if (connected) {
        setLoopStatus({ state: 'idle' });
      }
      return;
    }
    getSession(sessionId)
      .then((session) => {
        if (session.end_time) {
          setLoopStatus({
            state: 'complete',
            sessionId: session.session_id,
            nIterations: session.n_iterations,
            bestScore: session.final_score,
            stopReason: session.stop_reason ?? '',
          });
        } else {
          setLoopStatus({
            state: 'running',
            sessionId: session.session_id,
            currentIteration: session.n_iterations,
          });
        }
      })
      .catch(() => {
        // If fetch fails, SSE events will update status when they arrive
      });
  }, [loading, sessionId, connected]);

  const [basemap, setBasemap] = useState<BasemapType>('satellite');

  // Track user overrides per layer: visibility and opacity
  const [overrides, setOverrides] = useState<
    Record<string, { visible?: boolean; opacity?: number }>
  >({});

  // Merge resolved layers with any user overrides
  const layers: LayerState[] = resolvedLayers.map((layer) => {
    const override = overrides[layer.id];
    if (!override) return layer;
    return {
      ...layer,
      visible: override.visible ?? layer.visible,
      opacity: override.opacity ?? layer.opacity,
    };
  });

  const handleToggleLayer = useCallback((layerId: string) => {
    setOverrides((prev) => {
      const current = prev[layerId];
      return {
        ...prev,
        [layerId]: {
          ...current,
          visible: !(current?.visible ?? resolvedLayers.find((l) => l.id === layerId)?.visible ?? false),
        },
      };
    });
  }, [resolvedLayers]);

  const handleOpacityChange = useCallback((layerId: string, opacity: number) => {
    setOverrides((prev) => ({
      ...prev,
      [layerId]: { ...prev[layerId], opacity },
    }));
  }, []);

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* Map -- 70% width */}
      <div className="flex-[7] relative">
        <MapView basemap={basemap} layers={layers} />
      </div>

      {/* Side Panel -- 30% width */}
      <div className="flex-[3] border-l overflow-y-auto bg-background p-4 space-y-6">
        <ControlPanel
          basemap={basemap}
          onBasemapChange={setBasemap}
          layers={layers}
          onToggleLayer={handleToggleLayer}
          onOpacityChange={handleOpacityChange}
          healthy={healthy}
          loading={loading}
          error={error}
          loopStatus={loopStatus}
        />

        {/* Experiment Dashboard */}
        <div className="border-t pt-4">
          <ExperimentDashboard />
        </div>
      </div>
    </div>
  );
}
