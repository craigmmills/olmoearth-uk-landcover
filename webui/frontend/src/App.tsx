import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import MapView from '@/components/MapView';
import type { MapViewHandle } from '@/components/MapView';
import ComparisonView from '@/components/ComparisonView';
import ControlPanel from '@/components/ControlPanel';
import { ExperimentDashboard } from '@/components/ExperimentDashboard';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import AboutModal from '@/components/AboutModal';
import { useLatestSession } from '@/hooks/useLatestSession';
import { useBackendHealth } from '@/hooks/useBackendHealth';
import { useSessionSSE } from '@/hooks/useSessionSSE';
import { getSession, listIterations, queryPoint, buildWorldCoverTileUrl } from '@/api/client';
import { AOI_CENTER, DEFAULT_ZOOM } from '@/constants';
import { cn } from '@/lib/utils';
import type { ViewState, ViewStateChangeEvent } from '@vis.gl/react-maplibre';
import type {
  LayerState,
  BasemapType,
  ComparisonMode,
  LoopStatus,
  SSENewIterationEvent,
  SSESessionCompleteEvent,
  Theme,
  PixelQueryResult,
} from '@/types';

export default function App() {
  const {
    loading,
    error,
    sessionId,
    iterationNum,
    layers: resolvedLayers,
    refreshIteration,
    retry,
  } = useLatestSession();
  const { healthy } = useBackendHealth();

  // --- Theme ---
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('theme');
    return (saved === 'light' || saved === 'dark') ? saved : 'dark';
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('theme', theme);
  }, [theme]);

  const handleThemeToggle = useCallback(() => {
    setTheme(t => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  // --- Loop status ---
  const [loopStatus, setLoopStatus] = useState<LoopStatus>({ state: 'connecting' });

  const handleNewIteration = useCallback(
    (event: SSENewIterationEvent) => {
      if (sessionId && event.session_id !== sessionId) return;

      setLoopStatus({
        state: 'running',
        sessionId: event.session_id,
        currentIteration: event.iteration,
      });

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

  useEffect(() => {
    if (loading) return;
    if (!sessionId) {
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

  // --- Basemap & layers ---
  const [basemap, setBasemap] = useState<BasemapType>('satellite');

  const [overrides, setOverrides] = useState<
    Record<string, { visible?: boolean; opacity?: number }>
  >({});

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

  // --- Comparison mode ---
  const [comparisonEnabled, setComparisonEnabled] = useState(false);
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>('satellite-vs-classification');

  const [viewState, setViewState] = useState<ViewState>({
    longitude: AOI_CENTER.longitude,
    latitude: AOI_CENTER.latitude,
    zoom: DEFAULT_ZOOM,
    bearing: 0,
    pitch: 0,
    padding: { top: 0, bottom: 0, left: 0, right: 0 },
  });

  const handleToggleComparison = useCallback(() => {
    setComparisonEnabled((prev) => !prev);
  }, []);

  const handleComparisonModeChange = useCallback((mode: ComparisonMode) => {
    setComparisonMode(mode);
  }, []);

  const handleViewStateChange = useCallback((evt: ViewStateChangeEvent) => {
    setViewState(evt.viewState);
  }, []);

  // --- Panel state & modals ---
  const [controlPanelOpen, setControlPanelOpen] = useState(true);
  const [dashboardOpen, setDashboardOpen] = useState(true);
  const [showAbout, setShowAbout] = useState(false);

  // --- Keyboard shortcuts ---
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Suppress shortcuts when typing in form elements
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }

      // Suppress shortcuts when a modal is open
      if (showAbout) return;

      switch (e.key.toLowerCase()) {
        case 'l':
          setControlPanelOpen(prev => !prev);
          break;
        case 'd':
          setDashboardOpen(prev => !prev);
          break;
        case 'c':
          setComparisonEnabled(prev => !prev);
          break;
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showAbout]);

  // --- Click-to-query ---
  const [clickPopup, setClickPopup] = useState<{
    lngLat: { lng: number; lat: number };
    result: PixelQueryResult | null;
    loading: boolean;
    error: string | null;
  } | null>(null);

  const mapRef = useRef<MapViewHandle>(null);
  const clickAbortRef = useRef<AbortController | null>(null);

  const handleMapClick = useCallback(
    async (lngLat: { lng: number; lat: number }) => {
      if (!sessionId || !iterationNum) return;

      // Cancel any in-flight query
      clickAbortRef.current?.abort();
      const controller = new AbortController();
      clickAbortRef.current = controller;

      setClickPopup({ lngLat, result: null, loading: true, error: null });

      try {
        // Use the most specific visible classification year
        const year = layers.find(l => l.id === 'landcover-2023' && l.visible)
          ? '2023' as const : '2021' as const;
        const data = await queryPoint(
          sessionId, iterationNum, year, lngLat.lng, lngLat.lat, controller.signal,
        );
        if (controller.signal.aborted) return;
        setClickPopup({
          lngLat,
          result: {
            lng: data.lng,
            lat: data.lat,
            classIndex: data.class_index,
            className: data.class_name,
            color: data.color,
          },
          loading: false,
          error: null,
        });
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') return;
        setClickPopup({
          lngLat,
          result: null,
          loading: false,
          error: e instanceof Error ? e.message : 'Query failed',
        });
      }
    },
    [sessionId, iterationNum, layers],
  );

  const handleClosePopup = useCallback(() => setClickPopup(null), []);

  // --- Exports ---
  const handleExportPNG = useCallback(() => {
    const dataUrl = mapRef.current?.exportPNG();
    if (!dataUrl) return;

    const link = document.createElement('a');
    link.download = `landcover-${sessionId ?? 'map'}.png`;
    link.href = dataUrl;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [sessionId]);

  // --- Dashboard session change → update map tiles ---
  const handleDashboardSessionChange = useCallback(
    (newSessionId: string) => {
      listIterations(newSessionId).then(iterations => {
        if (iterations.length > 0) {
          const latest = iterations[iterations.length - 1];
          refreshIteration(newSessionId, latest.iteration);
        }
      });
    },
    [refreshIteration],
  );

  const handleOpenAbout = useCallback(() => setShowAbout(true), []);

  // --- Comparison sides ---
  const comparisonSides = useMemo(() => {
    const classification2021 = layers.find((l) => l.id === 'landcover-2021');
    const classification2023 = layers.find((l) => l.id === 'landcover-2023');
    const activeClassification = classification2021?.visible
      ? classification2021
      : classification2023?.visible
        ? classification2023
        : classification2021;

    switch (comparisonMode) {
      case 'satellite-vs-classification': {
        return {
          left: {
            basemap: 'satellite' as const,
            layers: [] as LayerState[],
            label: 'Satellite',
          },
          right: {
            basemap: 'satellite' as const,
            layers: activeClassification
              ? [{ ...activeClassification, visible: true, opacity: 1.0 }]
              : [],
            label: activeClassification?.label ?? 'Classification',
          },
        };
      }
      case '2021-vs-2023': {
        return {
          left: {
            basemap: 'satellite' as const,
            layers: classification2021
              ? [{ ...classification2021, visible: true, opacity: 1.0 }]
              : [],
            label: 'Classification 2021',
          },
          right: {
            basemap: 'satellite' as const,
            layers: classification2023
              ? [{ ...classification2023, visible: true, opacity: 1.0 }]
              : [],
            label: 'Classification 2023',
          },
        };
      }
      case 'classification-vs-worldcover': {
        const worldcoverLayer: LayerState = {
          id: 'worldcover',
          label: 'WorldCover',
          visible: true,
          opacity: 1.0,
          tileUrlTemplate: buildWorldCoverTileUrl(),
        };
        return {
          left: {
            basemap: 'satellite' as const,
            layers: classification2021
              ? [{ ...classification2021, visible: true, opacity: 1.0 }]
              : [],
            label: 'Classification 2021',
          },
          right: {
            basemap: 'satellite' as const,
            layers: [worldcoverLayer],
            label: 'WorldCover 2021',
          },
        };
      }
    }
  }, [comparisonMode, layers]);

  return (
    <>
      <div className="flex h-screen w-screen overflow-hidden">
        {/* Map area - takes all space when panel is hidden */}
        <div className={cn('relative', controlPanelOpen ? 'flex-[7]' : 'flex-1')}>
          <ErrorBoundary fallbackMessage="Map failed to load.">
            {comparisonEnabled ? (
              <ComparisonView
                leftSide={comparisonSides.left}
                rightSide={comparisonSides.right}
                viewState={viewState}
                onMove={handleViewStateChange}
                loading={loading}
              />
            ) : (
              <MapView
                ref={mapRef}
                basemap={basemap}
                layers={layers}
                viewState={viewState}
                onMove={handleViewStateChange}
                onMapClick={handleMapClick}
                clickPopup={clickPopup}
                onClosePopup={handleClosePopup}
              />
            )}
          </ErrorBoundary>
        </div>

        {/* Side panel - collapsible */}
        {controlPanelOpen && (
          <div className="flex-[3] min-w-[280px] max-w-[480px] border-l overflow-y-auto bg-background p-4 space-y-6">
            <ErrorBoundary fallbackMessage="Controls failed to load.">
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
                comparisonEnabled={comparisonEnabled}
                onToggleComparison={handleToggleComparison}
                comparisonMode={comparisonMode}
                onComparisonModeChange={handleComparisonModeChange}
                theme={theme}
                onThemeToggle={handleThemeToggle}
                onOpenAbout={handleOpenAbout}
                onExportPNG={handleExportPNG}
                onRetry={retry}
              />
            </ErrorBoundary>

            {dashboardOpen && (
              <div className="border-t pt-4">
                <ErrorBoundary fallbackMessage="Dashboard failed to load.">
                  <ExperimentDashboard
                    panelOpen={dashboardOpen}
                    onOpenChange={setDashboardOpen}
                    onSessionChange={handleDashboardSessionChange}
                  />
                </ErrorBoundary>
              </div>
            )}
          </div>
        )}
      </div>

      {/* About modal */}
      <AboutModal open={showAbout} onOpenChange={setShowAbout} />
    </>
  );
}
