import { useState, useCallback } from 'react';
import MapView from '@/components/MapView';
import ControlPanel from '@/components/ControlPanel';
import { ExperimentDashboard } from '@/components/ExperimentDashboard';
import { useLatestSession } from '@/hooks/useLatestSession';
import { useBackendHealth } from '@/hooks/useBackendHealth';
import type { LayerState, BasemapType } from '@/types';

export default function App() {
  const { loading, error, layers: resolvedLayers } = useLatestSession();
  const { healthy } = useBackendHealth();

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
        />

        {/* Experiment Dashboard */}
        <div className="border-t pt-4">
          <ExperimentDashboard />
        </div>
      </div>
    </div>
  );
}
