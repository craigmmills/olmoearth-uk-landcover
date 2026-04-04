import BasemapToggle from '@/components/BasemapToggle';
import LayerControls from '@/components/LayerControls';
import ConnectionStatus from '@/components/ConnectionStatus';
import type { LayerState, BasemapType } from '@/types';

interface ControlPanelProps {
  basemap: BasemapType;
  onBasemapChange: (basemap: BasemapType) => void;
  layers: LayerState[];
  onToggleLayer: (layerId: string) => void;
  onOpacityChange: (layerId: string, opacity: number) => void;
  healthy: boolean | null;
  loading: boolean;
  error: string | null;
}

export default function ControlPanel({
  basemap,
  onBasemapChange,
  layers,
  onToggleLayer,
  onOpacityChange,
  healthy,
  loading,
  error,
}: ControlPanelProps) {
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-lg font-semibold">UK Landcover</h1>
      <ConnectionStatus healthy={healthy} />

      {error && (
        <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading && (
        <div className="text-sm text-muted-foreground">Loading session data...</div>
      )}

      <BasemapToggle basemap={basemap} onChange={onBasemapChange} />

      <LayerControls
        layers={layers}
        onToggle={onToggleLayer}
        onOpacityChange={onOpacityChange}
      />
    </div>
  );
}
