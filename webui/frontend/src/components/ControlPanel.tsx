import BasemapToggle from '@/components/BasemapToggle';
import LayerControls from '@/components/LayerControls';
import ConnectionStatus from '@/components/ConnectionStatus';
import LoopStatus from '@/components/LoopStatus';
import ComparisonControls from '@/components/ComparisonControls';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { Sun, Moon, Info, Camera } from 'lucide-react';
import type { LayerState, BasemapType, ComparisonMode, LoopStatus as LoopStatusType, Theme } from '@/types';

interface ControlPanelProps {
  basemap: BasemapType;
  onBasemapChange: (basemap: BasemapType) => void;
  layers: LayerState[];
  onToggleLayer: (layerId: string) => void;
  onOpacityChange: (layerId: string, opacity: number) => void;
  healthy: boolean | null;
  loading: boolean;
  error: string | null;
  loopStatus: LoopStatusType;
  comparisonEnabled: boolean;
  onToggleComparison: () => void;
  comparisonMode: ComparisonMode;
  onComparisonModeChange: (mode: ComparisonMode) => void;
  theme: Theme;
  onThemeToggle: () => void;
  onOpenAbout: () => void;
  onExportPNG: () => void;
  onRetry?: () => void;
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
  loopStatus,
  comparisonEnabled,
  onToggleComparison,
  comparisonMode,
  onComparisonModeChange,
  theme,
  onThemeToggle,
  onOpenAbout,
  onExportPNG,
  onRetry,
}: ControlPanelProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">UK Landcover</h1>
        <div className="flex items-center gap-1">
          <button
            onClick={onOpenAbout}
            className="rounded-md p-1 hover:bg-accent"
            aria-label="About"
          >
            <Info className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Theme toggle */}
      <div className="flex items-center gap-2">
        <Sun className="h-4 w-4" />
        <Switch
          checked={theme === 'dark'}
          onCheckedChange={onThemeToggle}
          aria-label="Toggle dark mode"
        />
        <Moon className="h-4 w-4" />
      </div>

      <ConnectionStatus healthy={healthy} />
      <LoopStatus status={loopStatus} />

      {error && (
        <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          {error}
          {onRetry && (
            <button onClick={onRetry} className="ml-2 underline hover:no-underline">
              Retry
            </button>
          )}
        </div>
      )}

      {loading && (
        <div className="space-y-3">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-6 w-3/4" />
          <Skeleton className="h-24 w-full" />
        </div>
      )}

      <BasemapToggle basemap={basemap} onChange={onBasemapChange} />

      <ComparisonControls
        enabled={comparisonEnabled}
        onToggle={onToggleComparison}
        mode={comparisonMode}
        onModeChange={onComparisonModeChange}
      />

      <LayerControls
        layers={layers}
        onToggle={onToggleLayer}
        onOpacityChange={onOpacityChange}
      />

      {/* Export PNG button */}
      <div className="flex items-center gap-2">
        <button
          onClick={onExportPNG}
          disabled={comparisonEnabled}
          className="flex items-center gap-1.5 rounded-md bg-secondary px-3 py-1.5 text-sm text-secondary-foreground hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed"
          title={comparisonEnabled ? 'PNG export not available in comparison mode' : 'Export current map view as PNG'}
        >
          <Camera className="h-3.5 w-3.5" />
          Export PNG
        </button>
      </div>
    </div>
  );
}
