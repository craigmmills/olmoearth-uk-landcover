import { useState, useCallback } from 'react';
import { Map, Source, Layer, NavigationControl, ScaleControl } from '@vis.gl/react-maplibre';
import type { ViewState, ViewStateChangeEvent } from '@vis.gl/react-maplibre';
import Legend from '@/components/Legend';
import { buildMapStyle } from '@/mapStyle';
import { useSwipeDivider } from '@/hooks/useSwipeDivider';
import { AOI_CENTER, DEFAULT_ZOOM } from '@/constants';
import type { BasemapType, LayerState } from '@/types';

export interface ComparisonSide {
  basemap: BasemapType;
  layers: LayerState[];
  label: string;
}

interface ComparisonViewProps {
  leftSide: ComparisonSide;
  rightSide: ComparisonSide;
  viewState?: ViewState;
  onMove?: (evt: ViewStateChangeEvent) => void;
  loading?: boolean;
}

export default function ComparisonView({
  leftSide,
  rightSide,
  viewState: externalViewState,
  onMove: externalOnMove,
  loading,
}: ComparisonViewProps) {
  // Internal view state as fallback if not controlled externally
  const [internalViewState, setInternalViewState] = useState<ViewState>({
    longitude: AOI_CENTER.longitude,
    latitude: AOI_CENTER.latitude,
    zoom: DEFAULT_ZOOM,
    bearing: 0,
    pitch: 0,
    padding: { top: 0, bottom: 0, left: 0, right: 0 },
  });

  const viewState = externalViewState ?? internalViewState;

  const handleMove = useCallback(
    (evt: ViewStateChangeEvent) => {
      if (externalOnMove) {
        externalOnMove(evt);
      } else {
        setInternalViewState(evt.viewState);
      }
    },
    [externalOnMove],
  );

  const { position, isDragging, containerRef, onPointerDown } = useSwipeDivider(0.5);

  const leftStyle = buildMapStyle(leftSide.basemap);
  const rightStyle = buildMapStyle(rightSide.basemap);

  const hasLeftLayers = leftSide.layers.some((l) => l.tileUrlTemplate);
  const hasRightLayers = rightSide.layers.some((l) => l.tileUrlTemplate);
  const showLoading = loading && !hasLeftLayers && !hasRightLayers;

  return (
    <>
      <div
        ref={containerRef}
        className="relative w-full h-full overflow-hidden"
        style={{ cursor: isDragging ? 'col-resize' : undefined }}
      >
        {/* Left map -- full width, underneath */}
        <div className="absolute inset-0">
          <Map
            {...viewState}
            onMove={handleMove}
            style={{ width: '100%', height: '100%' }}
            mapStyle={leftStyle}
          >
            <NavigationControl position="top-left" />
            <ScaleControl position="bottom-left" />
            {leftSide.layers.map(
              (layer) =>
                layer.tileUrlTemplate && (
                  <Source
                    key={layer.id}
                    id={`left-${layer.id}`}
                    type="raster"
                    tiles={[layer.tileUrlTemplate]}
                    tileSize={256}
                  >
                    <Layer
                      id={`left-${layer.id}-layer`}
                      type="raster"
                      layout={{ visibility: layer.visible ? 'visible' : 'none' }}
                      paint={{ 'raster-opacity': layer.opacity }}
                    />
                  </Source>
                ),
            )}
          </Map>
        </div>

        {/* Right map -- clipped to reveal from divider position rightward */}
        <div
          className="absolute inset-0"
          style={{ clipPath: `inset(0 0 0 ${position}px)` }}
        >
          <Map
            {...viewState}
            onMove={handleMove}
            style={{ width: '100%', height: '100%' }}
            mapStyle={rightStyle}
          >
            {rightSide.layers.map(
              (layer) =>
                layer.tileUrlTemplate && (
                  <Source
                    key={layer.id}
                    id={`right-${layer.id}`}
                    type="raster"
                    tiles={[layer.tileUrlTemplate]}
                    tileSize={256}
                  >
                    <Layer
                      id={`right-${layer.id}-layer`}
                      type="raster"
                      layout={{ visibility: layer.visible ? 'visible' : 'none' }}
                      paint={{ 'raster-opacity': layer.opacity }}
                    />
                  </Source>
                ),
            )}
          </Map>
        </div>

        {/* Swipe divider handle */}
        <div
          className="absolute top-0 bottom-0 z-10"
          style={{ left: `${position}px`, width: '4px', transform: 'translateX(-50%)' }}
        >
          <div className="w-full h-full bg-white shadow-md" />
          {/* Drag handle circle */}
          <div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white shadow-lg border-2 border-gray-300 cursor-col-resize flex items-center justify-center"
            style={{ touchAction: 'none' }}
            onPointerDown={onPointerDown}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M4 1L1 6L4 11" stroke="#666" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M8 1L11 6L8 11" stroke="#666" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
        </div>

        {/* Side labels */}
        <div className="absolute top-3 left-3 z-10 bg-black/60 text-white text-xs px-2 py-1 rounded">
          {leftSide.label}
        </div>
        <div className="absolute top-3 right-3 z-10 bg-black/60 text-white text-xs px-2 py-1 rounded">
          {rightSide.label}
        </div>

        {/* Loading indicator when no layers are ready */}
        {showLoading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center">
            <div className="bg-black/60 text-white text-sm px-4 py-2 rounded">
              Loading comparison data...
            </div>
          </div>
        )}
      </div>

      {/* Legend overlay -- positioned outside Map, same pattern as MapView */}
      <Legend />
    </>
  );
}
