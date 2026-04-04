import { useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import { Map, Source, Layer, NavigationControl, ScaleControl } from '@vis.gl/react-maplibre';
import type { MapRef, ViewState, ViewStateChangeEvent } from '@vis.gl/react-maplibre';
import Legend from '@/components/Legend';
import ClickPopup from '@/components/ClickPopup';
import { Loader2 } from 'lucide-react';
import { buildMapStyle } from '@/mapStyle';
import { AOI_CENTER, DEFAULT_ZOOM } from '@/constants';
import type { LayerState, BasemapType, PixelQueryResult } from '@/types';

export interface MapViewHandle {
  exportPNG: () => string | null;
}

interface MapViewProps {
  basemap: BasemapType;
  layers: LayerState[];
  viewState?: ViewState;
  onMove?: (evt: ViewStateChangeEvent) => void;
  onMapClick?: (lngLat: { lng: number; lat: number }) => void;
  clickPopup?: {
    lngLat: { lng: number; lat: number };
    result: PixelQueryResult | null;
    loading: boolean;
    error: string | null;
  } | null;
  onClosePopup?: () => void;
}

const MapView = forwardRef<MapViewHandle, MapViewProps>(function MapView(
  { basemap, layers, viewState, onMove, onMapClick, clickPopup, onClosePopup },
  ref,
) {
  const mapRef = useRef<MapRef>(null);

  useImperativeHandle(ref, () => ({
    exportPNG: () => {
      const map = mapRef.current?.getMap();
      if (!map) return null;
      return map.getCanvas().toDataURL('image/png');
    },
  }));

  const handleClick = useCallback(
    (e: { lngLat: { lng: number; lat: number } }) => {
      onMapClick?.({ lng: e.lngLat.lng, lat: e.lngLat.lat });
    },
    [onMapClick],
  );

  const mapStyle = buildMapStyle(basemap);

  // If viewState is provided, use controlled mode; otherwise use initialViewState
  const mapProps = viewState
    ? { ...viewState, onMove }
    : {
        initialViewState: {
          longitude: AOI_CENTER.longitude,
          latitude: AOI_CENTER.latitude,
          zoom: DEFAULT_ZOOM,
        },
      };

  return (
    <>
      <Map
        ref={mapRef}
        {...mapProps}
        style={{ width: '100%', height: '100%' }}
        mapStyle={mapStyle}
        onClick={handleClick}
        canvasContextAttributes={{ preserveDrawingBuffer: true }}
      >
        <NavigationControl position="top-left" />
        <ScaleControl position="bottom-left" />

        {/* Classification overlay layers */}
        {layers.map(
          (layer) =>
            layer.tileUrlTemplate && (
              <Source
                key={layer.id}
                id={layer.id}
                type="raster"
                tiles={[layer.tileUrlTemplate]}
                tileSize={256}
              >
                <Layer
                  id={`${layer.id}-layer`}
                  type="raster"
                  layout={{ visibility: layer.visible ? 'visible' : 'none' }}
                  paint={{ 'raster-opacity': layer.opacity }}
                />
              </Source>
            ),
        )}

        {/* Click-to-query popup */}
        {clickPopup && (
          <ClickPopup
            lngLat={clickPopup.lngLat}
            result={clickPopup.result}
            loading={clickPopup.loading}
            error={clickPopup.error}
            onClose={() => onClosePopup?.()}
          />
        )}
      </Map>

      {/* Legend overlay -- positioned outside Map to avoid library compat issues */}
      <Legend />

      {/* Loading overlay when no tile URLs resolved yet */}
      {layers.every(l => !l.tileUrlTemplate) && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/50 z-20 pointer-events-none">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading map tiles...
          </div>
        </div>
      )}
    </>
  );
});

export default MapView;
