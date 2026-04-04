import { Map, Source, Layer, NavigationControl, ScaleControl } from '@vis.gl/react-maplibre';
import type { ViewState, ViewStateChangeEvent } from '@vis.gl/react-maplibre';
import Legend from '@/components/Legend';
import { buildMapStyle } from '@/mapStyle';
import { AOI_CENTER, DEFAULT_ZOOM } from '@/constants';
import type { LayerState, BasemapType } from '@/types';

interface MapViewProps {
  basemap: BasemapType;
  layers: LayerState[];
  viewState?: ViewState;
  onMove?: (evt: ViewStateChangeEvent) => void;
}

export default function MapView({ basemap, layers, viewState, onMove }: MapViewProps) {
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
        {...mapProps}
        style={{ width: '100%', height: '100%' }}
        mapStyle={mapStyle}
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
      </Map>

      {/* Legend overlay -- positioned outside Map to avoid library compat issues */}
      <Legend />
    </>
  );
}
