import { Map, Source, Layer, NavigationControl, ScaleControl } from '@vis.gl/react-maplibre';
import type { StyleSpecification } from 'maplibre-gl';
import Legend from '@/components/Legend';
import { AOI_CENTER, DEFAULT_ZOOM, ESRI_SATELLITE_TILES } from '@/constants';
import type { LayerState, BasemapType } from '@/types';

interface MapViewProps {
  basemap: BasemapType;
  layers: LayerState[];
}

/** Build a MapLibre style JSON for raster basemaps */
function buildMapStyle(basemap: BasemapType): StyleSpecification {
  const tileUrl =
    basemap === 'satellite'
      ? ESRI_SATELLITE_TILES
      : 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

  return {
    version: 8,
    sources: {
      basemap: {
        type: 'raster',
        tiles: [tileUrl],
        tileSize: 256,
        attribution:
          basemap === 'satellite'
            ? '&copy; Esri'
            : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      },
    },
    layers: [
      {
        id: 'basemap-tiles',
        type: 'raster',
        source: 'basemap',
        minzoom: 0,
        maxzoom: 19,
      },
    ],
  };
}

export default function MapView({ basemap, layers }: MapViewProps) {
  const mapStyle = buildMapStyle(basemap);

  return (
    <>
      <Map
        initialViewState={{
          longitude: AOI_CENTER.longitude,
          latitude: AOI_CENTER.latitude,
          zoom: DEFAULT_ZOOM,
        }}
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
