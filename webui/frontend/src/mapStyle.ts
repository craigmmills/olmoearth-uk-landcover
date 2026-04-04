import type { StyleSpecification } from 'maplibre-gl';
import type { BasemapType } from '@/types';
import { ESRI_SATELLITE_TILES } from '@/constants';

/** Build a MapLibre style JSON for raster basemaps */
export function buildMapStyle(basemap: BasemapType): StyleSpecification {
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
