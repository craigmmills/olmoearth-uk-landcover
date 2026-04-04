/**
 * AOI center from src/config.py -- hardcoded per PROMPT.md constraint.
 * TODO: Fetch dynamically if backend adds /api/aoi endpoint in future.
 */
export const AOI_CENTER = {
  longitude: -3.50,
  latitude: 50.72,
} as const;

export const DEFAULT_ZOOM = 14;

/** Backend API base URL -- override via VITE_API_URL env var */
export const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/** Landcover class names -- mirrors src/config.py LANDCOVER_CLASSES */
export const LANDCOVER_CLASSES: Record<number, string> = {
  0: 'Built-up',
  1: 'Cropland',
  2: 'Grassland',
  3: 'Tree cover',
  4: 'Water',
  5: 'Other',
};

/** Landcover class colors -- mirrors src/config.py LANDCOVER_COLORS */
export const LANDCOVER_COLORS: Record<number, string> = {
  0: '#FF0000',   // Built-up -- Red
  1: '#FFFF00',   // Cropland -- Yellow
  2: '#90EE90',   // Grassland -- Light green
  3: '#006400',   // Tree cover -- Dark green
  4: '#0000FF',   // Water -- Blue
  5: '#808080',   // Other -- Gray
};

/** Esri satellite tile URL -- same as src/app.py line 665 */
export const ESRI_SATELLITE_TILES =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';

/**
 * Default layer configuration.
 * Visibility defaults match the Streamlit app: 2021=on, 2023=off, change=off.
 * Opacity defaults: classification=0.7, change=0.8 (matching src/app.py).
 */
export const DEFAULT_LAYERS = [
  { id: 'landcover-2021', label: 'Landcover 2021', year: '2021' as const, visible: true, opacity: 0.7 },
  { id: 'landcover-2023', label: 'Landcover 2023', year: '2023' as const, visible: false, opacity: 0.7 },
  { id: 'change-map', label: 'Change Map', year: 'change' as const, visible: false, opacity: 0.8 },
] as const;
