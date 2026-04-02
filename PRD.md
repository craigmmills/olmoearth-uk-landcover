# OlmoEarth UK Landcover Demo — PRD

## Overview

Minimal end-to-end demonstration of AI2's OlmoEarth foundation model for satellite-based landcover classification and year-over-year change detection, running entirely on an M3 Pro MacBook (18GB RAM). The demo covers a ~5km x 5km area around Cambridge, UK — chosen for its mix of urban, agricultural, fenland, and woodland cover types that can be visually verified.

## Goals

1. Download real Sentinel-2 satellite imagery for Cambridge, UK for two time periods (summer 2021 and summer 2023)
2. Download ESA WorldCover 2021 labels as ground truth for the same area
3. Run OlmoEarth Tiny (8.1M params) to extract learned feature embeddings from the satellite imagery
4. Train a lightweight pixel-level classifier on those embeddings using WorldCover labels
5. Produce landcover classification maps for both 2021 and 2023
6. Compute a change detection map showing where landcover changed between the two years
7. Display all results in an interactive web UI with map overlays for visual inspection

## Target Area

- **Location**: Cambridge and surrounds, Cambridgeshire, UK
- **Center**: approximately 52.205°N, 0.12°E
- **Extent**: ~5km x 5km (512 x 512 pixels at 10m resolution)
- **Why**: Mix of built-up (Cambridge city), intensive arable farmland, fenland, scattered woodland, River Cam and water features

## Technical Constraints

- **Hardware**: Apple M3 Pro, 11 cores, 18GB RAM — no CUDA GPU
- **Model**: OlmoEarth Tiny (not Base/Large) to fit in memory and run at reasonable speed on CPU/MPS
- **Data source**: Element84 Earth Search STAC API for Sentinel-2 L2A (no authentication required). ESA WorldCover from the same or Planetary Computer.
- **Fully automated**: The entire pipeline from data download to web UI must run without human intervention
- **Execution time**: Pipeline should complete within ~30 minutes on target hardware

## Architecture

```
olmoearth-uk-landcover/
├── PRD.md                    # This document
├── pyproject.toml            # Python project config (uv)
├── src/
│   ├── config.py             # AOI bounds, paths, constants
│   ├── acquire.py            # Download Sentinel-2 + WorldCover data
│   ├── preprocess.py         # Cloud masking, compositing, normalization
│   ├── embeddings.py         # OlmoEarth Tiny feature extraction
│   ├── classify.py           # Train classifier + predict landcover
│   ├── change.py             # Year-over-year change detection
│   ├── pipeline.py           # Orchestrate full pipeline end-to-end
│   └── app.py                # Streamlit web UI
├── data/                     # Downloaded satellite imagery (gitignored)
│   ├── sentinel2/
│   └── worldcover/
├── output/                   # Classification results (gitignored)
│   ├── landcover_2021.tif
│   ├── landcover_2023.tif
│   └── change_map.tif
└── .gitignore
```

## Pipeline Stages

### Stage 1: Data Acquisition
- Query Element84 Earth Search STAC for Sentinel-2 L2A scenes covering the AOI
- Filter for summer months (June-August) with low cloud cover (<20%)
- Download the 12 bands OlmoEarth expects: B02, B03, B04, B08 (10m), B05, B06, B07, B8A, B11, B12 (20m), B01, B09 (60m)
- Resample all bands to 10m resolution, crop to AOI
- Download ESA WorldCover 2021 for the same extent

### Stage 2: Preprocessing
- Apply SCL (Scene Classification Layer) cloud/shadow mask
- Create a per-pixel median composite for each summer period
- Handle band resampling (20m and 60m bands upsampled to 10m via bilinear interpolation)
- Normalize using OlmoEarth's built-in normalizer (per-band statistics)
- Output: Two 12-band composites (2021, 2023) at 10m, 512x512 pixels

### Stage 3: OlmoEarth Embedding Extraction
- Load OlmoEarth Tiny from HuggingFace (auto-download, ~57MB)
- Tile the 512x512 composite into 64x64 patches (8x8 grid)
- For each patch: construct MaskedOlmoEarthSample with Sentinel-2 bands and timestamp
- Run encoder with patch_size=4 to get embeddings
- Pool embeddings spatially to get per-pixel feature vectors
- Reassemble patches into a full-AOI embedding map
- Output: Two embedding arrays of shape (512, 512, 192) for 2021 and 2023

### Stage 4: Classification
- Load WorldCover 2021, remap to 6 simplified classes:
  - 0: Built-up (WorldCover class 50)
  - 1: Cropland (WorldCover class 40)
  - 2: Grassland (WorldCover classes 30)
  - 3: Tree cover (WorldCover class 10)
  - 4: Water (WorldCover class 80)
  - 5: Other (WorldCover classes 20, 60, 70, 90, 95, 100)
- Sample training pixels from 2021 embeddings + WorldCover labels
- Train a scikit-learn RandomForest classifier on the embedding features
- Predict landcover for both 2021 and 2023 embeddings
- Output: Two classified GeoTIFFs at 10m resolution

### Stage 5: Change Detection
- Compare 2021 and 2023 classified maps pixel-by-pixel
- Compute transition matrix (from-class → to-class counts and areas)
- Generate a change map GeoTIFF: 0 = no change, 1+ = change type index
- Compute summary statistics (% area changed, dominant transitions)
- Output: Change map GeoTIFF + transition matrix JSON

### Stage 6: Web Visualization
- Streamlit app with folium map centered on Cambridge
- Layers (toggleable):
  - Satellite basemap (OpenStreetMap or satellite tiles)
  - Landcover 2021 (colored overlay)
  - Landcover 2023 (colored overlay)
  - Change map (highlight changed pixels)
- Sidebar:
  - Class legend with colors
  - Area statistics per class per year
  - Transition matrix heatmap
  - Top changes summary
- Side-by-side or swipe comparison mode for 2021 vs 2023

## Landcover Color Scheme

| Class | Color | Hex |
|-------|-------|-----|
| Built-up | Red | #FF0000 |
| Cropland | Yellow | #FFFF00 |
| Grassland | Light green | #90EE90 |
| Tree cover | Dark green | #006400 |
| Water | Blue | #0000FF |
| Other | Gray | #808080 |

## Dependencies

```
# Core
torch>=2.4
olmoearth-pretrain
numpy
scikit-learn

# Geospatial
rasterio
rioxarray
pystac-client
planetary-computer
shapely

# Web UI
streamlit
folium
streamlit-folium
matplotlib
```

## Success Criteria

1. Pipeline completes end-to-end without errors on M3 Pro 18GB
2. Landcover maps are visually plausible when compared to satellite imagery / Google Maps
3. Cambridge city center shows as Built-up, surrounding fields as Cropland, parks/commons as Grassland
4. Change detection shows some plausible changes (new development, crop rotation, etc.)
5. Web UI loads and displays all layers interactively
6. Total pipeline execution time < 30 minutes

## Out of Scope

- Multi-temporal (>2 timestep) analysis
- Sentinel-1 SAR fusion
- Custom model fine-tuning (we use a simple classifier on frozen embeddings)
- Accuracy validation against independent reference data
- Deployment or containerization
- Areas larger than the ~5km AOI
