"""Interactive web UI for landcover visualization.

Usage:
    uv run streamlit run src/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

from src import config

# Page config
st.set_page_config(
    page_title="OlmoEarth UK Landcover Demo",
    page_icon="🌍",
    layout="wide",
)


def check_outputs_exist() -> bool:
    """Check if pipeline outputs exist."""
    required = [
        config.OUTPUT_DIR / "landcover_2021.tif",
        config.OUTPUT_DIR / "landcover_2023.tif",
        config.OUTPUT_DIR / "change_map.tif",
        config.OUTPUT_DIR / "transitions.json",
    ]
    return all(p.exists() for p in required)


def load_landcover(year: str) -> tuple[np.ndarray, dict]:
    """Load classified landcover GeoTIFF and metadata."""
    path = config.OUTPUT_DIR / f"landcover_{year}.tif"
    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return data, {"bounds": bounds}


def load_change_map() -> tuple[np.ndarray, dict]:
    """Load change detection map."""
    path = config.OUTPUT_DIR / "change_map.tif"
    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return data, {"bounds": bounds}


def load_transitions() -> dict:
    """Load transition statistics."""
    path = config.OUTPUT_DIR / "transitions.json"
    with open(path) as f:
        return json.load(f)


def landcover_to_rgba(data: np.ndarray, alpha: int = 180) -> np.ndarray:
    """Convert landcover class array to RGBA image."""
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    color_map = {
        0: (255, 0, 0),      # Built-up - Red
        1: (255, 255, 0),    # Cropland - Yellow
        2: (144, 238, 144),  # Grassland - Light green
        3: (0, 100, 0),      # Tree cover - Dark green
        4: (0, 0, 255),      # Water - Blue
        5: (128, 128, 128),  # Other - Gray
    }

    for cls_idx, rgb in color_map.items():
        mask = data == cls_idx
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = alpha

    return rgba


def change_to_rgba(data: np.ndarray) -> np.ndarray:
    """Convert change map to RGBA (changed pixels highlighted)."""
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    changed = data == 1
    rgba[changed, 0] = 255   # Red
    rgba[changed, 1] = 0
    rgba[changed, 2] = 255   # Magenta
    rgba[changed, 3] = 200

    return rgba


def create_map(
    bounds: tuple,
    lc_2021_rgba: np.ndarray,
    lc_2023_rgba: np.ndarray,
    change_rgba: np.ndarray,
    active_layers: list[str],
) -> folium.Map:
    """Create folium map with landcover overlays."""
    # Center on AOI
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # Add satellite basemap option
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    # Bounds for image overlay: [[south, west], [north, east]]
    img_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    if "Landcover 2021" in active_layers:
        ImageOverlay(
            image=lc_2021_rgba,
            bounds=img_bounds,
            name="Landcover 2021",
            opacity=0.7,
        ).add_to(m)

    if "Landcover 2023" in active_layers:
        ImageOverlay(
            image=lc_2023_rgba,
            bounds=img_bounds,
            name="Landcover 2023",
            opacity=0.7,
        ).add_to(m)

    if "Change Map" in active_layers:
        ImageOverlay(
            image=change_rgba,
            bounds=img_bounds,
            name="Change 2021→2023",
            opacity=0.8,
        ).add_to(m)

    folium.LayerControl().add_to(m)

    return m


def main():
    st.title("OlmoEarth UK Landcover Demo")
    st.markdown("Cambridge, UK — 5km x 5km AOI — OlmoEarth Tiny embeddings")

    if not check_outputs_exist():
        st.error(
            "Pipeline outputs not found. Run the pipeline first:\n\n"
            "```\nuv run python -m src.pipeline\n```"
        )
        return

    # Load data
    lc_2021, meta_2021 = load_landcover("2021")
    lc_2023, meta_2023 = load_landcover("2023")
    change_map, meta_change = load_change_map()
    transitions = load_transitions()

    bounds = meta_2021["bounds"]

    # Convert to RGBA
    lc_2021_rgba = landcover_to_rgba(lc_2021)
    lc_2023_rgba = landcover_to_rgba(lc_2023)
    change_rgba = change_to_rgba(change_map)

    # Sidebar
    with st.sidebar:
        st.header("Layers")
        layers = []
        if st.checkbox("Landcover 2021", value=True):
            layers.append("Landcover 2021")
        if st.checkbox("Landcover 2023", value=False):
            layers.append("Landcover 2023")
        if st.checkbox("Change Map", value=False):
            layers.append("Change Map")

        st.divider()

        # Legend
        st.header("Legend")
        for idx, name in config.LANDCOVER_CLASSES.items():
            color = config.LANDCOVER_COLORS[idx]
            st.markdown(
                f'<span style="background-color:{color};padding:2px 12px;'
                f'margin-right:8px;border:1px solid #333;">&nbsp;</span> {name}',
                unsafe_allow_html=True,
            )

        st.divider()

        # Summary stats
        st.header("Change Summary")
        summary = transitions["summary"]
        st.metric("Total Area", f"{summary['total_area_ha']:.0f} ha")
        st.metric("Changed Area", f"{summary['changed_area_ha']:.1f} ha")
        st.metric("% Changed", f"{summary['pct_changed']:.1f}%")

    # Main content - two columns
    col1, col2 = st.columns([3, 1])

    with col1:
        m = create_map(bounds, lc_2021_rgba, lc_2023_rgba, change_rgba, layers)
        st_folium(m, width=800, height=600)

    with col2:
        # Per-class area comparison
        st.subheader("Class Areas (ha)")

        pixel_area = summary["pixel_area_ha"]
        class_data = []
        for idx, name in config.LANDCOVER_CLASSES.items():
            area_2021 = float((lc_2021 == idx).sum() * pixel_area)
            area_2023 = float((lc_2023 == idx).sum() * pixel_area)
            change_ha = area_2023 - area_2021
            class_data.append({
                "Class": name,
                "2021 (ha)": f"{area_2021:.1f}",
                "2023 (ha)": f"{area_2023:.1f}",
                "Change": f"{change_ha:+.1f}",
            })

        st.dataframe(class_data, hide_index=True, use_container_width=True)

        # Top transitions
        st.subheader("Top Transitions")
        for t in transitions["transitions"][:8]:
            st.markdown(
                f"**{t['from_name']}** → **{t['to_name']}**: "
                f"{t['area_ha']:.1f} ha"
            )

    # Transition matrix
    st.subheader("Transition Matrix (pixels)")
    matrix = np.array(transitions["matrix"])
    class_names = [config.LANDCOVER_CLASSES[i] for i in range(len(config.LANDCOVER_CLASSES))]

    import pandas as pd
    df = pd.DataFrame(
        matrix,
        index=[f"From: {n}" for n in class_names],
        columns=[f"To: {n}" for n in class_names],
    )
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
