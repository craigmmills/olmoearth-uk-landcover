"""Year-over-year landcover change detection.

Usage:
    uv run python -m src.change
"""
from __future__ import annotations

import json

from src import config


def compute_change_map(lc_2021, lc_2023):
    """Compute pixel-wise change map and transition matrix.

    Returns:
        change_map: (512, 512) uint8 — 0=no change, 1=changed
        transitions: dict with transition counts and areas
    """
    import numpy as np

    n_classes = len(config.LANDCOVER_CLASSES)
    change_map = (lc_2021 != lc_2023).astype(np.uint8)

    # Transition matrix
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for from_cls in range(n_classes):
        for to_cls in range(n_classes):
            count = ((lc_2021 == from_cls) & (lc_2023 == to_cls)).sum()
            matrix[from_cls, to_cls] = int(count)

    # Pixel area at 10m resolution = 100 m² = 0.01 ha
    pixel_area_ha = (config.AOI_RESOLUTION_M ** 2) / 10000

    total_pixels = lc_2021.size
    changed_pixels = change_map.sum()
    pct_changed = changed_pixels / total_pixels * 100

    print(f"[change] Total pixels: {total_pixels:,}")
    print(f"[change] Changed pixels: {changed_pixels:,} ({pct_changed:.1f}%)")
    print(f"[change] Unchanged pixels: {total_pixels - changed_pixels:,} "
          f"({100 - pct_changed:.1f}%)")

    # Build transitions list sorted by area
    transition_list = []
    for from_cls in range(n_classes):
        for to_cls in range(n_classes):
            if from_cls == to_cls:
                continue
            count = matrix[from_cls, to_cls]
            if count == 0:
                continue
            transition_list.append({
                "from_class": from_cls,
                "from_name": config.LANDCOVER_CLASSES[from_cls],
                "to_class": to_cls,
                "to_name": config.LANDCOVER_CLASSES[to_cls],
                "pixel_count": int(count),
                "area_ha": round(float(count * pixel_area_ha), 2),
            })

    transition_list.sort(key=lambda t: t["pixel_count"], reverse=True)

    print("\n[change] Top transitions:")
    for t in transition_list[:5]:
        print(f"  {t['from_name']} -> {t['to_name']}: "
              f"{t['pixel_count']:,} pixels ({t['area_ha']:.1f} ha)")

    transitions = {
        "summary": {
            "total_pixels": int(total_pixels),
            "changed_pixels": int(changed_pixels),
            "pct_changed": round(pct_changed, 2),
            "pixel_area_ha": pixel_area_ha,
            "total_area_ha": round(float(total_pixels * pixel_area_ha), 2),
            "changed_area_ha": round(float(changed_pixels * pixel_area_ha), 2),
        },
        "matrix": matrix.tolist(),
        "class_names": config.LANDCOVER_CLASSES,
        "transitions": transition_list,
    }

    return change_map, transitions


def save_change_map(change_map):
    """Save change map as GeoTIFF."""
    import rasterio

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.OUTPUT_DIR / "change_map.tif"

    ref_path = config.SENTINEL2_DIR / "2021" / "B02.tif"
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs = src.crs

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": change_map.shape[1],
        "height": change_map.shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(change_map, 1)

    print(f"[change] Saved change map to {output_path}")
    return output_path


def save_transitions(transitions):
    """Save transition statistics as JSON."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.OUTPUT_DIR / "transitions.json"

    with open(output_path, "w") as f:
        json.dump(transitions, f, indent=2, default=str)

    print(f"[change] Saved transitions to {output_path}")
    return output_path


def run_change_detection():
    """Run change detection on pre-classified landcover maps."""
    import rasterio

    lc_2021_path = config.OUTPUT_DIR / "landcover_2021.tif"
    lc_2023_path = config.OUTPUT_DIR / "landcover_2023.tif"

    for path in [lc_2021_path, lc_2023_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Landcover map not found: {path}. "
                f"Run `python -m src.classify` first."
            )

    with rasterio.open(lc_2021_path) as src:
        lc_2021 = src.read(1)
    with rasterio.open(lc_2023_path) as src:
        lc_2023 = src.read(1)

    change_map, transitions = compute_change_map(lc_2021, lc_2023)
    save_change_map(change_map)
    save_transitions(transitions)

    return change_map, transitions


def main() -> None:
    """CLI entry point."""
    run_change_detection()


if __name__ == "__main__":
    main()
