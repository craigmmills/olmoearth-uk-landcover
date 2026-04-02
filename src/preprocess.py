"""Preprocess Sentinel-2 composites: cloud masking, normalization.

Usage:
    uv run python -m src.preprocess
"""
from __future__ import annotations

from src import config


def _load_band(year: str, band: str):
    """Load a single Sentinel-2 band GeoTIFF as a float32 array."""
    import numpy as np
    import rasterio

    path = config.SENTINEL2_DIR / year / f"{band}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Band file not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

    if data.shape != (config.AOI_SIZE_PX, config.AOI_SIZE_PX):
        raise ValueError(
            f"Expected ({config.AOI_SIZE_PX}, {config.AOI_SIZE_PX}), got {data.shape}"
        )
    return data


def load_sentinel2_bands(year: str):
    """Load all 12 Sentinel-2 bands for a year, stacked in OlmoEarth order."""
    import numpy as np

    bands = [_load_band(year, band) for band in config.SENTINEL2_BANDS]
    composite = np.stack(bands, axis=-1)
    print(f"[preprocess] [{year}] Loaded 12 bands, shape {composite.shape}")
    return composite


def download_scl_if_missing(year: str):
    """Download the SCL band for cloud masking if not already present.

    Returns Path to SCL GeoTIFF, or None if download failed.
    """
    scl_path = config.SENTINEL2_DIR / year / "SCL.tif"
    if scl_path.exists():
        print(f"[preprocess] [{year}] SCL already exists")
        return scl_path

    try:
        import numpy as np
        import rasterio
        from rasterio.warp import reproject, Resampling

        from src.acquire import (
            compute_aoi_bbox,
            compute_target_grid,
            query_sentinel2_scenes,
            select_best_scene,
        )

        bbox_wgs84 = compute_aoi_bbox(
            config.AOI_CENTER_LAT, config.AOI_CENTER_LON, config.AOI_EXTENT_KM
        )
        bbox_utm, target_transform = compute_target_grid(
            bbox_wgs84, config.TARGET_CRS, config.AOI_SIZE_PX, config.AOI_SIZE_PX
        )

        date_start, date_end = config.TIME_RANGES[year]
        items = query_sentinel2_scenes(
            bbox_wgs84, date_start, date_end, config.MAX_CLOUD_COVER
        )
        item = select_best_scene(items, config.TARGET_CRS)

        asset_href = item.assets["scl"].href
        tmp_path = scl_path.with_suffix(".tif.tmp")

        env_options = {
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "AWS_NO_SIGN_REQUEST": "YES",
        }

        with rasterio.Env(**env_options):
            with rasterio.open(asset_href) as src:
                dst_array = np.zeros(
                    (config.AOI_SIZE_PX, config.AOI_SIZE_PX), dtype=src.dtypes[0]
                )
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=config.TARGET_CRS,
                    resampling=Resampling.nearest,
                )

        profile = {
            "driver": "GTiff",
            "dtype": dst_array.dtype,
            "width": config.AOI_SIZE_PX,
            "height": config.AOI_SIZE_PX,
            "count": 1,
            "crs": config.TARGET_CRS,
            "transform": target_transform,
            "compress": "lzw",
        }
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(dst_array, 1)
        tmp_path.rename(scl_path)

        print(f"[preprocess] [{year}] SCL downloaded: {scl_path}")
        return scl_path

    except Exception as e:
        print(f"[preprocess] WARNING: Could not download SCL for {year}: {e}. "
              "Skipping cloud masking.")
        return None


def apply_cloud_mask(composite, scl_data):
    """Apply SCL-based cloud/shadow mask. Masked pixels set to NaN."""
    import numpy as np

    if scl_data is None:
        print("[preprocess] WARNING: SCL not available, skipping cloud masking")
        return composite.copy()

    mask_2d = np.isin(scl_data, list(config.SCL_MASK_VALUES))
    pct = mask_2d.sum() / mask_2d.size * 100

    if pct > 90:
        raise RuntimeError(
            f"[preprocess] {pct:.1f}% of pixels masked by SCL. "
            "Scene is unusable. Check data quality."
        )
    if pct > 50:
        print(f"[preprocess] WARNING: {pct:.1f}% of pixels masked. "
              "Scene quality may be poor.")

    mask_3d = np.repeat(mask_2d[:, :, np.newaxis], composite.shape[2], axis=2)
    result = composite.copy()
    result[mask_3d] = np.nan
    print(f"[preprocess] Masked {pct:.1f}% of pixels as cloud/shadow")
    return result


def normalize_composite(composite):
    """Normalize composite using OlmoEarth COMPUTED strategy."""
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy

    normalizer = Normalizer(strategy=Strategy.COMPUTED)
    normalized = normalizer.normalize(Modality.SENTINEL2_L2A, composite)
    print("[preprocess] Normalized using OlmoEarth COMPUTED strategy")
    return normalized


def run_preprocessing():
    """Run full preprocessing pipeline for all years.

    Returns: {"2021": array(512, 512, 12), "2023": array(512, 512, 12)}
    """
    import numpy as np
    import rasterio

    results = {}
    errors = {}

    for year in config.TIME_RANGES:
        try:
            print(f"\n[preprocess] === Preprocessing {year} ===")
            scl_path = download_scl_if_missing(year)
            composite = load_sentinel2_bands(year)

            scl_data = None
            if scl_path is not None and scl_path.exists():
                with rasterio.open(scl_path) as src:
                    scl_data = src.read(1)

            composite = apply_cloud_mask(composite, scl_data)
            composite = normalize_composite(composite)
            composite = np.nan_to_num(composite, nan=0.0)
            results[year] = composite
        except Exception as e:
            errors[year] = str(e)
            print(f"[preprocess] ERROR processing {year}: {e}")

    if len(results) == 0:
        raise RuntimeError(f"Preprocessing failed for all years: {errors}")

    if errors:
        for year, err in errors.items():
            print(f"[preprocess] WARNING: {year} failed: {err}")

    print(f"\n[preprocess] Preprocessing complete for {len(results)} years")
    return results


def main() -> None:
    """CLI entry point."""
    run_preprocessing()


if __name__ == "__main__":
    main()
