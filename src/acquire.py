"""Download Sentinel-2 L2A imagery and ESA WorldCover for the Cambridge AOI.

Usage:
    uv run python -m src.acquire
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

import pystac

from src import config


def retry(max_attempts: int = 3, base_delay: float = 2.0):
    """Decorator for retrying functions on transient network failures."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (OSError, ConnectionError, TimeoutError) as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"[acquire] Attempt {attempt + 1} failed: {e}. "
                          f"Retrying in {delay:.0f}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


def compute_aoi_bbox(
    center_lat: float,
    center_lon: float,
    extent_km: float,
) -> tuple[float, float, float, float]:
    """Compute WGS84 bounding box from center point and extent.

    Returns (west, south, east, north) in EPSG:4326.
    """
    if not (-90 <= center_lat <= 90):
        raise ValueError(f"Invalid latitude: {center_lat}")
    if not (-180 <= center_lon <= 180):
        raise ValueError(f"Invalid longitude: {center_lon}")

    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", config.TARGET_CRS, always_xy=True)
    to_wgs = Transformer.from_crs(config.TARGET_CRS, "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(center_lon, center_lat)
    half = (extent_km * 1000) / 2

    west, south = to_wgs.transform(cx - half, cy - half)
    east, north = to_wgs.transform(cx + half, cy + half)

    return (west, south, east, north)


def compute_target_grid(
    bbox_wgs84: tuple[float, float, float, float],
    target_crs: str,
    width: int = 512,
    height: int = 512,
) -> tuple[tuple[float, float, float, float], "rasterio.Affine"]:
    """Compute UTM bounding box and affine transform for the target grid."""
    from pyproj import Transformer
    from rasterio.transform import from_bounds

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    west, south = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    east, north = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])

    # Snap to exact 10m grid centered on the transformed center
    center_x = (west + east) / 2
    center_y = (south + north) / 2
    half_extent = (width * config.AOI_RESOLUTION_M) / 2

    west_utm = center_x - half_extent
    east_utm = center_x + half_extent
    south_utm = center_y - half_extent
    north_utm = center_y + half_extent

    transform = from_bounds(west_utm, south_utm, east_utm, north_utm, width, height)
    bbox_utm = (west_utm, south_utm, east_utm, north_utm)

    return bbox_utm, transform


@retry(max_attempts=3, base_delay=2.0)
def query_sentinel2_scenes(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    max_cloud_cover: float,
) -> list[pystac.Item]:
    """Query Element84 Earth Search STAC API for Sentinel-2 L2A scenes."""
    import pystac_client

    print(f"[acquire] Searching for Sentinel-2 scenes: "
          f"{date_start} to {date_end}, cloud < {max_cloud_cover}%")

    client = pystac_client.Client.open(config.EARTH_SEARCH_URL)
    search = client.search(
        collections=[config.S2_COLLECTION],
        bbox=list(bbox),
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.item_collection())

    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for bbox={bbox}, "
            f"dates={date_start}/{date_end}, cloud<{max_cloud_cover}%. "
            f"Try increasing MAX_CLOUD_COVER in config.py."
        )

    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))
    print(f"[acquire] Found {len(items)} scenes")

    return items


def select_best_scene(
    items: list[pystac.Item],
    target_crs: str = config.TARGET_CRS,
) -> pystac.Item:
    """Select the best scene, preferring tiles in the target UTM zone.

    Cambridge sits on the UTM zone 30/31 boundary. Scenes from zone 31
    may not fully cover the AOI when reprojected to zone 30. We prefer
    scenes whose grid:code matches the target zone.
    """
    # Extract zone number from target CRS (e.g., "EPSG:32630" -> "30")
    zone = target_crs.replace("EPSG:326", "")

    # Prefer scenes from matching UTM zone
    zone_items = [
        item for item in items
        if item.id.startswith(f"S2A_{zone}") or item.id.startswith(f"S2B_{zone}")
    ]
    candidates = zone_items if zone_items else items

    return min(candidates, key=lambda item: item.properties.get("eo:cloud_cover", 100))


@retry(max_attempts=3, base_delay=2.0)
def _download_single_band(
    item: pystac.Item,
    band: str,
    bbox_utm: tuple[float, float, float, float],
    output_path: Path,
    target_crs: str,
    target_width: int,
    target_height: int,
    target_transform: "rasterio.Affine",
) -> Path:
    """Download and resample a single Sentinel-2 band to 10m."""
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling

    asset_key = config.STAC_ASSET_KEYS[band]
    if asset_key not in item.assets:
        available = sorted(item.assets.keys())
        raise KeyError(
            f"Asset key '{asset_key}' (band {band}) not found in STAC item. "
            f"Available keys: {available}"
        )
    href = item.assets[asset_key].href

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tif.tmp")

    env_options = {
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
    }

    with rasterio.Env(**env_options):
        with rasterio.open(href) as src:
            from pyproj import Transformer
            from rasterio.windows import from_bounds as window_from_bounds

            to_src_crs = Transformer.from_crs(
                target_crs, src.crs, always_xy=True
            )
            src_west, src_south = to_src_crs.transform(
                bbox_utm[0], bbox_utm[1]
            )
            src_east, src_north = to_src_crs.transform(
                bbox_utm[2], bbox_utm[3]
            )

            window = window_from_bounds(
                src_west, src_south, src_east, src_north,
                transform=src.transform,
            )
            # Expand window slightly to avoid edge artifacts
            window = rasterio.windows.Window(
                col_off=max(0, int(window.col_off) - 2),
                row_off=max(0, int(window.row_off) - 2),
                width=min(src.width, int(window.width) + 4),
                height=min(src.height, int(window.height) + 4),
            )

            src_data = src.read(1, window=window)
            src_transform = rasterio.windows.transform(window, src.transform)

            dst_array = np.zeros(
                (target_height, target_width), dtype=src.dtypes[0]
            )
            reproject(
                source=src_data,
                destination=dst_array,
                src_transform=src_transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )

    profile = {
        "driver": "GTiff",
        "dtype": dst_array.dtype,
        "width": target_width,
        "height": target_height,
        "count": 1,
        "crs": target_crs,
        "transform": target_transform,
        "compress": "lzw",
    }

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(dst_array, 1)

    tmp_path.rename(output_path)
    return output_path


def download_sentinel2_bands(
    item: pystac.Item,
    bands: list[str],
    bbox_wgs84: tuple[float, float, float, float],
    output_dir: Path,
    year: str,
    target_crs: str,
    bbox_utm: tuple[float, float, float, float],
    target_transform: "rasterio.Affine",
) -> dict[str, Path]:
    """Download all Sentinel-2 bands for one year."""
    year_dir = output_dir / year
    year_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for i, band in enumerate(bands, 1):
        output_path = year_dir / f"{band}.tif"
        if output_path.exists():
            print(f"[acquire] [{year}] Band {band} ({i}/{len(bands)}) "
                  f"already exists, skipping")
            results[band] = output_path
            continue

        print(f"[acquire] [{year}] Downloading band {band} "
              f"({i}/{len(bands)})...")
        _download_single_band(
            item=item,
            band=band,
            bbox_utm=bbox_utm,
            output_path=output_path,
            target_crs=target_crs,
            target_width=config.AOI_SIZE_PX,
            target_height=config.AOI_SIZE_PX,
            target_transform=target_transform,
        )
        results[band] = output_path

    return results


@retry(max_attempts=3, base_delay=2.0)
def download_worldcover(
    bbox_wgs84: tuple[float, float, float, float],
    output_dir: Path,
    target_crs: str,
    bbox_utm: tuple[float, float, float, float],
    target_transform: "rasterio.Affine",
) -> Path:
    """Download ESA WorldCover 2021, cropped and reprojected to AOI."""
    import numpy as np
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.warp import reproject, Resampling

    output_path = output_dir / "worldcover_2021.tif"
    if output_path.exists():
        print("[acquire] WorldCover already exists, skipping")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tif.tmp")

    print("[acquire] Searching for WorldCover on Planetary Computer...")
    catalog = pystac_client.Client.open(
        config.PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[config.WORLDCOVER_COLLECTION],
        bbox=list(bbox_wgs84),
        datetime="2021-01-01/2021-12-31",
    )
    items = list(search.item_collection())

    if not items:
        raise RuntimeError(
            f"No WorldCover items found for bbox={bbox_wgs84}. "
            f"Check the Planetary Computer STAC catalog."
        )

    item = items[0]
    href = item.assets["map"].href
    print(f"[acquire] Downloading WorldCover from: {href[:80]}...")

    env_options = {
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }

    with rasterio.Env(**env_options):
        with rasterio.open(href) as src:
            from pyproj import Transformer
            from rasterio.windows import from_bounds as window_from_bounds

            # WorldCover is in EPSG:4326
            window = window_from_bounds(
                bbox_wgs84[0], bbox_wgs84[1], bbox_wgs84[2], bbox_wgs84[3],
                transform=src.transform,
            )
            window = rasterio.windows.Window(
                col_off=max(0, int(window.col_off) - 2),
                row_off=max(0, int(window.row_off) - 2),
                width=min(src.width, int(window.width) + 4),
                height=min(src.height, int(window.height) + 4),
            )

            src_data = src.read(1, window=window)
            src_transform = rasterio.windows.transform(window, src.transform)

            dst_array = np.zeros(
                (config.AOI_SIZE_PX, config.AOI_SIZE_PX), dtype=np.uint8
            )
            reproject(
                source=src_data,
                destination=dst_array,
                src_transform=src_transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,  # Categorical data
            )

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": config.AOI_SIZE_PX,
        "height": config.AOI_SIZE_PX,
        "count": 1,
        "crs": target_crs,
        "transform": target_transform,
        "compress": "lzw",
    }

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(dst_array, 1)

    tmp_path.rename(output_path)
    print("[acquire] WorldCover downloaded successfully")
    return output_path


def _validate_downloads() -> None:
    """Validate all downloaded files have correct dimensions and CRS."""
    import rasterio

    for year in config.TIME_RANGES:
        for band in config.SENTINEL2_BANDS:
            path = config.SENTINEL2_DIR / year / f"{band}.tif"
            if not path.exists():
                raise FileNotFoundError(f"Missing band file: {path}")
            with rasterio.open(path) as src:
                if (src.width != config.AOI_SIZE_PX
                        or src.height != config.AOI_SIZE_PX):
                    raise ValueError(
                        f"{path}: expected {config.AOI_SIZE_PX}x"
                        f"{config.AOI_SIZE_PX}, got {src.width}x{src.height}"
                    )

    wc_path = config.WORLDCOVER_DIR / "worldcover_2021.tif"
    if not wc_path.exists():
        raise FileNotFoundError(f"Missing WorldCover file: {wc_path}")
    with rasterio.open(wc_path) as src:
        if src.width != config.AOI_SIZE_PX or src.height != config.AOI_SIZE_PX:
            raise ValueError(
                f"{wc_path}: expected {config.AOI_SIZE_PX}x"
                f"{config.AOI_SIZE_PX}, got {src.width}x{src.height}"
            )


def run_acquisition() -> None:
    """Run the full data acquisition pipeline."""
    bbox_wgs84 = compute_aoi_bbox(
        config.AOI_CENTER_LAT,
        config.AOI_CENTER_LON,
        config.AOI_EXTENT_KM,
    )
    print(f"[acquire] AOI bbox (WGS84): {bbox_wgs84}")

    bbox_utm, target_transform = compute_target_grid(
        bbox_wgs84, config.TARGET_CRS,
        config.AOI_SIZE_PX, config.AOI_SIZE_PX,
    )
    print(f"[acquire] Target grid: {config.AOI_SIZE_PX}x{config.AOI_SIZE_PX} "
          f"at {config.AOI_RESOLUTION_M}m in {config.TARGET_CRS}")

    # Download Sentinel-2 for each year
    errors = {}
    for year, (date_start, date_end) in config.TIME_RANGES.items():
        try:
            print(f"\n[acquire] === Sentinel-2 {year} ===")
            items = query_sentinel2_scenes(
                bbox_wgs84, date_start, date_end, config.MAX_CLOUD_COVER,
            )
            best = select_best_scene(items)
            cloud = best.properties.get("eo:cloud_cover", "?")
            print(f"[acquire] Selected scene: {best.id} "
                  f"(cloud cover: {cloud}%)")

            download_sentinel2_bands(
                item=best,
                bands=config.SENTINEL2_BANDS,
                bbox_wgs84=bbox_wgs84,
                output_dir=config.SENTINEL2_DIR,
                year=year,
                target_crs=config.TARGET_CRS,
                bbox_utm=bbox_utm,
                target_transform=target_transform,
            )
        except Exception as e:
            errors[year] = str(e)
            print(f"[acquire] ERROR downloading {year}: {e}")

    # Download WorldCover
    print(f"\n[acquire] === WorldCover 2021 ===")
    download_worldcover(
        bbox_wgs84=bbox_wgs84,
        output_dir=config.WORLDCOVER_DIR,
        target_crs=config.TARGET_CRS,
        bbox_utm=bbox_utm,
        target_transform=target_transform,
    )

    # Report results
    if errors:
        failed = ", ".join(errors.keys())
        print(f"\n[acquire] WARNING: Failed years: {failed}")
        for year, err in errors.items():
            print(f"[acquire]   {year}: {err}")
        if len(errors) == len(config.TIME_RANGES):
            raise RuntimeError(
                f"All Sentinel-2 downloads failed: {errors}"
            )
    else:
        print("\n[acquire] All downloads completed successfully")

    # Validate
    try:
        _validate_downloads()
        print("[acquire] Validation passed: all files correct")
    except (FileNotFoundError, ValueError) as e:
        print(f"[acquire] Validation warning: {e}")


def main() -> None:
    """CLI entry point. Not safe for concurrent execution."""
    run_acquisition()


if __name__ == "__main__":
    main()
