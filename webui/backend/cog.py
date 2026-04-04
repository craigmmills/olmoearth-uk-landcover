"""COG (Cloud-Optimized GeoTIFF) conversion utility.

Converts standard GeoTIFFs to Cloud-Optimized GeoTIFFs for efficient tile serving.
COG files are written alongside the source with a '_cog' suffix.
"""
from __future__ import annotations

import os
from pathlib import Path


def ensure_cog(source_tif: Path) -> Path:
    """Convert a GeoTIFF to COG format if not already converted.

    Writes the COG to a sibling file: landcover_2021.tif -> landcover_2021_cog.tif
    Returns the path to the COG file.

    If the COG file already exists and is newer than the source, returns it immediately
    (cache-by-mtime pattern).
    """
    cog_path = source_tif.with_name(source_tif.stem + "_cog.tif")

    # Skip if COG already exists and is up to date
    if cog_path.exists() and cog_path.stat().st_mtime >= source_tif.stat().st_mtime:
        return cog_path

    _convert_to_cog(source_tif, cog_path)
    return cog_path


def _convert_to_cog(source: Path, dest: Path) -> None:
    """Convert a GeoTIFF to Cloud-Optimized GeoTIFF using rasterio.

    Steps:
    1. Open source and build overviews on it (in-memory)
    2. Copy to dest with COG profile (tiled, 256x256 blocks, LZW compression)
       with copy_src_overviews=True so overviews are interleaved correctly
    3. Atomic rename from temp file to final path

    Uses PID-suffixed temp filename to prevent corruption from concurrent requests.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.shutil import copy as rio_copy

    # PID-unique temp path prevents concurrent write corruption
    tmp_path = dest.with_suffix(f".tif.tmp.{os.getpid()}")

    try:
        with rasterio.open(source, "r+") as src:
            # Build overviews on the source FIRST (in-memory for small files)
            # For 512x512, factor 2 -> 256x256, factor 4 -> 128x128
            overview_levels = [2, 4]
            src.build_overviews(overview_levels, Resampling.nearest)
            src.update_tags(ns="rio_overview", resampling="nearest")

            # Copy with COG profile; copy_src_overviews=True interleaves them correctly
            cog_profile = {
                "driver": "GTiff",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "lzw",
                "copy_src_overviews": True,
            }
            rio_copy(src, tmp_path, **cog_profile)

        # Atomic rename (first writer wins in concurrent scenarios)
        tmp_path.rename(dest)

    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise
