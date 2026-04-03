"""AOI bounds, time ranges, band lists, and file paths for the data acquisition pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# AOI Definition - Cambridge, UK
# ---------------------------------------------------------------------------
AOI_CENTER_LAT: float = 50.72    # degrees North
AOI_CENTER_LON: float = -3.50    # degrees West  (near Exeter, Devon — solidly in zone 30)
AOI_EXTENT_KM: float = 5.0       # km per side
AOI_RESOLUTION_M: int = 10       # meters per pixel
AOI_SIZE_PX: int = 512           # pixels per side (512 * 10m = 5120m)

# ---------------------------------------------------------------------------
# CRS
# Cambridge (0.12E) falls in UTM Zone 31N (0°-6°E). Zone 30 tiles only clip
# the western edge. Zone 31 tiles fully cover Cambridge.
# ---------------------------------------------------------------------------
TARGET_CRS: str = "EPSG:32631"

# ---------------------------------------------------------------------------
# Time Ranges - summer months only (June-August)
# ---------------------------------------------------------------------------
TIME_RANGES: dict[str, tuple[str, str]] = {
    "2021": ("2021-06-01", "2021-08-31"),
    "2023": ("2023-06-01", "2023-08-31"),
}

# ---------------------------------------------------------------------------
# Cloud Cover Filter
# ---------------------------------------------------------------------------
MAX_CLOUD_COVER: int = 20  # percent

# ---------------------------------------------------------------------------
# Sentinel-2 L2A Bands
# All 12 bands OlmoEarth expects, in OlmoEarth's expected order.
# Ordering matters for Issue 2 (embedding extraction), not for Issue 1
# (per-band files are independent).
# ---------------------------------------------------------------------------
SENTINEL2_BANDS: list[str] = [
    "B02", "B03", "B04", "B08",                    # 10m native
    "B05", "B06", "B07", "B8A", "B11", "B12",      # 20m native
    "B01", "B09",                                    # 60m native
]

SENTINEL2_BAND_NATIVE_RESOLUTION: dict[str, int] = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20,
    "B01": 60, "B09": 60,
}

# ---------------------------------------------------------------------------
# STAC Asset Key Mapping
# Element84 Earth Search uses descriptive asset keys, not band IDs.
# This mapping MUST be verified at first runtime. If keys differ, the
# error message from _download_single_band will list available keys.
# ---------------------------------------------------------------------------
STAC_ASSET_KEYS: dict[str, str] = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
}

# ---------------------------------------------------------------------------
# STAC API Configuration
# ---------------------------------------------------------------------------
EARTH_SEARCH_URL: str = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION: str = "sentinel-2-l2a"

PC_STAC_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
WORLDCOVER_COLLECTION: str = "esa-worldcover"

# ---------------------------------------------------------------------------
# Data Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SENTINEL2_DIR: Path = DATA_DIR / "sentinel2"
WORLDCOVER_DIR: Path = DATA_DIR / "worldcover"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Landcover Classes (used by downstream issues)
# ---------------------------------------------------------------------------
LANDCOVER_CLASSES: dict[int, str] = {
    0: "Built-up",
    1: "Cropland",
    2: "Grassland",
    3: "Tree cover",
    4: "Water",
    5: "Other",
}

LANDCOVER_COLORS: dict[int, str] = {
    0: "#FF0000",
    1: "#FFFF00",
    2: "#90EE90",
    3: "#006400",
    4: "#0000FF",
    5: "#808080",
}

# WorldCover class code -> our simplified class index
WORLDCOVER_REMAP: dict[int, int] = {
    10: 3,   # Tree cover
    20: 5,   # Shrubland -> Other
    30: 2,   # Grassland
    40: 1,   # Cropland
    50: 0,   # Built-up
    60: 5,   # Bare / sparse vegetation -> Other
    70: 5,   # Snow and ice -> Other
    80: 4,   # Permanent water bodies
    90: 5,   # Herbaceous wetland -> Other
    95: 5,   # Mangroves -> Other
    100: 5,  # Moss and lichen -> Other
}

# ---------------------------------------------------------------------------
# OlmoEarth Embedding Extraction (Issue 2)
# ---------------------------------------------------------------------------
PATCH_SIZE_PX: int = 64             # Tile size for encoder input (pixels)
ENCODER_PATCH_SIZE: int = 4         # ViT patch size within each tile
EMBEDDING_DIM: int = 192            # OlmoEarth Tiny output embedding dimension
OLMOEARTH_MODEL_ID: str = "OlmoEarth-v1-Tiny"

# SCL (Scene Classification Layer) values to MASK OUT
# 1=saturated, 3=cloud_shadows, 8=cloud_medium, 9=cloud_high, 10=thin_cirrus
# NOTE: SCL=0 (no_data) is excluded because reprojection fills out-of-extent
# pixels with 0. Those pixels may still have valid spectral data.
SCL_MASK_VALUES: set[int] = {1, 3, 8, 9, 10}

# Embedding output directory
EMBEDDINGS_DIR: Path = OUTPUT_DIR / "embeddings"

# ---------------------------------------------------------------------------
# Experiment Tracking
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

# ---------------------------------------------------------------------------
# Evaluation Pipeline (Issue 6)
# ---------------------------------------------------------------------------
EVALUATION_DIR: Path = OUTPUT_DIR / "evaluations"
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_MAX_RETRIES: int = 3
GEMINI_BASE_DELAY: float = 4.0   # seconds, base for exponential backoff
GEMINI_RATE_LIMIT_PAUSE: float = 5.0  # seconds between year evaluations
