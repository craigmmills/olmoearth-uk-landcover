"""VLM evaluation pipeline -- comparison images and Gemini Flash scoring.

Generates side-by-side comparison PNGs (satellite RGB | classification | WorldCover)
and optionally sends them to Gemini 2.5 Flash for structured quality scoring.

Gracefully handles missing GOOGLE_API_KEY by generating images without VLM evaluation.

Usage:
    uv run python -m src.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

from src import config


# ---------------------------------------------------------------------------
# Pydantic Models (lazy-loaded)
# ---------------------------------------------------------------------------

def _define_models():
    """Define Pydantic models. Called once at module load."""
    from pydantic import BaseModel, Field, field_validator

    class PerClassScore(BaseModel):
        """Score for a single landcover class."""
        class_name: str
        score: float = Field(default=5.0)
        notes: str = ""

        @field_validator("score", mode="before")
        @classmethod
        def clamp_score(cls, v):
            """Clamp to [1, 10] rather than rejecting."""
            return max(1.0, min(10.0, float(v)))

    class ErrorRegion(BaseModel):
        """A spatial region where classification appears incorrect."""
        location: str
        expected: str
        predicted: str
        severity: str = "medium"

        @field_validator("severity", mode="before")
        @classmethod
        def normalize_severity(cls, v: str) -> str:
            """Normalize severity to lowercase, default to medium if unknown."""
            v_lower = str(v).lower().strip()
            if v_lower not in {"low", "medium", "high"}:
                return "medium"
            return v_lower

    class VLMEvaluation(BaseModel):
        """Complete evaluation response from Gemini."""
        overall_score: int = Field(default=5)
        per_class: list[PerClassScore] = Field(default_factory=list)
        error_regions: list[ErrorRegion] = Field(default_factory=list)
        spatial_quality: str = ""
        confidence: float = Field(default=0.5)
        recommendations: list[str] = Field(default_factory=list)

        @field_validator("overall_score", mode="before")
        @classmethod
        def clamp_overall_score(cls, v):
            return max(1, min(10, int(v)))

        @field_validator("confidence", mode="before")
        @classmethod
        def clamp_confidence(cls, v):
            return max(0.0, min(1.0, float(v)))

    return PerClassScore, ErrorRegion, VLMEvaluation


# Module-level model references (lazy-loaded on first access)
_models_loaded = False
PerClassScore = None
ErrorRegion = None
VLMEvaluation = None


def _ensure_models():
    """Ensure Pydantic models are loaded."""
    global _models_loaded, PerClassScore, ErrorRegion, VLMEvaluation
    if not _models_loaded:
        PerClassScore, ErrorRegion, VLMEvaluation = _define_models()
        _models_loaded = True
    return PerClassScore, ErrorRegion, VLMEvaluation


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

def _validate_input_files(year: str) -> dict[str, Path]:
    """Validate all required input files exist and are readable.

    Reports ALL missing files at once (not one at a time).
    Raises FileNotFoundError with comprehensive remediation message.
    """
    import rasterio

    required = {
        "B04": config.SENTINEL2_DIR / year / "B04.tif",
        "B03": config.SENTINEL2_DIR / year / "B03.tif",
        "B02": config.SENTINEL2_DIR / year / "B02.tif",
        "classification": config.OUTPUT_DIR / f"landcover_{year}.tif",
        "worldcover": config.WORLDCOVER_DIR / "worldcover_2021.tif",
    }

    missing = []
    for name, path in required.items():
        if not path.exists():
            missing.append(f"  {name}: {path}")

    if missing:
        missing_str = "\n".join(missing)
        raise FileNotFoundError(
            f"Missing required input files for year {year}:\n{missing_str}\n"
            f"Run the pipeline first: `python -m src.pipeline`\n"
            f"Or download WorldCover: `python -m src.acquire`"
        )

    # Validate shapes (skip WorldCover -- may differ from AOI_SIZE_PX)
    shape_errors = []
    for name, path in required.items():
        if name == "worldcover":
            continue  # WorldCover may have different dimensions
        try:
            with rasterio.open(path) as src:
                h, w = src.height, src.width
                if h != config.AOI_SIZE_PX or w != config.AOI_SIZE_PX:
                    shape_errors.append(
                        f"  {name}: expected ({config.AOI_SIZE_PX}, {config.AOI_SIZE_PX}), "
                        f"got ({h}, {w}) -- {path}"
                    )
        except rasterio.errors.RasterioIOError as e:
            shape_errors.append(f"  {name}: corrupt or unreadable -- {path}: {e}")

    if shape_errors:
        errors_str = "\n".join(shape_errors)
        raise ValueError(
            f"Input file validation failed for year {year}:\n{errors_str}"
        )

    return required


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def _load_rgb(year: str):
    """Load Sentinel-2 B04/B03/B02 as uint8 RGB array (H, W, 3).

    Applies brightness stretch: clip to [0, 3000], scale to [0, 255].
    Same stretch as app.py:load_sentinel2_rgb().
    Returns RGB (3 channels), not RGBA (4 channels) -- intentional
    difference from app.py which returns RGBA for folium overlays.
    """
    import numpy as np
    import rasterio

    bands = {}
    for band_name in ["B04", "B03", "B02"]:
        path = config.SENTINEL2_DIR / year / f"{band_name}.tif"
        with rasterio.open(path) as src:
            bands[band_name] = src.read(1).astype(np.float32)

    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)

    # NaN handling -- per-pixel check (any channel NaN counts the pixel)
    if np.any(np.isnan(rgb)):
        nan_pixels = np.any(np.isnan(rgb), axis=-1).sum()
        total_pixels = rgb.shape[0] * rgb.shape[1]
        nan_pct = nan_pixels / total_pixels * 100
        print(f"[evaluate] WARNING: {nan_pct:.1f}% pixels have NaN in RGB for {year}")
        rgb = np.nan_to_num(rgb, nan=0.0)

    rgb = np.clip(rgb, 0, 3000) / 3000 * 255
    return rgb.astype(np.uint8)


def _load_classification(year: str):
    """Load classified landcover map as uint8 array (H, W).

    Values 0-5 correspond to config.LANDCOVER_CLASSES.
    Unexpected values (including nodata=255) are mapped to 5 (Other) with warning.
    """
    import numpy as np
    import rasterio

    path = config.OUTPUT_DIR / f"landcover_{year}.tif"
    with rasterio.open(path) as src:
        data = src.read(1)

    valid_classes = set(config.LANDCOVER_CLASSES.keys())
    unique_vals = set(np.unique(data))
    invalid = unique_vals - valid_classes
    if invalid:
        print(f"[evaluate] WARNING: Classification has unexpected values {invalid}, "
              "mapping to class 5 (Other)")
        for v in invalid:
            data[data == v] = 5

    return data


def _load_worldcover_labels():
    """Load WorldCover 2021, remapped to our 6-class scheme.

    Returns uint8 array (H, W) with values 0-5.
    """
    import numpy as np
    import rasterio

    path = config.WORLDCOVER_DIR / "worldcover_2021.tif"
    with rasterio.open(path) as src:
        wc = src.read(1)

    labels = np.full(wc.shape, 5, dtype=np.uint8)
    for wc_code, class_idx in config.WORLDCOVER_REMAP.items():
        labels[wc == wc_code] = class_idx

    return labels


def _colorize_classes(data):
    """Convert class index array (H, W) to RGB array (H, W, 3).

    Uses config.LANDCOVER_COLORS (hex strings) -- single source of truth.
    """
    import numpy as np

    h, w = data.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, hex_color in config.LANDCOVER_COLORS.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        mask = data == cls_idx
        rgb[mask] = [r, g, b]
    return rgb


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------

def _build_legend_text():
    """Return a human-readable legend string for prompts and titles.

    Example: 'Built-up (red) | Cropland (yellow) | ...'
    """
    color_names = {
        "#FF0000": "red", "#FFFF00": "yellow", "#90EE90": "light green",
        "#006400": "dark green", "#0000FF": "blue", "#808080": "gray",
    }
    parts = []
    for idx in sorted(config.LANDCOVER_CLASSES.keys()):
        name = config.LANDCOVER_CLASSES[idx]
        hex_color = config.LANDCOVER_COLORS[idx]
        cname = color_names.get(hex_color, hex_color)
        parts.append(f"{name} ({cname})")
    return " | ".join(parts)


def _extract_quadrant(arr, quadrant: str):
    """Extract a quadrant from an array. Works for (H,W) and (H,W,C).

    Computes midpoints dynamically (not hardcoded to 256).
    Assumes north-up row ordering (row 0 = north), which is consistent
    with all GeoTIFFs produced by this pipeline.
    """
    h, w = arr.shape[:2]
    mh, mw = h // 2, w // 2
    slices = {
        "nw": (slice(0, mh), slice(0, mw)),
        "ne": (slice(0, mh), slice(mw, w)),
        "sw": (slice(mh, h), slice(0, mw)),
        "se": (slice(mh, h), slice(mw, w)),
    }
    if quadrant not in slices:
        raise ValueError(f"Invalid quadrant '{quadrant}', must be one of {list(slices.keys())}")
    sy, sx = slices[quadrant]
    return arr[sy, sx]


def _generate_comparison_image(rgb, classification_rgb, worldcover_rgb, title, output_path):
    """Generate 3-panel comparison PNG with legend. Atomic write.

    Panels: Satellite RGB | Classification | WorldCover Labels
    Bottom: Color legend with all 6 class names.
    """
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        (axes[0], rgb, "Satellite RGB"),
        (axes[1], classification_rgb, "Classification"),
        (axes[2], worldcover_rgb, "WorldCover 2021"),
    ]
    for ax, img, panel_title in panels:
        ax.imshow(img)
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Color legend
    legend_patches = []
    for idx in sorted(config.LANDCOVER_CLASSES.keys()):
        hex_color = config.LANDCOVER_COLORS[idx]
        name = config.LANDCOVER_CLASSES[idx]
        legend_patches.append(mpatches.Patch(color=hex_color, label=name))

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    # Atomic write: save to tmp then rename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    fig.savefig(tmp_path, dpi=150, format="png")
    plt.close(fig)
    tmp_path.rename(output_path)
    print(f"[evaluate] Saved: {output_path.name}")
    return output_path


def generate_comparison_images(year: str, output_dir: Path) -> list[Path]:
    """Generate full comparison image and 4 quadrant detail images for one year.

    Returns list of 5 output paths: [full, nw, ne, sw, se].
    Can be called independently of VLM scoring.

    Args:
        year: "2021" or "2023"
        output_dir: directory where PNGs are saved
    """
    print(f"[evaluate] [{year}] Generating comparison images...")

    rgb = _load_rgb(year)
    classification = _load_classification(year)
    worldcover = _load_worldcover_labels()

    classification_rgb = _colorize_classes(classification)
    worldcover_rgb = _colorize_classes(worldcover)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    # Full image
    full_path = output_dir / f"comparison_{year}_full.png"
    _generate_comparison_image(
        rgb, classification_rgb, worldcover_rgb,
        f"Landcover Evaluation -- Exeter, Devon -- {year} -- 10m resolution",
        full_path,
    )
    output_paths.append(full_path)

    # Quadrant detail images
    for quadrant in ["nw", "ne", "sw", "se"]:
        q_rgb = _extract_quadrant(rgb, quadrant)
        q_cls = _extract_quadrant(classification_rgb, quadrant)
        q_wc = _extract_quadrant(worldcover_rgb, quadrant)
        q_path = output_dir / f"comparison_{year}_{quadrant}.png"
        _generate_comparison_image(
            q_rgb, q_cls, q_wc,
            f"Quadrant {quadrant.upper()} -- {year} -- Exeter, Devon",
            q_path,
        )
        output_paths.append(q_path)

    print(f"[evaluate] [{year}] Generated {len(output_paths)} comparison images")
    return output_paths


# ---------------------------------------------------------------------------
# Gemini API -- Prompt and Calling
# ---------------------------------------------------------------------------

def _build_evaluation_prompt(year: str) -> str:
    """Build the text prompt sent to Gemini alongside comparison images.

    Includes color legend, class definitions, geographic context, spatial
    resolution, multi-image context, and WorldCover temporal mismatch note.
    """
    legend = _build_legend_text()

    # Class definitions for Gemini context
    class_definitions = (
        "Class definitions:\n"
        "- Built-up: urban areas including buildings, roads, and impervious surfaces\n"
        "- Cropland: agricultural fields, arable land, actively cultivated areas\n"
        "- Grassland: natural and managed grassland, pastures, meadows\n"
        "- Tree cover: forests, woodlands, tree plantations, dense canopy\n"
        "- Water: rivers, lakes, reservoirs, permanent water bodies\n"
        "- Other: bare soil, shrubland, wetland, and unclassifiable areas"
    )

    # WorldCover temporal mismatch note
    temporal_note = ""
    if year != "2021":
        temporal_note = (
            f"\n\nIMPORTANT: The WorldCover reference (RIGHT panel) is from 2021, "
            f"but the satellite image and classification are from {year}. "
            f"Some differences between the classification and WorldCover may reflect "
            f"genuine land use changes, not classification errors."
        )

    return f"""You are evaluating a satellite landcover classification for a 5km x 5km area
near Exeter, Devon, UK (50.72N, 3.50W) at 10m spatial resolution ({config.AOI_SIZE_PX}x{config.AOI_SIZE_PX} pixels).

You will see 5 comparison images:
1. FULL AREA: The complete 5km x 5km classification comparison
2-5. QUADRANT DETAILS: NW, NE, SW, SE quadrant close-ups for fine-grained inspection

Each image shows three panels side by side:
- LEFT: Sentinel-2 satellite RGB composite ({year})
- CENTER: Our AI classification result ({year})
- RIGHT: ESA WorldCover 2021 reference labels

Color legend for classification and WorldCover panels:
{legend}

{class_definitions}
{temporal_note}

Your task:
1. Compare the CENTER panel (our classification) against BOTH the LEFT panel (what the satellite
   actually shows) and the RIGHT panel (WorldCover reference labels).
2. Use the quadrant detail images to identify fine-grained errors not visible in the full view.
3. Score the overall classification quality from 1 (terrible) to 10 (perfect).
4. Score each individual landcover class from 1 to 10.
5. Identify specific regions where the classification appears wrong -- describe their approximate
   location (e.g., "upper-left corner", "central band"), what class was predicted, and what
   class you think is correct based on the satellite image.
6. Assess spatial quality: are boundaries between classes sharp and accurate, or blobby and
   misaligned?
7. Rate your confidence in this evaluation from 0.0 to 1.0.
8. Provide 1-3 actionable recommendations for improving the classification.
"""


def _call_gemini(image_paths: list[Path], prompt: str):
    """Send images and prompt to Gemini, return parsed VLMEvaluation.

    Sends all images (full + quadrants) in a single API call.
    If multi-image fails, falls back to single full image.
    Uses exponential backoff retry with non-retryable error detection.

    Raises:
        RuntimeError: After all retries exhausted, or on non-retryable error.
    """
    import os
    import time
    from google import genai
    from google.genai import types
    from PIL import Image

    _, _, VLMEvaluation = _ensure_models()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable not set. "
            "Get a free API key at https://aistudio.google.com/app/apikey"
        )

    client = genai.Client(api_key=api_key)

    # Non-retryable error patterns
    non_retryable = [
        "401", "403", "invalid_api_key", "permission_denied",
        "404", "not_found", "safety", "blocked",
    ]

    # Load images as PIL objects with proper resource management
    pil_images = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img.load()  # Force full load so file handle can be released
        pil_images.append(img)

    contents = pil_images + [prompt]
    schema = VLMEvaluation.model_json_schema()

    max_attempts = config.GEMINI_MAX_RETRIES
    base_delay = config.GEMINI_BASE_DELAY
    last_error = None

    for attempt in range(max_attempts):
        try:
            print(f"[evaluate] Sending request to Gemini (attempt {attempt + 1}/{max_attempts})...")
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0,
                ),
            )

            # Guard against empty/blocked responses
            if not response.text:
                raise RuntimeError("Gemini returned empty response (possibly blocked by safety filters)")

            return VLMEvaluation.model_validate_json(response.text)

        except Exception as e:
            error_str = str(e).lower()
            last_error = e

            # Non-retryable errors -- fail immediately
            if any(code in error_str for code in non_retryable):
                raise RuntimeError(
                    f"Gemini API error (non-retryable): {type(e).__name__}. "
                    "Check your GOOGLE_API_KEY and model name."
                ) from e

            if attempt == max_attempts - 1:
                break

            delay = base_delay * (2 ** attempt)
            print(f"[evaluate] API attempt {attempt + 1}/{max_attempts} failed: {type(e).__name__}: {e}. "
                  f"Retrying in {delay:.0f}s...")
            time.sleep(delay)

    # Multi-image may have failed -- try fallback with just the full image
    if len(image_paths) > 1:
        print("[evaluate] Multi-image call failed. Falling back to single full image...")
        try:
            full_img = Image.open(image_paths[0])
            full_img.load()
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[full_img, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0,
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned empty response on fallback")
            return VLMEvaluation.model_validate_json(response.text)
        except Exception as fallback_error:
            print(f"[evaluate] Fallback also failed: {type(fallback_error).__name__}: {fallback_error}")

    raise RuntimeError(
        f"Gemini API failed after {max_attempts} attempts. "
        f"Last error type: {type(last_error).__name__}"
    )


# ---------------------------------------------------------------------------
# Scoring Orchestration and Result Persistence
# ---------------------------------------------------------------------------

def evaluate_with_gemini(year: str, image_paths: list[Path], output_dir: Path) -> dict:
    """Score classification quality using Gemini 2.5 Flash.

    Sends all comparison images in a single API call.
    Saves evaluation JSON with metadata.

    Args:
        year: "2021" or "2023"
        image_paths: list of Path objects for comparison PNGs
        output_dir: directory for evaluation JSON output

    Returns:
        Evaluation results dict with metadata.

    Raises:
        RuntimeError: If Gemini API fails after retries.
    """
    import datetime

    _, _, VLMEvaluation = _ensure_models()

    print(f"[evaluate] [{year}] Sending {len(image_paths)} images to Gemini "
          f"{config.GEMINI_MODEL}...")

    prompt = _build_evaluation_prompt(year)
    evaluation = _call_gemini(image_paths, prompt)

    print(f"[evaluate] [{year}] Overall score: {evaluation.overall_score}/10 "
          f"(confidence: {evaluation.confidence:.2f})")

    result = {
        "year": year,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": config.GEMINI_MODEL,
        "temperature": 0.0,
        "evaluation": evaluation.model_dump(),
        "image_paths": [str(p.name) for p in image_paths],
        "summary": {
            "overall_score": evaluation.overall_score,
            "confidence": evaluation.confidence,
            "num_error_regions": len(evaluation.error_regions),
            "num_recommendations": len(evaluation.recommendations),
        },
    }

    # Save evaluation JSON (atomic write)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"evaluation_{year}.json"
    tmp_path = json_path.with_name(json_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    tmp_path.rename(json_path)
    print(f"[evaluate] [{year}] Saved evaluation: {json_path.name}")

    return result


# ---------------------------------------------------------------------------
# Top-Level Orchestrator and CLI Entry Point
# ---------------------------------------------------------------------------

def run_evaluation(output_dir: Path | None = None) -> dict[str, dict] | None:
    """Run the full evaluation pipeline: generate images, then score.

    Evaluates each year that has classification output.
    If GOOGLE_API_KEY is not set (or google-genai not installed),
    generates images but skips VLM scoring.

    Args:
        output_dir: Override output directory (default: config.EVALUATION_DIR).
            Integration seam: pass experiment iteration directory here.

    Returns:
        Dict mapping year -> evaluation result dict, or None if VLM was skipped.
        Partial failures are included with an "error" key.
    """
    import os
    import time

    if output_dir is None:
        output_dir = config.EVALUATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover which years have classification outputs
    years = []
    for year in config.TIME_RANGES:
        lc_path = config.OUTPUT_DIR / f"landcover_{year}.tif"
        if lc_path.exists():
            years.append(year)
        else:
            print(f"[evaluate] No classification found for {year}, skipping.")

    if not years:
        raise FileNotFoundError(
            "No classification outputs found in output/. "
            "Run `python -m src.classify` first."
        )

    print(f"[evaluate] Evaluating years: {', '.join(years)}")

    # Phase 1: Generate comparison images (always runs)
    all_image_paths = {}
    for year in years:
        _validate_input_files(year)
        all_image_paths[year] = generate_comparison_images(year, output_dir)

    # Phase 2: Gemini evaluation (skipped if no API key or SDK not installed)
    api_key = os.environ.get("GOOGLE_API_KEY")

    sdk_available = True
    try:
        import google.genai  # noqa: F401
    except ImportError:
        sdk_available = False
        print("[evaluate] WARNING: google-genai package not installed. "
              "Install with: uv sync --all-extras")

    if not api_key:
        print("[evaluate] WARNING: GOOGLE_API_KEY not set. "
              "Skipping VLM evaluation. Comparison images were generated.")
        print("[evaluate] Set GOOGLE_API_KEY to enable Gemini Flash scoring.")
        print(f"[evaluate] Images saved to: {output_dir}")
        return None

    if not sdk_available:
        print("[evaluate] Skipping VLM evaluation (SDK not available). "
              "Comparison images were generated.")
        print(f"[evaluate] Images saved to: {output_dir}")
        return None

    print("[evaluate] Running Gemini evaluation...")
    results = {}
    errors = {}
    for year in years:
        try:
            results[year] = evaluate_with_gemini(year, all_image_paths[year], output_dir)
        except Exception as e:
            errors[year] = str(e)
            results[year] = {"year": year, "error": str(e)}
            print(f"[evaluate] ERROR for {year}: {e}")

        # Rate limit pause between years
        if year != years[-1]:
            print(f"[evaluate] Waiting {config.GEMINI_RATE_LIMIT_PAUSE}s between "
                  "evaluations (rate limit)...")
            time.sleep(config.GEMINI_RATE_LIMIT_PAUSE)

    if len(errors) == len(years):
        raise RuntimeError(f"Evaluation failed for all years: {errors}")

    if errors:
        print(f"[evaluate] WARNING: Partial failures: {errors}")

    print(f"[evaluate] Evaluation complete. Results in {output_dir}/")
    return results


def main() -> None:
    """CLI entry point."""
    run_evaluation()


if __name__ == "__main__":
    main()
