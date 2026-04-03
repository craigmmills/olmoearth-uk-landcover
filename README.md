# OlmoEarth UK Landcover

A self-correcting satellite landcover classification system that uses AI2's OlmoEarth foundation model to classify land use from Sentinel-2 imagery, then iteratively improves its own accuracy through an automated feedback loop powered by vision and language models.

The project covers a 5km x 5km area near Exeter, Devon, UK -- chosen for its mix of urban, agricultural, and woodland cover types -- and runs entirely on an Apple Silicon MacBook with no GPU required.

## What This Project Does

The system downloads real satellite imagery from the European Space Agency's Sentinel-2 constellation, extracts learned feature representations using OlmoEarth (a geospatial foundation model from the Allen Institute for AI), trains a classifier to identify six landcover types, and then enters an automated self-correction loop that diagnoses classification errors and proposes fixes -- all without human intervention.

**The six landcover classes:**

| Class | Description | Color |
|-------|-------------|-------|
| Built-up | Urban areas, buildings, roads, impervious surfaces | Red |
| Cropland | Agricultural fields, arable land | Yellow |
| Grassland | Natural and managed grassland, pastures | Light green |
| Tree cover | Forests, woodlands, dense canopy | Dark green |
| Water | Rivers, lakes, reservoirs | Blue |
| Other | Bare soil, shrubland, wetland | Gray |

## How It Works

The pipeline has two major phases: an initial classification pass (stages 1-5) and an iterative self-correction loop (stage 6) that reruns the classification with modified parameters until quality converges.

### Phase 1: Initial Classification

```
Sentinel-2 Satellite     ESA WorldCover 2021
  (12 spectral bands)      (ground truth labels)
         |                          |
         v                          |
  [1. Data Acquisition]             |
         |                          |
         v                          |
  [2. Preprocessing]                |
     Cloud masking +                |
     normalization                  |
         |                          |
         v                          |
  [3. OlmoEarth Embedding]         |
     Foundation model               |
     feature extraction             |
         |                          |
         v                          v
  [4. Classification] <--- Train on 2021 embeddings
     RandomForest on          + WorldCover labels
     embedding features             |
         |                          |
         +------> Landcover maps (2021 + 2023)
         |
         v
  [5. Change Detection]
     Compare 2021 vs 2023
     pixel by pixel
```

**Stage 1 -- Data Acquisition:** Downloads Sentinel-2 Level-2A (atmospherically corrected) imagery from the Element84 Earth Search STAC API, selecting the clearest summer scenes (June-August, <20% cloud cover). Also downloads ESA WorldCover 2021 labels from Microsoft Planetary Computer to use as training labels. All 12 spectral bands that OlmoEarth expects are downloaded and resampled to a common 10m grid (512x512 pixels).

**Stage 2 -- Preprocessing:** Downloads the Scene Classification Layer (SCL) to mask clouds and shadows. Normalizes band values using OlmoEarth's built-in normalizer, which applies per-band statistics that the model was trained with.

**Stage 3 -- Embedding Extraction:** Loads OlmoEarth Tiny (8.1M parameters, ~57MB) and tiles the 512x512 image into 64 patches of 64x64 pixels each. Each patch is passed through the OlmoEarth encoder, which produces a 192-dimensional feature vector per pixel position. These embeddings capture high-level semantic information about the landscape that the model learned during pretraining on massive satellite datasets. The result is a (512, 512, 192) feature map for each year.

**Stage 4 -- Classification:** Loads the WorldCover 2021 ground truth, remaps it from 11 ESA classes to our 6 simplified classes, and trains a scikit-learn RandomForest classifier on the 2021 embeddings using stratified sampling (up to 5,000 pixels per class by default). The trained classifier then predicts landcover for both 2021 and 2023, producing two GeoTIFF maps. Optional post-processing includes a spatial mode filter and minimum mapping unit enforcement.

**Stage 5 -- Change Detection:** Compares the 2021 and 2023 classified maps pixel-by-pixel to produce a change map and a transition matrix showing how many pixels changed from each class to every other class.

### Phase 2: The Self-Correction Loop

This is the distinctive feature of the project. After the initial classification, an automated feedback loop iteratively improves quality:

```
                    +---------------------------+
                    |                           |
                    v                           |
          [Classification]                      |
          (with current config)                 |
                    |                           |
                    v                           |
          [VLM Evaluation]                      |
          Gemini 2.5 Flash scores               |
          the classification                    |
          quality (1-10)                        |
                    |                           |
                    v                           |
          [LLM Diagnosis]                       |
          Claude analyzes errors                |
          and proposes a fix                    |
                    |                           |
                    v                           |
          [Accept or Revert?]                   |
          Did the score improve?                |
          Did any class regress?                |
                    |                           |
            yes/  no\                           |
              |      \---> Revert config        |
              v            and try again -------+
          Update best config                    |
          and continue -------------------------+
```

**How it decides when to stop:**
- The target score is reached (default: 8.5 out of 10)
- The patience budget is exhausted (default: 3 consecutive non-improving iterations)
- The maximum iteration count is reached (default: 10)

## The Self-Correction Methodology

The self-correction loop uses two AI models in complementary roles:

### Step 1: VLM Evaluation (Gemini 2.5 Flash)

After each classification run, the system generates side-by-side comparison images showing the satellite RGB, the AI classification, and the WorldCover reference labels. These images include a full-area view and four quadrant close-ups for fine-grained inspection.

All five images are sent to Google's Gemini 2.5 Flash vision-language model (VLM) with a detailed prompt asking it to:
- Score overall classification quality from 1 to 10
- Score each individual landcover class from 1 to 10
- Identify specific regions where the classification appears wrong
- Assess spatial quality (are boundaries sharp or blobby?)
- Provide actionable recommendations for improvement

The VLM returns structured JSON with scores, error regions, and recommendations.

### Step 2: LLM Diagnosis (Claude)

The VLM evaluation, classification metrics (precision, recall, F1 per class), confusion matrix, experiment history, and current configuration are all packaged into a prompt sent to Claude (Anthropic's LLM). Claude acts as a "remote sensing diagnostician" that:

- Analyzes which classes are performing poorly and why
- Reviews the experiment history to avoid repeating failed changes
- Proposes exactly one parameter change per iteration (to isolate the effect of each change)
- Assigns a confidence score and explains the reasoning

Changes are organized into tiers of increasing aggressiveness:
- **Tier 1 (safe):** Training data sampling, class weights, tree depth, post-processing filters
- **Tier 2 (moderate):** Number of trees, adding spectral indices (NDVI, NDWI)
- **Tier 3 (aggressive):** Structural changes

The system escalates from Tier 1 to Tier 2 only after three unsuccessful Tier 1 attempts, and from Tier 2 to Tier 3 after two unsuccessful Tier 2 attempts.

If neither the Claude CLI nor API is available, a rule-based fallback engine proposes changes based on simple heuristics (overfitting detection, class weight adjustment, etc.).

### Step 3: Acceptance Policy

A proposed change is accepted only if:
1. The overall score strictly improves
2. No individual class score drops by more than 1.0 point (Pareto-relaxed constraint)

If accepted, the new configuration becomes the baseline for the next iteration. If rejected, the previous best configuration is restored. The loop tracks all iterations in an experiment ledger with full config snapshots and metrics, so every decision is auditable.

## Architecture

```
olmoearth-uk-landcover/
|-- pyproject.toml               # Python project config (uv)
|-- experiment_config.json       # Current experiment configuration (auto-managed)
|-- PRD.md                       # Product requirements document
|-- README.md                    # This file
|-- src/
|   |-- config.py                # AOI bounds, paths, band lists, all constants
|   |-- acquire.py               # Download Sentinel-2 + WorldCover via STAC APIs
|   |-- preprocess.py            # Cloud masking (SCL), OlmoEarth normalization
|   |-- embeddings.py            # OlmoEarth Tiny feature extraction (64x64 patches)
|   |-- classify.py              # Train RandomForest, predict landcover, post-processing
|   |-- change.py                # Year-over-year change detection + transition matrix
|   |-- evaluate.py              # Generate comparison images + Gemini VLM scoring
|   |-- diagnose.py              # Claude-based error diagnosis + hypothesis generation
|   |-- autocorrect.py           # Self-correction loop orchestrator
|   |-- experiment.py            # Experiment config management + iteration ledger
|   |-- pipeline.py              # End-to-end pipeline (acquire -> classify -> evaluate)
|   +-- app.py                   # Streamlit web UI with folium map overlays
|-- data/                        # Downloaded satellite imagery (gitignored)
|   |-- sentinel2/
|   +-- worldcover/
|-- output/                      # Classification results (gitignored)
|   |-- landcover_2021.tif
|   |-- landcover_2023.tif
|   |-- change_map.tif
|   |-- transitions.json
|   |-- embeddings/
|   +-- evaluations/
+-- experiments/                  # Experiment iteration history (gitignored)
    |-- iteration_000/
    |-- iteration_001/
    +-- SUMMARY.md
```

## What Each Module Does

| Module | Purpose |
|--------|---------|
| `config.py` | Central configuration: AOI coordinates, CRS, time ranges, Sentinel-2 band definitions, STAC API URLs, landcover class scheme, OlmoEarth parameters, file paths. Single source of truth for all constants. |
| `acquire.py` | Downloads Sentinel-2 L2A imagery via Element84 Earth Search STAC API and ESA WorldCover 2021 via Microsoft Planetary Computer. Handles band resampling to 10m, UTM reprojection, and retry logic for transient network failures. |
| `preprocess.py` | Downloads the Scene Classification Layer (SCL) for cloud/shadow masking, stacks all 12 bands in OlmoEarth's expected order, and applies OlmoEarth's built-in normalization strategy. |
| `embeddings.py` | Loads OlmoEarth Tiny, tiles the AOI into 64x64 patches, extracts 192-dimensional embeddings per pixel, and reassembles into full-AOI feature maps. Includes MPS-to-CPU fallback. |
| `classify.py` | Loads embeddings and WorldCover labels, optionally augments features with spectral indices (NDVI, NDWI), trains a RandomForest classifier with configurable hyperparameters, predicts landcover for both years, applies optional mode filter and minimum mapping unit post-processing, and computes per-class metrics. |
| `change.py` | Compares 2021 and 2023 classified maps to produce a binary change map and a transition matrix with area statistics in hectares. |
| `evaluate.py` | Generates three-panel comparison images (satellite RGB, classification, WorldCover reference) at full and quadrant scales, then sends them to Gemini 2.5 Flash for structured quality scoring. Gracefully degrades if no API key is set. |
| `diagnose.py` | Analyzes VLM evaluations, classification metrics, confusion matrices, and experiment history to propose a single parameter change. Uses Claude (CLI or API) as the primary engine, with a rule-based fallback for environments without API access. |
| `autocorrect.py` | Orchestrates the self-correction loop: runs classify, evaluate, diagnose, applies the proposed change, checks acceptance criteria, and iterates. Handles SIGINT/SIGTERM gracefully by restoring the best configuration. |
| `experiment.py` | Manages experiment configuration (load, save, validate, merge), tracks iterations in a ledger with full config snapshots and metrics, and provides comparison utilities. |
| `pipeline.py` | Runs the full end-to-end pipeline sequentially: acquire, preprocess + embed, classify, change detection, evaluate. |
| `app.py` | Streamlit web application with a folium map centered on the AOI, togglable overlays for each year's classification, change map visualization, class legend, area statistics, and transition matrix display. |

## How to Run It

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- ~2GB disk space for satellite data and model weights
- Internet connection for initial data download and model download

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd olmoearth-uk-landcover

# Install all dependencies (including PyTorch, OlmoEarth, scikit-learn, etc.)
uv sync --all-extras
```

### Running the Full Pipeline

```bash
# Run the complete pipeline: download data -> extract embeddings -> classify -> evaluate
uv run python -m src.pipeline
```

This takes roughly 15-30 minutes on an Apple M-series Mac. The pipeline stages are:
1. Downloads ~500MB of Sentinel-2 imagery and WorldCover labels
2. Preprocesses and cloud-masks the imagery
3. Extracts OlmoEarth embeddings (the slowest step)
4. Trains a classifier and produces landcover maps
5. Generates evaluation images and (optionally) sends them to Gemini for scoring

### Running the Self-Correction Loop

```bash
# Run with defaults (10 iterations max, target score 8.5, patience 3)
uv run python -m src.autocorrect

# Customize parameters
uv run python -m src.autocorrect --max-iterations 5 --target-score 9.0 --patience 2
```

**Note:** The self-correction loop requires embeddings to already be extracted. Run the pipeline first, then use autocorrect for iterative improvement.

### Running Individual Stages

```bash
# Data acquisition only
uv run python -m src.acquire

# Preprocessing + embedding extraction
uv run python -m src.embeddings

# Classification only (uses existing embeddings)
uv run python -m src.classify

# Change detection only (uses existing classification maps)
uv run python -m src.change

# Evaluation only (generates comparison images + optional VLM scoring)
uv run python -m src.evaluate

# Diagnosis only (analyze latest experiment iteration)
uv run python -m src.diagnose
```

### Viewing Results

```bash
# Launch the interactive web UI
uv run streamlit run src/app.py
```

The Streamlit app displays:
- An interactive folium map centered on the AOI with satellite basemap
- Togglable overlays for the 2021 and 2023 landcover classifications
- Change map highlighting pixels that changed between years
- Class legend, area statistics per class per year, and a transition matrix

### API Keys (Optional)

The core pipeline (data acquisition through classification) requires no API keys. The self-correction loop's AI components are optional:

```bash
# For VLM evaluation (Gemini 2.5 Flash)
# Free API key from https://aistudio.google.com/app/apikey
export GOOGLE_API_KEY="your-key-here"

# For LLM diagnosis (Claude)
# Uses Claude Code CLI if installed (no API key needed), otherwise:
export ANTHROPIC_API_KEY="your-key-here"
```

Without these keys, the system falls back to metrics-only scoring and rule-based diagnosis.

## Key Findings and Gotchas

### MPS (Metal Performance Shaders) Does Not Work Reliably

OlmoEarth's encoder operations trigger errors on Apple's MPS backend. The embedding extraction code detects this and automatically falls back to CPU. If you see a warning like "MPS failed, switching to CPU," this is expected behavior, not a bug. CPU extraction is slower but produces correct results.

### UTM Zone Boundaries Matter

The original AOI (Cambridge, UK) sits almost exactly on the UTM zone 30/31 boundary (0 degrees longitude). Sentinel-2 tiles from the wrong zone may not fully cover the AOI when reprojected. The acquisition code prefers tiles from the target UTM zone, but the project was ultimately relocated to Exeter, Devon (3.5 degrees West), which sits comfortably within UTM zone 30.

### Embedding Extraction is the Bottleneck

Running OlmoEarth Tiny on CPU for 64 patches across two years takes the majority of the pipeline's runtime. Once embeddings are saved to disk as `.npy` files, reclassification during the self-correction loop is fast because it only retrains the lightweight RandomForest.

### WorldCover Temporal Mismatch

ESA WorldCover 2021 is used as ground truth, but the pipeline classifies both 2021 and 2023 imagery. Differences between the 2023 classification and WorldCover may reflect genuine land use changes (new construction, crop rotation) rather than classification errors. The VLM evaluation prompt explicitly accounts for this.

### Single-Scene Compositing

Rather than creating a multi-scene median composite, the pipeline uses the single best (lowest cloud cover) scene per year. This simplifies the pipeline but means that any remaining clouds or atmospheric effects in that scene will affect results.

### The Self-Correction Loop Can Only Tune, Not Restructure

The autocorrect loop modifies RandomForest hyperparameters, feature engineering toggles, and post-processing settings. It cannot change the underlying model (OlmoEarth Tiny is used frozen), swap classifiers, or modify the landcover class taxonomy. These are deliberate scope constraints that keep the loop tractable.

## Technical Details

### Model: OlmoEarth Tiny

[OlmoEarth](https://github.com/allenai/olmo-earth) is a family of geospatial foundation models from AI2 (Allen Institute for Artificial Intelligence). The Tiny variant has 8.1M parameters and produces 192-dimensional embeddings. It was pretrained on large-scale Sentinel-2 data and uses a Vision Transformer (ViT) architecture with flexible patch sizes.

The model expects 12 Sentinel-2 L2A bands in a specific order, along with a timestamp. The encoder processes each 64x64 pixel tile with a ViT patch size of 4 (producing 16x16 token positions), which are then upsampled back to 64x64 via nearest-neighbor interpolation.

### Data Sources

| Source | API | Authentication |
|--------|-----|----------------|
| Sentinel-2 L2A | Element84 Earth Search STAC | None required |
| ESA WorldCover 2021 | Microsoft Planetary Computer STAC | None required (auto-signed) |

### Classification Scheme

The 11 ESA WorldCover classes are remapped to 6 simplified classes. The remapping collapses several rare-in-the-UK classes (snow/ice, mangroves, moss/lichen) into "Other," and merges shrubland and wetland into "Other" as well.

### Experiment Tracking

Each classification run is saved as a numbered iteration in `experiments/iteration_NNN/` containing:
- `config.json` -- The exact configuration used
- `metrics.json` -- Per-class precision, recall, F1, confusion matrix
- `landcover_2021.tif` / `landcover_2023.tif` -- Backed-up GeoTIFF outputs
- Evaluation images and VLM scores (when available)

The experiment ledger enables full reproducibility and makes it possible to compare any two iterations.

## License

This project is a demonstration/research prototype. See the repository for license details.
