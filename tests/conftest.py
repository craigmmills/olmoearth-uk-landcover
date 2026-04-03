import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def sample_embeddings():
    """Small (8, 8, 4) embedding array for fast tests."""
    return np.random.RandomState(123).rand(8, 8, 4).astype(np.float32)


@pytest.fixture
def sample_embeddings_large():
    """Larger (32, 32, 4) embedding array for stratified sampling tests."""
    return np.random.RandomState(123).rand(32, 32, 4).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Small (8, 8) label array with all 6 classes."""
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[:2, :] = 0   # Built-up (16 pixels)
    labels[2:3, :] = 1  # Cropland (8 pixels)
    labels[3:4, :] = 2  # Grassland (8 pixels)
    labels[4:5, :] = 3  # Tree cover (8 pixels)
    labels[5:6, :] = 4  # Water (8 pixels)
    labels[6:, :] = 5   # Other (16 pixels)
    return labels


@pytest.fixture
def sample_labels_large():
    """Larger (32, 32) label array with enough pixels for subsampling tests."""
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[:6, :] = 0
    labels[6:11, :] = 1
    labels[11:16, :] = 2
    labels[16:22, :] = 3
    labels[22:27, :] = 4
    labels[27:, :] = 5
    return labels


@pytest.fixture
def default_config():
    """Return a copy of DEFAULT_CONFIG."""
    from src.experiment import DEFAULT_CONFIG
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


@pytest.fixture
def tmp_config(tmp_path):
    """Factory fixture: create a temporary config file with optional overrides."""
    def _make(overrides=None):
        from src.experiment import DEFAULT_CONFIG
        import copy
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        if overrides:
            for section, values in overrides.items():
                cfg.setdefault(section, {}).update(values)
        path = tmp_path / "experiment_config.json"
        with open(path, "w") as f:
            json.dump(cfg, f)
        return path
    return _make


@pytest.fixture
def sample_rgb():
    """512x512x3 uint8 RGB array."""
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_classification():
    """512x512 uint8 classification array with values 0-5."""
    return np.random.randint(0, 6, (512, 512), dtype=np.uint8)


@pytest.fixture
def sample_worldcover():
    """512x512 uint8 WorldCover array with ESA codes."""
    codes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    return np.random.choice(codes, (512, 512)).astype(np.uint8)


@pytest.fixture
def mock_rasterio_dataset():
    """Factory fixture returning a mock rasterio dataset context manager.

    When called with a band argument (e.g. src.read(1)), returns 2D data.
    When called without arguments, returns 3D (1, H, W) data.
    """
    def _make(data, height=512, width=512, crs="EPSG:32631"):
        mock_ds = MagicMock()

        def read_side_effect(band=None):
            if band is not None:
                # src.read(1) returns 2D array
                return data
            # src.read() returns 3D array
            return data.reshape(1, height, width) if data.ndim == 2 else data

        mock_ds.read.side_effect = read_side_effect
        mock_ds.height = height
        mock_ds.width = width
        mock_ds.crs = crs
        mock_ds.bounds = (0, 0, width * 10, height * 10)
        mock_ds.__enter__ = MagicMock(return_value=mock_ds)
        mock_ds.__exit__ = MagicMock(return_value=False)
        return mock_ds
    return _make


@pytest.fixture
def mock_gemini_response():
    """Factory fixture returning a mock Gemini response."""
    def _make(overall_score=7, confidence=0.8):
        return {
            "overall_score": overall_score,
            "per_class": [{"class_name": "Built-up", "score": 8, "notes": "Good"}],
            "error_regions": [],
            "spatial_quality": "Sharp boundaries",
            "confidence": confidence,
            "recommendations": ["More training data"],
        }
    return _make


@pytest.fixture
def sample_metrics():
    """Return sample classification metrics matching metrics.json schema."""
    return {
        "overall_accuracy": 0.7523,
        "evaluation_year": "2021",
        "per_class": {
            "Built-up": {"precision": 0.80, "recall": 0.70, "f1": 0.75, "support": 1000},
            "Cropland": {"precision": 0.90, "recall": 0.85, "f1": 0.87, "support": 2000},
            "Grassland": {"precision": 0.60, "recall": 0.55, "f1": 0.57, "support": 1500},
            "Tree cover": {"precision": 0.70, "recall": 0.65, "f1": 0.67, "support": 800},
            "Water": {"precision": 0.95, "recall": 0.90, "f1": 0.92, "support": 300},
            "Other": {"precision": 0.40, "recall": 0.30, "f1": 0.34, "support": 500},
        },
        "weighted_avg": {"precision": 0.75, "recall": 0.72, "f1": 0.73},
        "confusion_matrix": [
            [700, 50, 100, 50, 0, 100],
            [30, 1700, 150, 50, 0, 70],
            [50, 200, 825, 200, 25, 200],
            [30, 50, 150, 520, 0, 50],
            [0, 0, 10, 0, 270, 20],
            [100, 50, 100, 50, 20, 180],
        ],
        "confusion_matrix_axes": {"rows": "true_class", "columns": "predicted_class"},
        "class_names": ["Built-up", "Cropland", "Grassland", "Tree cover", "Water", "Other"],
        "training_accuracy": 0.95,
        "n_training_samples": 30000,
    }


@pytest.fixture
def sample_evaluation():
    """Return sample VLM evaluation result matching Issue #6 schema."""
    return {
        "year": "2021",
        "timestamp": "2026-04-03T12:00:00Z",
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "evaluation": {
            "overall_score": 7,
            "per_class": [
                {"class_name": "Built-up", "score": 8.0, "notes": "Well detected in urban core"},
                {"class_name": "Cropland", "score": 8.5, "notes": "Good coverage"},
                {"class_name": "Grassland", "score": 5.0, "notes": "Confused with cropland"},
                {"class_name": "Tree cover", "score": 6.0, "notes": "Underdetected in hedgerows"},
                {"class_name": "Water", "score": 9.0, "notes": "Excellent river detection"},
                {"class_name": "Other", "score": 3.0, "notes": "Miscellaneous, often missed"},
            ],
            "error_regions": [
                {"location": "NW quadrant", "expected": "Cropland",
                 "predicted": "Grassland", "severity": "high"},
                {"location": "SE rural fringe", "expected": "Tree cover",
                 "predicted": "Grassland", "severity": "medium"},
            ],
            "spatial_quality": "Moderate salt-and-pepper noise in rural areas",
            "confidence": 0.75,
            "recommendations": ["Enable post-processing to reduce noise",
                                "Consider boundary exclusion"],
        },
        "image_paths": ["comparison_2021.png"],
        "summary": {
            "overall_score": 7,
            "confidence": 0.75,
            "num_error_regions": 2,
            "num_recommendations": 2,
        },
    }
