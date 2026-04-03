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
