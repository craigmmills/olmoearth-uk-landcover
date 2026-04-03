import pytest
import numpy as np
import json
from pathlib import Path


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
