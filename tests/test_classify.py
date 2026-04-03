"""Tests for src.classify module: features, post-processing, metrics, backward compat."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.classify import (
    _apply_min_mapping_unit,
    _apply_mode_filter,
    _apply_post_processing,
    _augment_features,
    _compute_boundary_mask,
    _compute_metrics,
    _compute_spectral_index,
    predict_landcover,
    train_classifier,
)
from src.experiment import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Feature Augmentation Tests
# ---------------------------------------------------------------------------


class TestAugmentFeatures:
    def test_augment_features_no_augmentation(self, sample_embeddings, default_config):
        """Returns embeddings unchanged when both NDVI/NDWI disabled."""
        result = _augment_features(sample_embeddings, "2021", default_config)
        np.testing.assert_array_equal(result, sample_embeddings)

    def test_augment_features_ndvi_only(self, sample_embeddings, default_config, mocker):
        """Shape becomes (H, W, D+1) when add_ndvi=True."""
        default_config["features"]["add_ndvi"] = True
        h, w, d = sample_embeddings.shape
        mocker.patch(
            "src.classify._load_raw_band",
            return_value=np.ones((h, w), dtype=np.float32),
        )
        result = _augment_features(sample_embeddings, "2021", default_config)
        assert result.shape == (h, w, d + 1)

    def test_augment_features_ndwi_only(self, sample_embeddings, default_config, mocker):
        """Shape becomes (H, W, D+1) when add_ndwi=True."""
        default_config["features"]["add_ndwi"] = True
        h, w, d = sample_embeddings.shape
        mocker.patch(
            "src.classify._load_raw_band",
            return_value=np.ones((h, w), dtype=np.float32),
        )
        result = _augment_features(sample_embeddings, "2021", default_config)
        assert result.shape == (h, w, d + 1)

    def test_augment_features_both(self, sample_embeddings, default_config, mocker):
        """Shape becomes (H, W, D+2) when both enabled."""
        default_config["features"]["add_ndvi"] = True
        default_config["features"]["add_ndwi"] = True
        h, w, d = sample_embeddings.shape
        mocker.patch(
            "src.classify._load_raw_band",
            return_value=np.ones((h, w), dtype=np.float32),
        )
        result = _augment_features(sample_embeddings, "2021", default_config)
        assert result.shape == (h, w, d + 2)

    def test_augment_features_embeddings_disabled(self, sample_embeddings, default_config, mocker):
        """use_embeddings=False + add_ndvi=True gives shape (H, W, 1)."""
        default_config["features"]["use_embeddings"] = False
        default_config["features"]["add_ndvi"] = True
        h, w, d = sample_embeddings.shape
        mocker.patch(
            "src.classify._load_raw_band",
            return_value=np.ones((h, w), dtype=np.float32),
        )
        result = _augment_features(sample_embeddings, "2021", default_config)
        assert result.shape == (h, w, 1)


class TestComputeSpectralIndex:
    def test_compute_spectral_index_division_by_zero(self, mocker):
        """Zero denominator produces 0.0."""
        zeros = np.zeros((8, 8), dtype=np.float32)
        mocker.patch("src.classify._load_raw_band", return_value=zeros)
        result = _compute_spectral_index("2021", "ndvi", 8, 8)
        np.testing.assert_array_equal(result, 0.0)

    def test_compute_spectral_index_nan_cleanup(self, mocker):
        """NaN/Inf replaced with 0.0/1.0/-1.0 and clamped."""
        band_a = np.full((4, 4), 1.0, dtype=np.float32)
        band_b = np.full((4, 4), -1.0, dtype=np.float32)  # denom=0 -> NaN

        mocker.patch(
            "src.classify._load_raw_band",
            side_effect=[band_a, band_b],
        )
        result = _compute_spectral_index("2021", "ndvi", 4, 4)
        # All values should be finite and within [-1, 1]
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_compute_spectral_index_clamp_range(self, mocker):
        """Values clamped to [-1, 1]."""
        band_a = np.full((4, 4), 100.0, dtype=np.float32)
        band_b = np.full((4, 4), 1.0, dtype=np.float32)
        mocker.patch(
            "src.classify._load_raw_band",
            side_effect=[band_a, band_b],
        )
        result = _compute_spectral_index("2021", "ndvi", 4, 4)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_compute_spectral_index_shape_mismatch(self, mocker):
        """Wrong band shape raises RuntimeError."""
        band_ok = np.ones((8, 8), dtype=np.float32)
        band_bad = np.ones((4, 4), dtype=np.float32)
        mocker.patch(
            "src.classify._load_raw_band",
            side_effect=[band_ok, band_bad],
        )
        with pytest.raises(RuntimeError, match="shape"):
            _compute_spectral_index("2021", "ndvi", 8, 8)


# ---------------------------------------------------------------------------
# Boundary Exclusion Tests
# ---------------------------------------------------------------------------


class TestBoundaryMask:
    def test_compute_boundary_mask_uniform(self):
        """Uniform labels produce all-True mask."""
        labels = np.zeros((8, 8), dtype=np.uint8)
        mask = _compute_boundary_mask(labels, buffer_px=1)
        assert mask.all()

    def test_compute_boundary_mask_boundary_detected(self, sample_labels):
        """Boundaries correctly identified between adjacent classes."""
        mask = _compute_boundary_mask(sample_labels, buffer_px=1)
        # At boundaries between classes, mask should be False
        assert not mask.all()
        # Interior pixels far from boundaries should be True
        # Row 0 is solidly class 0, should have some True interior pixels
        assert mask[0, :].any()

    def test_boundary_exclusion_all_excluded_raises(self, sample_embeddings, sample_labels, default_config):
        """All pixels excluded raises ValueError (large buffer on small image)."""
        default_config["training"]["exclude_boundary_pixels"] = True
        default_config["training"]["boundary_buffer_px"] = 10  # bigger than 8x8 image
        with pytest.raises(ValueError, match="excluded ALL pixels"):
            train_classifier(sample_embeddings, sample_labels, cfg=default_config)


# ---------------------------------------------------------------------------
# Post-Processing Tests
# ---------------------------------------------------------------------------


class TestPostProcessing:
    def test_apply_post_processing_noop(self, default_config):
        """Both values 0 returns input unchanged."""
        landcover = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = _apply_post_processing(landcover, default_config, "2021")
        np.testing.assert_array_equal(result, landcover)

    def test_apply_mode_filter_no_scipy(self, mocker):
        """Missing scipy raises RuntimeError with helpful message."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "scipy.ndimage":
                raise ImportError("No module named 'scipy'")
            return real_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=mock_import)
        landcover = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="scipy is required"):
            _apply_mode_filter(landcover, 3, "2021")

    def test_apply_mode_filter_removes_noise(self):
        """Isolated noisy pixel smoothed to surrounding class."""
        landcover = np.zeros((5, 5), dtype=np.uint8)
        landcover[2, 2] = 1  # Single noisy pixel
        result = _apply_mode_filter(landcover, 3, "2021")
        assert result[2, 2] == 0  # Should be smoothed to 0

    def test_apply_mode_filter_preserves_uniform(self):
        """Uniform input unchanged."""
        landcover = np.ones((5, 5), dtype=np.uint8) * 3
        result = _apply_mode_filter(landcover, 3, "2021")
        np.testing.assert_array_equal(result, landcover)

    def test_apply_min_mapping_unit_removes_small(self):
        """Single-pixel patches replaced by neighbor."""
        landcover = np.zeros((5, 5), dtype=np.uint8)
        landcover[2, 2] = 1  # Single pixel of class 1
        result = _apply_min_mapping_unit(landcover, min_px=2, year="2021")
        assert result[2, 2] == 0  # Replaced by dominant neighbor

    def test_apply_min_mapping_unit_preserves_large(self):
        """Large patches kept unchanged."""
        landcover = np.zeros((5, 5), dtype=np.uint8)
        landcover[1:4, 1:4] = 1  # 9-pixel patch
        result = _apply_min_mapping_unit(landcover, min_px=2, year="2021")
        assert (result[1:4, 1:4] == 1).all()


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_compute_metrics_perfect_prediction(self, sample_labels):
        """F1=1.0 for all classes when predictions == labels."""
        metrics = _compute_metrics(sample_labels, sample_labels)
        assert metrics["overall_accuracy"] == 1.0
        for class_name, m in metrics["per_class"].items():
            if m["support"] > 0:
                assert m["f1"] == 1.0

    def test_compute_metrics_structure(self, sample_labels):
        """All expected keys present."""
        metrics = _compute_metrics(sample_labels, sample_labels)
        assert "overall_accuracy" in metrics
        assert "per_class" in metrics
        assert "weighted_avg" in metrics
        assert "confusion_matrix" in metrics
        assert "class_names" in metrics
        assert "confusion_matrix_axes" in metrics
        assert "evaluation_year" in metrics
        assert metrics["evaluation_year"] == "2021"
        assert metrics["confusion_matrix_axes"]["rows"] == "true_class"
        assert metrics["confusion_matrix_axes"]["columns"] == "predicted_class"

    def test_compute_metrics_class_names(self, sample_labels):
        """Class names match config.LANDCOVER_CLASSES values."""
        from src import config as cfg

        metrics = _compute_metrics(sample_labels, sample_labels)
        expected_names = [cfg.LANDCOVER_CLASSES[i] for i in sorted(cfg.LANDCOVER_CLASSES.keys())]
        assert metrics["class_names"] == expected_names

    def test_compute_metrics_weighted_avg(self, sample_labels):
        """Weighted average metrics present and between 0-1."""
        metrics = _compute_metrics(sample_labels, sample_labels)
        wavg = metrics["weighted_avg"]
        assert 0.0 <= wavg["precision"] <= 1.0
        assert 0.0 <= wavg["recall"] <= 1.0
        assert 0.0 <= wavg["f1"] <= 1.0


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_train_classifier_no_config(self, sample_embeddings, sample_labels, mocker):
        """cfg=None loads defaults, returns (clf, acc, n_samples)."""
        mocker.patch("src.experiment.load_config", return_value=copy.deepcopy(DEFAULT_CONFIG))
        result = train_classifier(sample_embeddings, sample_labels)
        assert len(result) == 3
        clf, acc, n_samples = result
        assert hasattr(clf, "predict")
        assert 0.0 <= acc <= 1.0
        assert n_samples > 0

    def test_predict_landcover_no_config(self, sample_embeddings, mocker):
        """cfg=None skips post-processing, returns valid array."""
        mock_clf = mocker.MagicMock()
        h, w, d = sample_embeddings.shape
        mock_clf.predict.return_value = np.zeros(h * w, dtype=np.uint8)
        result = predict_landcover(mock_clf, sample_embeddings, "2021")
        assert result.shape == (h, w)
        assert result.dtype == np.uint8

    def test_run_classification_returns_tuple(self, mocker):
        """Returns (lc_2021, lc_2023) tuple."""
        mock_labels = np.zeros((4, 4), dtype=np.uint8)
        mock_emb = np.random.RandomState(123).rand(4, 4, 4).astype(np.float32)

        # Patch lazy imports at their source module
        mocker.patch("src.experiment.load_config", return_value=copy.deepcopy(DEFAULT_CONFIG))
        mocker.patch("src.experiment.validate_prerequisites")
        mocker.patch("src.experiment.save_experiment", return_value=(None, 1))

        # Patch heavy functions in classify module
        mocker.patch("src.classify.load_worldcover_labels", return_value=mock_labels)
        mocker.patch("src.classify.load_embeddings", return_value=mock_emb)
        mocker.patch("src.classify.save_landcover_geotiff")

        from src.classify import run_classification
        result = run_classification()
        assert isinstance(result, tuple)
        assert len(result) == 2
