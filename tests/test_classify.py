"""Tests for src.classify module: features, post-processing, metrics, backward compat."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.classify import (
    _add_spatial_context,
    _apply_min_mapping_unit,
    _apply_mode_filter,
    _apply_post_processing,
    _augment_features,
    _build_classifier,
    _build_preprocessing_steps,
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

    def test_augment_features_spatial_context_mean(self, sample_embeddings, default_config):
        """Shape doubles when spatial context with mean is enabled."""
        default_config["features"]["add_spatial_context"] = True
        default_config["features"]["spatial_context_size"] = 3
        default_config["features"]["spatial_context_stats"] = ["mean"]
        h, w, d = sample_embeddings.shape
        result = _augment_features(sample_embeddings, "2021", default_config)
        assert result.shape == (h, w, d * 2)

    def test_augment_features_spatial_context_multiple_stats(self, sample_embeddings, default_config):
        """Shape grows by D * len(stats) with multiple stats."""
        default_config["features"]["add_spatial_context"] = True
        default_config["features"]["spatial_context_size"] = 3
        default_config["features"]["spatial_context_stats"] = ["mean", "std"]
        h, w, d = sample_embeddings.shape
        result = _augment_features(sample_embeddings, "2021", default_config)
        assert result.shape == (h, w, d + d * 2)

    def test_augment_features_spatial_context_with_ndvi(self, sample_embeddings, default_config, mocker):
        """Spatial context applies to full feature vector including NDVI."""
        default_config["features"]["add_spatial_context"] = True
        default_config["features"]["add_ndvi"] = True
        default_config["features"]["spatial_context_size"] = 3
        default_config["features"]["spatial_context_stats"] = ["mean"]
        h, w, d = sample_embeddings.shape
        mocker.patch(
            "src.classify._load_raw_band",
            return_value=np.ones((h, w), dtype=np.float32),
        )
        result = _augment_features(sample_embeddings, "2021", default_config)
        # Base: d+1 (embeddings + NDVI), context: (d+1) * 1 stat
        assert result.shape == (h, w, (d + 1) * 2)

    def test_augment_features_spatial_context_disabled(self, sample_embeddings, default_config):
        """Spatial context disabled leaves shape unchanged."""
        default_config["features"]["add_spatial_context"] = False
        default_config["features"]["spatial_context_size"] = 3
        default_config["features"]["spatial_context_stats"] = ["mean"]
        result = _augment_features(sample_embeddings, "2021", default_config)
        np.testing.assert_array_equal(result, sample_embeddings)

    def test_augment_features_spatial_context_single_channel(self, sample_embeddings, default_config, mocker):
        """Spatial context works with single-channel input (embeddings disabled, NDVI only)."""
        default_config["features"]["use_embeddings"] = False
        default_config["features"]["add_ndvi"] = True
        default_config["features"]["add_spatial_context"] = True
        default_config["features"]["spatial_context_size"] = 3
        default_config["features"]["spatial_context_stats"] = ["mean"]
        h, w, _ = sample_embeddings.shape
        mocker.patch(
            "src.classify._load_raw_band",
            return_value=np.ones((h, w), dtype=np.float32),
        )
        result = _augment_features(sample_embeddings, "2021", default_config)
        assert result.shape == (h, w, 2)  # 1 NDVI + 1 mean of NDVI


# ---------------------------------------------------------------------------
# Spatial Context Tests
# ---------------------------------------------------------------------------


class TestSpatialContext:
    def test_spatial_context_mean_uniform_input(self):
        """Mean of uniform array equals the array itself."""
        arr = np.ones((8, 8, 2), dtype=np.float32) * 5.0
        result = _add_spatial_context(arr, size=3, stats=["mean"], year="2021")
        # result[:, :, 0:2] = original, result[:, :, 2:4] = mean
        np.testing.assert_allclose(result[:, :, 2:4], 5.0, atol=1e-6)

    def test_spatial_context_no_nan_or_inf(self, sample_embeddings):
        """Result has no NaN or Inf values."""
        result = _add_spatial_context(
            sample_embeddings, size=3, stats=["mean", "std", "max", "min"], year="2021"
        )
        assert np.all(np.isfinite(result))

    def test_spatial_context_std_uniform_is_zero(self):
        """Std of uniform array is zero."""
        arr = np.ones((8, 8, 2), dtype=np.float32) * 3.0
        result = _add_spatial_context(arr, size=3, stats=["std"], year="2021")
        # std channels are at indices 2:4
        np.testing.assert_allclose(result[:, :, 2:4], 0.0, atol=1e-6)

    def test_spatial_context_max_min_known_values(self):
        """Max and min on a simple array with a single peak produce expected results."""
        arr = np.zeros((5, 5, 1), dtype=np.float32)
        arr[2, 2, 0] = 10.0  # single peak
        result = _add_spatial_context(arr, size=3, stats=["max", "min"], year="2021")
        # Shape: (5, 5, 1 + 1*2) = (5, 5, 3)
        assert result.shape == (5, 5, 3)
        # At (2,2): max should be 10.0, min should be 0.0
        assert result[2, 2, 1] == 10.0  # max channel
        assert result[2, 2, 2] == 0.0   # min channel
        # At (0,0): peak is outside 3x3 window, max should be 0.0
        assert result[0, 0, 1] == 0.0

    def test_spatial_context_edge_handling(self):
        """Edge pixels have valid values with reflect padding."""
        arr = np.arange(25, dtype=np.float32).reshape(5, 5, 1)
        result = _add_spatial_context(arr, size=3, stats=["mean"], year="2021")
        # All values must be finite (no NaN from edge effects)
        assert np.all(np.isfinite(result))
        # Corner (0,0) should have a valid mean
        assert result[0, 0, 1] > 0.0

    def test_spatial_context_preserves_original(self, sample_embeddings):
        """First D channels of result are the original features."""
        h, w, d = sample_embeddings.shape
        result = _add_spatial_context(
            sample_embeddings, size=3, stats=["mean"], year="2021"
        )
        np.testing.assert_array_equal(result[:, :, :d], sample_embeddings)

    def test_spatial_context_window_size_5(self, sample_embeddings):
        """5x5 window produces correct shape."""
        h, w, d = sample_embeddings.shape
        result = _add_spatial_context(
            sample_embeddings, size=5, stats=["mean"], year="2021"
        )
        assert result.shape == (h, w, d * 2)

    def test_spatial_context_unknown_stat_raises(self, sample_embeddings):
        """Unknown stat name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown spatial context stat"):
            _add_spatial_context(
                sample_embeddings, size=3, stats=["median"], year="2021"
            )


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

    def test_default_config_produces_bare_rf(self, sample_embeddings_large, sample_labels_large):
        """Default config returns a bare RandomForestClassifier (not Pipeline)."""
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        clf, acc, n = train_classifier(sample_embeddings_large, sample_labels_large, cfg=cfg)
        assert type(clf).__name__ == "RandomForestClassifier"


# ---------------------------------------------------------------------------
# Build Classifier Tests
# ---------------------------------------------------------------------------


class TestBuildClassifier:
    def test_random_forest(self, default_config):
        """RandomForest config returns RandomForestClassifier."""
        training = default_config["training"]
        clf = _build_classifier(training)
        assert type(clf).__name__ == "RandomForestClassifier"
        assert clf.n_estimators == 100
        assert clf.max_depth == 20

    def test_gradient_boosting(self, default_config):
        """GradientBoosting config returns GradientBoostingClassifier."""
        training = default_config["training"]
        training["classifier"] = "GradientBoosting"
        training["max_depth"] = 3
        clf = _build_classifier(training)
        assert type(clf).__name__ == "GradientBoostingClassifier"
        assert clf.learning_rate == 0.1
        assert clf.max_depth == 3

    def test_svm(self, default_config):
        """SVM config returns SVC."""
        training = default_config["training"]
        training["classifier"] = "SVM"
        clf = _build_classifier(training)
        assert type(clf).__name__ == "SVC"
        assert clf.C == 1.0
        assert clf.kernel == "rbf"

    def test_logistic_regression(self, default_config):
        """LogisticRegression config returns LogisticRegression."""
        training = default_config["training"]
        training["classifier"] = "LogisticRegression"
        clf = _build_classifier(training)
        assert type(clf).__name__ == "LogisticRegression"
        assert clf.max_iter == 1000

    def test_knn(self, default_config):
        """KNN config returns KNeighborsClassifier."""
        training = default_config["training"]
        training["classifier"] = "KNN"
        clf = _build_classifier(training)
        assert type(clf).__name__ == "KNeighborsClassifier"
        assert clf.n_neighbors == 5

    def test_mlp(self, default_config):
        """MLP config returns MLPClassifier with tuple hidden_layer_sizes."""
        training = default_config["training"]
        training["classifier"] = "MLP"
        clf = _build_classifier(training)
        assert type(clf).__name__ == "MLPClassifier"
        assert clf.hidden_layer_sizes == (100,)
        assert clf.alpha == 0.0001

    def test_unknown_classifier_raises(self, default_config):
        """Unknown classifier name raises ValueError."""
        training = default_config["training"]
        training["classifier"] = "XGBoost"
        with pytest.raises(ValueError, match="Unknown classifier"):
            _build_classifier(training)

    def test_gb_deep_max_depth_warns(self, default_config, capsys):
        """GradientBoosting with max_depth > 10 prints warning."""
        training = default_config["training"]
        training["classifier"] = "GradientBoosting"
        training["max_depth"] = 20
        _build_classifier(training)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "GradientBoosting" in captured.out


# ---------------------------------------------------------------------------
# Build Preprocessing Steps Tests
# ---------------------------------------------------------------------------


class TestBuildPreprocessingSteps:
    def test_no_preprocessing(self, default_config):
        """Default config returns empty steps list."""
        steps = _build_preprocessing_steps(default_config["training"])
        assert steps == []

    def test_l2_only(self, default_config):
        """l2_normalize=True returns single L2 step."""
        default_config["training"]["l2_normalize"] = True
        steps = _build_preprocessing_steps(default_config["training"])
        assert len(steps) == 1
        assert steps[0][0] == "l2_normalize"

    def test_scale_only(self, default_config):
        """scale_features=True returns single scaler step."""
        default_config["training"]["scale_features"] = True
        steps = _build_preprocessing_steps(default_config["training"])
        assert len(steps) == 1
        assert steps[0][0] == "scaler"

    def test_pca_only(self, default_config):
        """pca_components > 0 returns single PCA step."""
        default_config["training"]["pca_components"] = 50
        steps = _build_preprocessing_steps(default_config["training"])
        assert len(steps) == 1
        assert steps[0][0] == "pca"

    def test_all_preprocessing_order(self, default_config):
        """All three enabled returns steps in order: l2 -> scaler -> pca."""
        default_config["training"]["l2_normalize"] = True
        default_config["training"]["scale_features"] = True
        default_config["training"]["pca_components"] = 50
        steps = _build_preprocessing_steps(default_config["training"])
        assert len(steps) == 3
        assert [s[0] for s in steps] == ["l2_normalize", "scaler", "pca"]

    def test_l2_normalize_produces_unit_norms(self, default_config):
        """L2 normalization produces rows with approximately unit norm."""
        import numpy as np
        default_config["training"]["l2_normalize"] = True
        steps = _build_preprocessing_steps(default_config["training"])
        l2_transformer = steps[0][1]
        X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
        result = l2_transformer.transform(X)
        norms = np.linalg.norm(result, axis=1)
        # Row with nonzero norm should be ~1.0
        assert abs(norms[0] - 1.0) < 1e-6
        # Zero row should stay near zero
        assert norms[1] < 1e-6


# ---------------------------------------------------------------------------
# Classifier Integration Tests
# ---------------------------------------------------------------------------


class TestClassifierIntegration:
    """Test that all 6 classifiers work end-to-end via train_classifier."""

    @pytest.fixture
    def _cfg_for_classifier(self, default_config):
        """Factory: return config with specified classifier."""
        def _make(classifier_name, **overrides):
            cfg = copy.deepcopy(default_config)
            cfg["training"]["classifier"] = classifier_name
            for k, v in overrides.items():
                cfg["training"][k] = v
            return cfg
        return _make

    @pytest.mark.parametrize("classifier_name,overrides", [
        ("RandomForest", {}),
        ("GradientBoosting", {"max_depth": 3}),
        ("SVM", {}),
        ("LogisticRegression", {}),
        ("KNN", {}),
        ("MLP", {"max_iter": 200}),
    ])
    def test_classifier_trains_and_predicts(
        self, sample_embeddings_large, sample_labels_large,
        _cfg_for_classifier, classifier_name, overrides,
    ):
        """Each classifier produces (clf, acc, n_samples) and can predict."""
        cfg = _cfg_for_classifier(classifier_name, **overrides)
        clf, acc, n_samples = train_classifier(
            sample_embeddings_large, sample_labels_large, cfg=cfg,
        )
        assert hasattr(clf, "predict")
        assert 0.0 <= acc <= 1.0
        assert n_samples > 0

        # Verify predict works
        h, w, d = sample_embeddings_large.shape
        X = sample_embeddings_large.reshape(-1, d)
        predictions = clf.predict(X)
        assert len(predictions) == h * w

    def test_preprocessing_pipeline_trains(
        self, sample_embeddings_large, sample_labels_large, default_config,
    ):
        """Preprocessing (scale + PCA) with default RF produces Pipeline."""
        cfg = copy.deepcopy(default_config)
        cfg["training"]["scale_features"] = True
        # Use pca_components=3 which fits within 4-dim test features
        cfg["training"]["pca_components"] = 3
        clf, acc, n_samples = train_classifier(
            sample_embeddings_large, sample_labels_large, cfg=cfg,
        )
        assert hasattr(clf, "named_steps")
        assert "scaler" in clf.named_steps
        assert "pca" in clf.named_steps
        assert "classifier" in clf.named_steps

    def test_pca_fitted_on_training_only(
        self, sample_embeddings_large, sample_labels_large, default_config,
    ):
        """Scaler n_samples_seen_ equals training set size (not full dataset).

        StandardScaler tracks n_samples_seen_, which proves preprocessing
        was fitted on training data only (Pipeline handles this automatically).
        """
        cfg = copy.deepcopy(default_config)
        cfg["training"]["scale_features"] = True
        cfg["training"]["pca_components"] = 3
        clf, acc, n_samples = train_classifier(
            sample_embeddings_large, sample_labels_large, cfg=cfg,
        )
        scaler_step = clf.named_steps["scaler"]
        assert scaler_step.n_samples_seen_ == n_samples

    def test_pca_too_large_disabled_with_warning(
        self, sample_embeddings_large, sample_labels_large, default_config, capsys,
    ):
        """PCA components >= feature dim is disabled with warning."""
        cfg = copy.deepcopy(default_config)
        cfg["training"]["pca_components"] = 100  # Way more than 4 features
        clf, acc, n_samples = train_classifier(
            sample_embeddings_large, sample_labels_large, cfg=cfg,
        )
        captured = capsys.readouterr()
        assert "disabling PCA" in captured.out
        # Should still work, just without PCA
        assert hasattr(clf, "predict")
