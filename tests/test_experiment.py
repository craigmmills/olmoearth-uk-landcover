"""Tests for src.experiment module: config loading, validation, and experiment ledger."""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest

from src.experiment import (
    DEFAULT_CONFIG,
    _deep_merge,
    _flatten_dict,
    _migrate_legacy_experiments,
    _update_latest_symlink,
    compare_experiments,
    create_session,
    get_next_iteration_number,
    get_session_dir,
    list_iterations,
    list_sessions,
    load_config,
    load_experiment,
    save_config,
    save_experiment,
    update_experiment_status,
    update_session_meta,
    validate_config,
    validate_prerequisites,
)


# ---------------------------------------------------------------------------
# Config Loading Tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_config_no_file_returns_defaults(self, tmp_path, monkeypatch):
        """When no config file exists, returns DEFAULT_CONFIG."""
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        result = load_config()
        assert result == DEFAULT_CONFIG

    def test_load_config_partial_override_merges(self, tmp_config):
        """Partial config fills in missing defaults via deep merge."""
        path = tmp_config(overrides={"training": {"n_estimators": 200}})
        result = load_config(config_path=path)
        assert result["training"]["n_estimators"] == 200
        assert result["training"]["max_depth"] == 20  # default preserved

    def test_load_config_invalid_json_raises_valueerror(self, tmp_path):
        """Malformed JSON raises ValueError with line/col info."""
        bad_file = tmp_path / "experiment_config.json"
        bad_file.write_text("{bad json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config(config_path=bad_file)

    def test_load_config_deep_merge_nested(self, tmp_config):
        """Recursive merge handles nested dicts correctly."""
        path = tmp_config(overrides={
            "training": {"max_depth": 30},
            "features": {"add_ndvi": True},
        })
        result = load_config(config_path=path)
        assert result["training"]["max_depth"] == 30
        assert result["features"]["add_ndvi"] is True
        assert result["features"]["use_embeddings"] is True  # untouched default

    def test_load_config_unknown_key_warns(self, tmp_path, capsys):
        """Unknown keys produce warning print."""
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["training"]["unknown_param"] = 999
        path = tmp_path / "experiment_config.json"
        with open(path, "w") as f:
            json.dump(cfg, f)
        load_config(config_path=path)
        captured = capsys.readouterr()
        assert "WARNING: unknown key 'training.unknown_param'" in captured.out

    def test_load_config_prints_warning_when_missing(self, tmp_path, monkeypatch, capsys):
        """Prints info message when config file absent."""
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        load_config()
        captured = capsys.readouterr()
        assert "No config file found" in captured.out


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_validate_config_valid_defaults(self, default_config):
        """DEFAULT_CONFIG passes validation without error."""
        validate_config(default_config)

    def test_validate_config_invalid_estimators(self, default_config):
        """n_estimators <= 0 raises ValueError."""
        default_config["training"]["n_estimators"] = 0
        with pytest.raises(ValueError, match="n_estimators"):
            validate_config(default_config)

    def test_validate_config_estimators_too_large(self, default_config):
        """n_estimators > 1000 raises ValueError."""
        default_config["training"]["n_estimators"] = 1001
        with pytest.raises(ValueError, match="n_estimators"):
            validate_config(default_config)

    def test_validate_config_invalid_max_depth(self, default_config):
        """Non-int/non-None max_depth raises ValueError."""
        default_config["training"]["max_depth"] = "deep"
        with pytest.raises(ValueError, match="max_depth"):
            validate_config(default_config)

    def test_validate_config_max_depth_null(self, default_config):
        """max_depth=None (JSON null) is accepted."""
        default_config["training"]["max_depth"] = None
        validate_config(default_config)  # should not raise

    def test_validate_config_even_mode_filter_size(self, default_config):
        """Even filter size raises ValueError."""
        default_config["post_processing"]["mode_filter_size"] = 4
        with pytest.raises(ValueError, match="mode_filter_size"):
            validate_config(default_config)

    def test_validate_config_mode_filter_size_one(self, default_config):
        """mode_filter_size=1 raises ValueError."""
        default_config["post_processing"]["mode_filter_size"] = 1
        with pytest.raises(ValueError, match="mode_filter_size"):
            validate_config(default_config)

    def test_validate_config_boundary_buffer_zero_with_exclude(self, default_config):
        """exclude_boundary=True + buffer=0 raises ValueError."""
        default_config["training"]["exclude_boundary_pixels"] = True
        default_config["training"]["boundary_buffer_px"] = 0
        with pytest.raises(ValueError, match="boundary_buffer_px"):
            validate_config(default_config)

    def test_validate_config_no_features_enabled(self, default_config):
        """All features disabled raises ValueError."""
        default_config["features"]["use_embeddings"] = False
        default_config["features"]["add_ndvi"] = False
        default_config["features"]["add_ndwi"] = False
        with pytest.raises(ValueError, match="At least one"):
            validate_config(default_config)

    def test_validate_config_class_weight_none_string(self, default_config):
        """'none' accepted, maps to None."""
        default_config["training"]["class_weight"] = "none"
        validate_config(default_config)
        assert default_config["training"]["class_weight"] is None

    def test_validate_config_class_weight_null(self, default_config):
        """JSON null accepted."""
        default_config["training"]["class_weight"] = None
        validate_config(default_config)  # should not raise

    def test_validate_config_negative_random_state(self, default_config):
        """Negative random_state raises ValueError."""
        default_config["training"]["random_state"] = -1
        with pytest.raises(ValueError, match="random_state"):
            validate_config(default_config)

    def test_validate_config_spatial_context_enabled(self, default_config):
        """add_spatial_context=True passes validation with valid size and stats."""
        default_config["features"]["add_spatial_context"] = True
        default_config["features"]["spatial_context_size"] = 3
        default_config["features"]["spatial_context_stats"] = ["mean"]
        validate_config(default_config)  # should not raise

    def test_validate_config_spatial_context_size_even(self, default_config):
        """Even spatial_context_size raises ValueError."""
        default_config["features"]["spatial_context_size"] = 4
        with pytest.raises(ValueError, match="spatial_context_size"):
            validate_config(default_config)

    def test_validate_config_spatial_context_size_too_small(self, default_config):
        """spatial_context_size < 3 raises ValueError."""
        default_config["features"]["spatial_context_size"] = 1
        with pytest.raises(ValueError, match="spatial_context_size"):
            validate_config(default_config)

    def test_validate_config_spatial_context_size_valid(self, default_config):
        """Odd spatial_context_size >= 3 passes validation."""
        default_config["features"]["spatial_context_size"] = 5
        validate_config(default_config)  # should not raise

    def test_validate_config_spatial_context_size_too_large(self, default_config):
        """spatial_context_size > 99 raises ValueError."""
        default_config["features"]["spatial_context_size"] = 101
        with pytest.raises(ValueError, match="spatial_context_size"):
            validate_config(default_config)

    def test_validate_config_spatial_context_stats_empty(self, default_config):
        """Empty spatial_context_stats raises ValueError."""
        default_config["features"]["spatial_context_stats"] = []
        with pytest.raises(ValueError, match="spatial_context_stats"):
            validate_config(default_config)

    def test_validate_config_spatial_context_stats_invalid(self, default_config):
        """Invalid stat name raises ValueError."""
        default_config["features"]["spatial_context_stats"] = ["mean", "median"]
        with pytest.raises(ValueError, match="spatial_context_stats"):
            validate_config(default_config)

    def test_validate_config_spatial_context_stats_duplicates(self, default_config):
        """Duplicate stats raise ValueError."""
        default_config["features"]["spatial_context_stats"] = ["mean", "mean"]
        with pytest.raises(ValueError, match="duplicates"):
            validate_config(default_config)

    def test_validate_config_spatial_context_all_stats(self, default_config):
        """All four valid stats pass validation."""
        default_config["features"]["spatial_context_stats"] = ["mean", "std", "max", "min"]
        validate_config(default_config)  # should not raise


# ---------------------------------------------------------------------------
# Classifier Selection Validation Tests
# ---------------------------------------------------------------------------


class TestValidateClassifierSelection:
    """Tests for multi-classifier validation in validate_config."""

    @pytest.mark.parametrize("classifier", [
        "RandomForest", "GradientBoosting", "SVM",
        "LogisticRegression", "KNN", "MLP",
    ])
    def test_all_valid_classifiers_accepted(self, default_config, classifier):
        """All 6 classifier types pass validation."""
        default_config["training"]["classifier"] = classifier
        validate_config(default_config)  # should not raise

    def test_invalid_classifier_rejected(self, default_config):
        """Unknown classifier name raises ValueError."""
        default_config["training"]["classifier"] = "XGBoost"
        with pytest.raises(ValueError, match="training.classifier"):
            validate_config(default_config)

    def test_gb_learning_rate_validated(self, default_config):
        """GradientBoosting with invalid learning_rate raises ValueError."""
        default_config["training"]["classifier"] = "GradientBoosting"
        default_config["training"]["learning_rate"] = 0
        with pytest.raises(ValueError, match="learning_rate"):
            validate_config(default_config)

    def test_svm_kernel_validated(self, default_config):
        """SVM with invalid kernel raises ValueError."""
        default_config["training"]["classifier"] = "SVM"
        default_config["training"]["kernel"] = "quantum"
        with pytest.raises(ValueError, match="kernel"):
            validate_config(default_config)

    def test_svm_gamma_validated(self, default_config):
        """SVM with invalid gamma raises ValueError."""
        default_config["training"]["classifier"] = "SVM"
        default_config["training"]["gamma"] = "invalid"
        with pytest.raises(ValueError, match="gamma"):
            validate_config(default_config)

    def test_svm_gamma_float_accepted(self, default_config):
        """SVM with float gamma > 0 is accepted."""
        default_config["training"]["classifier"] = "SVM"
        default_config["training"]["gamma"] = 0.5
        validate_config(default_config)  # should not raise

    def test_lr_c_validated(self, default_config):
        """LogisticRegression with C <= 0 raises ValueError."""
        default_config["training"]["classifier"] = "LogisticRegression"
        default_config["training"]["C"] = -1.0
        with pytest.raises(ValueError, match="training.C"):
            validate_config(default_config)

    def test_knn_n_neighbors_validated(self, default_config):
        """KNN with invalid n_neighbors raises ValueError."""
        default_config["training"]["classifier"] = "KNN"
        default_config["training"]["n_neighbors"] = 0
        with pytest.raises(ValueError, match="n_neighbors"):
            validate_config(default_config)

    def test_knn_weights_validated(self, default_config):
        """KNN with invalid weights raises ValueError."""
        default_config["training"]["classifier"] = "KNN"
        default_config["training"]["weights"] = "cosine"
        with pytest.raises(ValueError, match="weights"):
            validate_config(default_config)

    def test_mlp_hidden_layer_sizes_validated(self, default_config):
        """MLP with empty hidden_layer_sizes raises ValueError."""
        default_config["training"]["classifier"] = "MLP"
        default_config["training"]["hidden_layer_sizes"] = []
        with pytest.raises(ValueError, match="hidden_layer_sizes"):
            validate_config(default_config)

    def test_mlp_alpha_validated(self, default_config):
        """MLP with negative alpha raises ValueError."""
        default_config["training"]["classifier"] = "MLP"
        default_config["training"]["alpha"] = -0.01
        with pytest.raises(ValueError, match="alpha"):
            validate_config(default_config)

    def test_max_iter_validated_for_mlp(self, default_config):
        """MLP with max_iter out of range raises ValueError."""
        default_config["training"]["classifier"] = "MLP"
        default_config["training"]["max_iter"] = 50
        with pytest.raises(ValueError, match="max_iter"):
            validate_config(default_config)

    def test_class_weight_only_validated_for_supported_classifiers(self, default_config):
        """class_weight is NOT validated for GradientBoosting (silently ignored)."""
        default_config["training"]["classifier"] = "GradientBoosting"
        default_config["training"]["class_weight"] = "invalid_value"
        default_config["training"]["max_depth"] = 3
        # Should not raise -- class_weight is ignored for GB
        validate_config(default_config)

    def test_n_estimators_only_validated_for_tree_classifiers(self, default_config):
        """n_estimators is NOT validated for SVM."""
        default_config["training"]["classifier"] = "SVM"
        default_config["training"]["n_estimators"] = 99999  # invalid range
        # Should not raise -- n_estimators is ignored for SVM
        validate_config(default_config)


# ---------------------------------------------------------------------------
# Preprocessing Validation Tests
# ---------------------------------------------------------------------------


class TestValidatePreprocessing:
    """Tests for preprocessing parameter validation."""

    def test_pca_zero_accepted(self, default_config):
        """pca_components=0 (disabled) is accepted."""
        default_config["training"]["pca_components"] = 0
        validate_config(default_config)  # should not raise

    def test_pca_valid_range_accepted(self, default_config):
        """pca_components in 2-500 accepted."""
        default_config["training"]["pca_components"] = 100
        validate_config(default_config)  # should not raise

    def test_pca_one_rejected(self, default_config):
        """pca_components=1 rejected (below minimum 2)."""
        default_config["training"]["pca_components"] = 1
        with pytest.raises(ValueError, match="pca_components"):
            validate_config(default_config)

    def test_pca_too_large_rejected(self, default_config):
        """pca_components > 500 rejected."""
        default_config["training"]["pca_components"] = 501
        with pytest.raises(ValueError, match="pca_components"):
            validate_config(default_config)

    def test_pca_negative_rejected(self, default_config):
        """Negative pca_components rejected."""
        default_config["training"]["pca_components"] = -1
        with pytest.raises(ValueError, match="pca_components"):
            validate_config(default_config)

    def test_scale_features_bool_required(self, default_config):
        """Non-bool scale_features rejected."""
        default_config["training"]["scale_features"] = 1
        with pytest.raises(ValueError, match="scale_features"):
            validate_config(default_config)

    def test_l2_normalize_bool_required(self, default_config):
        """Non-bool l2_normalize rejected."""
        default_config["training"]["l2_normalize"] = "yes"
        with pytest.raises(ValueError, match="l2_normalize"):
            validate_config(default_config)


# ---------------------------------------------------------------------------
# Prerequisite Validation Tests
# ---------------------------------------------------------------------------


class TestValidatePrerequisites:
    def test_validate_prerequisites_ndvi_missing_band(self, default_config, tmp_path, monkeypatch):
        """FileNotFoundError when B04.tif missing for NDVI."""
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "SENTINEL2_DIR", tmp_path)
        default_config["features"]["add_ndvi"] = True
        with pytest.raises(FileNotFoundError, match="B04"):
            validate_prerequisites(default_config, years=["2021"])

    def test_validate_prerequisites_ndwi_missing_band(self, default_config, tmp_path, monkeypatch):
        """FileNotFoundError when B03.tif missing for NDWI."""
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "SENTINEL2_DIR", tmp_path)
        default_config["features"]["add_ndwi"] = True
        # Create B08 but not B03
        (tmp_path / "2021").mkdir()
        (tmp_path / "2021" / "B08.tif").touch()
        with pytest.raises(FileNotFoundError, match="B03"):
            validate_prerequisites(default_config, years=["2021"])

    def test_validate_prerequisites_all_present(self, default_config, tmp_path, monkeypatch):
        """No error when all required files exist."""
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "SENTINEL2_DIR", tmp_path)
        default_config["features"]["add_ndvi"] = True
        default_config["features"]["add_ndwi"] = True
        for year in ["2021", "2023"]:
            (tmp_path / year).mkdir(exist_ok=True)
            for band in ["B03", "B04", "B08"]:
                (tmp_path / year / f"{band}.tif").touch()
        validate_prerequisites(default_config, years=["2021", "2023"])

    def test_validate_prerequisites_no_indices(self, default_config):
        """No file checks when add_ndvi=False, add_ndwi=False."""
        default_config["features"]["add_ndvi"] = False
        default_config["features"]["add_ndwi"] = False
        # Should not raise even if SENTINEL2_DIR doesn't exist
        validate_prerequisites(default_config, years=["2021"])


# ---------------------------------------------------------------------------
# Experiment Ledger Tests
# ---------------------------------------------------------------------------


class TestExperimentLedger:
    @pytest.fixture(autouse=True)
    def _setup_experiments_dir(self, tmp_path, monkeypatch):
        """Point EXPERIMENTS_DIR to a session dir via symlink for all ledger tests."""
        import src.config as cfg_mod

        base_dir = tmp_path / "experiments"
        base_dir.mkdir()
        session_dir = base_dir / "session_test"
        session_dir.mkdir()
        latest_link = base_dir / "latest"
        latest_link.symlink_to(session_dir.name)

        self.experiments_dir = latest_link
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_BASE_DIR", base_dir)
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", latest_link)

    def _sample_metrics(self, overall_acc=0.85):
        return {
            "overall_accuracy": overall_acc,
            "evaluation_year": "2021",
            "per_class": {
                "Built-up": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 100},
                "Cropland": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 200},
            },
            "weighted_avg": {"precision": 0.85, "recall": 0.8, "f1": 0.82},
            "confusion_matrix": [[70, 30], [15, 170]],
            "confusion_matrix_axes": {"rows": "true_class", "columns": "predicted_class"},
            "class_names": ["Built-up", "Cropland"],
        }

    def test_get_next_iteration_empty(self):
        """Returns 1 when no iterations exist."""
        assert get_next_iteration_number() == 1

    def test_get_next_iteration_sequential(self):
        """Returns N+1 after creating N iterations."""
        (self.experiments_dir / "iteration_001").mkdir()
        (self.experiments_dir / "iteration_002").mkdir()
        assert get_next_iteration_number() == 3

    def test_get_next_iteration_skips_corrupted(self):
        """Non-matching dirs ignored."""
        (self.experiments_dir / "iteration_001").mkdir()
        (self.experiments_dir / "not_an_iteration").mkdir()
        assert get_next_iteration_number() == 2

    def test_save_experiment_creates_all_files(self, default_config):
        """config.json, metrics.json, metadata.json all created."""
        metrics = self._sample_metrics()
        iteration_dir, num = save_experiment(default_config, metrics)
        assert (iteration_dir / "config.json").exists()
        assert (iteration_dir / "metrics.json").exists()
        assert (iteration_dir / "metadata.json").exists()
        assert num == 1

    def test_save_experiment_collision_raises(self, default_config, monkeypatch):
        """Existing dir raises RuntimeError."""
        metrics = self._sample_metrics()
        # Pre-create the directory that get_next_iteration_number will try to use
        (self.experiments_dir / "iteration_001").mkdir()
        # Monkeypatch get_next_iteration_number to always return 1 (collision)
        monkeypatch.setattr(
            "src.experiment.get_next_iteration_number", lambda: 1
        )
        with pytest.raises(RuntimeError, match="already exists"):
            save_experiment(default_config, metrics)

    def test_save_experiment_default_pending_status(self, default_config):
        """metadata.json has status 'pending'."""
        metrics = self._sample_metrics()
        iteration_dir, _ = save_experiment(default_config, metrics)
        with open(iteration_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["status"] == "pending"

    def test_save_experiment_timestamp_is_iso8601(self, default_config):
        """Timestamp parses as valid ISO 8601."""
        metrics = self._sample_metrics()
        iteration_dir, _ = save_experiment(default_config, metrics)
        with open(iteration_dir / "metadata.json") as f:
            metadata = json.load(f)
        # Should not raise
        datetime.fromisoformat(metadata["timestamp"])

    def test_load_experiment_roundtrip(self, default_config):
        """save then load produces same data."""
        metrics = self._sample_metrics()
        _, num = save_experiment(default_config, metrics)
        loaded = load_experiment(num)
        assert loaded["config"] == default_config
        assert loaded["metrics"]["overall_accuracy"] == metrics["overall_accuracy"]
        assert loaded["metadata"]["iteration"] == num

    def test_load_experiment_missing_raises(self):
        """FileNotFoundError for non-existent iteration."""
        with pytest.raises(FileNotFoundError):
            load_experiment(999)

    def test_compare_experiments_metric_deltas(self, default_config):
        """Correct delta computation between two experiments."""
        metrics_a = self._sample_metrics(overall_acc=0.80)
        metrics_b = self._sample_metrics(overall_acc=0.85)
        save_experiment(default_config, metrics_a)
        save_experiment(default_config, metrics_b)
        result = compare_experiments(1, 2)
        assert result["metrics_diff"]["overall_accuracy"]["delta"] == pytest.approx(0.05, abs=0.001)

    def test_compare_experiments_config_diff(self, default_config):
        """Identifies changed config fields with dot-notation."""
        cfg_a = copy.deepcopy(default_config)
        cfg_b = copy.deepcopy(default_config)
        cfg_b["training"]["n_estimators"] = 200

        metrics = self._sample_metrics()
        save_experiment(cfg_a, metrics)
        save_experiment(cfg_b, metrics)
        result = compare_experiments(1, 2)
        assert "training.n_estimators" in result["config_diff"]
        assert result["config_diff"]["training.n_estimators"]["a"] == 100
        assert result["config_diff"]["training.n_estimators"]["b"] == 200

    def test_update_status_valid(self, default_config):
        """accepted/reverted/pending transitions work."""
        metrics = self._sample_metrics()
        _, num = save_experiment(default_config, metrics)

        update_experiment_status(num, "accepted")
        loaded = load_experiment(num)
        assert loaded["metadata"]["status"] == "accepted"

        update_experiment_status(num, "reverted")
        loaded = load_experiment(num)
        assert loaded["metadata"]["status"] == "reverted"

    def test_update_status_invalid(self, default_config):
        """Invalid status raises ValueError."""
        metrics = self._sample_metrics()
        _, num = save_experiment(default_config, metrics)
        with pytest.raises(ValueError, match="Status must be one of"):
            update_experiment_status(num, "invalid_status")

    def test_list_iterations_sorted(self, default_config):
        """Iterations returned in order."""
        metrics = self._sample_metrics()
        save_experiment(default_config, metrics)
        save_experiment(default_config, metrics)
        save_experiment(default_config, metrics)
        iterations = list_iterations()
        assert len(iterations) == 3
        assert [i["iteration"] for i in iterations] == [1, 2, 3]

    def test_save_config_roundtrip(self, tmp_path, default_config):
        """save_config then load_config returns same values."""
        path = tmp_path / "experiment_config.json"
        save_config(default_config, config_path=path)
        loaded = load_config(config_path=path)
        assert loaded == default_config


# ---------------------------------------------------------------------------
# Session Management Tests
# ---------------------------------------------------------------------------


class TestSessionManagement:
    @pytest.fixture(autouse=True)
    def _setup_base_dir(self, tmp_path, monkeypatch):
        """Point EXPERIMENTS_BASE_DIR and EXPERIMENTS_DIR to tmp_path."""
        import src.config as cfg_mod

        self.base_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_BASE_DIR", self.base_dir)
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", self.base_dir / "latest")

    def test_create_session_creates_dir_and_symlink(self, default_config):
        """create_session creates session dir and latest/ symlink."""
        session_dir = create_session(default_config)
        assert session_dir.exists()
        assert session_dir.name.startswith("session_")
        latest = self.base_dir / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == session_dir.resolve()

    def test_create_session_writes_meta(self, default_config):
        """session_meta.json created with correct fields."""
        session_dir = create_session(default_config)
        meta_path = session_dir / "session_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["session_id"] == session_dir.name
        assert meta["start_time"] is not None
        assert meta["end_time"] is None
        assert meta["stop_reason"] is None
        assert meta["parameter_count"] == len(_flatten_dict(default_config))
        assert meta["n_iterations"] == 0

    def test_create_session_preserves_previous(self, default_config):
        """Multiple sessions preserved; latest symlink updated."""
        import time
        session_1 = create_session(default_config)
        time.sleep(1.1)  # Ensure different timestamp
        session_2 = create_session(default_config)
        assert session_1.exists()
        assert session_2.exists()
        latest = self.base_dir / "latest"
        assert latest.resolve() == session_2.resolve()

    def test_update_latest_symlink_atomic(self, default_config):
        """Symlink update is relative and atomic."""
        session_dir = create_session(default_config)
        # Verify symlink is relative
        latest = self.base_dir / "latest"
        target = Path(str(latest.parent / latest.readlink()))
        assert target.name == session_dir.name

    def test_list_sessions_returns_sorted(self, default_config):
        """list_sessions returns sessions sorted by start_time descending."""
        import time
        create_session(default_config)
        time.sleep(1.1)
        create_session(default_config)
        sessions = list_sessions()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0]["start_time"] > sessions[1]["start_time"]

    def test_list_sessions_skips_symlink(self, default_config):
        """list_sessions does not include the latest/ symlink."""
        create_session(default_config)
        sessions = list_sessions()
        session_ids = [s["session_id"] for s in sessions]
        assert "latest" not in session_ids

    def test_update_session_meta(self, default_config):
        """update_session_meta merges updates into existing meta."""
        session_dir = create_session(default_config)
        update_session_meta(session_dir, {
            "end_time": "2026-04-04T12:00:00+00:00",
            "stop_reason": "max_iterations",
            "n_iterations": 5,
        })
        with open(session_dir / "session_meta.json") as f:
            meta = json.load(f)
        assert meta["end_time"] == "2026-04-04T12:00:00+00:00"
        assert meta["stop_reason"] == "max_iterations"
        assert meta["n_iterations"] == 5
        # Original fields preserved
        assert meta["session_id"] == session_dir.name

    def test_update_session_meta_missing_raises(self, tmp_path):
        """FileNotFoundError when session_meta.json missing."""
        with pytest.raises(FileNotFoundError):
            update_session_meta(tmp_path, {"end_time": "now"})

    def test_get_session_dir_none_returns_latest(self):
        """get_session_dir(None) returns config.EXPERIMENTS_DIR."""
        import src.config as cfg_mod
        result = get_session_dir(None)
        assert result == cfg_mod.EXPERIMENTS_DIR

    def test_get_session_dir_by_id(self, default_config):
        """get_session_dir(id) returns correct path."""
        session_dir = create_session(default_config)
        result = get_session_dir(session_dir.name)
        assert result == session_dir

    def test_get_session_dir_missing_raises(self):
        """FileNotFoundError for non-existent session."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            get_session_dir("session_nonexistent")

    def test_save_experiment_guard_no_symlink(self, default_config):
        """save_experiment raises when latest/ symlink doesn't exist."""
        # Don't create any session - latest/ doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"overall_accuracy": 0.85}
        with pytest.raises(RuntimeError, match="does not exist"):
            save_experiment(default_config, metrics)

    def test_load_experiment_with_session_dir(self, default_config):
        """load_experiment with session_dir loads from correct directory."""
        session_dir = create_session(default_config)
        # Create an iteration inside the session
        iter_dir = session_dir / "iteration_001"
        iter_dir.mkdir()
        for fname, data in [
            ("config.json", default_config),
            ("metrics.json", {"overall_accuracy": 0.85}),
            ("metadata.json", {"iteration": 1, "timestamp": "2026-04-04T12:00:00Z", "status": "accepted"}),
        ]:
            with open(iter_dir / fname, "w") as f:
                json.dump(data, f)

        # Load via explicit session_dir
        result = load_experiment(1, session_dir=session_dir)
        assert result["metrics"]["overall_accuracy"] == 0.85

    def test_compare_experiments_with_session_dir(self, default_config):
        """compare_experiments with session_dir works for non-latest sessions."""
        session_dir = create_session(default_config)
        cfg_a = copy.deepcopy(default_config)
        cfg_b = copy.deepcopy(default_config)
        cfg_b["training"]["n_estimators"] = 200

        for num, cfg in [(1, cfg_a), (2, cfg_b)]:
            iter_dir = session_dir / f"iteration_{num:03d}"
            iter_dir.mkdir()
            for fname, data in [
                ("config.json", cfg),
                ("metrics.json", {"overall_accuracy": 0.80 + num * 0.01}),
                ("metadata.json", {"iteration": num, "timestamp": "2026-04-04T12:00:00Z", "status": "accepted"}),
            ]:
                with open(iter_dir / fname, "w") as f:
                    json.dump(data, f)

        result = compare_experiments(1, 2, session_dir=session_dir)
        assert "training.n_estimators" in result["config_diff"]


class TestLegacyMigration:
    @pytest.fixture(autouse=True)
    def _setup_base_dir(self, tmp_path, monkeypatch):
        """Point config to tmp_path."""
        import src.config as cfg_mod

        self.base_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_BASE_DIR", self.base_dir)
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", self.base_dir / "latest")

    def test_migrates_flat_iterations(self):
        """Flat iteration_NNN/ dirs moved to session_legacy/."""
        self.base_dir.mkdir()
        # Create flat iteration dirs
        for i in range(1, 4):
            iter_dir = self.base_dir / f"iteration_{i:03d}"
            iter_dir.mkdir()
            with open(iter_dir / "metadata.json", "w") as f:
                json.dump({"iteration": i, "timestamp": f"2026-04-0{i}T12:00:00Z", "status": "accepted"}, f)
            with open(iter_dir / "metrics.json", "w") as f:
                json.dump({"overall_accuracy": 0.7 + i * 0.01}, f)

        # Create SUMMARY.md
        (self.base_dir / "SUMMARY.md").write_text("# Summary")

        _migrate_legacy_experiments(self.base_dir)

        legacy = self.base_dir / "session_legacy"
        assert legacy.exists()
        assert (legacy / "iteration_001").exists()
        assert (legacy / "iteration_002").exists()
        assert (legacy / "iteration_003").exists()
        assert (legacy / "SUMMARY.md").exists()
        assert not (self.base_dir / "iteration_001").exists()

    def test_creates_session_meta_for_legacy(self):
        """session_meta.json created with best iteration data."""
        self.base_dir.mkdir()
        for i in range(1, 3):
            iter_dir = self.base_dir / f"iteration_{i:03d}"
            iter_dir.mkdir()
            with open(iter_dir / "metadata.json", "w") as f:
                json.dump({"iteration": i, "timestamp": f"2026-04-0{i}T12:00:00Z", "status": "accepted"}, f)
            with open(iter_dir / "metrics.json", "w") as f:
                json.dump({"overall_accuracy": 0.7 + i * 0.05}, f)

        _migrate_legacy_experiments(self.base_dir)

        meta_path = self.base_dir / "session_legacy" / "session_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["session_id"] == "session_legacy"
        assert meta["n_iterations"] == 2
        assert meta["best_iteration"] == 2
        assert meta["stop_reason"] == "migrated_from_flat"

    def test_migration_idempotent(self):
        """Second migration call is a no-op."""
        self.base_dir.mkdir()
        (self.base_dir / "iteration_001").mkdir()
        with open(self.base_dir / "iteration_001" / "metadata.json", "w") as f:
            json.dump({"iteration": 1, "timestamp": "2026-04-01T12:00:00Z"}, f)

        _migrate_legacy_experiments(self.base_dir)
        # session_legacy now exists; second call should be no-op
        _migrate_legacy_experiments(self.base_dir)
        assert (self.base_dir / "session_legacy").exists()

    def test_no_migration_when_no_flat_iters(self):
        """No migration when no iteration_* dirs exist."""
        self.base_dir.mkdir()
        _migrate_legacy_experiments(self.base_dir)
        assert not (self.base_dir / "session_legacy").exists()

    def test_legacy_visible_in_list_sessions(self):
        """session_legacy appears in list_sessions() after migration."""
        self.base_dir.mkdir()
        iter_dir = self.base_dir / "iteration_001"
        iter_dir.mkdir()
        with open(iter_dir / "metadata.json", "w") as f:
            json.dump({"iteration": 1, "timestamp": "2026-04-01T12:00:00Z"}, f)
        with open(iter_dir / "metrics.json", "w") as f:
            json.dump({"overall_accuracy": 0.75}, f)

        sessions = list_sessions()  # Triggers migration
        session_ids = [s["session_id"] for s in sessions]
        assert "session_legacy" in session_ids
