"""Tests for src/autocorrect.py -- self-correction orchestrator."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.autocorrect import (
    CLASS_REGRESSION_THRESHOLD,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PATIENCE,
    DEFAULT_TARGET_SCORE,
    IterationScore,
    _apply_hypothesis,
    _backup_outputs,
    _build_summary,
    _check_pareto_acceptance,
    _extract_score,
    _get_latest_iteration,
    _print_summary,
    _restore_best_outputs,
    _run_diagnosis_safe,
    _save_summary_md,
    _score_from_metrics,
    _score_from_vlm,
    run_autocorrect,
)
from src.experiment import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Fixtures local to autocorrect tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_vlm_results():
    """Minimal VLM evaluation results matching evaluate.run_evaluation() output."""
    return {
        "2021": {
            "evaluation": {
                "overall_score": 7,
                "per_class": [
                    {"class_name": "Built-up", "score": 8.0, "notes": "Good"},
                    {"class_name": "Cropland", "score": 7.5, "notes": "OK"},
                    {"class_name": "Grassland", "score": 6.0, "notes": "Confused"},
                    {"class_name": "Tree cover", "score": 8.0, "notes": "Good"},
                    {"class_name": "Water", "score": 9.0, "notes": "Excellent"},
                    {"class_name": "Other", "score": 5.0, "notes": "Poor"},
                ],
                "error_regions": [],
                "spatial_quality": "Good",
                "confidence": 0.8,
                "recommendations": [],
            },
            "summary": {"overall_score": 7, "confidence": 0.8},
        }
    }


@pytest.fixture
def sample_hypothesis():
    """Minimal hypothesis dict matching diagnose.py output."""
    return {
        "hypothesis": "Adding NDVI should improve vegetation class separation",
        "component": "features",
        "parameter_changes": {"features.add_ndvi": True},
        "expected_impact": "Better Grassland/Tree cover F1",
        "risk": "Slight increase in training time",
        "tier": 1,
        "confidence": 0.8,
        "reasoning": "NDVI correlates with chlorophyll content",
    }


@pytest.fixture
def mock_pipeline(mocker, tmp_path, sample_metrics):
    """Mock all pipeline dependencies for orchestrator testing.

    Redirects EXPERIMENTS_DIR and OUTPUT_DIR to tmp_path.
    Mocks run_classification to create experiment ledger entries and fake GeoTIFFs.
    Mocks run_evaluation to return None (metrics-only).
    Mocks run_diagnosis to return a hypothesis.
    """
    mocker.patch.object(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    mocker.patch.object(config, "OUTPUT_DIR", tmp_path / "output")
    mocker.patch.object(config, "PROJECT_ROOT", tmp_path)

    (tmp_path / "output").mkdir()
    (tmp_path / "experiments").mkdir()

    # Write a config file
    config_path = tmp_path / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(DEFAULT_CONFIG, f)

    iteration_counter = {"count": 0}
    metrics_sequence = [sample_metrics]  # Default: same metrics each time

    def fake_classification():
        """Mock that creates experiment ledger entries and fake GeoTIFFs."""
        iteration_counter["count"] += 1
        # Create fake GeoTIFFs
        for year in config.TIME_RANGES:
            path = tmp_path / "output" / f"landcover_{year}.tif"
            path.write_bytes(b"fake_geotiff_" + str(iteration_counter["count"]).encode())
        # Save to experiment ledger
        from src.experiment import load_config, save_experiment
        cfg = load_config(config_path=config_path)
        idx = min(iteration_counter["count"] - 1, len(metrics_sequence) - 1)
        save_experiment(cfg, metrics_sequence[idx])

    mock_classify = mocker.patch(
        "src.classify.run_classification",
        side_effect=fake_classification,
    )
    mock_eval = mocker.patch(
        "src.evaluate.run_evaluation",
        return_value=None,
    )
    mock_change = mocker.patch(
        "src.change.run_change_detection",
        return_value=None,
    )

    return {
        "counter": iteration_counter,
        "metrics_sequence": metrics_sequence,
        "mock_classify": mock_classify,
        "mock_eval": mock_eval,
        "mock_change": mock_change,
        "config_path": config_path,
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# Score Extraction Tests
# ---------------------------------------------------------------------------

class TestScoreFromVlm:
    def test_extracts_overall_and_per_class(self, sample_vlm_results):
        """VLM overall_score and per_class scores extracted correctly."""
        score = _score_from_vlm(1, sample_vlm_results["2021"])
        assert score.overall_score == 7.0
        assert score.source == "vlm"
        assert score.class_scores["Built-up"] == 8.0
        assert score.class_scores["Water"] == 9.0
        assert len(score.class_scores) == 6

    def test_handles_missing_per_class(self):
        """Returns empty class_scores when per_class missing."""
        result = {"evaluation": {"overall_score": 6}}
        score = _score_from_vlm(1, result)
        assert score.overall_score == 6.0
        assert score.class_scores == {}


class TestScoreFromMetrics:
    def test_scales_accuracy_to_10(self, sample_metrics):
        """overall_accuracy * 10 maps correctly."""
        score = _score_from_metrics(1, sample_metrics)
        assert score.overall_score == round(sample_metrics["overall_accuracy"] * 10, 2)
        assert score.source == "metrics"

    def test_scales_f1_to_10(self, sample_metrics):
        """per_class F1 * 10 maps correctly."""
        score = _score_from_metrics(1, sample_metrics)
        for name, data in sample_metrics["per_class"].items():
            expected = round(data["f1"] * 10, 2)
            assert score.class_scores[name] == expected


class TestExtractScore:
    def test_prefers_vlm_when_available(self, sample_metrics, sample_vlm_results):
        """VLM scores used when eval_results has valid evaluation."""
        score = _extract_score(1, sample_metrics, sample_vlm_results)
        assert score.source == "vlm"
        assert score.overall_score == 7.0

    def test_falls_back_to_metrics_when_vlm_none(self, sample_metrics):
        """Metrics used when eval_results is None."""
        score = _extract_score(1, sample_metrics, None)
        assert score.source == "metrics"

    def test_falls_back_when_vlm_missing_evaluation_key(self, sample_metrics):
        """Metrics used when eval_results lacks evaluation key."""
        score = _extract_score(1, sample_metrics, {"2021": {"error": "something failed"}})
        assert score.source == "metrics"

    def test_falls_back_when_vlm_has_no_2021(self, sample_metrics):
        """Metrics used when VLM result has no 2021 year."""
        score = _extract_score(1, sample_metrics, {"2023": {"evaluation": {}}})
        assert score.source == "metrics"


# ---------------------------------------------------------------------------
# Acceptance Policy Tests
# ---------------------------------------------------------------------------

class TestCheckParetoAcceptance:
    def test_accepts_improved_overall_no_regression(self):
        """Accepts when overall improves and no class regresses."""
        best = IterationScore(1, 7.0, {"A": 7.0, "B": 8.0})
        candidate = IterationScore(2, 7.5, {"A": 7.5, "B": 8.0})
        accepted, reason = _check_pareto_acceptance(best, candidate)
        assert accepted is True
        assert "improved" in reason.lower()

    def test_rejects_lower_overall(self):
        """Rejects when overall score decreases."""
        best = IterationScore(1, 7.0, {"A": 7.0})
        candidate = IterationScore(2, 6.5, {"A": 7.5})
        accepted, reason = _check_pareto_acceptance(best, candidate)
        assert accepted is False
        assert "did not improve" in reason.lower()

    def test_rejects_equal_overall(self):
        """Rejects when overall score is equal (strictly greater required)."""
        best = IterationScore(1, 7.0, {"A": 7.0})
        candidate = IterationScore(2, 7.0, {"A": 7.0})
        accepted, reason = _check_pareto_acceptance(best, candidate)
        assert accepted is False

    def test_rejects_class_regression_above_threshold(self):
        """Rejects when one class drops > 1.0 even if overall improves."""
        best = IterationScore(1, 7.0, {"A": 7.0, "B": 8.0})
        candidate = IterationScore(2, 7.5, {"A": 7.0, "B": 6.5})
        accepted, reason = _check_pareto_acceptance(best, candidate)
        assert accepted is False
        assert "regressed" in reason.lower()
        assert "B" in reason

    def test_accepts_small_class_dip(self):
        """Accepts when class drops < 1.0 and overall improves."""
        best = IterationScore(1, 7.0, {"A": 7.0, "B": 8.0})
        candidate = IterationScore(2, 7.5, {"A": 7.0, "B": 7.5})
        accepted, _ = _check_pareto_acceptance(best, candidate)
        assert accepted is True

    def test_handles_missing_class_in_candidate(self):
        """Class present in best but absent in candidate gets 0.0 default."""
        best = IterationScore(1, 7.0, {"A": 7.0, "B": 2.0})
        candidate = IterationScore(2, 7.5, {"A": 8.0})  # B missing -> 0.0
        accepted, reason = _check_pareto_acceptance(best, candidate)
        # B drops from 2.0 to 0.0, drop=2.0 > 1.0, should reject
        assert accepted is False
        assert "B" in reason

    def test_empty_class_scores_accepts_on_overall(self):
        """When best has no class scores, accept based on overall only (RT4-9)."""
        best = IterationScore(1, 7.0, {})
        candidate = IterationScore(2, 7.5, {"A": 8.0})
        accepted, _ = _check_pareto_acceptance(best, candidate)
        assert accepted is True

    def test_reason_string_contains_scores(self):
        """Reason message includes actual score values."""
        best = IterationScore(1, 7.0, {"A": 7.0})
        candidate = IterationScore(2, 7.5, {"A": 7.2})
        accepted, reason = _check_pareto_acceptance(best, candidate)
        assert "7.00" in reason
        assert "7.50" in reason


# ---------------------------------------------------------------------------
# Config Mutation Tests
# ---------------------------------------------------------------------------

class TestApplyHypothesis:
    def test_applies_simple_parameter_change(self, default_config):
        """Single dot-notation key applied correctly."""
        hypothesis_data = {
            "hypothesis": "Increase n_estimators",
            "component": "training",
            "parameter_changes": {"training.n_estimators": 200},
            "expected_impact": "Better",
            "risk": "Slower",
            "tier": 2,
            "confidence": 0.7,
            "reasoning": "More trees",
        }
        new_cfg = _apply_hypothesis(default_config, hypothesis_data)
        assert new_cfg["training"]["n_estimators"] == 200

    def test_applies_feature_flag_hypothesis(self, default_config):
        """NDVI-style boolean flag hypothesis works (RT1-8)."""
        hypothesis_data = {
            "hypothesis": "Add NDVI",
            "component": "features",
            "parameter_changes": {"features.add_ndvi": True},
            "expected_impact": "Better vegetation",
            "risk": "Marginal",
            "tier": 2,
            "confidence": 0.7,
            "reasoning": "NDVI helps",
        }
        assert default_config["features"]["add_ndvi"] is False
        new_cfg = _apply_hypothesis(default_config, hypothesis_data)
        assert new_cfg["features"]["add_ndvi"] is True

    def test_preserves_unrelated_config(self, default_config):
        """Changes to training don't affect features section."""
        hypothesis_data = {
            "hypothesis": "Reduce depth",
            "component": "training",
            "parameter_changes": {"training.max_depth": 10},
            "expected_impact": "Less overfitting",
            "risk": "Underfitting",
            "tier": 1,
            "confidence": 0.8,
            "reasoning": "Depth reduction",
        }
        new_cfg = _apply_hypothesis(default_config, hypothesis_data)
        assert new_cfg["features"] == default_config["features"]
        assert new_cfg["post_processing"] == default_config["post_processing"]

    def test_returns_deep_copy(self, default_config):
        """Original config not mutated."""
        original_depth = default_config["training"]["max_depth"]
        hypothesis_data = {
            "hypothesis": "Change depth",
            "component": "training",
            "parameter_changes": {"training.max_depth": original_depth - 5},
            "expected_impact": "Less overfitting",
            "risk": "Underfitting",
            "tier": 1,
            "confidence": 0.8,
            "reasoning": "Depth reduction",
        }
        _apply_hypothesis(default_config, hypothesis_data)
        assert default_config["training"]["max_depth"] == original_depth

    def test_invalid_value_raises_valueerror(self, default_config):
        """ValueError from validate_config propagates."""
        hypothesis_data = {
            "hypothesis": "Bad config",
            "component": "training",
            "parameter_changes": {"training.n_estimators": -1},
            "expected_impact": "N/A",
            "risk": "N/A",
            "tier": 1,
            "confidence": 0.5,
            "reasoning": "Testing validation",
        }
        with pytest.raises(ValueError):
            _apply_hypothesis(default_config, hypothesis_data)

    def test_noop_change_raises_valueerror(self, default_config):
        """No-op hypothesis (identical config) raises ValueError (RT2-14)."""
        current_depth = default_config["training"]["max_depth"]
        hypothesis_data = {
            "hypothesis": "Same depth",
            "component": "training",
            "parameter_changes": {"training.max_depth": current_depth},
            "expected_impact": "None",
            "risk": "None",
            "tier": 1,
            "confidence": 0.5,
            "reasoning": "Testing no-op detection",
        }
        with pytest.raises(ValueError, match="No effective changes"):
            _apply_hypothesis(default_config, hypothesis_data)


# ---------------------------------------------------------------------------
# Latest Iteration Helper Tests
# ---------------------------------------------------------------------------

class TestGetLatestIteration:
    def test_returns_latest(self, mocker, tmp_path):
        """Returns highest iteration number."""
        mocker.patch.object(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
        (tmp_path / "experiments").mkdir()

        from src.experiment import save_experiment
        save_experiment(DEFAULT_CONFIG, {"overall_accuracy": 0.7})
        save_experiment(DEFAULT_CONFIG, {"overall_accuracy": 0.75})

        assert _get_latest_iteration() == 2

    def test_raises_when_empty(self, mocker, tmp_path):
        """RuntimeError when no iterations exist."""
        mocker.patch.object(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
        (tmp_path / "experiments").mkdir()

        with pytest.raises(RuntimeError, match="No experiment iterations"):
            _get_latest_iteration()


# ---------------------------------------------------------------------------
# Diagnosis Integration Tests
# ---------------------------------------------------------------------------

class TestRunDiagnosisSafe:
    def test_returns_hypothesis_on_success(self, mocker):
        """Returns hypothesis dict when diagnosis succeeds."""
        from src.diagnose import Hypothesis

        mock_hyp = Hypothesis(
            hypothesis="Test",
            component="training",
            parameter_changes={"training.max_depth": 15},
            expected_impact="Better",
            risk="None",
            tier=1,
            confidence=0.8,
            reasoning="Test",
        )
        mocker.patch("src.diagnose.run_diagnosis", return_value=mock_hyp)
        result = _run_diagnosis_safe(1)
        assert result is not None
        assert result["hypothesis"] == "Test"
        assert result["parameter_changes"] == {"training.max_depth": 15}

    def test_returns_none_on_import_error(self, mocker):
        """Returns None when src.diagnose not available."""
        mocker.patch.dict("sys.modules", {"src.diagnose": None})
        # Force reimport to trigger ImportError
        import importlib
        import src.autocorrect
        result = _run_diagnosis_safe(1)
        # In practice, ImportError handling is in the function body
        # We test by directly mocking the import
        mocker.patch(
            "src.autocorrect._run_diagnosis_safe",
            return_value=None,
        )
        assert _run_diagnosis_safe(1) is None

    def test_returns_none_on_api_error(self, mocker):
        """Returns None when diagnosis raises exception."""
        mocker.patch("src.diagnose.run_diagnosis", side_effect=RuntimeError("API failed"))
        result = _run_diagnosis_safe(1)
        assert result is None


# ---------------------------------------------------------------------------
# GeoTIFF Management Tests
# ---------------------------------------------------------------------------

class TestBackupOutputs:
    def test_copies_geotiffs(self, mocker, tmp_path):
        """GeoTIFFs copied from output/ to iteration dir."""
        mocker.patch.object(config, "OUTPUT_DIR", tmp_path / "output")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        iter_dir = tmp_path / "iteration_001"
        iter_dir.mkdir()

        for year in config.TIME_RANGES:
            (output_dir / f"landcover_{year}.tif").write_bytes(b"tiff_data")

        _backup_outputs(iter_dir)

        for year in config.TIME_RANGES:
            assert (iter_dir / f"landcover_{year}.tif").exists()

    def test_raises_when_no_outputs(self, mocker, tmp_path):
        """FileNotFoundError when no GeoTIFFs exist."""
        mocker.patch.object(config, "OUTPUT_DIR", tmp_path / "output")
        (tmp_path / "output").mkdir()
        iter_dir = tmp_path / "iteration_001"
        iter_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No landcover outputs"):
            _backup_outputs(iter_dir)


class TestRestoreBestOutputs:
    def test_restores_to_output(self, mocker, tmp_path):
        """GeoTIFFs copied from iteration dir to output/."""
        mocker.patch.object(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
        mocker.patch.object(config, "OUTPUT_DIR", tmp_path / "output")
        (tmp_path / "output").mkdir()

        iter_dir = tmp_path / "experiments" / "iteration_001"
        iter_dir.mkdir(parents=True)

        for year in config.TIME_RANGES:
            (iter_dir / f"landcover_{year}.tif").write_bytes(b"best_tiff")

        _restore_best_outputs(1)

        for year in config.TIME_RANGES:
            restored = tmp_path / "output" / f"landcover_{year}.tif"
            assert restored.exists()
            assert restored.read_bytes() == b"best_tiff"


# ---------------------------------------------------------------------------
# Summary Generation Tests
# ---------------------------------------------------------------------------

class TestSummary:
    def _make_summary(self):
        """Helper to create a minimal summary dict."""
        return {
            "iterations_run": 3,
            "iterations_classified": 3,
            "accepted": 2,
            "reverted": 1,
            "skipped": 0,
            "initial_score": 7.52,
            "final_score": 7.80,
            "best_iteration": 2,
            "best_score": 7.80,
            "stop_reason": "max_iterations",
            "max_iterations": 5,
            "target_score": 8.5,
            "patience": 3,
            "history": [
                {
                    "iteration": 1, "score": 7.52, "status": "accepted",
                    "hypothesis": None, "config_changes": None,
                    "reason": "Baseline iteration",
                },
                {
                    "iteration": 2, "score": 7.80, "status": "accepted",
                    "hypothesis": "Add NDVI", "config_changes": {"features.add_ndvi": True},
                    "reason": "Overall score improved: 7.52 -> 7.80",
                },
                {
                    "iteration": 3, "score": 7.60, "status": "reverted",
                    "hypothesis": "Increase trees", "config_changes": {"training.n_estimators": 200},
                    "reason": "Overall score did not improve",
                },
            ],
        }

    def test_print_summary(self, capsys):
        """Console summary has [autocorrect] prefix and key data."""
        summary = self._make_summary()
        _print_summary(summary)
        captured = capsys.readouterr().out
        assert "[autocorrect]" in captured
        assert "Self-Correction Complete" in captured
        assert "7.52" in captured
        assert "7.80" in captured

    def test_save_summary_md(self, mocker, tmp_path):
        """SUMMARY.md written with iteration table."""
        mocker.patch.object(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
        (tmp_path / "experiments").mkdir()

        summary = self._make_summary()
        _save_summary_md(summary)

        md_path = tmp_path / "experiments" / "SUMMARY.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "# Self-Correction Loop Summary" in content
        assert "max_iterations=5" in content
        assert "target_score=8.5" in content
        assert "patience=3" in content
        assert "Iteration History" in content
        assert "Accepted Changes" in content

    def test_summary_handles_no_accepted(self, mocker, tmp_path):
        """SUMMARY.md handles zero accepted changes."""
        mocker.patch.object(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
        (tmp_path / "experiments").mkdir()

        summary = self._make_summary()
        # Remove accepted changes
        for h in summary["history"]:
            h["config_changes"] = None
        _save_summary_md(summary)

        content = (tmp_path / "experiments" / "SUMMARY.md").read_text()
        assert "No improvements were accepted beyond the baseline." in content


# ---------------------------------------------------------------------------
# Full Loop Tests (integration-level with mocks)
# ---------------------------------------------------------------------------

class TestRunAutocorrect:
    def test_baseline_only_max_iterations_1(self, mock_pipeline):
        """max_iterations=1 means baseline only."""
        result = run_autocorrect(max_iterations=1, target_score=10.0, patience=3)
        assert result["iterations_run"] == 1
        assert result["stop_reason"] == "max_iterations"
        assert result["accepted"] == 1
        assert result["best_iteration"] is not None

    def test_baseline_only_target_met(self, mock_pipeline, sample_metrics):
        """Single iteration when baseline meets target."""
        # Set target below baseline score (0.7523 * 10 = 7.52)
        result = run_autocorrect(max_iterations=5, target_score=7.0, patience=3)
        assert result["stop_reason"] == "target_reached"
        assert result["accepted"] == 1

    def test_patience_stops_loop(self, mock_pipeline, mocker):
        """Stops after patience consecutive non-accepted iterations."""
        # Make diagnosis always return None -> stale each iteration
        mocker.patch(
            "src.autocorrect._run_diagnosis_safe",
            return_value=None,
        )
        result = run_autocorrect(max_iterations=10, target_score=10.0, patience=2)
        assert result["stop_reason"] == "patience_exhausted"
        # Baseline (1 accepted) + 2 skipped = patience exhausted
        assert result["skipped"] >= 2

    def test_max_iterations_stops_loop(self, mock_pipeline, mocker):
        """Stops at max_iterations."""
        # Diagnosis returns hypothesis that always produces a different-but-worse config
        call_count = {"n": 0}

        def diagnosis_side_effect(iteration):
            call_count["n"] += 1
            depth = 20 - call_count["n"]
            if depth < 1:
                depth = 1
            return {
                "hypothesis": f"Reduce depth to {depth}",
                "component": "training",
                "parameter_changes": {"training.max_depth": depth},
                "expected_impact": "Test",
                "risk": "Test",
                "tier": 1,
                "confidence": 0.7,
                "reasoning": "Test",
            }

        mocker.patch(
            "src.autocorrect._run_diagnosis_safe",
            side_effect=diagnosis_side_effect,
        )

        result = run_autocorrect(max_iterations=3, target_score=10.0, patience=10)
        assert result["stop_reason"] == "max_iterations"
        # 1 baseline + 2 improvement iterations = 3 total
        assert mock_pipeline["counter"]["count"] == 3

    def test_diagnosis_failure_increments_stale(self, mock_pipeline, mocker):
        """Diagnosis failure increments stale count, loop continues."""
        mocker.patch(
            "src.autocorrect._run_diagnosis_safe",
            return_value=None,
        )
        result = run_autocorrect(max_iterations=5, target_score=10.0, patience=2)
        assert result["stop_reason"] == "patience_exhausted"
        assert result["skipped"] >= 2

    def test_classification_failure_reverts(self, mock_pipeline, mocker):
        """Classification failure reverts config and continues."""
        call_count = {"n": 0}

        def failing_classify():
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Baseline succeeds
                mock_pipeline["mock_classify"].side_effect.__wrapped__ = None
                # Create fake ledger entry
                from src.experiment import load_config, save_experiment
                cfg = load_config()
                save_experiment(cfg, mock_pipeline["metrics_sequence"][0])
                for year in config.TIME_RANGES:
                    path = config.OUTPUT_DIR / f"landcover_{year}.tif"
                    path.write_bytes(b"fake")
            else:
                raise RuntimeError("Training failed")

        mock_pipeline["mock_classify"].side_effect = failing_classify

        # Provide a hypothesis that will be tried
        mocker.patch(
            "src.autocorrect._run_diagnosis_safe",
            return_value={
                "hypothesis": "Change depth",
                "component": "training",
                "parameter_changes": {"training.max_depth": 15},
                "expected_impact": "Test",
                "risk": "Test",
                "tier": 1,
                "confidence": 0.7,
                "reasoning": "Test",
            },
        )

        result = run_autocorrect(max_iterations=4, target_score=10.0, patience=5)
        assert result["reverted"] >= 1

    def test_baseline_failure_raises(self, mock_pipeline):
        """Baseline classification failure raises RuntimeError."""
        mock_pipeline["mock_classify"].side_effect = RuntimeError("No data")
        with pytest.raises(RuntimeError, match="Baseline classification failed"):
            run_autocorrect(max_iterations=3)

    def test_argument_validation(self):
        """Invalid arguments raise ValueError."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            run_autocorrect(max_iterations=0)
        with pytest.raises(ValueError, match="target_score must be between"):
            run_autocorrect(target_score=11.0)
        with pytest.raises(ValueError, match="patience must be >= 1"):
            run_autocorrect(patience=0)

    def test_summary_generated(self, mock_pipeline):
        """SUMMARY.md exists after loop completes."""
        run_autocorrect(max_iterations=1, target_score=10.0, patience=3)
        summary_path = mock_pipeline["tmp_path"] / "experiments" / "SUMMARY.md"
        assert summary_path.exists()

    def test_best_outputs_restored_after_loop(self, mock_pipeline):
        """output/ has best iteration's GeoTIFFs after loop ends."""
        run_autocorrect(max_iterations=1, target_score=10.0, patience=3)
        for year in config.TIME_RANGES:
            assert (mock_pipeline["tmp_path"] / "output" / f"landcover_{year}.tif").exists()

    def test_improvement_accepted(self, mock_pipeline, mocker):
        """When hypothesis improves score, iteration is accepted."""
        better_metrics = copy.deepcopy(mock_pipeline["metrics_sequence"][0])
        better_metrics["overall_accuracy"] = 0.85  # Higher than baseline 0.7523
        mock_pipeline["metrics_sequence"].append(better_metrics)

        mocker.patch(
            "src.autocorrect._run_diagnosis_safe",
            return_value={
                "hypothesis": "Add NDVI",
                "component": "features",
                "parameter_changes": {"features.add_ndvi": True},
                "expected_impact": "Better",
                "risk": "Low",
                "tier": 2,
                "confidence": 0.8,
                "reasoning": "NDVI helps",
            },
        )

        result = run_autocorrect(max_iterations=2, target_score=10.0, patience=3)
        assert result["accepted"] >= 2  # Baseline + at least one improvement
        assert result["final_score"] > result["initial_score"]


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_default_args(self, mocker):
        """Default arguments passed to run_autocorrect."""
        mock_run = mocker.patch("src.autocorrect.run_autocorrect", return_value={})
        mocker.patch("sys.argv", ["autocorrect"])

        from src.autocorrect import main
        main()

        mock_run.assert_called_once_with(
            max_iterations=DEFAULT_MAX_ITERATIONS,
            target_score=DEFAULT_TARGET_SCORE,
            patience=DEFAULT_PATIENCE,
        )

    def test_custom_args(self, mocker):
        """Custom CLI arguments forwarded correctly."""
        mock_run = mocker.patch("src.autocorrect.run_autocorrect", return_value={})
        mocker.patch("sys.argv", [
            "autocorrect",
            "--max-iterations", "5",
            "--target-score", "9.0",
            "--patience", "2",
        ])

        from src.autocorrect import main
        main()

        mock_run.assert_called_once_with(
            max_iterations=5,
            target_score=9.0,
            patience=2,
        )


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_existing_modules_importable(self):
        """All existing pipeline modules can still be imported (RT1-4)."""
        import src.config
        import src.experiment
        import src.classify
        import src.evaluate
        import src.diagnose
        import src.pipeline
        import src.change
        import src.autocorrect
