"""Tests for src.diagnose module: failure analysis and hypothesis generation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.diagnose import (
    Hypothesis,
    TIER_PARAMS,
    _apply_hypothesis_to_config,
    _build_diagnosis_prompt,
    _determine_tier,
    _load_evaluation_results,
    _parse_hypothesis_response,
    _rule_based_diagnosis,
    _save_hypothesis,
    _summarize_experiment_history,
    _validate_hypothesis,
    run_diagnosis,
)
from src.experiment import DEFAULT_CONFIG, validate_config


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _create_experiment_dir(
    base_dir: Path,
    iteration: int,
    config_data: dict,
    metrics_data: dict,
    status: str = "accepted",
):
    """Create a complete experiment iteration directory for testing."""
    iter_dir = base_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    with open(iter_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    with open(iter_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    metadata = {
        "iteration": iteration,
        "timestamp": "2026-04-01T00:00:00Z",
        "status": status,
    }
    with open(iter_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return iter_dir


def _make_valid_hypothesis_data(**overrides) -> dict:
    """Return a minimal valid hypothesis dict with optional overrides."""
    data = {
        "hypothesis": "Test hypothesis",
        "component": "training",
        "parameter_changes": {"training.n_estimators": 200},
        "expected_impact": "Better accuracy",
        "risk": "Slower training",
        "tier": 1,
        "confidence": 0.8,
        "reasoning": "Test reasoning",
    }
    data.update(overrides)
    return data


@pytest.fixture
def mock_anthropic_response():
    """Return a mock Anthropic API response with valid hypothesis JSON."""
    hypothesis_json = json.dumps(_make_valid_hypothesis_data())

    mock_text_block = MagicMock()
    mock_text_block.text = hypothesis_json

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]

    return mock_response


# ---------------------------------------------------------------------------
# Class 1: TestHypothesisModel
# ---------------------------------------------------------------------------


class TestHypothesisModel:
    def test_valid_creation(self):
        """Well-formed data creates a valid Hypothesis."""
        h = Hypothesis(**_make_valid_hypothesis_data())
        assert h.hypothesis == "Test hypothesis"
        assert h.component == "training"
        assert h.tier == 1
        assert h.confidence == 0.8

    def test_invalid_component_rejected(self):
        """Component not in allowed set raises ValidationError."""
        with pytest.raises(Exception, match="component must be one of"):
            Hypothesis(**_make_valid_hypothesis_data(component="invalid"))

    def test_invalid_tier_rejected(self):
        """Tier outside 1-3 raises ValidationError."""
        with pytest.raises(Exception, match="tier must be 1, 2, or 3"):
            Hypothesis(**_make_valid_hypothesis_data(tier=5))

    def test_confidence_out_of_range_rejected(self):
        """Confidence outside 0.0-1.0 raises ValidationError."""
        with pytest.raises(Exception, match="confidence must be 0.0-1.0"):
            Hypothesis(**_make_valid_hypothesis_data(confidence=1.5))

    def test_dot_notation_required(self):
        """Parameter keys without dots raise ValidationError."""
        with pytest.raises(Exception, match="dot notation"):
            Hypothesis(**_make_valid_hypothesis_data(
                parameter_changes={"n_estimators": 200}
            ))

    def test_cross_component_rejected(self):
        """Parameter in different component than declared raises ValidationError."""
        with pytest.raises(Exception, match="One change per component"):
            Hypothesis(**_make_valid_hypothesis_data(
                component="training",
                parameter_changes={"features.add_ndvi": True},
            ))

    def test_empty_parameter_changes_rejected(self):
        """Empty parameter_changes dict raises ValidationError."""
        with pytest.raises(Exception, match="must not be empty"):
            Hypothesis(**_make_valid_hypothesis_data(parameter_changes={}))

    def test_extra_fields_rejected(self):
        """Extra fields from Claude response are rejected (extra='forbid')."""
        data = _make_valid_hypothesis_data()
        data["extra_field"] = "should not be here"
        with pytest.raises(Exception):
            Hypothesis(**data)


# ---------------------------------------------------------------------------
# Class 2: TestLoadEvaluationResults
# ---------------------------------------------------------------------------


class TestLoadEvaluationResults:
    def test_no_eval_dir_returns_none(self, tmp_path, monkeypatch):
        """Missing evaluation directory returns None."""
        from src import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "EVALUATION_DIR", tmp_path / "nonexistent")
        result = _load_evaluation_results()
        assert result is None

    def test_empty_eval_dir_returns_none(self, tmp_path, monkeypatch):
        """Empty evaluation directory returns None."""
        from src import config as cfg_mod
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir()
        monkeypatch.setattr(cfg_mod, "EVALUATION_DIR", eval_dir)
        result = _load_evaluation_results()
        assert result is None

    def test_loads_evaluation_files(self, tmp_path, monkeypatch, sample_evaluation):
        """Loads and returns evaluation JSONs sorted by filename."""
        from src import config as cfg_mod
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir()
        with open(eval_dir / "evaluation_2021.json", "w") as f:
            json.dump(sample_evaluation, f)
        monkeypatch.setattr(cfg_mod, "EVALUATION_DIR", eval_dir)
        result = _load_evaluation_results()
        assert result is not None
        assert len(result) == 1
        assert result[0]["year"] == "2021"

    def test_corrupt_file_skipped(self, tmp_path, monkeypatch, capsys):
        """Corrupt evaluation file is skipped with warning, not crash."""
        from src import config as cfg_mod
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir()
        with open(eval_dir / "evaluation_2021.json", "w") as f:
            f.write("not json{{{")
        monkeypatch.setattr(cfg_mod, "EVALUATION_DIR", eval_dir)
        result = _load_evaluation_results()
        assert result is None
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


# ---------------------------------------------------------------------------
# Class 3: TestSummarizeExperimentHistory
# ---------------------------------------------------------------------------


class TestSummarizeExperimentHistory:
    def test_no_iterations_returns_first_message(self, tmp_path, monkeypatch):
        """Empty ledger returns 'first iteration' message."""
        from src import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
        result = _summarize_experiment_history()
        assert "first iteration" in result.lower()

    def test_single_iteration_summary(self, tmp_path, monkeypatch):
        """One iteration formatted correctly with accuracy and status."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        metrics = {"overall_accuracy": 0.75}
        _create_experiment_dir(exp_dir, 1, cfg, metrics, status="accepted")

        result = _summarize_experiment_history()
        assert "Iteration 1" in result
        assert "0.75" in result
        assert "accepted" in result

    def test_multi_iteration_with_deltas(self, tmp_path, monkeypatch):
        """Shows accuracy deltas and config changes using diff['a']/diff['b']."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)

        cfg1 = copy.deepcopy(DEFAULT_CONFIG)
        metrics1 = {"overall_accuracy": 0.70}
        _create_experiment_dir(exp_dir, 1, cfg1, metrics1, status="accepted")

        cfg2 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2["training"]["n_estimators"] = 200
        metrics2 = {"overall_accuracy": 0.75}
        _create_experiment_dir(exp_dir, 2, cfg2, metrics2, status="accepted")

        result = _summarize_experiment_history()
        assert "Iteration 1" in result
        assert "Iteration 2" in result
        assert "+0.05" in result or "0.0500" in result
        assert "n_estimators" in result

    def test_history_capped_at_max_entries(self, tmp_path, monkeypatch):
        """History exceeding MAX_HISTORY_ENTRIES is truncated to most recent."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)
        monkeypatch.setattr(cfg_mod, "MAX_HISTORY_ENTRIES", 3)

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        for i in range(1, 6):
            metrics = {"overall_accuracy": 0.70 + i * 0.01}
            _create_experiment_dir(exp_dir, i, cfg, metrics, status="accepted")

        result = _summarize_experiment_history()
        # Should only include iterations 3, 4, 5
        assert "Iteration 3" in result
        assert "Iteration 5" in result
        assert "Iteration 1" not in result


# ---------------------------------------------------------------------------
# Class 4: TestDetermineTier
# ---------------------------------------------------------------------------


class TestDetermineTier:
    def test_no_history_returns_tier1(self):
        """Empty history returns Tier 1."""
        assert _determine_tier([]) == 1

    def test_few_reverts_stays_tier1(self, tmp_path, monkeypatch):
        """1-2 Tier 1 reverts stay at Tier 1."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)

        base_cfg = copy.deepcopy(DEFAULT_CONFIG)
        base_metrics = {"overall_accuracy": 0.70}
        _create_experiment_dir(exp_dir, 1, base_cfg, base_metrics, status="accepted")

        cfg2 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2["training"]["class_weight"] = "balanced_subsample"
        _create_experiment_dir(exp_dir, 2, cfg2, {"overall_accuracy": 0.68}, status="reverted")

        from src.experiment import list_iterations
        iterations = list_iterations()
        assert _determine_tier(iterations) == 1

    def test_three_tier1_reverts_escalates_tier2(self, tmp_path, monkeypatch):
        """3+ Tier 1 reverts with no improvement escalates to Tier 2."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)

        base_cfg = copy.deepcopy(DEFAULT_CONFIG)
        _create_experiment_dir(exp_dir, 1, base_cfg, {"overall_accuracy": 0.70}, status="accepted")

        cfg2 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2["training"]["class_weight"] = "balanced_subsample"
        _create_experiment_dir(exp_dir, 2, cfg2, {"overall_accuracy": 0.68}, status="reverted")

        cfg3 = copy.deepcopy(DEFAULT_CONFIG)
        cfg3["training"]["boundary_buffer_px"] = 2
        cfg3["training"]["exclude_boundary_pixels"] = True
        _create_experiment_dir(exp_dir, 3, cfg3, {"overall_accuracy": 0.67}, status="reverted")

        cfg4 = copy.deepcopy(DEFAULT_CONFIG)
        cfg4["post_processing"]["mode_filter_size"] = 5
        _create_experiment_dir(exp_dir, 4, cfg4, {"overall_accuracy": 0.65}, status="reverted")

        from src.experiment import list_iterations
        iterations = list_iterations()
        assert _determine_tier(iterations) == 2

    def test_tier2_reverts_escalates_tier3(self, tmp_path, monkeypatch):
        """2+ Tier 2 reverts with no improvement escalates to Tier 3."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)

        base_cfg = copy.deepcopy(DEFAULT_CONFIG)
        _create_experiment_dir(exp_dir, 1, base_cfg, {"overall_accuracy": 0.70}, status="accepted")

        # 3 Tier 1 reverts -- each iteration differs from previous by one Tier 1 param
        cfg2 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2["training"]["class_weight"] = "balanced_subsample"
        _create_experiment_dir(exp_dir, 2, cfg2, {"overall_accuracy": 0.68}, status="reverted")

        cfg3 = copy.deepcopy(cfg2)
        cfg3["training"]["max_depth"] = 15
        _create_experiment_dir(exp_dir, 3, cfg3, {"overall_accuracy": 0.67}, status="reverted")

        cfg4 = copy.deepcopy(cfg3)
        cfg4["training"]["max_samples_per_class"] = 10000
        _create_experiment_dir(exp_dir, 4, cfg4, {"overall_accuracy": 0.66}, status="reverted")

        # 2 Tier 2 reverts -- each differs from previous by ONLY a Tier 2 param
        cfg5 = copy.deepcopy(cfg4)
        cfg5["features"]["add_ndvi"] = True
        _create_experiment_dir(exp_dir, 5, cfg5, {"overall_accuracy": 0.64}, status="reverted")

        cfg6 = copy.deepcopy(cfg5)
        cfg6["training"]["n_estimators"] = 300
        _create_experiment_dir(exp_dir, 6, cfg6, {"overall_accuracy": 0.63}, status="reverted")

        from src.experiment import list_iterations
        iterations = list_iterations()
        assert _determine_tier(iterations) == 3


# ---------------------------------------------------------------------------
# Class 5: TestBuildDiagnosisPrompt
# ---------------------------------------------------------------------------


class TestBuildDiagnosisPrompt:
    def test_prompt_contains_all_sections(self, default_config, sample_metrics):
        """System and user prompts contain all required sections."""
        system, user = _build_diagnosis_prompt(
            sample_metrics, None, "Experiment History:\n  Iteration 1: ...",
            default_config, 1,
        )
        assert "RULES" in system
        assert "OUTPUT FORMAT" in system
        assert "Current Classification Metrics" in user
        assert "Confusion Matrix" in user
        assert "Experiment History" in user
        assert "Current Configuration" in user
        assert "Current Tier" in user

    def test_prompt_without_evaluation(self, default_config, sample_metrics):
        """evaluation=None produces valid prompt without VLM section."""
        _, user = _build_diagnosis_prompt(
            sample_metrics, None, "No experiment history yet.",
            default_config, 1,
        )
        assert "VLM Evaluation Results" not in user

    def test_prompt_includes_tier_constraints(self, default_config, sample_metrics):
        """Prompt includes tier-appropriate parameter restrictions."""
        _, user = _build_diagnosis_prompt(
            sample_metrics, None, "No experiment history yet.",
            default_config, 2,
        )
        assert "Tier 2" in user
        assert "features.add_ndvi" in user

    def test_confusion_matrix_has_class_labels(self, default_config, sample_metrics):
        """Confusion matrix in prompt includes class name labels."""
        _, user = _build_diagnosis_prompt(
            sample_metrics, None, "No experiment history yet.",
            default_config, 1,
        )
        assert "Built-up" in user
        assert "Cropland" in user
        assert "Predicted ->" in user


# ---------------------------------------------------------------------------
# Class 6: TestParseHypothesisResponse
# ---------------------------------------------------------------------------


class TestParseHypothesisResponse:
    def test_parse_raw_json(self):
        """Parses raw JSON string."""
        data = _make_valid_hypothesis_data()
        result = _parse_hypothesis_response(json.dumps(data))
        assert result["hypothesis"] == "Test hypothesis"

    def test_parse_json_in_code_block(self):
        """Extracts JSON from ```json ... ``` fences."""
        data = _make_valid_hypothesis_data()
        text = f"Here is my analysis:\n```json\n{json.dumps(data)}\n```\nDone."
        result = _parse_hypothesis_response(text)
        assert result["hypothesis"] == "Test hypothesis"

    def test_parse_json_with_preamble(self):
        """Extracts JSON when surrounded by explanatory text."""
        data = _make_valid_hypothesis_data()
        text = f"Based on the evaluation, I propose: {json.dumps(data)} That should help."
        result = _parse_hypothesis_response(text)
        assert result["hypothesis"] == "Test hypothesis"

    def test_invalid_json_raises(self):
        """Malformed JSON raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_hypothesis_response("{not: valid json!!")

    def test_no_json_at_all_raises(self):
        """Text with no JSON raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_hypothesis_response("This is just plain text with no JSON at all.")

    def test_multiple_code_blocks_uses_last(self):
        """With multiple ```json blocks, parses the LAST one."""
        first = {"hypothesis": "Wrong one"}
        second = _make_valid_hypothesis_data(hypothesis="Correct one")
        text = (
            "```json\n" + json.dumps(first) + "\n```\n"
            "But actually:\n"
            "```json\n" + json.dumps(second) + "\n```"
        )
        result = _parse_hypothesis_response(text)
        assert result["hypothesis"] == "Correct one"

    def test_missing_closing_fence_falls_through(self):
        """Missing closing ``` falls through to next strategy."""
        data = _make_valid_hypothesis_data()
        text = f"```json\n{json.dumps(data)}"  # No closing fence
        result = _parse_hypothesis_response(text)
        assert result["hypothesis"] == "Test hypothesis"


# ---------------------------------------------------------------------------
# Class 7: TestRuleBasedDiagnosis
# ---------------------------------------------------------------------------


class TestRuleBasedDiagnosis:
    def test_overfitting_detection(self, default_config, sample_metrics):
        """High training-test gap proposes max_depth reduction."""
        sample_metrics["training_accuracy"] = 0.98
        sample_metrics["overall_accuracy"] = 0.60
        h = _rule_based_diagnosis(sample_metrics, default_config, tier=1)
        assert "max_depth" in str(h.parameter_changes)
        assert h.component == "training"
        assert "overfitting" in h.hypothesis.lower() or "Overfitting" in h.hypothesis

    def test_low_recall_proposes_class_weight(self, default_config, sample_metrics):
        """Low recall class proposes balanced class weights."""
        default_config["training"]["class_weight"] = "balanced_subsample"
        sample_metrics["per_class"]["Other"]["recall"] = 0.1
        sample_metrics["per_class"]["Other"]["f1"] = 0.15
        # Make training_accuracy close to overall to skip Rule 1
        sample_metrics["training_accuracy"] = 0.76
        h = _rule_based_diagnosis(sample_metrics, default_config, tier=1)
        assert h.parameter_changes.get("training.class_weight") == "balanced"

    def test_no_postprocessing_proposes_mode_filter(self, default_config, sample_metrics):
        """mode_filter_size=0 proposes enabling it."""
        # Skip Rule 1 (overfitting), Rule 2 (class_weight already balanced),
        # Rule 3 (boundary exclusion -- need worst_f1 >= 0.6 or boundary already set)
        sample_metrics["training_accuracy"] = 0.76
        # All classes have f1 >= 0.6 to skip Rule 3
        for cls in sample_metrics["per_class"].values():
            cls["f1"] = max(cls["f1"], 0.65)
            cls["recall"] = max(cls["recall"], 0.6)
        h = _rule_based_diagnosis(sample_metrics, default_config, tier=1)
        assert h.component == "post_processing"
        assert "mode_filter_size" in str(h.parameter_changes)

    def test_tier_escalation_to_ndvi(self, sample_metrics):
        """Tier 2 with all Tier 1 options exhausted proposes NDVI."""
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        # Make training_accuracy close to overall to skip Rule 1
        sample_metrics["training_accuracy"] = 0.76
        h = _rule_based_diagnosis(sample_metrics, cfg, tier=2)
        assert h.parameter_changes.get("features.add_ndvi") is True

    def test_empty_per_class_returns_sample_increase(self, default_config):
        """Empty per_class dict returns max_samples_per_class increase."""
        metrics = {"overall_accuracy": 0.5, "per_class": {}}
        h = _rule_based_diagnosis(metrics, default_config, tier=1)
        assert "max_samples_per_class" in str(h.parameter_changes)

    def test_ndwi_respects_tier_guard(self, default_config, sample_metrics):
        """NDWI rule only fires at tier <= 2."""
        # Configure to reach Rule 8: tier=3, all earlier rules skipped
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["features"]["add_ndvi"] = True
        cfg["features"]["add_ndwi"] = False
        cfg["training"]["n_estimators"] = 300
        sample_metrics["training_accuracy"] = 0.76
        h = _rule_based_diagnosis(sample_metrics, cfg, tier=3)
        # At tier 3, NDWI rule (tier <= 2) should NOT fire; should hit exhaustion
        assert "add_ndwi" not in str(h.parameter_changes)

    def test_always_returns_valid_hypothesis(self, default_config, sample_metrics):
        """Every rule output passes validate_config when applied to config."""
        test_cases = [
            # Rule 1: overfitting
            ({"training_accuracy": 0.98, "overall_accuracy": 0.60}, {}, 1),
            # Rule 2: class weight (need class_weight != balanced)
            ({"training_accuracy": 0.76}, {"training": {"class_weight": None}}, 1),
            # Rule 3: boundary exclusion (need worst_f1 < 0.6)
            ({"training_accuracy": 0.76}, {}, 1),
            # Rule 4: mode filter
            ({"training_accuracy": 0.76}, {"training": {"exclude_boundary_pixels": True, "boundary_buffer_px": 2}}, 1),
            # Rule 5: samples
            ({"training_accuracy": 0.76}, {
                "training": {"exclude_boundary_pixels": True, "boundary_buffer_px": 2},
                "post_processing": {"mode_filter_size": 5},
            }, 1),
            # Rule 6: NDVI (tier 2)
            ({"training_accuracy": 0.76}, {}, 2),
            # Rule 7: n_estimators (tier 2, NDVI already on)
            ({"training_accuracy": 0.76}, {"features": {"add_ndvi": True}}, 2),
            # Rule 8: NDWI (tier 2, NDVI on, n_estimators high)
            ({"training_accuracy": 0.76}, {
                "features": {"add_ndvi": True},
                "training": {"n_estimators": 300},
            }, 2),
        ]

        for metric_overrides, cfg_overrides, tier in test_cases:
            metrics = copy.deepcopy(sample_metrics)
            metrics.update(metric_overrides)
            cfg = copy.deepcopy(default_config)
            for section, vals in cfg_overrides.items():
                cfg.setdefault(section, {}).update(vals)

            h = _rule_based_diagnosis(metrics, cfg, tier)
            merged = _apply_hypothesis_to_config(cfg, h)
            # Should not raise
            validate_config(merged)


# ---------------------------------------------------------------------------
# Class 8: TestApplyHypothesisToConfig
# ---------------------------------------------------------------------------


class TestApplyHypothesisToConfig:
    def test_single_change_applied(self, default_config):
        """Single dot-notation key correctly modifies nested config."""
        h = Hypothesis(**_make_valid_hypothesis_data(
            parameter_changes={"training.n_estimators": 500}
        ))
        merged = _apply_hypothesis_to_config(default_config, h)
        assert merged["training"]["n_estimators"] == 500

    def test_multiple_changes_same_component(self, default_config):
        """Multiple changes in same component all applied."""
        h = Hypothesis(**_make_valid_hypothesis_data(
            parameter_changes={
                "training.exclude_boundary_pixels": True,
                "training.boundary_buffer_px": 3,
            }
        ))
        merged = _apply_hypothesis_to_config(default_config, h)
        assert merged["training"]["exclude_boundary_pixels"] is True
        assert merged["training"]["boundary_buffer_px"] == 3


# ---------------------------------------------------------------------------
# Class 9: TestValidateHypothesis
# ---------------------------------------------------------------------------


class TestValidateHypothesis:
    def test_valid_hypothesis_passes(self, default_config):
        """Valid parameter changes pass validation."""
        h = Hypothesis(**_make_valid_hypothesis_data(
            parameter_changes={"training.n_estimators": 500}
        ))
        merged = _validate_hypothesis(h, default_config)
        assert merged["training"]["n_estimators"] == 500

    def test_invalid_parameter_range_rejected(self, default_config):
        """n_estimators > 1000 rejected by validate_config."""
        h = Hypothesis(**_make_valid_hypothesis_data(
            parameter_changes={"training.n_estimators": 5000}
        ))
        with pytest.raises(ValueError):
            _validate_hypothesis(h, default_config)

    def test_unknown_parameter_detected(self, default_config):
        """Parameter key not in config schema raises ValueError."""
        h = Hypothesis(**_make_valid_hypothesis_data(
            parameter_changes={"training.nonexistent_param": 42}
        ))
        with pytest.raises(ValueError, match="None of the proposed parameter changes"):
            _validate_hypothesis(h, default_config)

    def test_class_weight_none_string_accepted(self, default_config):
        """class_weight: 'none' is a valid config value."""
        h = Hypothesis(**_make_valid_hypothesis_data(
            parameter_changes={"training.class_weight": "none"}
        ))
        merged = _validate_hypothesis(h, default_config)
        # validate_config maps "none" -> None
        assert merged["training"]["class_weight"] is None


# ---------------------------------------------------------------------------
# Class 10: TestSaveHypothesis
# ---------------------------------------------------------------------------


class TestSaveHypothesis:
    def test_saves_to_correct_path(self, tmp_path, monkeypatch):
        """Hypothesis saved to experiments/iteration_NNN/hypothesis.json."""
        from src import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")

        h = Hypothesis(**_make_valid_hypothesis_data())
        path = _save_hypothesis(h, 1, "rule_based")
        assert path.exists()
        assert path.name == "hypothesis.json"
        assert "iteration_001" in str(path)

        with open(path) as f:
            data = json.load(f)
        assert data["source"] == "rule_based"
        assert "timestamp" in data

    def test_atomic_write_no_tmp_left(self, tmp_path, monkeypatch):
        """No .tmp file remains after successful save."""
        from src import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")

        h = Hypothesis(**_make_valid_hypothesis_data())
        _save_hypothesis(h, 1, "claude")
        iter_dir = tmp_path / "experiments" / "iteration_001"
        tmp_files = list(iter_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        """Creates iteration directory if it doesn't exist yet."""
        from src import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")

        h = Hypothesis(**_make_valid_hypothesis_data())
        path = _save_hypothesis(h, 5, "rule_based")
        assert path.exists()
        assert "iteration_005" in str(path)


# ---------------------------------------------------------------------------
# Class 11: TestCallClaude
# ---------------------------------------------------------------------------


class TestCallClaude:
    def test_successful_api_call(self, mocker, monkeypatch, mock_anthropic_response):
        """Valid API response returns Hypothesis."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-sk-xxxxx")

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response

        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        # Ensure exception types are proper exception classes
        mock_anthropic_mod.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_anthropic_mod.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_anthropic_mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_anthropic_mod.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_anthropic_mod.APIError = type("APIError", (Exception,), {})

        mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_mod})

        from src.diagnose import _call_claude
        h = _call_claude("system prompt", "user prompt")
        assert isinstance(h, Hypothesis)
        assert h.hypothesis == "Test hypothesis"

    def test_retries_on_transient_error(self, mocker, monkeypatch):
        """Retries once on RateLimitError, succeeds on second attempt."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-sk-xxxxx")

        RateLimitError = type("RateLimitError", (Exception,), {})

        valid_json = json.dumps(_make_valid_hypothesis_data())
        mock_text_block = MagicMock()
        mock_text_block.text = valid_json
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            RateLimitError("rate limited"),
            mock_response,
        ]

        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_anthropic_mod.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_anthropic_mod.RateLimitError = RateLimitError
        mock_anthropic_mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_anthropic_mod.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_anthropic_mod.APIError = type("APIError", (Exception,), {})

        mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_mod})
        mock_sleep = mocker.patch("time.sleep")

        from src.diagnose import _call_claude
        h = _call_claude("system", "user")
        assert isinstance(h, Hypothesis)
        mock_sleep.assert_called_once_with(3)

    def test_auth_error_raises_immediately(self, mocker, monkeypatch):
        """AuthenticationError raises RuntimeError without retry."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-sk-xxxxx")

        AuthenticationError = type("AuthenticationError", (Exception,), {})

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = AuthenticationError("bad key")

        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_anthropic_mod.AuthenticationError = AuthenticationError
        mock_anthropic_mod.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_anthropic_mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_anthropic_mod.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_anthropic_mod.APIError = type("APIError", (Exception,), {})

        mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic_mod})

        from src.diagnose import _call_claude
        with pytest.raises(RuntimeError, match="authentication failed"):
            _call_claude("system", "user")


# ---------------------------------------------------------------------------
# Class 12: TestRunDiagnosis
# ---------------------------------------------------------------------------


class TestRunDiagnosis:
    def test_no_experiments_raises(self, tmp_path, monkeypatch):
        """Empty ledger raises RuntimeError."""
        from src import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
        monkeypatch.setattr(cfg_mod, "EVALUATION_DIR", tmp_path / "evaluations")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="No experiments"):
            run_diagnosis()

    def test_invalid_iteration_raises(self, tmp_path, monkeypatch):
        """Negative iteration raises ValueError."""
        with pytest.raises(ValueError, match="iteration must be >= 1"):
            run_diagnosis(iteration=-1)

    def test_rule_based_end_to_end(self, tmp_path, monkeypatch, sample_metrics):
        """Full flow without API key uses rule-based and saves hypothesis."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)
        monkeypatch.setattr(cfg_mod, "EVALUATION_DIR", tmp_path / "evaluations")
        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        _create_experiment_dir(exp_dir, 1, cfg, sample_metrics, status="pending")

        h = run_diagnosis(iteration=1)
        assert isinstance(h, Hypothesis)

        # Check hypothesis.json was saved
        hyp_path = exp_dir / "iteration_001" / "hypothesis.json"
        assert hyp_path.exists()
        with open(hyp_path) as f:
            saved = json.load(f)
        assert saved["source"] == "rule_based"

    def test_claude_fallback_on_api_error(self, mocker, tmp_path, monkeypatch, sample_metrics):
        """Falls back to rule-based when Claude API fails."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)
        monkeypatch.setattr(cfg_mod, "EVALUATION_DIR", tmp_path / "evaluations")
        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-sk-xxxxx")

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        _create_experiment_dir(exp_dir, 1, cfg, sample_metrics, status="pending")

        mocker.patch("src.diagnose._call_claude", side_effect=RuntimeError("API down"))

        h = run_diagnosis(iteration=1)
        assert isinstance(h, Hypothesis)

        hyp_path = exp_dir / "iteration_001" / "hypothesis.json"
        with open(hyp_path) as f:
            saved = json.load(f)
        assert saved["source"] == "rule_based"

    def test_handles_missing_evaluation(self, mocker, tmp_path, monkeypatch, sample_metrics):
        """Proceeds with metrics only when VLM evaluation is missing."""
        from src import config as cfg_mod
        exp_dir = tmp_path / "experiments"
        monkeypatch.setattr(cfg_mod, "EXPERIMENTS_DIR", exp_dir)
        monkeypatch.setattr(cfg_mod, "EVALUATION_DIR", tmp_path / "nonexistent_evaluations")
        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        _create_experiment_dir(exp_dir, 1, cfg, sample_metrics, status="pending")

        h = run_diagnosis(iteration=1)
        assert isinstance(h, Hypothesis)
