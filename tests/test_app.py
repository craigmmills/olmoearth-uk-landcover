"""Tests for experiment dashboard helpers in src/app.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCheckExperimentsExist:
    """Tests for check_experiments_exist()."""

    def test_returns_false_when_dir_missing(self, tmp_path):
        """Should return False when experiments directory does not exist."""
        from src.app import check_experiments_exist

        with patch("src.config.EXPERIMENTS_DIR", tmp_path / "nonexistent"):
            assert check_experiments_exist() is False

    def test_returns_false_when_dir_empty(self, tmp_path):
        """Should return False when experiments directory exists but is empty."""
        from src.app import check_experiments_exist

        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            assert check_experiments_exist() is False

    def test_returns_false_when_no_iteration_dirs(self, tmp_path):
        """Should return False when directory has files but no iteration_* dirs."""
        from src.app import check_experiments_exist

        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        (experiments_dir / "SUMMARY.md").write_text("summary")
        (experiments_dir / "random_file.json").write_text("{}")

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            assert check_experiments_exist() is False

    def test_returns_true_when_iteration_dir_exists(self, tmp_path):
        """Should return True when at least one iteration_* directory exists."""
        from src.app import check_experiments_exist

        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        (experiments_dir / "iteration_001").mkdir()

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            assert check_experiments_exist() is True

    def test_returns_true_with_multiple_iterations(self, tmp_path):
        """Should return True with multiple iteration directories."""
        from src.app import check_experiments_exist

        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        (experiments_dir / "iteration_001").mkdir()
        (experiments_dir / "iteration_002").mkdir()
        (experiments_dir / "iteration_003").mkdir()

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            assert check_experiments_exist() is True

    def test_ignores_non_iteration_files(self, tmp_path):
        """Should not count non-iteration files as iterations."""
        from src.app import check_experiments_exist

        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        # File named iteration_001 (not a directory) should not count
        (experiments_dir / "iteration_001").write_text("not a dir")

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            assert check_experiments_exist() is False


class TestLoadIterationDetails:
    """Tests for load_iteration_details()."""

    def _create_iteration(
        self,
        experiments_dir: Path,
        num: int,
        *,
        status: str = "accepted",
        accuracy: float = 0.75,
        hypothesis: dict | None = None,
        evaluation_2021: dict | None = None,
    ) -> Path:
        """Helper to create a minimal iteration directory with required files."""
        iteration_dir = experiments_dir / f"iteration_{num:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "iteration": num,
            "timestamp": "2026-04-03T12:00:00Z",
            "status": status,
        }
        with open(iteration_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        metrics = {
            "overall_accuracy": accuracy,
            "per_class": {
                "Built-up": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 100},
                "Cropland": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 200},
            },
        }
        with open(iteration_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

        config_data = {
            "training": {"n_estimators": 100, "max_depth": 20},
            "features": {"use_embeddings": True},
        }
        with open(iteration_dir / "config.json", "w") as f:
            json.dump(config_data, f)

        if hypothesis is not None:
            with open(iteration_dir / "hypothesis.json", "w") as f:
                json.dump(hypothesis, f)

        if evaluation_2021 is not None:
            with open(iteration_dir / "evaluation_2021.json", "w") as f:
                json.dump(evaluation_2021, f)

        return iteration_dir

    def test_baseline_has_no_hypothesis(self, tmp_path):
        """Iteration 1 (baseline) should have hypothesis=None."""
        from src.app import load_iteration_details

        experiments_dir = tmp_path / "experiments"
        self._create_iteration(experiments_dir, 1)

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            details = load_iteration_details(1)

        assert details["hypothesis"] is None
        assert details["metadata"]["iteration"] == 1

    def test_reads_hypothesis_from_previous_iteration(self, tmp_path):
        """Iteration N should read hypothesis from iteration N-1."""
        from src.app import load_iteration_details

        experiments_dir = tmp_path / "experiments"
        hypothesis = {
            "component": "training",
            "hypothesis": "Increase trees",
            "parameter_changes": {"n_estimators": 200},
            "expected_impact": "Better accuracy",
        }
        self._create_iteration(experiments_dir, 1, hypothesis=hypothesis)
        self._create_iteration(experiments_dir, 2)

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            details = load_iteration_details(2)

        assert details["hypothesis"] is not None
        assert details["hypothesis"]["component"] == "training"
        assert details["hypothesis"]["hypothesis"] == "Increase trees"

    def test_loads_evaluation_data(self, tmp_path):
        """Should load evaluation JSON when present."""
        from src.app import load_iteration_details

        experiments_dir = tmp_path / "experiments"
        eval_data = {
            "evaluation": {"overall_score": 7, "per_class": [], "error_regions": []},
        }
        self._create_iteration(experiments_dir, 1, evaluation_2021=eval_data)

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            details = load_iteration_details(1)

        assert details["evaluations"]["2021"] is not None
        assert details["evaluations"]["2021"]["evaluation"]["overall_score"] == 7
        assert details["evaluations"]["2023"] is None

    def test_missing_evaluation_returns_none(self, tmp_path):
        """Should return None for missing evaluation files."""
        from src.app import load_iteration_details

        experiments_dir = tmp_path / "experiments"
        self._create_iteration(experiments_dir, 1)

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            details = load_iteration_details(1)

        assert details["evaluations"]["2021"] is None
        assert details["evaluations"]["2023"] is None

    def test_metrics_diff_for_second_iteration(self, tmp_path):
        """Should compute metrics_diff between iteration 1 and 2."""
        from src.app import load_iteration_details

        experiments_dir = tmp_path / "experiments"
        self._create_iteration(experiments_dir, 1, accuracy=0.70)
        self._create_iteration(experiments_dir, 2, accuracy=0.75)

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            details = load_iteration_details(2)

        assert details["metrics_diff"] is not None
        acc_diff = details["metrics_diff"]["overall_accuracy"]
        assert acc_diff["a"] == 0.70
        assert acc_diff["b"] == 0.75
        assert acc_diff["delta"] == pytest.approx(0.05, abs=0.001)

    def test_no_metrics_diff_for_baseline(self, tmp_path):
        """Baseline (iteration 1) should have no metrics_diff."""
        from src.app import load_iteration_details

        experiments_dir = tmp_path / "experiments"
        self._create_iteration(experiments_dir, 1)

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            details = load_iteration_details(1)

        assert details["metrics_diff"] is None
        assert details["config_diff"] is None

    def test_missing_images_return_none(self, tmp_path):
        """Should return None for missing comparison images."""
        from src.app import load_iteration_details

        experiments_dir = tmp_path / "experiments"
        self._create_iteration(experiments_dir, 1)

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            details = load_iteration_details(1)

        for key, value in details["images"].items():
            assert value is None

    def test_raises_for_missing_iteration(self, tmp_path):
        """Should raise FileNotFoundError for nonexistent iteration."""
        from src.app import load_iteration_details

        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        with patch("src.config.EXPERIMENTS_DIR", experiments_dir):
            with pytest.raises(FileNotFoundError):
                load_iteration_details(999)
