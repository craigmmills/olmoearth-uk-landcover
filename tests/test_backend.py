"""Tests for the FastAPI backend."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_experiments(tmp_path, monkeypatch):
    """Create a minimal experiment directory structure for testing.

    Creates:
    - session_20260404_093509/ with session_meta.json
    - iteration_001/ with config, metadata, metrics, comparison image
    - iteration_002/ with config, metadata, metrics (for diff testing)
    - hypothesis.json in iteration_001 (describes change for iteration_002)
    - latest symlink -> session_20260404_093509
    """
    experiments_dir = tmp_path / "experiments"
    session_dir = experiments_dir / "session_20260404_093509"
    session_dir.mkdir(parents=True)

    # Session meta
    (session_dir / "session_meta.json").write_text(json.dumps({
        "session_id": "session_20260404_093509",
        "start_time": "2026-04-04T09:35:09.131676+00:00",
        "end_time": "2026-04-04T09:45:45.964423+00:00",
        "stop_reason": "patience_exhausted",
        "best_iteration": 1,
        "final_score": 7.65,
        "n_iterations": 2,
        "initial_config": {"training": {"n_estimators": 100}},
        "parameter_count": 28,
    }))

    # Iteration 1
    iter1 = session_dir / "iteration_001"
    iter1.mkdir()
    (iter1 / "config.json").write_text(json.dumps({"training": {"n_estimators": 100}}))
    (iter1 / "metadata.json").write_text(json.dumps({
        "iteration": 1,
        "timestamp": "2026-04-04T09:36:18.536183+00:00",
        "status": "accepted",
    }))
    (iter1 / "metrics.json").write_text(json.dumps({
        "overall_accuracy": 0.918,
        "per_class": {"Built-up": {"precision": 0.928, "recall": 0.902, "f1": 0.915, "support": 103504}},
        "weighted_avg": {"precision": 0.919, "recall": 0.918, "f1": 0.918},
    }))
    # Hypothesis in iter 1 describes the change that created iter 2
    (iter1 / "hypothesis.json").write_text(json.dumps({
        "hypothesis": "Add mode filter for spatial smoothing",
        "component": "post_processing",
        "parameter_changes": {"post_processing.mode_filter_size": 5},
        "expected_impact": "Reduce salt-and-pepper noise",
        "risk": "May smooth small features",
        "tier": 1,
        "confidence": 0.7,
        "reasoning": "Rule-based: mode_filter_size=0",
    }))
    # Comparison image
    (iter1 / "comparison_2021_full.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (iter1 / "comparison_2023_full.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    # Iteration 2
    iter2 = session_dir / "iteration_002"
    iter2.mkdir()
    (iter2 / "config.json").write_text(json.dumps({
        "training": {"n_estimators": 100},
        "post_processing": {"mode_filter_size": 5},
    }))
    (iter2 / "metadata.json").write_text(json.dumps({
        "iteration": 2,
        "timestamp": "2026-04-04T09:40:00.000000+00:00",
        "status": "accepted",
    }))
    (iter2 / "metrics.json").write_text(json.dumps({
        "overall_accuracy": 0.925,
        "per_class": {"Built-up": {"precision": 0.935, "recall": 0.910, "f1": 0.922, "support": 103504}},
        "weighted_avg": {"precision": 0.926, "recall": 0.925, "f1": 0.925},
    }))

    # Create latest symlink
    (experiments_dir / "latest").symlink_to(session_dir.name)

    # Point config paths at tmp_path
    monkeypatch.setattr("src.config.EXPERIMENTS_BASE_DIR", experiments_dir)
    monkeypatch.setattr("src.config.EXPERIMENTS_DIR", experiments_dir / "latest")

    return tmp_path


@pytest.fixture
def client(mock_experiments):
    """TestClient with mock experiments directory."""
    from webui.backend.main import app
    return TestClient(app)


class TestListSessions:
    def test_returns_sessions_list(self, client):
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "session_id" in data[0]
        assert "path" not in data[0]  # Must not expose filesystem paths

    def test_empty_experiments_dir(self, client, monkeypatch, tmp_path):
        empty_dir = tmp_path / "empty_experiments"
        empty_dir.mkdir()
        monkeypatch.setattr("src.config.EXPERIMENTS_BASE_DIR", empty_dir)
        response = client.get("/api/sessions")
        assert response.status_code == 200
        assert response.json() == []


class TestGetSession:
    def test_returns_session_detail(self, client):
        response = client.get("/api/sessions/session_20260404_093509")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_20260404_093509"
        assert "initial_config" in data
        assert data["n_iterations"] == 2

    def test_session_not_found(self, client):
        response = client.get("/api/sessions/session_99990101_000000")
        assert response.status_code == 404

    def test_invalid_session_id_format(self, client):
        response = client.get("/api/sessions/not_a_valid_session")
        assert response.status_code == 400


class TestListIterations:
    def test_returns_iterations(self, client):
        response = client.get("/api/sessions/session_20260404_093509/iterations")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["iteration"] == 1
        assert data[0]["status"] == "accepted"
        assert data[0]["overall_accuracy"] == pytest.approx(0.918)

    def test_session_not_found(self, client):
        response = client.get("/api/sessions/session_99990101_000000/iterations")
        assert response.status_code == 404


class TestGetIterationDetail:
    def test_returns_full_detail(self, client):
        response = client.get("/api/sessions/session_20260404_093509/iterations/1")
        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert "metrics" in data
        assert "config" in data
        assert "images" in data
        assert data["metadata"]["iteration"] == 1
        assert data["metrics"]["overall_accuracy"] == pytest.approx(0.918)

    def test_iteration_2_has_hypothesis_from_iter_1(self, client):
        """hypothesis.json is in iteration N-1; iteration 2 should get hypothesis from iter 1."""
        response = client.get("/api/sessions/session_20260404_093509/iterations/2")
        assert response.status_code == 200
        data = response.json()
        assert data["hypothesis"] is not None
        assert data["hypothesis"]["component"] == "post_processing"

    def test_iteration_1_has_no_hypothesis(self, client):
        """Iteration 1 has no previous iteration, so hypothesis should be null."""
        response = client.get("/api/sessions/session_20260404_093509/iterations/1")
        assert response.status_code == 200
        data = response.json()
        assert data["hypothesis"] is None

    def test_images_are_url_strings(self, client):
        response = client.get("/api/sessions/session_20260404_093509/iterations/1")
        data = response.json()
        images = data["images"]
        assert "2021_full" in images
        if images["2021_full"]:
            assert images["2021_full"].startswith("/api/sessions/")

    def test_iteration_not_found(self, client):
        response = client.get("/api/sessions/session_20260404_093509/iterations/999")
        assert response.status_code == 404

    def test_invalid_iteration_num(self, client):
        response = client.get("/api/sessions/session_20260404_093509/iterations/0")
        assert response.status_code == 400


class TestGetImage:
    def test_serves_png(self, client):
        response = client.get(
            "/api/sessions/session_20260404_093509/iterations/1/images/comparison_2021_full.png"
        )
        assert response.status_code == 200
        assert "image/png" in response.headers["content-type"]

    def test_invalid_filename_rejected(self, client):
        response = client.get(
            "/api/sessions/session_20260404_093509/iterations/1/images/secret.txt"
        )
        assert response.status_code == 400

    def test_image_not_found(self, client):
        response = client.get(
            "/api/sessions/session_20260404_093509/iterations/1/images/comparison_2021_nw.png"
        )
        assert response.status_code == 404


class TestHealthCheck:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestCogConversion:
    def test_ensure_cog_creates_cog_file(self, tmp_path):
        """Test that ensure_cog creates a COG from a GeoTIFF."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "landcover_2021.tif"
        data = np.random.randint(0, 6, (1, 64, 64), dtype=np.uint8)
        transform = from_bounds(0, 0, 640, 640, 64, 64)

        with rasterio.open(
            tif_path, "w", driver="GTiff", width=64, height=64,
            count=1, dtype="uint8", crs="EPSG:32631", transform=transform,
        ) as dst:
            dst.write(data)

        from webui.backend.cog import ensure_cog
        cog_path = ensure_cog(tif_path)

        assert cog_path.exists()
        assert cog_path.name == "landcover_2021_cog.tif"

        # Verify it's tiled (COG property)
        with rasterio.open(cog_path) as src:
            profile = src.profile
            assert profile.get("tiled") is True or profile.get("blockxsize") == 256

    def test_ensure_cog_uses_cache(self, tmp_path):
        """Test that ensure_cog returns cached COG without reconverting."""
        tif_path = tmp_path / "test.tif"
        cog_path = tmp_path / "test_cog.tif"
        tif_path.write_bytes(b"source data")
        cog_path.write_bytes(b"cached cog data")

        from webui.backend.cog import ensure_cog
        result = ensure_cog(tif_path)
        assert result == cog_path
        assert result.read_bytes() == b"cached cog data"  # Not overwritten

    def test_ensure_cog_reconverts_when_source_newer(self, tmp_path):
        """Test that ensure_cog reconverts when source is newer than cached COG."""
        import time
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "landcover.tif"
        cog_path = tmp_path / "landcover_cog.tif"

        # Create COG first (older)
        cog_path.write_bytes(b"old cog")
        time.sleep(0.1)

        # Create source (newer)
        data = np.random.randint(0, 6, (1, 32, 32), dtype=np.uint8)
        transform = from_bounds(0, 0, 320, 320, 32, 32)
        with rasterio.open(
            tif_path, "w", driver="GTiff", width=32, height=32,
            count=1, dtype="uint8", crs="EPSG:32631", transform=transform,
        ) as dst:
            dst.write(data)

        from webui.backend.cog import ensure_cog
        result = ensure_cog(tif_path)
        assert result.read_bytes() != b"old cog"  # Should be reconverted


class TestTileEndpoint:
    def test_invalid_year_rejected(self, client):
        response = client.get(
            "/api/sessions/session_20260404_093509/iterations/1/tiles/2022/10/512/340.png"
        )
        assert response.status_code == 400

    def test_session_not_found(self, client):
        response = client.get(
            "/api/sessions/session_99990101_000000/iterations/1/tiles/2021/10/512/340.png"
        )
        assert response.status_code == 404

    def test_tiff_not_found(self, client):
        """Iteration exists but no GeoTIFF for the year."""
        response = client.get(
            "/api/sessions/session_20260404_093509/iterations/1/tiles/2021/10/512/340.png"
        )
        # 404 because mock_experiments doesn't create actual GeoTIFFs
        assert response.status_code == 404


class TestSSEEvents:
    def test_sse_generator_helpers(self, mock_experiments):
        """Test the SSE scanning helper functions directly."""
        from webui.backend.main import _scan_all_sessions, _get_iteration_nums, _scan_for_new_iterations
        from src import config

        # Test _scan_all_sessions populates known iterations
        known: dict[str, set[int]] = {}
        _scan_all_sessions(known)
        assert "session_20260404_093509" in known
        assert known["session_20260404_093509"] == {1, 2}

        # Test _scan_for_new_iterations returns empty when nothing changed
        events = _scan_for_new_iterations(known)
        assert events == []

        # Create a new iteration and verify it's detected
        session_dir = config.EXPERIMENTS_BASE_DIR / "session_20260404_093509"
        iter3 = session_dir / "iteration_003"
        iter3.mkdir()
        (iter3 / "metadata.json").write_text(json.dumps({
            "iteration": 3,
            "timestamp": "2026-04-04T10:00:00.000000+00:00",
            "status": "pending",
        }))

        events = _scan_for_new_iterations(known)
        assert len(events) == 1
        assert events[0]["session_id"] == "session_20260404_093509"
        assert events[0]["iteration"] == 3

    def test_get_iteration_nums(self, mock_experiments):
        """Test _get_iteration_nums returns correct iteration set."""
        from webui.backend.main import _get_iteration_nums
        from src import config

        session_dir = config.EXPERIMENTS_BASE_DIR / "session_20260404_093509"
        nums = _get_iteration_nums(session_dir)
        assert nums == {1, 2}


class TestSSESessionComplete:
    """Tests for session_complete SSE event detection."""

    def test_scan_completed_sessions_initial(self, mock_experiments):
        """Already-complete sessions are recorded in known_completed."""
        from webui.backend.main import _scan_completed_sessions

        known: set[str] = set()
        _scan_completed_sessions(known)
        # mock_experiments fixture has end_time set, so session is complete
        assert "session_20260404_093509" in known

    def test_scan_completed_sessions_empty_dir(self, mock_experiments, tmp_path):
        """No crash when experiments dir doesn't exist."""
        from webui.backend.main import _scan_completed_sessions
        import src.config as config

        orig = config.EXPERIMENTS_BASE_DIR
        config.EXPERIMENTS_BASE_DIR = tmp_path / "nonexistent"
        known: set[str] = set()
        _scan_completed_sessions(known)
        assert len(known) == 0
        config.EXPERIMENTS_BASE_DIR = orig

    def test_scan_for_completed_sessions_detects_new(self, mock_experiments, tmp_path):
        """When end_time appears in session_meta.json, emit event."""
        from webui.backend.main import _scan_for_completed_sessions, _scan_completed_sessions
        import src.config as config

        base = config.EXPERIMENTS_BASE_DIR

        # Create a second session that is still running (end_time is null)
        running_session = base / "session_20260404_100000"
        running_session.mkdir(parents=True)
        (running_session / "session_meta.json").write_text(json.dumps({
            "session_id": "session_20260404_100000",
            "start_time": "2026-04-04T10:00:00+00:00",
            "end_time": None,
            "stop_reason": None,
            "best_iteration": None,
            "final_score": None,
            "n_iterations": 1,
        }))
        iter1 = running_session / "iteration_001"
        iter1.mkdir()
        (iter1 / "metadata.json").write_text(json.dumps({
            "iteration": 1,
            "timestamp": "2026-04-04T10:01:00+00:00",
            "status": "accepted",
        }))

        # Initial scan: the first session (from fixture) is complete; new session is running
        known: set[str] = set()
        _scan_completed_sessions(known)
        assert "session_20260404_093509" in known
        assert "session_20260404_100000" not in known

        # First poll: running session is not yet complete
        events = _scan_for_completed_sessions(known)
        assert len(events) == 0

        # Now simulate the autocorrect loop finishing
        (running_session / "session_meta.json").write_text(json.dumps({
            "session_id": "session_20260404_100000",
            "start_time": "2026-04-04T10:00:00+00:00",
            "end_time": "2026-04-04T10:15:00+00:00",
            "stop_reason": "target_reached",
            "best_iteration": 1,
            "final_score": 0.95,
            "n_iterations": 1,
        }))

        # Second poll: should detect the completion
        events = _scan_for_completed_sessions(known)
        assert len(events) == 1
        assert events[0]["session_id"] == "session_20260404_100000"
        assert events[0]["stop_reason"] == "target_reached"
        assert events[0]["best_iteration"] == 1
        assert events[0]["final_score"] == 0.95
        assert events[0]["n_iterations"] == 1

        # Third poll: should not re-emit (now in known_completed)
        events = _scan_for_completed_sessions(known)
        assert len(events) == 0
        assert "session_20260404_100000" in known

    def test_scan_for_completed_sessions_handles_corrupt_meta(self, mock_experiments):
        """Corrupt session_meta.json is skipped without crashing."""
        from webui.backend.main import _scan_for_completed_sessions
        import src.config as config

        base = config.EXPERIMENTS_BASE_DIR
        corrupt_session = base / "session_20260404_110000"
        corrupt_session.mkdir(parents=True)
        (corrupt_session / "session_meta.json").write_text("NOT JSON{{{")

        known: set[str] = set()
        events = _scan_for_completed_sessions(known)
        # Should not crash; corrupt session is skipped
        assert "session_20260404_110000" not in known
