"""Tests for src/evaluate.py -- VLM evaluation pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import config


# ---------------------------------------------------------------------------
# Category 1: Pydantic Model Validation
# ---------------------------------------------------------------------------

class TestPydanticModels:
    """Tests for VLMEvaluation, PerClassScore, ErrorRegion models."""

    def test_vlm_evaluation_valid(self):
        """Valid data round-trips through model_dump / model_validate."""
        from src.evaluate import _ensure_models
        PerClassScore, ErrorRegion, VLMEvaluation = _ensure_models()

        data = {
            "overall_score": 7,
            "per_class": [{"class_name": "Built-up", "score": 8.0, "notes": "Good"}],
            "error_regions": [
                {"location": "NW corner", "expected": "Tree cover",
                 "predicted": "Grassland", "severity": "high"}
            ],
            "spatial_quality": "Sharp edges",
            "confidence": 0.85,
            "recommendations": ["More data"],
        }
        obj = VLMEvaluation.model_validate(data)
        dumped = obj.model_dump()
        assert dumped["overall_score"] == 7
        assert dumped["confidence"] == 0.85
        assert len(dumped["per_class"]) == 1
        assert dumped["per_class"][0]["score"] == 8.0

    def test_vlm_evaluation_score_clamping(self):
        """overall_score=0 clamped to 1, overall_score=11 clamped to 10."""
        from src.evaluate import _ensure_models
        _, _, VLMEvaluation = _ensure_models()

        low = VLMEvaluation.model_validate({"overall_score": 0})
        assert low.overall_score == 1

        high = VLMEvaluation.model_validate({"overall_score": 11})
        assert high.overall_score == 10

    def test_vlm_evaluation_confidence_clamping(self):
        """confidence=-0.5 clamped to 0.0, confidence=1.5 clamped to 1.0."""
        from src.evaluate import _ensure_models
        _, _, VLMEvaluation = _ensure_models()

        low = VLMEvaluation.model_validate({"confidence": -0.5})
        assert low.confidence == 0.0

        high = VLMEvaluation.model_validate({"confidence": 1.5})
        assert high.confidence == 1.0

    def test_error_region_severity_normalization(self):
        """'HIGH' -> 'high', 'unknown' -> 'medium'."""
        from src.evaluate import _ensure_models
        _, ErrorRegion, _ = _ensure_models()

        region_high = ErrorRegion.model_validate(
            {"location": "NW", "expected": "Tree", "predicted": "Grass", "severity": "HIGH"}
        )
        assert region_high.severity == "high"

        region_unknown = ErrorRegion.model_validate(
            {"location": "NW", "expected": "Tree", "predicted": "Grass", "severity": "unknown"}
        )
        assert region_unknown.severity == "medium"

    def test_per_class_score_float(self):
        """score=7.5 accepted as float, score=0 clamped to 1.0."""
        from src.evaluate import _ensure_models
        PerClassScore, _, _ = _ensure_models()

        normal = PerClassScore.model_validate({"class_name": "Built-up", "score": 7.5})
        assert normal.score == 7.5

        clamped = PerClassScore.model_validate({"class_name": "Water", "score": 0})
        assert clamped.score == 1.0


# ---------------------------------------------------------------------------
# Category 2: Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Tests for _validate_input_files."""

    def test_validate_input_files_all_present(self, mock_rasterio_dataset):
        """All files exist, returns path dict with 5 keys."""
        from src.evaluate import _validate_input_files

        band_data = np.zeros((512, 512), dtype=np.float32)
        mock_ds = mock_rasterio_dataset(band_data, height=512, width=512)

        with patch("src.evaluate.Path.exists", return_value=True), \
             patch("rasterio.open", return_value=mock_ds):
            # Need to patch it at the module where it's imported
            result = _validate_input_files("2021")
            assert len(result) == 5
            assert "B04" in result
            assert "worldcover" in result

    def test_validate_input_files_multiple_missing(self, tmp_path):
        """2+ files missing, error message lists ALL."""
        from src.evaluate import _validate_input_files

        # Point config at tmp_path so no files exist
        with patch.object(config, "SENTINEL2_DIR", tmp_path / "s2"), \
             patch.object(config, "OUTPUT_DIR", tmp_path / "out"), \
             patch.object(config, "WORLDCOVER_DIR", tmp_path / "wc"):
            with pytest.raises(FileNotFoundError, match="Missing required input files"):
                _validate_input_files("2021")

    def test_validate_input_files_wrong_shape(self, mock_rasterio_dataset):
        """256x256 GeoTIFF raises ValueError."""
        from src.evaluate import _validate_input_files

        band_data = np.zeros((256, 256), dtype=np.float32)
        mock_ds = mock_rasterio_dataset(band_data, height=256, width=256)

        with patch("src.evaluate.Path.exists", return_value=True), \
             patch("rasterio.open", return_value=mock_ds):
            with pytest.raises(ValueError, match="Input file validation failed"):
                _validate_input_files("2021")

    def test_validate_input_files_worldcover_shape_ok(self, mock_rasterio_dataset):
        """WorldCover with non-512 dimensions does NOT raise."""
        from src.evaluate import _validate_input_files

        # All non-worldcover files are 512x512
        band_ds_512 = mock_rasterio_dataset(
            np.zeros((512, 512), dtype=np.float32), height=512, width=512
        )
        # WorldCover is 480x480 -- should NOT cause an error
        wc_ds_480 = mock_rasterio_dataset(
            np.zeros((480, 480), dtype=np.float32), height=480, width=480
        )

        call_count = 0

        def mock_open(path, *args, **kwargs):
            nonlocal call_count
            path_str = str(path)
            if "worldcover" in path_str:
                return wc_ds_480
            return band_ds_512

        with patch("src.evaluate.Path.exists", return_value=True), \
             patch("rasterio.open", side_effect=mock_open):
            result = _validate_input_files("2021")
            assert "worldcover" in result


# ---------------------------------------------------------------------------
# Category 3: Data Loading
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Tests for _load_rgb, _load_classification, _colorize_classes."""

    def test_load_rgb_normal(self, mock_rasterio_dataset):
        """Returns shape (512, 512, 3), dtype uint8, values in [0, 255]."""
        from src.evaluate import _load_rgb

        band_data = np.full((512, 512), 1500.0, dtype=np.float32)
        mock_ds = mock_rasterio_dataset(band_data)

        with patch("rasterio.open", return_value=mock_ds):
            result = _load_rgb("2021")
            assert result.shape == (512, 512, 3)
            assert result.dtype == np.uint8
            assert result.min() >= 0
            assert result.max() <= 255

    def test_load_rgb_nan_handling(self, mock_rasterio_dataset, capsys):
        """NaN replaced with 0, warning printed."""
        from src.evaluate import _load_rgb

        band_data = np.full((512, 512), 1500.0, dtype=np.float32)
        band_data[0, 0] = np.nan
        mock_ds = mock_rasterio_dataset(band_data)

        with patch("rasterio.open", return_value=mock_ds):
            result = _load_rgb("2021")
            captured = capsys.readouterr()
            assert "NaN" in captured.out or "nan" in captured.out.lower()
            assert result[0, 0, 0] == 0

    def test_load_classification_valid(self, mock_rasterio_dataset):
        """Values 0-5 unchanged."""
        from src.evaluate import _load_classification

        data = np.random.randint(0, 6, (512, 512), dtype=np.uint8)
        mock_ds = mock_rasterio_dataset(data)

        with patch("rasterio.open", return_value=mock_ds):
            result = _load_classification("2021")
            np.testing.assert_array_equal(result, data)

    def test_load_classification_invalid_values(self, mock_rasterio_dataset, capsys):
        """Value 7 mapped to 5 with warning."""
        from src.evaluate import _load_classification

        data = np.full((512, 512), 7, dtype=np.uint8)
        mock_ds = mock_rasterio_dataset(data)

        with patch("rasterio.open", return_value=mock_ds):
            result = _load_classification("2021")
            captured = capsys.readouterr()
            assert "WARNING" in captured.out
            assert (result == 5).all()

    def test_colorize_classes(self):
        """Output colors match config.LANDCOVER_COLORS hex values."""
        from src.evaluate import _colorize_classes

        # Create a 4x4 array with all 6 classes
        data = np.array([
            [0, 1, 2, 3],
            [4, 5, 0, 1],
            [2, 3, 4, 5],
            [0, 1, 2, 3],
        ], dtype=np.uint8)

        result = _colorize_classes(data)
        assert result.shape == (4, 4, 3)
        assert result.dtype == np.uint8

        # Verify specific colors
        # Class 0 = #FF0000 = (255, 0, 0)
        assert list(result[0, 0]) == [255, 0, 0]
        # Class 1 = #FFFF00 = (255, 255, 0)
        assert list(result[0, 1]) == [255, 255, 0]
        # Class 4 = #0000FF = (0, 0, 255)
        assert list(result[0, 3]) == [0, 100, 0]  # class 3 = #006400


# ---------------------------------------------------------------------------
# Category 4: Image Generation
# ---------------------------------------------------------------------------

class TestImageGeneration:
    """Tests for _extract_quadrant, _generate_comparison_image, etc."""

    def test_extract_quadrant_all_four(self):
        """For 512x512 input: each quadrant is (256, 256)."""
        from src.evaluate import _extract_quadrant

        arr_2d = np.zeros((512, 512), dtype=np.uint8)
        arr_3d = np.zeros((512, 512, 3), dtype=np.uint8)

        for quadrant in ["nw", "ne", "sw", "se"]:
            q2d = _extract_quadrant(arr_2d, quadrant)
            assert q2d.shape == (256, 256), f"{quadrant} 2D shape: {q2d.shape}"

            q3d = _extract_quadrant(arr_3d, quadrant)
            assert q3d.shape == (256, 256, 3), f"{quadrant} 3D shape: {q3d.shape}"

    def test_extract_quadrant_invalid_name(self):
        """_extract_quadrant(arr, 'xx') raises ValueError."""
        from src.evaluate import _extract_quadrant

        arr = np.zeros((512, 512), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid quadrant"):
            _extract_quadrant(arr, "xx")

    def test_generate_comparison_image(self, tmp_path, sample_rgb, sample_classification):
        """Verify PNG file created at expected path, tmp file removed."""
        from src.evaluate import _generate_comparison_image, _colorize_classes

        cls_rgb = _colorize_classes(sample_classification)
        # Use classification as stand-in for worldcover
        wc_rgb = _colorize_classes(sample_classification)

        output_path = tmp_path / "test_comparison.png"
        result = _generate_comparison_image(
            sample_rgb, cls_rgb, wc_rgb, "Test Title", output_path
        )

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        # Verify tmp file was cleaned up
        tmp_file = output_path.with_name(output_path.name + ".tmp")
        assert not tmp_file.exists()

    def test_generate_comparison_images_returns_five(self, tmp_path):
        """Returns list of 5 Path objects with expected filenames."""
        from src.evaluate import generate_comparison_images

        rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        cls_data = np.zeros((512, 512), dtype=np.uint8)
        wc_data = np.zeros((512, 512), dtype=np.uint8)

        with patch("src.evaluate._load_rgb", return_value=rgb), \
             patch("src.evaluate._load_classification", return_value=cls_data), \
             patch("src.evaluate._load_worldcover_labels", return_value=wc_data):
            paths = generate_comparison_images("2021", tmp_path)

        assert len(paths) == 5
        names = [p.name for p in paths]
        assert "comparison_2021_full.png" in names
        assert "comparison_2021_nw.png" in names
        assert "comparison_2021_ne.png" in names
        assert "comparison_2021_sw.png" in names
        assert "comparison_2021_se.png" in names
        for p in paths:
            assert p.exists()

    def test_legend_color_coverage(self):
        """Every key in config.LANDCOVER_COLORS has a mapping in _build_legend_text."""
        from src.evaluate import _build_legend_text

        legend = _build_legend_text()

        # All class names should appear in legend
        for idx, name in config.LANDCOVER_CLASSES.items():
            assert name in legend, f"Class '{name}' (idx={idx}) missing from legend"

        # All colors should be represented (not hex codes, but color names)
        # If a color has no name mapping, its hex code should appear
        color_names = {
            "#FF0000": "red", "#FFFF00": "yellow", "#90EE90": "light green",
            "#006400": "dark green", "#0000FF": "blue", "#808080": "gray",
        }
        for idx in config.LANDCOVER_COLORS:
            hex_color = config.LANDCOVER_COLORS[idx]
            cname = color_names.get(hex_color, hex_color)
            assert cname in legend, f"Color '{cname}' for class {idx} missing from legend"


# ---------------------------------------------------------------------------
# Category 5: Gemini API
# ---------------------------------------------------------------------------

class TestGeminiAPI:
    """Tests for _call_gemini and related functions."""

    def test_call_gemini_success(self, mock_gemini_response):
        """Valid response parsed to VLMEvaluation."""
        from src.evaluate import _call_gemini, _ensure_models
        _ensure_models()

        response_data = mock_gemini_response()
        mock_response = MagicMock()
        mock_response.text = json.dumps(response_data)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        # Create dummy image files
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "test.png"
            # Create a minimal valid PNG
            _create_test_png(img_path)

            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
                 patch("google.genai.Client", return_value=mock_client):
                result = _call_gemini([img_path], "test prompt")

            assert result.overall_score == 7
            assert result.confidence == 0.8

    def test_call_gemini_retry_on_transient(self, mock_gemini_response):
        """Fail twice then succeed, verify 3 attempts."""
        from src.evaluate import _call_gemini, _ensure_models
        _ensure_models()

        response_data = mock_gemini_response()
        mock_response = MagicMock()
        mock_response.text = json.dumps(response_data)

        mock_client = MagicMock()
        # Fail twice with transient error, succeed on third
        mock_client.models.generate_content.side_effect = [
            Exception("connection timeout"),
            Exception("server overloaded"),
            mock_response,
        ]

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "test.png"
            _create_test_png(img_path)

            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
                 patch("google.genai.Client", return_value=mock_client), \
                 patch("time.sleep"):  # Don't actually sleep
                result = _call_gemini([img_path], "test prompt")

            assert result.overall_score == 7
            assert mock_client.models.generate_content.call_count == 3

    def test_call_gemini_no_retry_on_auth(self):
        """401 error: immediate RuntimeError, only 1 API call."""
        from src.evaluate import _call_gemini, _ensure_models
        _ensure_models()

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("401 Unauthorized")

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "test.png"
            _create_test_png(img_path)

            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
                 patch("google.genai.Client", return_value=mock_client):
                with pytest.raises(RuntimeError, match="non-retryable"):
                    _call_gemini([img_path], "test prompt")

            assert mock_client.models.generate_content.call_count == 1

    def test_call_gemini_fallback_to_single(self, mock_gemini_response):
        """Multi-image fails, single-image succeeds."""
        from src.evaluate import _call_gemini, _ensure_models
        _ensure_models()

        response_data = mock_gemini_response()
        mock_response = MagicMock()
        mock_response.text = json.dumps(response_data)

        mock_client = MagicMock()
        # Fail all multi-image attempts, succeed on fallback
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            contents = kwargs.get("contents", args[0] if args else [])
            # Fallback sends only 1 image + prompt (2 items)
            if len(contents) <= 2:
                return mock_response
            raise Exception("too many images")

        mock_client.models.generate_content.side_effect = side_effect

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i in range(5):
                p = Path(td) / f"test_{i}.png"
                _create_test_png(p)
                paths.append(p)

            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
                 patch("google.genai.Client", return_value=mock_client), \
                 patch("time.sleep"):
                result = _call_gemini(paths, "test prompt")

            assert result.overall_score == 7

    def test_call_gemini_missing_key(self):
        """No GOOGLE_API_KEY env var: RuntimeError."""
        from src.evaluate import _call_gemini, _ensure_models
        _ensure_models()

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
                _call_gemini([Path("/fake.png")], "test")


# ---------------------------------------------------------------------------
# Category 6: Orchestration
# ---------------------------------------------------------------------------

class TestOrchestration:
    """Tests for run_evaluation, _build_evaluation_prompt, and imports."""

    def test_run_evaluation_no_api_key(self, tmp_path):
        """Images generated, VLM skipped, returns None, no exception."""
        from src.evaluate import run_evaluation

        # Create fake landcover files
        lc_2021 = tmp_path / "output" / "landcover_2021.tif"
        lc_2021.parent.mkdir(parents=True, exist_ok=True)
        lc_2021.touch()

        image_paths = [tmp_path / f"img_{i}.png" for i in range(5)]

        with patch.object(config, "OUTPUT_DIR", tmp_path / "output"), \
             patch.object(config, "TIME_RANGES", {"2021": ("2021-06-01", "2021-08-31")}), \
             patch("src.evaluate._validate_input_files"), \
             patch("src.evaluate.generate_comparison_images", return_value=image_paths), \
             patch.dict(os.environ, {}, clear=True):
            result = run_evaluation(output_dir=tmp_path)

        assert result is None

    def test_run_evaluation_no_classification(self, tmp_path):
        """No landcover files: FileNotFoundError."""
        from src.evaluate import run_evaluation

        with patch.object(config, "OUTPUT_DIR", tmp_path / "output"), \
             patch.object(config, "TIME_RANGES", {"2021": ("2021-06-01", "2021-08-31")}):
            with pytest.raises(FileNotFoundError, match="No classification outputs"):
                run_evaluation(output_dir=tmp_path)

    def test_run_evaluation_sdk_not_installed(self, tmp_path):
        """ImportError on import google.genai: graceful skip, returns None."""
        from src.evaluate import run_evaluation
        import builtins

        # Create fake landcover file
        lc_2021 = tmp_path / "output" / "landcover_2021.tif"
        lc_2021.parent.mkdir(parents=True, exist_ok=True)
        lc_2021.touch()

        image_paths = [tmp_path / f"img_{i}.png" for i in range(5)]

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "google.genai":
                raise ImportError("No module named 'google.genai'")
            return original_import(name, *args, **kwargs)

        with patch.object(config, "OUTPUT_DIR", tmp_path / "output"), \
             patch.object(config, "TIME_RANGES", {"2021": ("2021-06-01", "2021-08-31")}), \
             patch("src.evaluate._validate_input_files"), \
             patch("src.evaluate.generate_comparison_images", return_value=image_paths), \
             patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
             patch("builtins.__import__", side_effect=mock_import):
            result = run_evaluation(output_dir=tmp_path)

        assert result is None

    def test_build_evaluation_prompt(self):
        """Contains 'Exeter', legend text, year, class definitions, '5 comparison images'."""
        from src.evaluate import _build_evaluation_prompt

        prompt = _build_evaluation_prompt("2021")
        assert "Exeter" in prompt
        assert "Devon" in prompt
        assert "2021" in prompt
        assert "5 comparison images" in prompt
        assert "Built-up" in prompt
        assert "Cropland" in prompt
        assert "Tree cover" in prompt
        assert "Class definitions" in prompt

        # 2023 should include temporal mismatch note
        prompt_2023 = _build_evaluation_prompt("2023")
        assert "IMPORTANT" in prompt_2023
        assert "2023" in prompt_2023
        assert "WorldCover reference" in prompt_2023

    def test_pipeline_import(self):
        """from src.evaluate import run_evaluation succeeds."""
        from src.evaluate import run_evaluation
        assert callable(run_evaluation)


# ---------------------------------------------------------------------------
# Test Helper
# ---------------------------------------------------------------------------

def _create_test_png(path: Path):
    """Create a minimal valid PNG file for testing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png")
    plt.close(fig)
