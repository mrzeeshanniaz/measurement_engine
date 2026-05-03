"""
Tests for MeasurementEngine.

MediaPipe is mocked so that no GPU/model download is required during CI.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from measurement_engine import MeasurementEngine, MeasurementResult
from measurement_engine.pose_detector import Landmark, PoseResult


# ------------------------------------------------------------------ #
# Shared fixtures                                                       #
# ------------------------------------------------------------------ #

def _make_mock_pose(
    image_width: int = 480, image_height: int = 640
) -> PoseResult:
    """Synthetic full-body PoseResult used across engine tests."""
    landmarks = {
        0:  Landmark(x=0.50, y=0.05, z=0.00, visibility=0.95),
        11: Landmark(x=0.38, y=0.25, z=0.00, visibility=0.95),
        12: Landmark(x=0.62, y=0.25, z=0.00, visibility=0.95),
        13: Landmark(x=0.32, y=0.40, z=0.00, visibility=0.90),
        14: Landmark(x=0.68, y=0.40, z=0.00, visibility=0.90),
        15: Landmark(x=0.30, y=0.55, z=0.00, visibility=0.85),
        16: Landmark(x=0.70, y=0.55, z=0.00, visibility=0.85),
        23: Landmark(x=0.41, y=0.56, z=0.00, visibility=0.90),
        24: Landmark(x=0.59, y=0.56, z=0.00, visibility=0.90),
        25: Landmark(x=0.42, y=0.72, z=0.00, visibility=0.85),
        26: Landmark(x=0.58, y=0.72, z=0.00, visibility=0.85),
        27: Landmark(x=0.43, y=0.90, z=0.00, visibility=0.80),
        28: Landmark(x=0.57, y=0.90, z=0.00, visibility=0.80),
    }
    return PoseResult(
        landmarks=landmarks,
        image_width=image_width,
        image_height=image_height,
    )


@pytest.fixture()
def engine():
    return MeasurementEngine()


@pytest.fixture()
def blank_image():
    return np.zeros((640, 480, 3), dtype=np.uint8)


# ------------------------------------------------------------------ #
# Instantiation                                                         #
# ------------------------------------------------------------------ #

class TestMeasurementEngineInit:
    def test_default_instantiation(self):
        eng = MeasurementEngine()
        assert eng is not None

    def test_custom_confidence(self):
        eng = MeasurementEngine(min_detection_confidence=0.7)
        assert eng._detector.min_detection_confidence == 0.7

    def test_custom_tracking_confidence(self):
        eng = MeasurementEngine(min_tracking_confidence=0.8)
        assert eng._detector.min_tracking_confidence == 0.8


# ------------------------------------------------------------------ #
# analyze() – success paths                                            #
# ------------------------------------------------------------------ #

class TestMeasurementEngineAnalyze:
    @patch("measurement_engine.pose_detector.PoseDetector.detect")
    def test_returns_measurement_result(self, mock_detect, engine, blank_image):
        mock_detect.return_value = _make_mock_pose()
        result = engine.analyze(blank_image)
        assert isinstance(result, MeasurementResult)

    @patch("measurement_engine.pose_detector.PoseDetector.detect")
    def test_pixel_mode_without_height(self, mock_detect, engine, blank_image):
        mock_detect.return_value = _make_mock_pose()
        result = engine.analyze(blank_image)
        assert result.units == "pixels"

    @patch("measurement_engine.pose_detector.PoseDetector.detect")
    def test_cm_mode_with_height(self, mock_detect, engine, blank_image):
        mock_detect.return_value = _make_mock_pose()
        result = engine.analyze(blank_image, person_height_cm=175.0)
        assert result.units == "cm"
        assert result.scale_factor is not None

    @patch("measurement_engine.pose_detector.PoseDetector.detect")
    def test_confidence_above_zero(self, mock_detect, engine, blank_image):
        mock_detect.return_value = _make_mock_pose()
        result = engine.analyze(blank_image)
        assert result.confidence > 0.0

    @patch("measurement_engine.pose_detector.PoseDetector.detect")
    def test_shoulder_width_present(self, mock_detect, engine, blank_image):
        mock_detect.return_value = _make_mock_pose()
        result = engine.analyze(blank_image)
        assert result.shoulder_width is not None

    @patch("measurement_engine.pose_detector.PoseDetector.detect")
    def test_accepts_path_object(self, mock_detect, engine, tmp_path):
        # Write a tiny valid JPEG so load_image succeeds
        import cv2  # noqa: PLC0415

        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        mock_detect.return_value = _make_mock_pose(100, 100)
        result = engine.analyze(img_path)
        assert isinstance(result, MeasurementResult)


# ------------------------------------------------------------------ #
# analyze() – error paths                                              #
# ------------------------------------------------------------------ #

class TestMeasurementEngineAnalyzeErrors:
    @patch("measurement_engine.pose_detector.PoseDetector.detect")
    def test_no_person_raises_value_error(self, mock_detect, engine, blank_image):
        mock_detect.return_value = None
        with pytest.raises(ValueError, match="No person detected"):
            engine.analyze(blank_image)

    def test_missing_file_raises_file_not_found(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.analyze("/nonexistent/path/image.jpg")

    def test_string_path_missing_raises_file_not_found(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.analyze("does_not_exist.png")


# ------------------------------------------------------------------ #
# CLI                                                                   #
# ------------------------------------------------------------------ #

class TestCLI:
    def test_no_command_exits_zero(self):
        from measurement_engine.cli import main  # noqa: PLC0415

        assert main([]) == 0

    @patch("measurement_engine.engine.MeasurementEngine.analyze")
    def test_analyze_command_success(self, mock_analyze, tmp_path):
        import cv2  # noqa: PLC0415
        from measurement_engine.cli import main  # noqa: PLC0415

        img_path = tmp_path / "person.jpg"
        cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

        mock_analyze.return_value = MeasurementResult(
            shoulder_width=45.0,
            units="pixels",
            confidence=0.9,
        )
        rc = main(["analyze", str(img_path)])
        assert rc == 0

    def test_analyze_missing_image_exits_one(self):
        from measurement_engine.cli import main  # noqa: PLC0415

        rc = main(["analyze", "/no/such/file.jpg"])
        assert rc == 1
