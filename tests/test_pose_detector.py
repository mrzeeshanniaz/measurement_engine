"""
Tests for pose_detector module (data-classes and PoseResult helpers).

MediaPipe is NOT imported here; tests exercise only the pure-Python
data structures.
"""

import pytest

from measurement_engine.pose_detector import Landmark, PoseResult


def _make_pose_result() -> PoseResult:
    """Return a minimal PoseResult with a handful of landmarks."""
    landmarks = {
        0: Landmark(x=0.50, y=0.05, z=0.00, visibility=0.95),   # Nose
        11: Landmark(x=0.35, y=0.25, z=0.00, visibility=0.90),  # Left shoulder
        12: Landmark(x=0.65, y=0.25, z=0.00, visibility=0.90),  # Right shoulder
        23: Landmark(x=0.38, y=0.55, z=0.00, visibility=0.85),  # Left hip
        24: Landmark(x=0.62, y=0.55, z=0.00, visibility=0.85),  # Right hip
    }
    return PoseResult(landmarks=landmarks, image_width=400, image_height=600)


class TestLandmark:
    def test_fields_stored(self):
        lm = Landmark(x=0.3, y=0.7, z=-0.1, visibility=0.8)
        assert lm.x == 0.3
        assert lm.y == 0.7
        assert lm.z == -0.1
        assert lm.visibility == 0.8


class TestPoseResultPixelCoords:
    def test_nose_pixel_coords(self):
        pose = _make_pose_result()
        x, y = pose.get_pixel_coords(0)
        assert x == pytest.approx(400 * 0.50)
        assert y == pytest.approx(600 * 0.05)

    def test_left_shoulder_pixel_coords(self):
        pose = _make_pose_result()
        x, y = pose.get_pixel_coords(11)
        assert x == pytest.approx(400 * 0.35)
        assert y == pytest.approx(600 * 0.25)

    def test_right_hip_pixel_coords(self):
        pose = _make_pose_result()
        x, y = pose.get_pixel_coords(24)
        assert x == pytest.approx(400 * 0.62)
        assert y == pytest.approx(600 * 0.55)


class TestPoseResultIsVisible:
    def test_visible_above_default_threshold(self):
        pose = _make_pose_result()
        assert pose.is_visible(11) is True   # visibility 0.90 > 0.50

    def test_not_visible_above_strict_threshold(self):
        pose = _make_pose_result()
        assert pose.is_visible(11, threshold=0.95) is False  # 0.90 < 0.95

    def test_missing_landmark_not_visible(self):
        pose = _make_pose_result()
        assert pose.is_visible(99) is False

    def test_exact_threshold_boundary(self):
        pose = _make_pose_result()
        # visibility == 0.95 exactly for landmark 0
        assert pose.is_visible(0, threshold=0.95) is True
        assert pose.is_visible(0, threshold=0.96) is False
