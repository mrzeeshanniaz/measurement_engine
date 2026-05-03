"""
Tests for measurement_engine.utils.
"""

import math

import pytest

from measurement_engine.utils import (
    distance_2d,
    estimate_circumference,
    midpoint,
    pixel_to_cm,
)


class TestDistance2D:
    def test_horizontal_distance(self):
        assert distance_2d((0, 0), (5, 0)) == pytest.approx(5.0)

    def test_vertical_distance(self):
        assert distance_2d((0, 0), (0, 3)) == pytest.approx(3.0)

    def test_pythagorean_triple(self):
        # 3-4-5 right triangle
        assert distance_2d((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_same_point_is_zero(self):
        assert distance_2d((7, 7), (7, 7)) == pytest.approx(0.0)

    def test_float_coordinates(self):
        result = distance_2d((0.5, 0.5), (1.5, 1.5))
        assert result == pytest.approx(math.sqrt(2))

    def test_negative_coordinates(self):
        assert distance_2d((-1, -1), (2, 3)) == pytest.approx(5.0)


class TestMidpoint:
    def test_basic_midpoint(self):
        assert midpoint((0, 0), (4, 4)) == (2.0, 2.0)

    def test_float_midpoint(self):
        assert midpoint((1, 1), (2, 2)) == (1.5, 1.5)

    def test_same_points(self):
        assert midpoint((3, 7), (3, 7)) == (3.0, 7.0)

    def test_asymmetric(self):
        mx, my = midpoint((0, 0), (10, 20))
        assert mx == pytest.approx(5.0)
        assert my == pytest.approx(10.0)


class TestPixelToCm:
    def test_basic_conversion(self):
        assert pixel_to_cm(100, 0.5) == pytest.approx(50.0)

    def test_zero_pixels(self):
        assert pixel_to_cm(0, 0.5) == pytest.approx(0.0)

    def test_identity_scale(self):
        assert pixel_to_cm(42, 1.0) == pytest.approx(42.0)

    def test_fractional_scale(self):
        assert pixel_to_cm(200, 0.175) == pytest.approx(35.0)


class TestEstimateCircumference:
    def test_circle_case(self):
        # When depth_ratio == 1.0 the ellipse degenerates to a circle.
        # Circumference of circle with diameter *width* = π * width.
        width = 10.0
        result = estimate_circumference(width, depth_ratio=1.0)
        assert result == pytest.approx(math.pi * width, rel=1e-3)

    def test_positive_result(self):
        assert estimate_circumference(40.0, depth_ratio=0.70) > 0

    def test_wider_gives_larger_circumference(self):
        wide = estimate_circumference(50.0, depth_ratio=0.70)
        narrow = estimate_circumference(30.0, depth_ratio=0.70)
        assert wide > narrow

    def test_higher_depth_ratio_gives_larger_circumference(self):
        round_body = estimate_circumference(40.0, depth_ratio=0.90)
        flat_body = estimate_circumference(40.0, depth_ratio=0.50)
        assert round_body > flat_body

    def test_realistic_chest_range(self):
        # Chest width ~40 cm with depth_ratio 0.65 → circumference ~100 cm
        result = estimate_circumference(40.0, depth_ratio=0.65)
        assert 85 < result < 125

    def test_realistic_hip_range(self):
        # Hip width ~36 cm with depth_ratio 0.70 → realistic hip circumference
        result = estimate_circumference(36.0, depth_ratio=0.70)
        assert 80 < result < 120
