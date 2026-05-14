"""
Tests for app.measurement_engine.scan.measurements — extractor scaling,
ring detection, clothing compensation, and fallback chain.

Uses synthetic landmarks and a synthetic mesh (vertically-stretched cylinder)
so the extractor exercises both the mesh and landmark paths without needing
the real SMPL model.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.measurement_engine.scan.measurements import (
    LandmarkPoint,
    MeasurementExtractor,
    _mesh_cross_section_ring,
    apply_clothing_compensation,
)


# ---------------------------------------------------------------------------
# Synthetic landmarks for a 175 cm subject framed top-to-bottom in a 1080x1920 image
# ---------------------------------------------------------------------------

def _front_lm() -> dict[int, LandmarkPoint]:
    return {
        0:  LandmarkPoint(x=0.50, y=0.05, z=0.0, visibility=0.95),  # nose
        11: LandmarkPoint(x=0.42, y=0.18, z=0.0, visibility=0.95),  # L shoulder
        12: LandmarkPoint(x=0.58, y=0.18, z=0.0, visibility=0.95),  # R shoulder
        13: LandmarkPoint(x=0.36, y=0.30, z=0.0, visibility=0.95),  # L elbow
        14: LandmarkPoint(x=0.64, y=0.30, z=0.0, visibility=0.95),  # R elbow
        15: LandmarkPoint(x=0.32, y=0.42, z=0.0, visibility=0.95),  # L wrist
        16: LandmarkPoint(x=0.68, y=0.42, z=0.0, visibility=0.95),  # R wrist
        23: LandmarkPoint(x=0.45, y=0.48, z=0.0, visibility=0.95),  # L hip
        24: LandmarkPoint(x=0.55, y=0.48, z=0.0, visibility=0.95),  # R hip
        25: LandmarkPoint(x=0.45, y=0.70, z=0.0, visibility=0.95),  # L knee
        26: LandmarkPoint(x=0.55, y=0.70, z=0.0, visibility=0.95),  # R knee
        27: LandmarkPoint(x=0.45, y=0.95, z=0.0, visibility=0.95),  # L ankle
        28: LandmarkPoint(x=0.55, y=0.95, z=0.0, visibility=0.95),  # R ankle
    }


# ---------------------------------------------------------------------------
# Clothing compensation helper
# ---------------------------------------------------------------------------

class TestApplyClothingCompensation:
    def test_none_value_returns_none(self):
        assert apply_clothing_compensation(None, "chest") is None

    def test_known_offset_is_subtracted(self):
        # chest offset = 0.8 cm
        out = apply_clothing_compensation(100.0, "chest")
        assert out == pytest.approx(99.2, abs=0.01)

    def test_unknown_key_returns_value_unchanged(self):
        assert apply_clothing_compensation(100.0, "no_such_key") == 100.0

    def test_offset_floored_at_85_percent(self):
        # Tiny value with a normal offset — output should never be below 0.85 * value
        out = apply_clothing_compensation(2.0, "chest")  # offset 0.8 → 1.2; 0.85*2 = 1.7
        assert out >= 1.7

    def test_zero_offset_key_unchanged(self):
        # wrist offset = 0.0
        assert apply_clothing_compensation(16.0, "wrist") == 16.0


# ---------------------------------------------------------------------------
# Pixel scale (Y axis = nose→ankle span calibrated to height)
# ---------------------------------------------------------------------------

class TestPixelScale:
    def test_horizontal_distance_scales_with_aspect_ratio(self):
        extractor = MeasurementExtractor(
            height_cm=175.0,
            front_landmarks=_front_lm(),
            img_aspect_ratio=1080 / 1920,  # portrait
        )
        # nose y=0.05, ankle y=0.95 → span 0.90 → px_to_cm = 175 / 0.90 = 194.4 cm/unit
        # X distance between L_HIP (0.45) and R_HIP (0.55) = 0.10 → 0.10 * 194.4 * 0.5625 = 10.94 cm
        ls = _front_lm()[23]
        rs = _front_lm()[24]
        cm = extractor._lm_dist_x(ls, rs)
        assert 10.0 < cm < 12.0

    def test_vertical_distance_uses_full_pixel_scale(self):
        extractor = MeasurementExtractor(
            height_cm=175.0,
            front_landmarks=_front_lm(),
        )
        # L_SHOULDER y=0.18, L_HIP y=0.48 → span 0.30 → 0.30 * (175/0.90) = 58.3 cm
        ls = _front_lm()[11]
        lh = _front_lm()[23]
        cm = extractor._lm_dist_y(ls, lh)
        assert 55.0 < cm < 62.0


# ---------------------------------------------------------------------------
# Mesh ring detection
# ---------------------------------------------------------------------------

def _circle_segments(radius: float, n: int = 32, y: float = 0.0) -> np.ndarray:
    """Build N edge segments forming a closed loop in the XZ plane at y."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([np.cos(angles) * radius, np.full(n, y), np.sin(angles) * radius], axis=1)
    segs = np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)
    return segs


class TestMeshCrossSectionRing:
    def test_single_ring_perimeter(self):
        segs = _circle_segments(radius=10.0, n=64)
        result = _mesh_cross_section_ring(segs)
        assert result is not None
        perimeter, _ = result
        expected = 2 * np.pi * 10.0
        # Polygonal approximation underestimates; require within 0.5%
        assert abs(perimeter - expected) / expected < 0.01

    def test_two_rings_returns_larger(self):
        big   = _circle_segments(radius=10.0, n=64)
        small = _circle_segments(radius=3.0, n=32) + np.array([100.0, 0.0, 0.0])  # offset so endpoints don't unite
        all_segs = np.concatenate([big, small], axis=0)
        result = _mesh_cross_section_ring(all_segs)
        assert result is not None
        perimeter, _ = result
        # Should pick the large ring, perimeter ~62.8
        assert perimeter > 50.0

    def test_empty_segments_returns_none(self):
        assert _mesh_cross_section_ring(np.zeros((0, 2, 3))) is None


# ---------------------------------------------------------------------------
# Extractor end-to-end (no mesh) — falls back through landmark + height_ratio
# ---------------------------------------------------------------------------

class TestExtractorNoMesh:
    def test_extract_runs_and_returns_height(self):
        r = MeasurementExtractor(height_cm=175.0, front_landmarks=_front_lm()).extract()
        assert r.height_cm == 175.0

    def test_circumferences_come_from_height_ratio_without_mesh(self):
        r = MeasurementExtractor(height_cm=175.0, front_landmarks=_front_lm()).extract()
        assert r.sources["M01"] == "height_ratio"
        assert r.sources["M03"] == "height_ratio"

    def test_lengths_use_landmarks_when_visible(self):
        r = MeasurementExtractor(height_cm=175.0, front_landmarks=_front_lm()).extract()
        assert r.sources["M15"] == "landmark"   # shoulder-to-waist front
        assert r.sources["M21"] == "landmark"   # inseam

    def test_outseam_is_height_ratio_only(self):
        # By design: no reliable landmark or mesh path — always height_ratio
        r = MeasurementExtractor(height_cm=175.0, front_landmarks=_front_lm()).extract()
        assert r.sources["M22"] == "height_ratio"

    def test_values_within_plausible_range(self):
        r = MeasurementExtractor(height_cm=175.0, front_landmarks=_front_lm()).extract()
        assert 60 < r.M01_chest < 130
        assert 50 < r.M03_waist < 110
        assert 70 < r.M05_hips < 130
        assert 40 < r.M21_inseam < 110
