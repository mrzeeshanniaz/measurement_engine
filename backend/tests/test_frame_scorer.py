"""
Tests for app.measurement_engine.scan.frame_scorer — composite formula
weights, individual dimension scorers, and SCAN-04 usability threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image

from app.measurement_engine.scan.frame_scorer import (
    FrameScorer,
    USABLE_THRESHOLD,
    _W_BLUR,
    _W_LIGHT,
    _W_POSE,
)
from app.measurement_engine.scan.schemas import PoseID


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


def _solid_image(width: int = 256, height: int = 512, color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _noisy_image(width: int = 256, height: int = 512, seed: int = 0) -> Image.Image:
    """Random-noise image — high Laplacian variance, high contrast."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _full_visibility_landmarks() -> dict[int, _LM]:
    # Front pose: shoulders and hips spread symmetrically; ankles at bottom.
    return {
        11: _LM(0.40, 0.30, visibility=0.95),  # L shoulder
        12: _LM(0.60, 0.30, visibility=0.95),  # R shoulder
        23: _LM(0.42, 0.55, visibility=0.95),  # L hip
        24: _LM(0.58, 0.55, visibility=0.95),  # R hip
        27: _LM(0.45, 0.92, visibility=0.95),  # L ankle
        28: _LM(0.55, 0.92, visibility=0.95),  # R ankle
    }


# ---------------------------------------------------------------------------
# Composite formula (SCAN-04)
# ---------------------------------------------------------------------------

class TestCompositeWeights:
    def test_weights_sum_to_one(self):
        assert _W_BLUR + _W_POSE + _W_LIGHT == pytest.approx(1.0)

    def test_usable_threshold_matches_spec(self):
        assert USABLE_THRESHOLD == 0.60

    def test_composite_is_weighted_sum_of_blur_pose_light(self):
        scorer = FrameScorer()
        # A neutral-gray, perfectly-uniform image: blur≈0, lighting=1.0.
        # Pose visibility=0.95 → pose≈0.95.
        score = scorer.score(_solid_image(), PoseID.FRONT, _full_visibility_landmarks())
        expected = (
            _W_BLUR * score.blur_score
            + _W_POSE * score.pose_confidence
            + _W_LIGHT * score.lighting_score
        )
        assert score.composite == pytest.approx(round(expected, 3), abs=0.002)

    def test_angle_and_occlusion_not_in_composite(self):
        """Composite must depend only on blur, pose, lighting — not angle/occlusion."""
        scorer = FrameScorer()
        # Same image + pose visibility — only landmark X-spread differs (changes angle_match).
        baseline_lm = _full_visibility_landmarks()
        skewed_lm  = dict(baseline_lm)
        skewed_lm[11] = _LM(0.20, 0.30, visibility=0.95)   # shoulders pulled wide
        skewed_lm[12] = _LM(0.80, 0.30, visibility=0.95)

        s1 = scorer.score(_solid_image(), PoseID.FRONT, baseline_lm)
        s2 = scorer.score(_solid_image(), PoseID.FRONT, skewed_lm)
        assert s1.composite == pytest.approx(s2.composite, abs=0.002)


# ---------------------------------------------------------------------------
# Individual dimensions
# ---------------------------------------------------------------------------

class TestBlurScore:
    def test_uniform_image_is_zero(self):
        scorer = FrameScorer()
        s = scorer.score(_solid_image(), PoseID.FRONT, _full_visibility_landmarks())
        assert s.blur_score == pytest.approx(0.0, abs=0.01)

    def test_high_noise_image_is_one(self):
        scorer = FrameScorer()
        s = scorer.score(_noisy_image(), PoseID.FRONT, _full_visibility_landmarks())
        assert s.blur_score == 1.0


class TestPoseConfidence:
    def test_no_landmarks_is_zero(self):
        scorer = FrameScorer()
        s = scorer.score(_solid_image(), PoseID.FRONT, None)
        assert s.pose_confidence == 0.0

    def test_full_visibility_is_near_one(self):
        scorer = FrameScorer()
        s = scorer.score(_solid_image(), PoseID.FRONT, _full_visibility_landmarks())
        assert s.pose_confidence == pytest.approx(0.95, abs=0.01)

    def test_partial_visibility_is_averaged(self):
        scorer = FrameScorer()
        lm = _full_visibility_landmarks()
        # Drop one shoulder's visibility — average should drop accordingly.
        lm[11] = _LM(0.40, 0.30, visibility=0.10)
        s = scorer.score(_solid_image(), PoseID.FRONT, lm)
        # Mean of [0.10, 0.95, 0.95, 0.95, 0.95, 0.95] = 0.808
        assert s.pose_confidence == pytest.approx(0.808, abs=0.01)


class TestLightingScore:
    def test_mid_grey_uniform_returns_high_contrast_band(self):
        # mean=128 is in [60,220]; std=0 fails contrast → fallback 0.6
        scorer = FrameScorer()
        s = scorer.score(_solid_image(color=(128, 128, 128)), PoseID.FRONT, _full_visibility_landmarks())
        assert s.lighting_score == pytest.approx(0.6, abs=0.001)

    def test_very_dark_image_is_low(self):
        scorer = FrameScorer()
        s = scorer.score(_solid_image(color=(10, 10, 10)), PoseID.FRONT, _full_visibility_landmarks())
        assert s.lighting_score == pytest.approx(0.2, abs=0.001)

    def test_very_bright_image_is_low(self):
        scorer = FrameScorer()
        s = scorer.score(_solid_image(color=(245, 245, 245)), PoseID.FRONT, _full_visibility_landmarks())
        assert s.lighting_score == pytest.approx(0.2, abs=0.001)

    def test_noisy_image_has_good_contrast_and_brightness(self):
        scorer = FrameScorer()
        s = scorer.score(_noisy_image(), PoseID.FRONT, _full_visibility_landmarks())
        assert s.lighting_score == 1.0


# ---------------------------------------------------------------------------
# SCAN-04 usability threshold via FrameScore.is_usable
# ---------------------------------------------------------------------------

class TestIsUsable:
    def test_noisy_well_lit_front_frame_is_usable(self):
        scorer = FrameScorer()
        s = scorer.score(_noisy_image(), PoseID.FRONT, _full_visibility_landmarks())
        assert s.is_usable

    def test_uniform_grey_front_frame_is_unusable(self):
        scorer = FrameScorer()
        s = scorer.score(_solid_image(color=(128, 128, 128)), PoseID.FRONT, _full_visibility_landmarks())
        # blur≈0, pose=0.95, lighting=0.6  →  0.0*0.3 + 0.95*0.4 + 0.6*0.3 = 0.56 < 0.60
        assert not s.is_usable

    def test_threshold_boundary(self):
        scorer = FrameScorer()
        # Construct: composite exactly at threshold by mocking dimensions via fields
        from app.measurement_engine.scan.schemas import FrameScore
        fs = FrameScore(
            pose_id=PoseID.FRONT,
            blur_score=0.6, pose_confidence=0.6, angle_match=0.0,
            occlusion_score=0.0, lighting_score=0.6, composite=0.60,
        )
        assert fs.is_usable
        fs2 = fs.model_copy(update={"composite": 0.599})
        assert not fs2.is_usable


# ---------------------------------------------------------------------------
# Occlusion (diagnostic only; not in composite)
# ---------------------------------------------------------------------------

class TestOcclusion:
    def test_uses_landmark_visibility_when_no_mask(self):
        scorer = FrameScorer()
        # 3/6 visible joints → score 0.5
        lm = {
            11: _LM(0.4, 0.3, visibility=0.9),
            12: _LM(0.6, 0.3, visibility=0.9),
            23: _LM(0.4, 0.6, visibility=0.9),
            24: _LM(0.6, 0.6, visibility=0.2),
            27: _LM(0.4, 0.9, visibility=0.2),
            28: _LM(0.6, 0.9, visibility=0.2),
        }
        s = scorer.score(_solid_image(), PoseID.FRONT, lm)
        assert s.occlusion_score == pytest.approx(0.5, abs=0.01)

    def test_zero_landmarks_yields_zero(self):
        scorer = FrameScorer()
        s = scorer.score(_solid_image(), PoseID.FRONT, None)
        assert s.occlusion_score == 0.0

    def test_mask_blends_with_landmark_score(self):
        scorer = FrameScorer()
        lm = _full_visibility_landmarks()           # landmark score = 1.0
        # Mask coverage = 0.30 (ideal band 0.15–0.50 → mask_score=1.0)
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[10:55, 30:70] = 255  # 45 * 40 = 1800 / 10000 = 0.18 → ideal band
        s = scorer.score(_solid_image(), PoseID.FRONT, lm, body_mask=mask)
        # 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert s.occlusion_score == pytest.approx(1.0, abs=0.01)
