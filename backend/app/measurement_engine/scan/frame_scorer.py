"""
Frame quality scorer.

Implements SCAN-04 from the TailorSync PRD exactly:
  composite = sharpness × 0.30 + pose_quality × 0.40 + lighting × 0.30
  Frames with composite < 0.60 are rejected (is_usable = False).

angle_match and occlusion_score are retained as diagnostic fields in
FrameScore but do NOT contribute to the composite — keeping the formula
100% spec-compliant while preserving useful debug signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.measurement_engine.scan.schemas import FrameScore, PoseID

logger = logging.getLogger(__name__)

# SCAN-04 composite weights (spec-exact)
_W_BLUR    = 0.30
_W_POSE    = 0.40
_W_LIGHT   = 0.30

# SCAN-04: frames below this threshold are rejected
USABLE_THRESHOLD = 0.60

# Expected shoulder-hip vector angle (degrees from vertical) per pose
_EXPECTED_ANGLES: dict[PoseID, float] = {
    PoseID.FRONT:         0.0,
    PoseID.QUARTER_LEFT:  45.0,
    PoseID.SIDE_LEFT:     90.0,
    PoseID.THREE_QUARTER: 135.0,
    PoseID.BACK:          180.0,
    PoseID.SIDE_RIGHT:    90.0,   # right profile, same side-angle magnitude
    PoseID.ARMS_OUT:      0.0,
}

# MediaPipe landmark indices for key joints
_KEY_JOINT_INDICES = [11, 12, 23, 24, 27, 28]   # shoulders, hips, ankles


@dataclass
class _LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: float


class FrameScorer:
    """Scores a single decoded frame for measurement suitability."""

    def score(
        self,
        pil_image: Image.Image,
        pose_id: PoseID,
        landmarks: Optional[dict[int, _LandmarkPoint]],
    ) -> FrameScore:
        img_rgb = np.array(pil_image.convert("RGB"))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        blur  = self._blur_score(img_gray)
        pose  = self._pose_confidence(landmarks)
        light = self._lighting_score(img_gray)
        # Diagnostic only — not in composite per SCAN-04
        angle = self._angle_match(landmarks, pose_id)
        occ   = self._occlusion_score(landmarks)

        # SCAN-04 formula: sharpness×0.30 + pose_quality×0.40 + lighting×0.30
        composite = _W_BLUR * blur + _W_POSE * pose + _W_LIGHT * light

        return FrameScore(
            pose_id=pose_id,
            blur_score=round(blur, 3),
            pose_confidence=round(pose, 3),
            angle_match=round(angle, 3),
            occlusion_score=round(occ, 3),
            lighting_score=round(light, 3),
            composite=round(composite, 3),
        )

    # ------------------------------------------------------------------
    # Individual dimension scorers
    # ------------------------------------------------------------------

    def _blur_score(self, gray: np.ndarray) -> float:
        """Laplacian variance normalised to [0, 1]."""
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Empirically: < 50 is very blurry, > 500 is sharp
        return float(np.clip(variance / 500.0, 0.0, 1.0))

    def _pose_confidence(self, landmarks: Optional[dict]) -> float:
        if not landmarks:
            return 0.0
        scores = [
            landmarks[i].visibility
            for i in _KEY_JOINT_INDICES
            if i in landmarks
        ]
        return float(np.mean(scores)) if scores else 0.0

    def _angle_match(
        self,
        landmarks: Optional[dict],
        pose_id: PoseID,
    ) -> float:
        """
        Estimate body orientation from shoulder midpoint vs hip midpoint displacement
        along the X axis, then compare to the expected angle for this pose.

        For a front-facing pose the shoulders and hips have similar X midpoints.
        For a side pose one shoulder is occluded and X spread collapses.
        We use the ratio of visible shoulder spread to hip spread as a rough proxy.
        """
        if not landmarks:
            return 0.5  # neutral — no info

        expected = _EXPECTED_ANGLES.get(pose_id, 0.0)

        ls = landmarks.get(11)
        rs = landmarks.get(12)
        lh = landmarks.get(23)
        rh = landmarks.get(24)

        if not all([ls, rs, lh, rh]):
            return 0.5

        shoulder_spread = abs(ls.x - rs.x)
        hip_spread = abs(lh.x - rh.x)

        # At 0° (front) both spreads are similar → ratio ≈ 1
        # At 90° (side) shoulder/hip X spread collapses → ratio ≈ 0
        ratio = shoulder_spread / (hip_spread + 1e-6)
        ratio = float(np.clip(ratio, 0.0, 2.0)) / 2.0  # normalise to 0-1

        # Map expected angle to expected ratio: 0° → 1.0, 90° → 0.0
        expected_ratio = 1.0 - (min(expected, 180.0) / 180.0)

        angle_error = abs(ratio - expected_ratio)
        return float(np.clip(1.0 - angle_error, 0.0, 1.0))

    def _occlusion_score(self, landmarks: Optional[dict]) -> float:
        """Fraction of key joints with visibility > 0.5."""
        if not landmarks:
            return 0.0
        visible = sum(
            1 for i in _KEY_JOINT_INDICES
            if i in landmarks and landmarks[i].visibility > 0.5
        )
        return visible / len(_KEY_JOINT_INDICES)

    def _lighting_score(self, gray: np.ndarray) -> float:
        """
        Penalise frames that are too dark (mean < 60) or overexposed (mean > 220),
        and frames with very low contrast (std < 20).
        """
        mean = float(gray.mean())
        std  = float(gray.std())

        brightness_ok = 60 <= mean <= 220
        contrast_ok   = std >= 20

        if brightness_ok and contrast_ok:
            return 1.0
        elif brightness_ok or contrast_ok:
            return 0.6
        return 0.2
