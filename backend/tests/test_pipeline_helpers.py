"""
Tests for the stateless helpers in app.measurement_engine.scan.pipeline.

The full pipeline.run() path needs loaded ML models, so these tests target
only the deterministic helpers:
  - _decode_image                (base64 + EXIF rotation)
  - _convert_landmarks
  - _compute_mesh_fit_score      (multi-view silhouette IoU)
  - _landmark_visibilities       (M-code → mean visibility)
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
from PIL import Image

from app.measurement_engine.scan.measurements import LandmarkPoint
from app.measurement_engine.scan.pipeline import ScanPipeline, _ProcessedFrame
from app.measurement_engine.scan.schemas import FrameScore, PoseID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_jpeg(width: int = 64, height: int = 128, color: tuple[int, int, int] = (200, 150, 100)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_frame(
    pose: PoseID,
    image_size: tuple[int, int] = (256, 512),
    body_mask: Optional[np.ndarray] = None,
) -> _ProcessedFrame:
    img = Image.new("RGB", image_size, color=(0, 0, 0))
    fs = FrameScore(
        pose_id=pose, blur_score=0.8, pose_confidence=0.8, angle_match=0.5,
        occlusion_score=0.8, lighting_score=1.0, composite=0.85,
    )
    return _ProcessedFrame(pose_id=pose, image=img, landmarks=None, score=fs, body_mask=body_mask)


# ---------------------------------------------------------------------------
# _decode_image
# ---------------------------------------------------------------------------

class TestDecodeImage:
    def test_decodes_jpeg_to_rgb(self):
        img = ScanPipeline._decode_image(_b64_jpeg())
        assert img.mode == "RGB"
        assert img.size == (64, 128)

    def test_converts_non_rgb_to_rgb(self):
        # Encode a single-channel PNG
        gray = Image.new("L", (32, 64), color=128)
        buf = io.BytesIO()
        gray.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        img = ScanPipeline._decode_image(b64)
        assert img.mode == "RGB"

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            ScanPipeline._decode_image("not-valid-base64-!@#$%")


# ---------------------------------------------------------------------------
# _convert_landmarks
# ---------------------------------------------------------------------------

class TestConvertLandmarks:
    def test_none_returns_none(self):
        assert ScanPipeline._convert_landmarks(None) is None

    def test_converts_raw_dict_to_landmark_points(self):
        @dataclass
        class _Raw:
            x: float; y: float; z: float; visibility: float

        raw = {0: _Raw(0.5, 0.1, 0.0, 0.9), 11: _Raw(0.4, 0.3, 0.05, 0.95)}
        out = ScanPipeline._convert_landmarks(raw)
        assert out is not None
        assert isinstance(out[0], LandmarkPoint)
        assert out[0].x == 0.5
        assert out[11].visibility == 0.95


# ---------------------------------------------------------------------------
# _compute_mesh_fit_score — multi-view silhouette IoU
# ---------------------------------------------------------------------------

class TestComputeMeshFitScore:
    def _make_mesh(self, height_cm: float = 175.0) -> np.ndarray:
        """A vertically-stretched 'body' point cloud — 175 cm tall, ±25 cm wide,
        ±15 cm deep, centered at origin."""
        rng = np.random.default_rng(7)
        n = 1000
        xs = rng.uniform(-25.0, 25.0, n)
        ys = rng.uniform(0.0, height_cm, n)
        zs = rng.uniform(-15.0, 15.0, n)
        return np.stack([xs, ys, zs], axis=1)

    def _matching_mask(self, height_cm: float = 175.0) -> np.ndarray:
        """Rectangular silhouette matching the synthetic mesh's projected width."""
        H, W = 800, 500
        mask = np.zeros((H, W), dtype=np.uint8)
        # Body fills 80% of the height, centered horizontally
        body_h = int(H * 0.80)
        top = H - body_h
        cx = W // 2
        body_w_px = int(body_h / height_cm * 50)  # 50 cm wide (±25 cm)
        mask[top:H, cx - body_w_px // 2: cx + body_w_px // 2] = 255
        return mask

    def test_returns_1_when_no_vertices(self):
        frames = [_make_frame(PoseID.FRONT, body_mask=self._matching_mask())]
        assert ScanPipeline._compute_mesh_fit_score(None, frames, 175.0) == 1.0

    def test_returns_1_when_no_masks_present(self):
        mesh = self._make_mesh()
        frames = [_make_frame(PoseID.FRONT, body_mask=None)]
        assert ScanPipeline._compute_mesh_fit_score(mesh, frames, 175.0) == 1.0

    def test_returns_iou_for_matching_silhouette(self):
        mesh = self._make_mesh()
        frames = [_make_frame(PoseID.FRONT, body_mask=self._matching_mask())]
        iou = ScanPipeline._compute_mesh_fit_score(mesh, frames, 175.0)
        # Convex hull of point cloud vs rectangular mask should overlap heavily
        assert 0.3 < iou <= 1.0

    def test_averages_iou_across_multiple_views(self):
        mesh = self._make_mesh()
        frames = [
            _make_frame(PoseID.FRONT,     body_mask=self._matching_mask()),
            _make_frame(PoseID.SIDE_LEFT, body_mask=self._matching_mask()),
        ]
        iou = ScanPipeline._compute_mesh_fit_score(mesh, frames, 175.0)
        assert 0.0 < iou <= 1.0

    def test_unrelated_view_does_not_break(self):
        mesh = self._make_mesh()
        # FRONT mask present; SIDE_LEFT mask absent — should still produce a result
        frames = [
            _make_frame(PoseID.FRONT,     body_mask=self._matching_mask()),
            _make_frame(PoseID.SIDE_LEFT, body_mask=None),
        ]
        iou = ScanPipeline._compute_mesh_fit_score(mesh, frames, 175.0)
        assert iou > 0.0


# ---------------------------------------------------------------------------
# _landmark_visibilities — M-code → mean visibility from front frame
# ---------------------------------------------------------------------------

class TestLandmarkVisibilities:
    def test_no_front_frame_returns_empty(self):
        frames = [_make_frame(PoseID.BACK)]
        out = ScanPipeline._landmark_visibilities(frames)
        assert out == {}

    def test_front_landmarks_keyed_by_measurement_code(self):
        front = _make_frame(PoseID.FRONT)
        front.landmarks = {
            11: LandmarkPoint(0.4, 0.3, 0.0, 0.9),
            12: LandmarkPoint(0.6, 0.3, 0.0, 0.9),
            23: LandmarkPoint(0.4, 0.6, 0.0, 0.8),
            24: LandmarkPoint(0.6, 0.6, 0.0, 0.8),
        }
        out = ScanPipeline._landmark_visibilities([front])
        # M01 averages indices 11,12,23,24 → mean visibility = (0.9+0.9+0.8+0.8)/4 = 0.85
        assert out["M01"] == pytest.approx(0.85, abs=0.01)
        # M03 averages 23,24 → 0.80
        assert out["M03"] == pytest.approx(0.80, abs=0.01)

    def test_empty_landmarks_returns_empty_dict(self):
        front = _make_frame(PoseID.FRONT)
        front.landmarks = {}
        out = ScanPipeline._landmark_visibilities([front])
        assert out == {}

    def test_partial_landmarks_use_only_present_indices(self):
        front = _make_frame(PoseID.FRONT)
        front.landmarks = {11: LandmarkPoint(0.4, 0.3, 0.0, 0.9)}
        out = ScanPipeline._landmark_visibilities([front])
        # M01 expects 11,12,23,24 — only 11 present → mean of [0.9]
        assert out["M01"] == pytest.approx(0.9, abs=0.01)
