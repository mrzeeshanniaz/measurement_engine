"""
Tests for app.measurement_engine.scan.frame_selector — SCAN-06.

Covers:
  - Top-N retention per pose (FRONT/SIDE_LEFT/ARMS_OUT = 3; rest = 1)
  - Resize to long-edge ≤ 1024 px
  - Aspect-ratio preservation
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.measurement_engine.scan.frame_selector import (
    MAX_LONG_EDGE_PX,
    _resize,
    select_and_resize,
)
from app.measurement_engine.scan.pipeline import _ProcessedFrame
from app.measurement_engine.scan.schemas import FrameScore, PoseID


def _frame(pose: PoseID, composite: float, width: int = 256, height: int = 512) -> _ProcessedFrame:
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    score = FrameScore(
        pose_id=pose,
        blur_score=composite, pose_confidence=composite, angle_match=0.0,
        occlusion_score=0.0, lighting_score=composite, composite=composite,
    )
    return _ProcessedFrame(pose_id=pose, image=img, landmarks=None, score=score)


# ---------------------------------------------------------------------------
# Top-N retention
# ---------------------------------------------------------------------------

class TestTopN:
    def test_front_keeps_three_best(self):
        frames = [
            _frame(PoseID.FRONT, 0.3),
            _frame(PoseID.FRONT, 0.9),
            _frame(PoseID.FRONT, 0.5),
            _frame(PoseID.FRONT, 0.8),
            _frame(PoseID.FRONT, 0.7),
        ]
        selected = select_and_resize(frames)
        composites = sorted(f.score.composite for f in selected if f.pose_id == PoseID.FRONT)
        assert composites == [0.7, 0.8, 0.9]

    def test_side_left_keeps_three(self):
        frames = [_frame(PoseID.SIDE_LEFT, c) for c in (0.2, 0.4, 0.6, 0.8)]
        selected = select_and_resize(frames)
        assert len([f for f in selected if f.pose_id == PoseID.SIDE_LEFT]) == 3

    def test_arms_out_keeps_three(self):
        frames = [_frame(PoseID.ARMS_OUT, c) for c in (0.1, 0.5, 0.9, 0.4, 0.8)]
        selected = select_and_resize(frames)
        assert len([f for f in selected if f.pose_id == PoseID.ARMS_OUT]) == 3

    def test_back_keeps_one(self):
        frames = [_frame(PoseID.BACK, c) for c in (0.3, 0.7, 0.9, 0.4)]
        selected = select_and_resize(frames)
        back_frames = [f for f in selected if f.pose_id == PoseID.BACK]
        assert len(back_frames) == 1
        assert back_frames[0].score.composite == 0.9

    def test_quarter_left_keeps_one(self):
        frames = [_frame(PoseID.QUARTER_LEFT, c) for c in (0.3, 0.7)]
        selected = select_and_resize(frames)
        kept = [f for f in selected if f.pose_id == PoseID.QUARTER_LEFT]
        assert len(kept) == 1 and kept[0].score.composite == 0.7

    def test_three_quarter_keeps_one(self):
        frames = [_frame(PoseID.THREE_QUARTER, c) for c in (0.4, 0.8)]
        selected = select_and_resize(frames)
        kept = [f for f in selected if f.pose_id == PoseID.THREE_QUARTER]
        assert len(kept) == 1 and kept[0].score.composite == 0.8

    def test_single_frame_per_pose_passes_through(self):
        frames = [_frame(PoseID.FRONT, 0.7), _frame(PoseID.BACK, 0.8)]
        selected = select_and_resize(frames)
        assert len(selected) == 2

    def test_mixed_poses_preserved(self):
        frames = [
            _frame(PoseID.FRONT, 0.9),
            _frame(PoseID.SIDE_LEFT, 0.8),
            _frame(PoseID.BACK, 0.7),
            _frame(PoseID.ARMS_OUT, 0.85),
        ]
        selected = select_and_resize(frames)
        poses = {f.pose_id for f in selected}
        assert poses == {PoseID.FRONT, PoseID.SIDE_LEFT, PoseID.BACK, PoseID.ARMS_OUT}


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

class TestResize:
    def test_no_resize_when_already_below_max(self):
        img = Image.new("RGB", (800, 1000), color=(0, 0, 0))
        out = _resize(img)
        assert out.size == (800, 1000)

    def test_resize_preserves_aspect_ratio_portrait(self):
        img = Image.new("RGB", (1080, 1920), color=(0, 0, 0))
        out = _resize(img)
        assert max(out.size) == MAX_LONG_EDGE_PX
        # Aspect ratio preserved within rounding
        assert abs((out.width / out.height) - (1080 / 1920)) < 0.01

    def test_resize_preserves_aspect_ratio_landscape(self):
        img = Image.new("RGB", (2000, 1000), color=(0, 0, 0))
        out = _resize(img)
        assert out.width == MAX_LONG_EDGE_PX
        assert out.height == 512  # 1000 * (1024/2000)

    def test_square_image_at_exact_limit(self):
        img = Image.new("RGB", (MAX_LONG_EDGE_PX, MAX_LONG_EDGE_PX), color=(0, 0, 0))
        out = _resize(img)
        assert out.size == (MAX_LONG_EDGE_PX, MAX_LONG_EDGE_PX)


class TestResizeAppliedInSelector:
    def test_oversized_frame_is_resized(self):
        f = _frame(PoseID.FRONT, 0.9, width=2048, height=4096)
        selected = select_and_resize([f])
        assert max(selected[0].image.size) <= MAX_LONG_EDGE_PX
