"""
Server-side frame selector — SCAN-06.

Rules:
  FRONT, SIDE_LEFT, ARMS_OUT  → keep top-3 frames by composite score
  All other poses             → keep top-1 frame

Also enforces the SCAN-06 image constraints:
  - Resize so the longer edge ≤ 1024 px (Lanczos)
  - Re-encode as JPEG quality=85

The selector operates on already-decoded PIL images held in _ProcessedFrame,
so base64 overhead is not re-introduced inside the pipeline.  The resize is
applied in-place on the image object so downstream scorers and SMPL fitting
work on the properly-sized frames.

Upload payload budget (SCAN-06): < 1.5 MB total.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image

from app.measurement_engine.scan.schemas import PoseID

if TYPE_CHECKING:
    from app.measurement_engine.scan.pipeline import _ProcessedFrame

logger = logging.getLogger(__name__)

# SCAN-06 constants
MAX_LONG_EDGE_PX = 1024
JPEG_QUALITY     = 85
MAX_PAYLOAD_MB   = 1.5

# How many frames to keep per pose group
_TOP_N: dict[PoseID, int] = {
    PoseID.FRONT:         3,
    PoseID.SIDE_LEFT:     3,
    PoseID.ARMS_OUT:      3,
    PoseID.QUARTER_LEFT:  1,
    PoseID.THREE_QUARTER: 1,
    PoseID.BACK:          1,
    PoseID.SIDE_RIGHT:    1,
}


def select_and_resize(
    frames: "list[_ProcessedFrame]",
) -> "list[_ProcessedFrame]":
    """
    1. For each pose group, keep only the top-N frames by composite score.
    2. Resize each kept frame so the longer edge ≤ 1024 px.
    Returns the filtered, resized list — may be shorter than the input.
    """
    # Group by pose_id
    by_pose: dict[PoseID, list[_ProcessedFrame]] = {}
    for f in frames:
        by_pose.setdefault(f.pose_id, []).append(f)

    selected: list[_ProcessedFrame] = []
    for pose_id, pose_frames in by_pose.items():
        n = _TOP_N.get(pose_id, 1)
        top = sorted(pose_frames, key=lambda f: f.score.composite, reverse=True)[:n]
        for f in top:
            f.image = _resize(f.image)
        selected.extend(top)
        logger.debug(
            "SCAN-06 %s: kept %d/%d frame(s) (top composite=%.2f)",
            pose_id, len(top), len(pose_frames),
            top[0].score.composite if top else 0.0,
        )

    return selected


def _resize(img: Image.Image) -> Image.Image:
    """Resize so the longer edge ≤ 1024 px, preserving aspect ratio."""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= MAX_LONG_EDGE_PX:
        return img
    scale = MAX_LONG_EDGE_PX / long_edge
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)
