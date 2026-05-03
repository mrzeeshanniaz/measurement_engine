"""
Generate ready-to-use JSON payloads for the TailorSync scan API.

Usage:
  # Dummy image payloads (works in dev — pipeline uses fallback landmarks)
  python generate_test_payload.py

  # Real photo as FRONT frame
  python generate_test_payload.py --photo /path/to/photo.jpg

  # Sensor-fusion payload (no height_cm required)
  python generate_test_payload.py --mode sensor_fusion --camera-height 120 --tilt 20

Output files (written to ./test_payloads/):
  payload_minimum.json          — height_cm + front frame only
  payload_full_7pose.json       — height_cm + all 7 poses
  payload_sensor_fusion.json    — camera_metadata + front frame
  payload_sensor_fusion_full.json — camera_metadata + all 7 poses
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Install Pillow first:  pip install Pillow")

OUT_DIR = Path(__file__).parent / "test_payloads"

POSE_IDS = ["front", "quarter_left", "side_left", "three_quarter", "back", "side_right", "arms_out"]

POSE_COLORS = {
    "front":         (70,  130, 180),
    "quarter_left":  (60,  179, 113),
    "side_left":     (255, 165,   0),
    "three_quarter": (186,  85, 211),
    "back":          (220,  20,  60),
    "side_right":    (255, 215,   0),
    "arms_out":      (64,  224, 208),
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _dummy_image_b64(pose_id: str, width: int = 240, height: int = 480) -> str:
    """Create a coloured placeholder image and return as base64 JPEG."""
    colour = POSE_COLORS.get(pose_id, (128, 128, 128))
    img = Image.new("RGB", (width, height), colour)
    draw = ImageDraw.Draw(img)

    # Draw a rough stick-figure body outline so MediaPipe has *something*
    cx = width // 2
    draw.ellipse([cx - 20, 30, cx + 20, 70], outline="white", width=3)   # head
    draw.line([cx, 70, cx, 220], fill="white", width=4)                   # torso
    draw.line([cx, 110, cx - 50, 180], fill="white", width=3)             # L arm
    draw.line([cx, 110, cx + 50, 180], fill="white", width=3)             # R arm
    draw.line([cx, 220, cx - 30, 370], fill="white", width=3)             # L leg
    draw.line([cx, 220, cx + 30, 370], fill="white", width=3)             # R leg

    # Label
    draw.text((10, 10), pose_id.replace("_", "\n"), fill="white")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _real_image_b64(path: str) -> str:
    """Load a real photo and return as base64 JPEG."""
    img = Image.open(path).convert("RGB")
    # Resize to a reasonable size to keep payload manageable
    img.thumbnail((640, 1280), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _frame(pose_id: str, b64: str, quality: float = 0.87) -> dict:
    return {"pose_id": pose_id, "image_b64": b64, "quality_score": quality}


def build_minimum(height_cm: float, front_b64: str) -> dict:
    return {
        "height_cm": height_cm,
        "frames": [_frame("front", front_b64, 0.87)],
    }


def build_full_7pose(height_cm: float, frames_b64: dict[str, str]) -> dict:
    quality = {"front": 0.92, "quarter_left": 0.88, "side_left": 0.85,
               "three_quarter": 0.83, "back": 0.91, "side_right": 0.87, "arms_out": 0.90}
    return {
        "height_cm": height_cm,
        "frames": [_frame(p, frames_b64[p], quality[p]) for p in POSE_IDS],
    }


def build_sensor_fusion(camera_height: float, tilt: float,
                         focal_px: float | None, front_b64: str) -> dict:
    meta = {"camera_height_cm": camera_height, "tilt_angle_deg": tilt}
    if focal_px is not None:
        meta["focal_length_px"] = focal_px
    return {
        "camera_metadata": meta,
        "frames": [_frame("front", front_b64, 0.87)],
    }


def build_sensor_fusion_full(camera_height: float, tilt: float,
                              focal_px: float | None, frames_b64: dict[str, str]) -> dict:
    meta = {"camera_height_cm": camera_height, "tilt_angle_deg": tilt}
    if focal_px is not None:
        meta["focal_length_px"] = focal_px
    quality = {"front": 0.92, "quarter_left": 0.88, "side_left": 0.85,
               "three_quarter": 0.83, "back": 0.91, "side_right": 0.87, "arms_out": 0.90}
    return {
        "camera_metadata": meta,
        "frames": [_frame(p, frames_b64[p], quality[p]) for p in POSE_IDS],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TailorSync API test payloads")
    parser.add_argument("--photo", help="Path to a real full-body photo (used as FRONT frame)")
    parser.add_argument("--height", type=float, default=175.0, help="Height in cm (default 175.0)")
    parser.add_argument("--camera-height", type=float, default=120.0,
                        help="Camera height above floor in cm for sensor fusion (default 120)")
    parser.add_argument("--tilt", type=float, default=20.0,
                        help="Camera tilt angle in degrees for sensor fusion (default 20)")
    parser.add_argument("--focal-px", type=float, default=None,
                        help="Focal length in pixels for sensor fusion (optional)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("Generating images...")
    frames_b64: dict[str, str] = {}
    for pose in POSE_IDS:
        if pose == "front" and args.photo:
            frames_b64[pose] = _real_image_b64(args.photo)
            print(f"  {pose}: loaded from {args.photo}")
        else:
            frames_b64[pose] = _dummy_image_b64(pose)
            print(f"  {pose}: dummy placeholder image")

    payloads = {
        "payload_minimum.json": build_minimum(args.height, frames_b64["front"]),
        "payload_full_7pose.json": build_full_7pose(args.height, frames_b64),
        "payload_sensor_fusion.json": build_sensor_fusion(
            args.camera_height, args.tilt, args.focal_px, frames_b64["front"]
        ),
        "payload_sensor_fusion_full.json": build_sensor_fusion_full(
            args.camera_height, args.tilt, args.focal_px, frames_b64
        ),
    }

    for filename, payload in payloads.items():
        path = OUT_DIR / filename
        path.write_text(json.dumps(payload, indent=2))
        size_kb = path.stat().st_size // 1024
        print(f"  Written: {path}  ({size_kb} KB)")

    print(f"\nDone. Import into Postman or use with curl:")
    print(f"  curl -s -X POST http://localhost:8000/api/v1/scan/submit \\")
    print(f"       -H 'Content-Type: application/json' \\")
    print(f"       -d @{OUT_DIR}/payload_minimum.json | python -m json.tool")


if __name__ == "__main__":
    main()
