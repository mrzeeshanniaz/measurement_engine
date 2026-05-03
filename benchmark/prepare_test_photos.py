"""
Prepare raw HEIC scan photos for the TailorSync accuracy test.

Steps applied to every image:
  1. Decode HEIC  (pillow-heif)
  2. Correct EXIF rotation
  3. Resize to ≤ 1080×1440 for faster rembg inference
  4. Remove background  (rembg u2net_human_seg)
  5. Normalise lighting on the person region only
     - Gamma-corrects if mean person brightness is outside [110, 170]
     - Stretches per-channel histogram to use full [0, 255] range
  6. Composite onto clean white background
  7. Final resize: long-edge ≤ 1024 px  (matches SCAN-06 server constraint)
  8. Save as PNG

Usage:
    python prepare_test_photos.py
    python prepare_test_photos.py --force   # re-process even if output exists
    python prepare_test_photos.py --subjects S001 S002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    sys.exit("Missing pillow-heif. Run: pip install pillow-heif")

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Missing Pillow. Run: pip install Pillow")

try:
    from rembg import new_session, remove as rembg_remove
except ImportError:
    sys.exit("Missing rembg. Run: pip install 'rembg[cpu]'")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent
PHOTO_DIR = BENCH_DIR / "test_photos"
OUT_DIR   = BENCH_DIR / "test_photos_processed"

POSES = [
    "front", "quarter_left", "side_left", "three_quarter",
    "back", "side_right", "arms_out",
]

# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def _normalise_lighting(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Adjust brightness and contrast on the person region only.

    Two passes:
      a) Per-channel histogram stretch so the person uses the full tonal range.
      b) Global gamma correction if mean person brightness drifts outside [110, 170].
    """
    person = mask > 128
    if person.sum() < 500:
        return rgb

    out = rgb.astype(np.float32)

    # a) Per-channel stretch
    for c in range(3):
        ch = out[:, :, c]
        vals = ch[person]
        lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        if hi - lo > 10:
            ch = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
            out[:, :, c] = ch

    # b) Gamma correction
    mean_brightness = float(out[person].mean())
    if mean_brightness < 110:
        gamma = np.log(130.0 / 255.0) / np.log(max(mean_brightness, 1) / 255.0)
        gamma = float(np.clip(gamma, 0.5, 2.0))
        out = (out / 255.0) ** gamma * 255.0

    elif mean_brightness > 170:
        gamma = np.log(150.0 / 255.0) / np.log(mean_brightness / 255.0)
        gamma = float(np.clip(gamma, 0.5, 2.0))
        out = (out / 255.0) ** gamma * 255.0

    return out.clip(0, 255).astype(np.uint8)


def prepare_image(src: Path, dst: Path, session) -> bool:
    """
    Full preparation pipeline for one photo.
    Returns True on success.
    """
    try:
        # 1-2. Open + EXIF rotation
        img = Image.open(src)
        img = ImageOps.exif_transpose(img).convert("RGB")

        # 3. Resize to ≤ 1080×1440 before rembg (speeds inference ~4×)
        img.thumbnail((1080, 1440), Image.LANCZOS)

        # 4. Remove background
        rgba: Image.Image = rembg_remove(img, session=session, post_process_mask=True)
        mask = np.array(rgba.split()[3])   # alpha = person mask

        # 5. Normalise lighting on the person region
        rgb_fixed = _normalise_lighting(np.array(img), mask)
        img_fixed = Image.fromarray(rgb_fixed)

        # Composite onto white
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img_fixed, mask=Image.fromarray(mask))

        # 7. Final resize: long-edge ≤ 1024 px
        white.thumbnail((1024, 1024), Image.LANCZOS)

        # 8. Save
        dst.parent.mkdir(parents=True, exist_ok=True)
        white.save(dst, "PNG", optimize=True)
        return True

    except Exception as exc:
        print(f"    ERROR: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TailorSync test photos")
    parser.add_argument("--subjects", nargs="+", default=["S001", "S002"])
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    print("Loading rembg u2net_human_seg model (downloads ~176 MB on first run)...")
    session = new_session("u2net_human_seg")
    print("Model ready.\n")

    for subject in args.subjects:
        print(f"── {subject}")
        for pose in POSES:
            src = PHOTO_DIR / f"{subject}_{pose}.heic"
            dst = OUT_DIR / subject / f"{pose}.png"

            if not src.exists():
                print(f"  [missing] {src.name}")
                continue

            if dst.exists() and not args.force:
                sz = dst.stat().st_size // 1024
                print(f"  [skip]    {pose}.png  ({sz} KB already exists)")
                continue

            print(f"  [process] {src.name} → {dst} ...", end="", flush=True)
            ok = prepare_image(src, dst, session)
            if ok:
                sz = dst.stat().st_size // 1024
                print(f" done ({sz} KB)")
            else:
                print(" FAILED")

    print("\nDone.")


if __name__ == "__main__":
    main()
