"""
Accuracy benchmark for the TailorSync measurement engine.

Usage (server must be running):
  # With a real photo and known measurements:
  python accuracy_test.py \\
      --photo /path/to/full_body_photo.jpg \\
      --height 175.0 \\
      --ground-truth ground_truth.csv

  # Quick sanity check with a dummy image:
  python accuracy_test.py --height 175.0 --dummy

Ground truth CSV format (comma-separated, header required):
  code,value_cm
  M01_chest,96.0
  M03_waist,80.0
  M05_hips,100.0
  M26_shoulder_width,42.0
  M19_sleeve_length,60.0
  M21_inseam,76.0

Output: per-measurement absolute error and % error, overall MAE summary.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Install httpx:  pip install httpx")

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Install Pillow:  pip install Pillow")

BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _photo_b64(path: str) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((640, 1280), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _dummy_b64() -> str:
    img = Image.new("RGB", (240, 480), (70, 130, 180))
    draw = ImageDraw.Draw(img)
    cx = 120
    draw.ellipse([cx - 20, 30, cx + 20, 70], outline="white", width=3)
    draw.line([cx, 70, cx, 220], fill="white", width=4)
    draw.line([cx, 110, cx - 50, 180], fill="white", width=3)
    draw.line([cx, 110, cx + 50, 180], fill="white", width=3)
    draw.line([cx, 220, cx - 30, 370], fill="white", width=3)
    draw.line([cx, 220, cx + 30, 370], fill="white", width=3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: str) -> dict[str, float]:
    gt: dict[str, float] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["code"].strip()
            value = float(row["value_cm"].strip())
            gt[code] = value
    return gt


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def submit_scan(height_cm: float, image_b64: str, pose_id: str = "front") -> dict:
    payload = {
        "height_cm": height_cm,
        "frames": [{"pose_id": pose_id, "image_b64": image_b64, "quality_score": 0.90}],
    }
    resp = httpx.post(
        f"{BASE_URL}/api/v1/scan/submit",
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Accuracy report
# ---------------------------------------------------------------------------

def flatten_measurements(response: dict) -> dict[str, dict]:
    """Extract {code: {value_cm, confidence, source}} from the API response."""
    meas = response.get("measurements", {})
    out: dict[str, dict] = {}
    for code, field in meas.items():
        if isinstance(field, dict) and field.get("value_cm") is not None:
            out[code] = field
    return out


def print_report(predicted: dict[str, dict], ground_truth: dict[str, float]) -> None:
    COL = 32

    print("\n" + "=" * 72)
    print(f"{'Measurement':<{COL}} {'Predicted':>10} {'Actual':>8} {'Error':>8} {'%Err':>7}  {'Conf':<8} {'Source'}")
    print("-" * 72)

    errors: list[float] = []
    missing: list[str] = []

    for code, actual in sorted(ground_truth.items()):
        field = predicted.get(code)
        if field is None:
            missing.append(code)
            continue

        pred = field["value_cm"]
        err = abs(pred - actual)
        pct = err / actual * 100
        conf = field.get("confidence", "?")
        source = field.get("source", "?")

        flag = "  ✓" if err <= 2.0 else ("  !" if err <= 5.0 else "  ✗")
        print(f"{code:<{COL}} {pred:>10.1f} {actual:>8.1f} {err:>7.1f}cm {pct:>6.1f}%  {conf:<8} {source}{flag}")
        errors.append(err)

    print("-" * 72)

    if errors:
        mae = sum(errors) / len(errors)
        within_2 = sum(1 for e in errors if e <= 2.0)
        within_5 = sum(1 for e in errors if e <= 5.0)
        print(f"\nMAE (mean absolute error): {mae:.2f} cm  over {len(errors)} measurements")
        print(f"Within ±2 cm: {within_2}/{len(errors)} ({within_2/len(errors)*100:.0f}%)")
        print(f"Within ±5 cm: {within_5}/{len(errors)} ({within_5/len(errors)*100:.0f}%)")

        # Accuracy rating
        if mae <= 1.5:
            rating = "EXCELLENT (tailor-grade)"
        elif mae <= 3.0:
            rating = "GOOD (ready-to-wear grade)"
        elif mae <= 5.0:
            rating = "ACCEPTABLE (needs improvement)"
        else:
            rating = "POOR (significant calibration needed)"
        print(f"Accuracy rating: {rating}")

    if missing:
        print(f"\nNot in ground truth: {', '.join(missing)}")

    print("=" * 72)

    # Show all predicted values (not just those with ground truth)
    print("\n--- All predicted measurements ---")
    print(f"{'Code':<{COL}} {'Value':>8}  {'Confidence':<8}  Source")
    print("-" * 60)
    for code, field in sorted(predicted.items()):
        print(f"{code:<{COL}} {field['value_cm']:>8.1f}  {field.get('confidence','?'):<8}  {field.get('source','?')}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TailorSync measurement accuracy benchmark")
    parser.add_argument("--photo", help="Path to a real full-body photo")
    parser.add_argument("--height", type=float, required=True, help="Known height in cm")
    parser.add_argument("--ground-truth", help="CSV file with known measurements (code,value_cm)")
    parser.add_argument("--dummy", action="store_true", help="Use a dummy stick-figure image")
    parser.add_argument("--url", default=BASE_URL, help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--save-response", help="Save raw API response to this JSON file")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    if not args.photo and not args.dummy:
        parser.error("Provide --photo <path> or --dummy")

    # Health check
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        print(f"Server: {health.json()}")
        scan_health = httpx.get(f"{BASE_URL}/api/v1/scan/health", timeout=5.0)
        print(f"Pipeline: {scan_health.json()}")
    except Exception as e:
        sys.exit(f"Cannot reach server at {BASE_URL}: {e}\nStart it with: uvicorn app.main:app --reload")

    # Prepare image
    if args.photo:
        print(f"\nLoading photo: {args.photo}")
        image_b64 = _photo_b64(args.photo)
    else:
        print("\nUsing dummy stick-figure image (landmark quality will be LOW)")
        image_b64 = _dummy_b64()

    # Submit
    print(f"Submitting scan (height={args.height} cm)...")
    try:
        response = submit_scan(args.height, image_b64)
    except httpx.HTTPStatusError as e:
        sys.exit(f"API error {e.response.status_code}: {e.response.text}")

    if args.save_response:
        Path(args.save_response).write_text(json.dumps(response, indent=2))
        print(f"Response saved to {args.save_response}")

    status = response.get("status")
    overall_conf = response.get("overall_confidence")
    print(f"\nScan status: {status}  |  Overall confidence: {overall_conf}")

    if status == "failed":
        sys.exit(f"Pipeline failed: {response.get('error')}")

    predicted = flatten_measurements(response)

    # Load ground truth if provided
    if args.ground_truth:
        ground_truth = load_ground_truth(args.ground_truth)
        print(f"Ground truth loaded: {len(ground_truth)} measurements from {args.ground_truth}")
        print_report(predicted, ground_truth)
    else:
        # No ground truth — just print all predictions
        COL = 32
        print(f"\n--- Predicted measurements (no ground truth provided) ---")
        print(f"{'Code':<{COL}} {'Value':>8}  {'Confidence':<8}  Source")
        print("-" * 60)
        for code, field in sorted(predicted.items()):
            print(f"{code:<{COL}} {field['value_cm']:>8.1f}  {field.get('confidence','?'):<8}  {field.get('source','?')}")
        print(f"\nTip: provide --ground-truth ground_truth.csv to see accuracy metrics.")


if __name__ == "__main__":
    main()
