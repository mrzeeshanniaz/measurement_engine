"""
TailorSync Scan API — Photo Test Runner

Submits prepared photos for each subject to the scan API and prints a
full measurement report.  When ground-truth values are present in the CSV,
also prints per-measurement accuracy (MAE / ≤1 cm / ≤2 cm).

Usage:
    python run_test.py
    python run_test.py --api http://192.168.1.10:8000
    python run_test.py --csv ground_truth_test.csv --poses front side_left back arms_out
    python run_test.py --subject S001          # single subject
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
    import requests
except ImportError:
    sys.exit("Install requests:  pip install requests")

try:
    from PIL import Image
except ImportError:
    sys.exit("Install Pillow:  pip install Pillow")

BENCH_DIR        = Path(__file__).parent
SUBMIT_ENDPOINT  = "/api/v1/scan/submit"
STATUS_ENDPOINT  = "/api/v1/scan/status/{session_id}"
RESULT_ENDPOINT  = "/api/v1/scan/result/{session_id}"
POLL_INTERVAL_S  = 2.0
MAX_POLL_S       = 180.0

# All 7 pose column names in the CSV
POSE_PHOTO_COLS = {
    "front":         "front_photo",
    "quarter_left":  "quarter_left_photo",
    "side_left":     "side_left_photo",
    "three_quarter": "three_quarter_photo",
    "back":          "back_photo",
    "side_right":    "side_right_photo",
    "arms_out":      "arms_out_photo",
}

ALL_MEASUREMENTS = [
    "M01_chest", "M02_under_bust", "M03_waist", "M04_abdomen", "M05_hips",
    "M06_neck", "M07_bicep", "M08_wrist",
    "M09_thigh", "M10_mid_thigh", "M11_knee", "M12_calf", "M13_ankle",
    "M15_shoulder_to_waist_front", "M16_shoulder_to_waist_back",
    "M17_kameez_length", "M18_dress_length",
    "M19_sleeve_length", "M20_sleeve_length_elbow",
    "M21_inseam", "M22_outseam",
    "M23_crotch_depth_front", "M24_crotch_depth_back",
    "M25_torso_length",
    "M26_shoulder_width", "M27_chest_width", "M28_back_width", "M29_hip_width",
    "M30_chest_depth", "M31_waist_depth", "M32_armhole_depth",
]

CONF_EMOJI = {"HIGH": "✓", "MEDIUM": "~", "LOW": "⚠"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(path: str) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((640, 1280), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _call_api(base_url: str, height_cm: float, frames: list[dict]) -> dict:
    """
    Submit a scan, then poll /status until COMPLETE or FAILED.  Returns the
    final ScanResponse dict from /result, or raises on timeout/failure.
    """
    import time

    base = base_url.rstrip("/")
    submit = requests.post(
        base + SUBMIT_ENDPOINT,
        json={"height_cm": height_cm, "frames": frames},
        timeout=60,
    )
    submit.raise_for_status()
    session_id = submit.json()["session_id"]

    deadline = time.monotonic() + MAX_POLL_S
    while time.monotonic() < deadline:
        st = requests.get(
            base + STATUS_ENDPOINT.format(session_id=session_id),
            timeout=10,
        )
        st.raise_for_status()
        status = st.json()["status"]
        if status == "COMPLETE":
            res = requests.get(
                base + RESULT_ENDPOINT.format(session_id=session_id),
                timeout=30,
            )
            res.raise_for_status()
            return res.json()
        if status == "FAILED":
            raise RuntimeError(f"Pipeline FAILED: {st.json().get('error', '?')}")
        time.sleep(POLL_INTERVAL_S)

    raise TimeoutError(f"Scan {session_id} did not complete within {MAX_POLL_S}s")


# ---------------------------------------------------------------------------
# Per-subject test
# ---------------------------------------------------------------------------

def run_subject(row: dict, base_url: str, poses_to_use: list[str]) -> dict | None:
    subject_id = row["subject_id"]
    height_cm  = float(row["height_cm"])

    frames = []
    for pose_id in poses_to_use:
        col      = POSE_PHOTO_COLS.get(pose_id)
        path_str = row.get(col, "").strip()
        if not path_str:
            continue
        photo_path = BENCH_DIR / path_str
        if not photo_path.exists():
            print(f"  [warn] {subject_id}: photo not found — {photo_path}")
            continue
        frames.append({
            "pose_id":      pose_id,
            "image_b64":    _encode(str(photo_path)),
            "quality_score": 0.90,
        })

    if not frames:
        print(f"  [skip] {subject_id}: no photos found")
        return None

    print(f"  {subject_id}: submitting {len(frames)} frame(s) ...", end="", flush=True)
    try:
        resp = _call_api(base_url, height_cm, frames)
    except Exception as exc:
        print(f"  FAILED — {exc}")
        return None

    status = (resp.get("status") or "").lower()
    if status != "complete":
        print(f"  pipeline {status}: {resp.get('error','')}")
        return None

    print(f" {status.upper()}")
    return resp


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(subject_id: str, height_cm: float, resp: dict, gt_row: dict) -> None:
    meas  = resp.get("measurements", {})
    valid = resp.get("validation", {})
    conf  = resp.get("overall_confidence", "?")
    hsrc  = resp.get("height_source", "?")

    print()
    print("=" * 72)
    print(f"  {subject_id}  —  height={height_cm:.1f} cm ({hsrc})  "
          f"overall_confidence={conf}")
    print("=" * 72)
    print(f"  {'Measurement':<32} {'Value':>8}  {'Conf':>8}  {'Source':<20}  {'GT diff':>8}")
    print("-" * 72)

    errors_all: list[float] = []

    for code in ALL_MEASUREMENTS:
        field = meas.get(code, {})
        val   = field.get("value_cm")
        fc    = field.get("confidence", "LOW")
        src   = field.get("source", "—")
        sym   = CONF_EMOJI.get(fc, "?")

        gt_str = gt_row.get(code, "").strip()
        diff_str = ""
        if gt_str and val is not None:
            diff = abs(val - float(gt_str))
            errors_all.append(diff)
            diff_str = f"{diff:+.1f}" if (val - float(gt_str)) >= 0 else f"{diff:+.1f}"
            diff_str = f"{val - float(gt_str):+.1f} cm"

        val_s = f"{val:.1f}" if val is not None else "—"
        print(f"  {code:<32} {val_s:>8}  {sym} {fc:<7}  {src:<20}  {diff_str:>8}")

    print("-" * 72)
    if errors_all:
        import statistics
        mae  = statistics.mean(errors_all)
        rmse = (sum(e ** 2 for e in errors_all) / len(errors_all)) ** 0.5
        pct1 = 100 * sum(1 for e in errors_all if e <= 1.0) / len(errors_all)
        pct2 = 100 * sum(1 for e in errors_all if e <= 2.0) / len(errors_all)
        print(f"  Ground-truth fields: {len(errors_all)}  "
              f"MAE={mae:.2f} cm  RMSE={rmse:.2f} cm  "
              f"≤1cm={pct1:.0f}%  ≤2cm={pct2:.0f}%")
        print("-" * 72)

    # Validation
    print(f"\n  Validation: is_valid={valid.get('is_valid')}  "
          f"can_order={valid.get('can_order')}")
    summary = valid.get("summary", "")
    if summary:
        print(f"  {summary}")
    for issue in valid.get("issues", []):
        sev = issue["severity"].upper()
        print(f"    [{sev}] {issue['code']}: {issue['message']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TailorSync scan API test runner")
    parser.add_argument("--csv",     default="ground_truth_test.csv")
    parser.add_argument("--api",     default="http://localhost:8000")
    parser.add_argument("--poses",   nargs="+",
                        default=list(POSE_PHOTO_COLS.keys()),
                        choices=list(POSE_PHOTO_COLS.keys()),
                        help="Pose IDs to include (default: all 7)")
    parser.add_argument("--subject", default=None, help="Run only this subject ID")
    args = parser.parse_args()

    csv_path = BENCH_DIR / args.csv
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.subject:
        rows = [r for r in rows if r["subject_id"] == args.subject]
        if not rows:
            sys.exit(f"Subject '{args.subject}' not found in {args.csv}")

    # Health check
    try:
        import requests as _req
        hc = _req.get(args.api.rstrip("/") + "/api/v1/scan/health", timeout=5)
        hc_data = hc.json()
        if not hc_data.get("models_loaded"):
            print(f"[warn] Server models not fully loaded: {hc_data}")
        else:
            print(f"Server ready — {args.api}\n")
    except Exception as exc:
        sys.exit(f"Cannot reach server at {args.api}: {exc}")

    print(f"Subjects: {len(rows)}  |  Poses: {args.poses}\n")

    results = []
    for row in rows:
        resp = run_subject(row, args.api, args.poses)
        if resp:
            print_report(row["subject_id"], float(row["height_cm"]), resp, row)
            results.append((row["subject_id"], resp))

    # Summary table across all subjects
    if len(results) > 1:
        print("\n" + "=" * 72)
        print("  SUMMARY")
        print("=" * 72)
        print(f"  {'Subject':<10}  {'Height':>8}  {'Status':>10}  {'Overall Conf':>14}  {'Validation'}")
        print("-" * 72)
        for sid, r in results:
            v = r.get("validation", {})
            print(f"  {sid:<10}  {r.get('height_cm',0):>8.1f}  "
                  f"{r.get('status','?'):>10}  {r.get('overall_confidence','?'):>14}  "
                  f"valid={v.get('is_valid')}  can_order={v.get('can_order')}")
        print("=" * 72)


if __name__ == "__main__":
    main()
