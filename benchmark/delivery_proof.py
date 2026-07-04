"""
End-to-end delivery proof for the TailorSync measurement engine.

Loads real ML models (MediaPipe pose, MediaPipe segmenter, SMPL) and runs
the full ScanPipeline on real photos for each subject in test_photos_processed/.

For each subject we report:
  - All 32 measurements with confidence + source
  - Validator outcome (is_valid / can_order / issues)
  - Population-norm fit (|Z| against ANSUR-II ratios for the subject's height)
  - Internal-consistency checks (chest > waist, inseam < outseam, etc.)

Ground-truth tape measurements are not available, so we cannot report MAE/RMSE.
What we *can* prove:
  (a) the pipeline runs on real images without crashing
  (b) produces all 32 measurements
  (c) every value is inside its physiological hard limit
  (d) every value is within ±3σ of the ANSUR-II height-relative norm
  (e) the validator gives can_order=True

Usage:
    cd backend && PYTHONPATH=. python ../benchmark/delivery_proof.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from PIL import Image

ROOT = Path(__file__).resolve().parent
SUBJECTS = {
    "S001": 162.56,
    "S002": 180.339,
}
POSE_FILES = [
    ("front",         "front.png"),
    ("quarter_left",  "quarter_left.png"),
    ("side_left",     "side_left.png"),
    ("three_quarter", "three_quarter.png"),
    ("back",          "back.png"),
    ("side_right",    "side_right.png"),
    ("arms_out",      "arms_out.png"),
]


def _encode_to_b64(path: Path, max_long_edge: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_long_edge, max_long_edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


async def _main() -> int:
    # Imports deferred so model load happens after this script starts.
    from app.measurement_engine.models.model_manager import ModelManager
    from app.measurement_engine.scan.pipeline import ScanPipeline
    from app.measurement_engine.scan.schemas import PoseFrame, PoseID
    from app.measurement_engine.scan.norms import HARD_LIMITS, NORMS

    _print_section("Loading models (MediaPipe pose + MediaPipe segmenter + SMPL)")
    t0 = time.time()
    mgr = ModelManager()
    await mgr.load()
    print(f"  Loaded in {time.time() - t0:.1f}s.  pose={mgr.pose.is_loaded}  "
          f"segmenter={mgr.segmenter.is_loaded}  smpl={mgr.smpl.is_loaded}")

    pipeline = ScanPipeline(
        pose_model=mgr.pose,
        smpl_model=mgr.smpl,
        segmenter_model=mgr.segmenter,
    )

    all_subject_reports: dict = {}

    for sid, height_cm in SUBJECTS.items():
        photo_dir = ROOT / "test_photos_processed" / sid
        _print_section(f"{sid}  —  height={height_cm:.1f} cm")
        if not photo_dir.exists():
            print(f"  [skip] photo dir not found: {photo_dir}")
            continue

        frames: list[PoseFrame] = []
        for pose_id, fname in POSE_FILES:
            p = photo_dir / fname
            if not p.exists():
                print(f"  [warn] missing {fname}")
                continue
            frames.append(PoseFrame(
                pose_id=PoseID(pose_id),
                image_b64=_encode_to_b64(p),
                quality_score=0.90,
            ))
        print(f"  Frames submitted: {len(frames)}")

        t1 = time.time()
        resp = pipeline.run(frames=frames, height_cm=height_cm)
        dt = time.time() - t1
        print(f"  Pipeline completed in {dt:.1f}s  →  status={resp.status.value}  "
              f"overall_confidence={resp.overall_confidence.value}")

        if resp.status.value != "complete":
            print(f"  ERROR: {resp.error}")
            all_subject_reports[sid] = {"status": resp.status.value, "error": resp.error}
            continue

        # ---- Measurement table ----
        print()
        print(f"  {'Code':<30} {'Value':>9}  {'Conf':<8}  {'Source':<22}  {'|Z|':>5}")
        print("  " + "-" * 76)
        m = resp.measurements
        rows: list[dict] = []
        norm_zs: list[float] = []
        hard_violations: list[str] = []
        for attr_name in type(m).model_fields:
            field = getattr(m, attr_name)
            code = attr_name.split("_")[0]
            val = field.value_cm
            z_str = ""
            z_val: Optional[float] = None
            if val is not None and code in NORMS:
                z_val = NORMS[code].z_score(val, height_cm)
                z_str = f"{abs(z_val):.2f}"
                norm_zs.append(abs(z_val))
            if val is not None and code in HARD_LIMITS:
                lo, hi = HARD_LIMITS[code]
                if not (lo <= val <= hi):
                    hard_violations.append(f"{code}={val} not in [{lo},{hi}]")
            print(f"  {attr_name:<30} {(f'{val:.1f}' if val is not None else '—'):>9}  "
                  f"{field.confidence.value:<8}  {field.source:<22}  {z_str:>5}")
            rows.append({
                "code": code,
                "attr": attr_name,
                "value_cm": val,
                "confidence": field.confidence.value,
                "source": field.source,
                "abs_z": z_val if z_val is None else round(abs(z_val), 2),
            })

        # ---- Validator outcome ----
        v = resp.validation
        print()
        print(f"  Validator: is_valid={v.is_valid}  can_order={v.can_order}  "
              f"{len(v.issues)} issue(s)")
        for issue in v.issues:
            print(f"    [{issue.severity.value.upper():<7}] {issue.code}: {issue.message[:90]}")

        # ---- Norm-fit summary ----
        outliers = sum(1 for z in norm_zs if z > 3.0)
        warns    = sum(1 for z in norm_zs if 2.0 < z <= 3.0)
        median_z = statistics.median(norm_zs) if norm_zs else 0.0
        print()
        print(f"  Norm fit  —  median |Z|={median_z:.2f}  "
              f"warnings (|Z|>2)={warns}  errors (|Z|>3)={outliers}")
        if hard_violations:
            print(f"  HARD LIMIT VIOLATIONS: {hard_violations}")

        all_subject_reports[sid] = {
            "status": "complete",
            "height_cm": height_cm,
            "frames_submitted": len(frames),
            "pipeline_seconds": round(dt, 2),
            "overall_confidence": resp.overall_confidence.value,
            "validation": {
                "is_valid": v.is_valid,
                "can_order": v.can_order,
                "issues": [
                    {"severity": i.severity.value, "code": i.code, "message": i.message}
                    for i in v.issues
                ],
            },
            "norm_fit": {
                "median_abs_z": round(median_z, 2),
                "warnings_z_gt_2": warns,
                "errors_z_gt_3": outliers,
                "hard_limit_violations": hard_violations,
            },
            "measurements": rows,
        }

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------
    _print_section("DELIVERY SUMMARY")
    for sid, rep in all_subject_reports.items():
        if rep.get("status") != "complete":
            print(f"  {sid}: FAILED — {rep.get('error')}")
            continue
        v = rep["validation"]
        n = rep["norm_fit"]
        produced = sum(1 for r in rep["measurements"] if r["value_cm"] is not None)
        print(f"  {sid}: {produced}/32 produced  "
              f"can_order={v['can_order']}  is_valid={v['is_valid']}  "
              f"median|Z|={n['median_abs_z']}  outliers={n['errors_z_gt_3']}  "
              f"runtime={rep['pipeline_seconds']}s")

    # Persist JSON for the user / CI
    out_path = ROOT / "delivery_proof_report.json"
    out_path.write_text(json.dumps(all_subject_reports, indent=2, default=str))
    print(f"\n  Full JSON report → {out_path}")

    # Exit non-zero if any subject failed the can_order gate
    any_failure = any(
        rep.get("status") != "complete" or not rep["validation"]["can_order"]
        for rep in all_subject_reports.values()
    )
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
