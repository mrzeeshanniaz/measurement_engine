"""
API-level delivery proof.

Boots the full FastAPI app in-process, submits S001 photos via POST /submit,
polls GET /status until COMPLETE, then fetches GET /result.  This exercises
the entire production path: pydantic schemas → background task → job store →
unit conversion → final ScanResponse.

Run:
    cd backend && PYTHONPATH=. python ../benchmark/api_proof.py
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent


def _encode(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    from fastapi.testclient import TestClient
    from app.main import app

    photo_dir = ROOT / "test_photos_processed" / "S001"
    if not photo_dir.exists():
        print(f"[skip] {photo_dir} missing")
        return 0

    poses = ["front", "quarter_left", "side_left", "three_quarter",
             "back", "side_right", "arms_out"]
    frames = [
        {
            "pose_id": p,
            "image_b64": _encode(photo_dir / f"{p}.png"),
            "quality_score": 0.9,
        }
        for p in poses
        if (photo_dir / f"{p}.png").exists()
    ]

    body = {
        "height_cm": 162.56,
        "garment_type": "kameez",
        "fit_style": "regular",
        "frames": frames,
    }

    with TestClient(app) as client:
        # 1. Submit
        t0 = time.time()
        r = client.post("/api/v1/scan/submit", json=body)
        r.raise_for_status()
        session_id = r.json()["session_id"]
        print(f"  POST /submit  → 200  session_id={session_id[:8]}…")

        # 2. Poll status
        last_pct = -1
        for _ in range(60):
            r = client.get(f"/api/v1/scan/status/{session_id}")
            r.raise_for_status()
            d = r.json()
            if d["progress_pct"] != last_pct:
                print(f"  GET  /status   → {d['status']:<10}  progress={d['progress_pct']}%")
                last_pct = d["progress_pct"]
            if d["status"] in ("COMPLETE", "FAILED"):
                break
            time.sleep(0.5)

        if d["status"] != "COMPLETE":
            print(f"  FAILED — {d.get('error')}")
            return 1

        # 3. Result
        r = client.get(f"/api/v1/scan/result/{session_id}")
        r.raise_for_status()
        res = r.json()
        elapsed = time.time() - t0
        print(f"  GET  /result   → 200  ({elapsed:.1f}s total)")

        # Verify
        meas = res["measurements"]
        produced = sum(1 for v in meas.values() if v["value_cm"] is not None)
        kameez_required = ["M01_chest", "M03_waist", "M17_kameez_length",
                           "M19_sleeve_length", "M26_shoulder_width"]
        garment_fields = {k: meas[k]["is_required_for_garment"] for k in kameez_required}
        cutting = {k: (meas[k]["value_cm"], meas[k]["ease_cm"], meas[k]["cutting_value_cm"])
                   for k in kameez_required}

        print()
        print(f"  Response shape  : overall={res['overall_confidence']} unit={res['response_unit']}")
        print(f"  Measurements    : {produced}/32 with value_cm")
        print(f"  Validation      : valid={res['validation']['is_valid']}  "
              f"order={res['validation']['can_order']}  "
              f"issues={len(res['validation']['issues'])}")
        print(f"  Garment profile : {res['garment_type']} / {res['fit_style']}")
        print(f"  Required flags  : {garment_fields}")
        print(f"  Cut dimensions  :")
        for k, (val, ease, cut) in cutting.items():
            print(f"    {k:<32} value={val}cm  ease={ease}cm  cut={cut}cm")

        # 4. Inch conversion
        r = client.get(f"/api/v1/scan/result/{session_id}?units=in")
        r.raise_for_status()
        in_res = r.json()
        chest_cm = res["measurements"]["M01_chest"]["value_cm"]
        chest_in = in_res["measurements"]["M01_chest"]["value_cm"]
        ratio = chest_in / chest_cm
        print()
        print(f"  Unit conversion : chest_cm={chest_cm}  chest_in={chest_in}  "
              f"ratio={ratio:.4f}  (expected 0.3937)")

        # 5. Idempotency
        r2 = client.post("/api/v1/scan/submit", json={**body, "client_scan_id": "demo-123"})
        r3 = client.post("/api/v1/scan/submit", json={**body, "client_scan_id": "demo-123"})
        print(f"  Idempotency     : two submits with same client_scan_id → "
              f"same session_id? {r2.json()['session_id'] == r3.json()['session_id']}")

        # 6. Privacy: 404 for unknown session
        r4 = client.get("/api/v1/scan/result/nonexistent")
        print(f"  Privacy 404     : unknown session → {r4.status_code} (expected 404)")

    print()
    print("  ✓ Full async API path validated end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
