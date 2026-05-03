"""
Personal accuracy test — compares scan output against your own tape measurements.
Usage:  python personal_accuracy_test.py --front /path/front.jpg --side /path/side.jpg --height 170
"""
import argparse, base64, json, urllib.request, sys

GROUND_TRUTH = {
    "M01_chest":          105.41,
    "M06_neck":           43.18,
    "M13_ankle":          20.32,
    "M17_kameez_length":  109.22,
    "M19_sleeve_length":  59.69,
    "M22_outseam":        104.14,
    "M26_shoulder_width": 53.34,
    "M29_hip_width":      53.34,
}

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def run(front_path, side_path, height_cm, url="http://localhost:8000"):
    frames = [{"image_b64": b64(front_path), "pose_id": "front", "quality_score": 0.9}]
    if side_path:
        frames.append({"image_b64": b64(side_path), "pose_id": "side_left", "quality_score": 0.9})

    req = urllib.request.Request(
        f"{url}/api/v1/scan/submit",
        data=json.dumps({"frames": frames, "height_cm": height_cm}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front",  required=True, help="Path to front-view photo")
    ap.add_argument("--side",   default=None,  help="Path to side-left photo (optional but recommended)")
    ap.add_argument("--height", required=True, type=float, help="Your height in cm")
    args = ap.parse_args()

    print(f"\nSubmitting scan (height={args.height} cm)...")
    resp = run(args.front, args.side, args.height)

    if resp.get("status") != "complete":
        print("Scan failed:", resp.get("error"))
        sys.exit(1)

    m = resp["measurements"]

    print(f"\n{'Measurement':<28} {'Ground Truth':>13} {'Predicted':>10} {'Error':>8} {'Confidence':>12} {'Source'}")
    print("-" * 90)

    errors = []
    for code, gt in GROUND_TRUTH.items():
        field = m.get(code)
        if not field:
            print(f"{code:<28} {gt:>13.1f} {'—':>10} {'—':>8}")
            continue
        pred  = field["value_cm"]
        err   = abs(pred - gt)
        conf  = field.get("confidence", "?")
        src   = field.get("source", "?")
        flag  = " ⚠" if err > 3.0 else ""
        errors.append(err)
        print(f"{code:<28} {gt:>13.1f} {pred:>10.1f} {err:>7.1f}cm  {conf:>10}  {src}{flag}")

    if errors:
        mae = sum(errors) / len(errors)
        print(f"\n{'MAE across measured fields:':<40} {mae:.1f} cm")
        print(f"{'Max error:':<40} {max(errors):.1f} cm")
        ok = sum(1 for e in errors if e <= 2.0)
        print(f"{'Within ±2 cm (PRD Phase 1 target):':<40} {ok}/{len(errors)} measurements")

if __name__ == "__main__":
    main()
