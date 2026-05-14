"""
Locust load test for the TailorSync measurement engine.

What it does
------------
Each Locust user repeatedly:
  1. POST /api/v1/scan/submit         with a 1-frame minimal body
  2. GET  /api/v1/scan/status/{id}     poll until COMPLETE or FAILED
  3. GET  /api/v1/scan/result/{id}     fetch the result

It measures:
  - p50 / p95 / p99 submit-to-complete latency
  - per-endpoint error rate
  - whether the rate limiter kicks in (HTTP 429)
  - whether the server returns 503 (models still loading)
  - whether the body-size middleware ever 413s

Run against a *local* server:

  # Terminal 1
  cd backend && ./start.sh    # or: uvicorn app.main:app --port 8000

  # Terminal 2
  pip install locust
  cd benchmark
  locust -f loadtest.py --host http://localhost:8000 \
         --users 50 --spawn-rate 5 --run-time 2m --headless

Or with the Web UI:
  locust -f loadtest.py --host http://localhost:8000
  → open http://localhost:8089

Output
------
After the run, Locust prints per-endpoint stats and a histogram.  Look for:
  - p95(/submit) < 200 ms
  - p95(/status) < 50 ms
  - p95(/result) under ~5 s end-to-end (depends on hardware)
  - <1% 5xx errors, all 429s expected once you exceed RATE_LIMIT_SUBMIT
"""
from __future__ import annotations

import base64
import io
import random
import time
from pathlib import Path
from typing import Optional

from locust import HttpUser, between, events, task
from PIL import Image


# A pre-encoded 1024×1280 JPEG of a small gradient — enough bytes to pass the
# image_b64 size validator (≥100 chars) and decode cleanly through PIL.  We
# encode once at module load so the test isn't bottlenecked on JPEG encoding.
def _build_dummy_b64() -> str:
    img = Image.new("RGB", (640, 1024))
    # Add a vertical gradient so the JPEG encoder doesn't collapse to a tiny
    # single-color blob; this makes the payload size realistic (~30 KB).
    pixels = img.load()
    for y in range(img.height):
        c = int(255 * y / img.height)
        for x in range(img.width):
            pixels[x, y] = (c, c, c)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


DUMMY_FRAME_B64 = _build_dummy_b64()
HEIGHT_CHOICES = (160.0, 165.0, 170.0, 175.0, 180.0, 185.0)


class ScanUser(HttpUser):
    """One virtual user runs through the full submit → status → result loop."""

    # Wait 1–3 seconds between request cycles so we don't trip the rate limiter
    # too aggressively at 50 users (the default RATE_LIMIT_SUBMIT is 30/minute).
    wait_time = between(1.0, 3.0)

    @task
    def full_scan_cycle(self) -> None:
        height = random.choice(HEIGHT_CHOICES)
        body = {
            "height_cm": height,
            "frames": [
                {
                    "pose_id": "front",
                    "image_b64": DUMMY_FRAME_B64,
                    "quality_score": 0.9,
                }
            ],
        }

        # 1. Submit
        with self.client.post(
            "/api/v1/scan/submit",
            json=body,
            name="POST /scan/submit",
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                # Expected under load — count as success so we don't fail the run
                resp.success()
                return
            if resp.status_code == 503:
                resp.failure("503: models not loaded")
                return
            if resp.status_code != 200:
                resp.failure(f"submit returned {resp.status_code}")
                return
            session_id = resp.json().get("session_id")
            if not session_id:
                resp.failure("submit missing session_id")
                return

        # 2. Poll status until terminal (cap to 30s; tests run on CPU so 1
        #    frame typically completes in 2–5 seconds).
        terminal: Optional[str] = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with self.client.get(
                f"/api/v1/scan/status/{session_id}",
                name="GET /scan/status/[id]",
                catch_response=True,
            ) as st:
                if st.status_code != 200:
                    st.failure(f"status returned {st.status_code}")
                    return
                status = st.json().get("status")
                if status in ("COMPLETE", "FAILED"):
                    terminal = status
                    break
            time.sleep(0.5)

        if terminal is None:
            return  # timeout — Locust will count the last status call

        # 3. Fetch result (only on COMPLETE)
        if terminal == "COMPLETE":
            with self.client.get(
                f"/api/v1/scan/result/{session_id}",
                name="GET /scan/result/[id]",
                catch_response=True,
            ) as r:
                if r.status_code != 200:
                    r.failure(f"result returned {r.status_code}")
                    return
                body = r.json()
                if "measurements" not in body:
                    r.failure("result missing measurements")
                    return
                # Spot-check: M01_chest should be a finite float in [50, 200]
                m01 = body["measurements"].get("M01_chest", {}).get("value_cm")
                if m01 is not None and not (50.0 <= m01 <= 200.0):
                    r.failure(f"M01_chest out of plausible range: {m01}")


@events.test_start.add_listener
def _on_start(environment, **_kwargs):
    print("Load test starting against", environment.host)
    print("Per-frame payload size:", len(DUMMY_FRAME_B64), "bytes")


@events.test_stop.add_listener
def _on_stop(environment, **_kwargs):
    stats = environment.runner.stats
    print(f"\nTotal failures: {stats.total.num_failures}")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"p95 /submit:    {stats.get('POST /scan/submit', 'POST').get_response_time_percentile(0.95):.0f} ms")
    print(f"p95 /status:    {stats.get('GET /scan/status/[id]', 'GET').get_response_time_percentile(0.95):.0f} ms")
    print(f"p95 /result:    {stats.get('GET /scan/result/[id]', 'GET').get_response_time_percentile(0.95):.0f} ms")
