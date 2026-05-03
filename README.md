# TailorSync Measurement Engine

AI-powered body measurement backend. Accepts height and 1–7 pose photos from the mobile app, runs a MediaPipe + SMPL mesh pipeline, and returns 32 garment measurements with per-field confidence levels.

---

## Table of Contents

- [Architecture](#architecture)
- [Measurements](#measurements)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Benchmark & Accuracy Testing](#benchmark--accuracy-testing)
- [Project Structure](#project-structure)
- [Large Model Files](#large-model-files)

---

## Architecture

```
Mobile App
    │  height_cm + base64 frames (1–7 poses)
    ▼
POST /api/v1/scan/submit
    │
    ├─ Frame Scorer       — blur / pose confidence / lighting / occlusion
    ├─ Frame Selector     — picks best frame per pose
    ├─ MediaPipe Pose     — 33 body landmarks (PoseLandmarker heavy model)
    ├─ Height Estimator   — user input / sensor fusion / population mean
    ├─ SMPL Mesh Fitter   — estimates body shape betas from landmark proportions
    ├─ SMPL-Anthropometry — circumferences via plane intersection + largest-ring isolation
    └─ Validator          — physiological range checks + cross-measurement rules
    │
    ▼
ScanResponse — 32 measurements, confidence, validation issues
```

**Measurement sources (priority order):**

| Source | Description |
|---|---|
| `smpl_anthro_full` | SMPL-Anthropometry with body-part segmentation — best accuracy |
| `smpl_anthro_trimesh` | Trimesh plane intersection fallback |
| `smpl_mesh` | Direct mesh geometry (width, depth) |
| `landmark` | MediaPipe joint distances |
| `height_ratio` | Anthropometric proportion × height (last resort) |
| `manual` | Customer-entered value |

---

## Measurements

32 measurements across 4 sections:

| Code | Name | Section |
|---|---|---|
| M01 | Chest circumference | A — Upper body |
| M02 | Under-bust circumference | A |
| M03 | Waist circumference | A |
| M04 | Abdomen circumference | A |
| M05 | Hips circumference | A |
| M06 | Neck circumference | A |
| M07 | Bicep circumference | A |
| M08 | Wrist circumference | A |
| M09 | Thigh circumference | B — Lower body |
| M10 | Mid-thigh circumference | B |
| M11 | Knee circumference | B |
| M12 | Calf circumference | B |
| M13 | Ankle circumference | B |
| M14 | Total height | C — Lengths |
| M15 | Shoulder to waist (front) | C |
| M16 | Shoulder to waist (back) | C |
| M17 | Kameez length | C |
| M18 | Dress length | C |
| M19 | Sleeve length (full) | C |
| M20 | Sleeve length (elbow) | C |
| M21 | Inseam | C |
| M22 | Outseam | C |
| M23 | Crotch depth (front) | C |
| M24 | Crotch depth (back) | C |
| M25 | Torso length | C |
| M26 | Shoulder width | D — Widths & depths |
| M27 | Chest width | D |
| M28 | Back width | D |
| M29 | Hip width | D |
| M30 | Chest depth | D |
| M31 | Waist depth | D |
| M32 | Armhole depth | D |

**Confidence levels:**

| Level | Accuracy | Meaning |
|---|---|---|
| `HIGH` | ±0.5–1.0 cm | Tailor can cut directly |
| `MEDIUM` | ±1.0–2.0 cm | Acceptable, flag on tailor sheet |
| `LOW` | > 2.0 cm | Requires manual confirmation before order |

---

## Prerequisites

- Python 3.10–3.13
- pip
- SMPL model files (see [Large Model Files](#large-model-files))

---

## Setup

**1. Clone the repo**

```bash
git clone git@github.com:mrzeeshanniaz/measurement_engine.git
cd measurement_engine
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
cd backend
pip install -r requirements.txt
```

For benchmark photo preparation (background removal), also install:

```bash
pip install pillow-heif "rembg[cpu]"
```

**4. Place SMPL model files**

The SMPL `.pkl` files are not included in the repo (too large). Copy them to:

```
backend/app/measurement_engine/smpl_anthropometry/data/smpl/
    SMPL_NEUTRAL.pkl          # 236 MB — full SMPL model
    SMPL_NEUTRAL_clean.pkl    #  40 MB — cleaned version (used by default)
```

See [Large Model Files](#large-model-files) for download instructions.

**5. Create the environment file**

```bash
cp backend/.env.example backend/.env   # if .env.example exists
# or create backend/.env manually:
```

```env
DEBUG=false
DEVICE=cpu
MODEL_CACHE_DIR=./models
```

The MediaPipe pose model (`pose_landmarker_heavy.task`, ~29 MB) is downloaded automatically on first startup.

---

## Configuration

All settings are in `backend/.env` (or environment variables):

| Variable | Default | Description |
|---|---|---|
| `DEBUG` | `false` | Enable debug logging |
| `DEVICE` | `cpu` | Inference device (`cpu` or `cuda`) |
| `MODEL_CACHE_DIR` | `./models` | Directory for downloaded model files |
| `DEFAULT_FOCAL_LENGTH_MM` | `4.25` | Fallback focal length for height estimation |
| `DEFAULT_SENSOR_WIDTH_MM` | `4.8` | Fallback sensor width for height estimation |

---

## Running the Server

```bash
cd backend
./start.sh
```

Or directly:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server prints the LAN address on startup. On first run, the MediaPipe model (~29 MB) downloads automatically.

**Health check:**

```bash
curl http://localhost:8000/api/v1/scan/health
# {"pipeline":"ok","models_loaded":true,"pose_model":true,"smpl_model":true}
```

---

## API Reference

Base URL: `http://localhost:8000`

### POST `/api/v1/scan/submit`

Submit height + pose frames; receive 32 measurements.

**Query parameters:**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `units` | `cm`, `in` | `cm` | Response unit for all measurement values |

**Request body:**

```json
{
  "height_cm": 175.0,
  "frames": [
    {
      "pose_id": "front",
      "image_b64": "<base64-encoded JPEG or PNG>",
      "quality_score": 0.90
    }
  ]
}
```

Supported `pose_id` values: `front`, `quarter_left`, `side_left`, `three_quarter`, `back`, `side_right`, `arms_out`.

Minimum viable request: `height_cm` + one `front` frame. For best accuracy supply all 7 poses.

**Response:**

```json
{
  "scan_id": "uuid",
  "status": "complete",
  "overall_confidence": "HIGH | MEDIUM | LOW",
  "frames_received": 7,
  "height_cm": 175.0,
  "height_source": "user_input",
  "response_unit": "cm",
  "measurements": {
    "M01_chest": {
      "value_cm": 102.3,
      "unit": "cm",
      "confidence": "MEDIUM",
      "source": "smpl_anthro_full",
      "is_manual_override": false
    }
    // ... 31 more fields
  },
  "validation": {
    "is_valid": true,
    "can_order": true,
    "issues": [],
    "rescan_poses": [],
    "summary": "All measurements valid."
  },
  "error": null
}
```

**Inches response (`?units=in`):**

```json
{
  "height_cm": 68.9,
  "response_unit": "in",
  "measurements": {
    "M01_chest": { "value_cm": 40.26, "unit": "in", ... }
  }
}
```

> Note: The field name `value_cm` and `height_cm` are retained for backwards compatibility; the actual unit is indicated by `response_unit` and each field's `unit`.

---

### POST `/api/v1/scan/manual`

Submit all 32 measurements entered manually (SCAN-09).

**Query parameters:** same `units` param as above.

**Request body:**

```json
{
  "height_cm": 175.0,
  "M01_chest": 102.0,
  "M03_waist": 88.0,
  "M05_hips": 104.0
}
```

All measurement fields are optional. Missing fields default to `null` with `LOW` confidence. Supplied fields are flagged `is_manual_override: true` with `MEDIUM` confidence.

---

### GET `/api/v1/scan/validation-rules`

Returns all physiological range limits and cross-measurement consistency rules as JSON. Flutter app fetches this once per launch for client-side pre-validation.

---

### GET `/api/v1/scan/health`

```json
{
  "pipeline": "ok",
  "models_loaded": true,
  "pose_model": true,
  "smpl_model": true
}
```

---

### GET `/health`

Basic liveness check.

---

## Benchmark & Accuracy Testing

The `benchmark/` directory contains tools for measuring pipeline accuracy against tape-measured ground truth.

### Prepare test photos

Raw HEIC photos go in `benchmark/test_photos/` named `<SUBJECT>_<POSE>.heic`  
(e.g. `S001_front.heic`, `S001_side_left.heic`).

```bash
cd benchmark
python prepare_test_photos.py --subjects S001 S002
```

This runs each photo through: EXIF rotation → resize → background removal (rembg `u2net_human_seg`) → lighting normalisation → white background composite → PNG save in `benchmark/test_photos_processed/<SUBJECT>/`.

Use `--force` to re-process existing outputs.

### Run the accuracy test

```bash
cd benchmark
python run_test.py --api http://localhost:8000
```

Options:

| Flag | Description |
|---|---|
| `--api URL` | Server base URL (default: `http://localhost:8000`) |
| `--csv FILE` | Ground-truth CSV (default: `ground_truth_test.csv`) |
| `--poses` | Subset of poses to send (default: all 7) |
| `--subject S001` | Run a single subject only |

Output includes per-measurement accuracy and summary statistics (MAE, RMSE, ≤1 cm %, ≤2 cm %) when ground-truth values are filled in `ground_truth_test.csv`.

### Postman collection

Import `TailorSync_API.postman_collection.json` into Postman. Set the `base_url` collection variable to your server address.

Pre-built test bodies for S001 (162.56 cm) and S002 (180.339 cm) with all 7 processed poses are in `benchmark/postman_body_S001.json` and `benchmark/postman_body_S002.json`.

---

## Project Structure

```
measurement_engine/
├── backend/
│   ├── app/
│   │   ├── api/v1/
│   │   │   └── scan.py                  # Route handlers, units conversion
│   │   ├── config.py                    # Pydantic settings
│   │   ├── main.py                      # FastAPI app, lifespan model loading
│   │   └── measurement_engine/
│   │       ├── models/
│   │       │   ├── model_manager.py     # Loads pose + SMPL at startup
│   │       │   ├── pose.py              # MediaPipe PoseLandmarker wrapper
│   │       │   └── smpl.py              # SMPL mesh loader
│   │       ├── scan/
│   │       │   ├── schemas.py           # Pydantic request/response models
│   │       │   ├── pipeline.py          # Orchestrates all pipeline stages
│   │       │   ├── frame_scorer.py      # Blur / pose / lighting / occlusion scoring
│   │       │   ├── frame_selector.py    # Picks best frame per pose
│   │       │   ├── height_estimator.py  # User input / sensor fusion / population mean
│   │       │   ├── measurements.py      # 32-measurement extraction from SMPL mesh
│   │       │   ├── confidence.py        # Per-field confidence assignment
│   │       │   ├── validator.py         # Range + cross-measurement validation rules
│   │       │   └── norms.py             # ANSUR II population norms
│   │       └── smpl_anthropometry/      # SMPL-Anthropometry integration
│   ├── models/                          # Downloaded model files (gitignored)
│   ├── requirements.txt
│   └── start.sh
├── benchmark/
│   ├── prepare_test_photos.py           # HEIC → processed PNG pipeline
│   ├── run_test.py                      # Accuracy test runner
│   ├── ground_truth_test.csv            # Fill in tape measurements here
│   ├── test_photos_processed/           # Background-removed PNGs (S001, S002)
│   └── images/                          # Sample single-subject test images
├── TailorSync_API.postman_collection.json
└── README.md
```

---

## Large Model Files

These files are excluded from the repo and must be obtained separately.

### SMPL models

Register at [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and download the SMPL neutral model. Place the files at:

```
backend/app/measurement_engine/smpl_anthropometry/data/smpl/
    SMPL_NEUTRAL.pkl
    SMPL_NEUTRAL_clean.pkl
```

### MediaPipe pose model

Downloaded automatically on first server start from Google's CDN (~29 MB). Cached at `backend/models/pose_landmarker_heavy.task`.

To pre-download manually:

```bash
mkdir -p backend/models
curl -L -o backend/models/pose_landmarker_heavy.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
```
