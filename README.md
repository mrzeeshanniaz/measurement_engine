# TailorSync Measurement Engine

AI-powered body measurement backend. Accepts height and 1–7 pose photos from the mobile app, runs a MediaPipe + SMPL mesh pipeline, and returns 32 garment measurements with per-field confidence levels, ease allowances, and cutting values.

---

## Table of Contents

- [Architecture](#architecture)
- [Measurements](#measurements)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Firebase Auth](#firebase-auth)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Benchmark & Accuracy Testing](#benchmark--accuracy-testing)
- [Project Structure](#project-structure)
- [Large Model Files](#large-model-files)

---

## Architecture

```
Mobile App
    │  height_cm + base64 frames (1–7 poses) + garment_type + fit_style
    ▼
POST /api/v1/scan/submit
    │
    ├─ Frame Scorer            — blur / pose confidence / lighting / occlusion
    ├─ Frame Selector          — picks best frame per pose
    ├─ Segmentation            — body mask extraction per frame (DeepLabV3)
    ├─ MediaPipe Pose          — 33 body landmarks (PoseLandmarker heavy model)
    ├─ Height Estimator        — user input / sensor fusion / population mean
    ├─ SMPL Mesh Fitter        — initial betas from landmark proportions
    ├─ Multi-view Beta Optim.  — silhouette IoU refinement across all views
    ├─ SMPL-Anthropometry      — circumferences via plane intersection
    ├─ Confidence Scorer       — per-field confidence (source × frame quality × mesh fit)
    ├─ Garment Profiler        — required-field flags + ease/cutting values per fit style
    └─ Validator               — 5-pass physiological checks + mesh quality gate
    │
    ▼
ScanResponse — 32 measurements, confidence, ease, cutting values, validation issues
    │
    ▼ (async, when AUTH_ENABLED)
Firestore — scan profile persisted under customer_id / scan_id
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

**Garment types supported:** `kameez`, `kurta`, `shalwar`, `trouser`, `shirt`, `sherwani`, `dress`, `suit_jacket`, `blouse`, `skirt`, `lehenga_skirt`, `coat`

**Fit styles:** `fitted`, `regular`, `relaxed` — control ease allowances added to each circumference measurement.

---

## Prerequisites

- Python 3.10–3.13
- pip
- SMPL model files (see [Large Model Files](#large-model-files))
- Firebase project (optional — see [Firebase Auth](#firebase-auth))

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
# or create backend/.env manually
```

Minimum `.env` for local development:

```env
DEBUG=false
DEVICE=cpu
MODEL_CACHE_DIR=./models
AUTH_ENABLED=false
CORS_ORIGINS=*
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
| `AUTH_ENABLED` | `false` | Require Firebase ID token on all scan + profile endpoints |
| `FIREBASE_CREDENTIALS_PATH` | _(none)_ | Path to Firebase service account JSON (omit on GCP — ADC is used) |
| `FIREBASE_PROJECT_ID` | _(none)_ | GCP project ID (inferred from credentials when not set) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins; `*` allows all (dev default) |

**Production CORS example:**

```env
CORS_ORIGINS=https://tailorsync.app,https://www.tailorsync.app
```

---

## Firebase Auth

Authentication is implemented via Firebase Admin SDK. Setting `AUTH_ENABLED=true` makes the server verify a Firebase ID token on all scan submission and profile endpoints.

**Setup:**

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com).
2. Download a service account key: Project Settings → Service Accounts → Generate new private key.
3. Set `FIREBASE_CREDENTIALS_PATH=/path/to/service-account.json` in `.env`.
4. Set `AUTH_ENABLED=true`.

**Mobile client:** Send the Firebase ID token as a Bearer token:

```http
Authorization: Bearer <firebase-id-token>
```

**Development:** `AUTH_ENABLED=false` (default) allows unauthenticated requests — no token required.

**On Cloud Run / GCP:** Omit `FIREBASE_CREDENTIALS_PATH`; Application Default Credentials are used automatically.

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
# {"pipeline":"ok","models_loaded":true,"pose_model":true,"smpl_model":true,
#  "jobs_queued":0,"jobs_processing":0,"jobs_complete":0,"jobs_failed":0}
```

---

## API Reference

Base URL: `http://localhost:8000`

### POST `/api/v1/scan/submit`

Submit height + pose frames; receive 32 measurements with confidence, ease, and cutting values.

**Query parameters:**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `units` | `cm`, `in` | `cm` | Response unit for all measurement values |

**Request body:**

```json
{
  "height_cm": 175.0,
  "garment_type": "kameez",
  "fit_style": "regular",
  "scale_tier": "TIER2",
  "client_scan_id": "optional-client-uuid",
  "frames": [
    {
      "pose_id": "front",
      "image_b64": "<base64-encoded JPEG or PNG>",
      "quality_score": 0.90
    }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `height_cm` | Yes | Customer height in cm (100–250) |
| `garment_type` | No | Activates required-field flags and validation |
| `fit_style` | No | Activates ease and cutting value calculation |
| `scale_tier` | No | `TIER1` / `TIER2` (default) / `TIER3` — processing scale hint |
| `client_scan_id` | No | Client-provided idempotency key |
| `frames` | Yes | 1–7 pose frames |

Supported `pose_id` values: `front`, `quarter_left`, `side_left`, `three_quarter`, `back`, `side_right`, `arms_out`.

Minimum viable request: `height_cm` + one `front` frame. For best accuracy supply all 7 poses.

**Response:**

```json
{
  "scan_id": "uuid",
  "status": "complete",
  "overall_confidence": "HIGH",
  "frames_received": 7,
  "height_cm": 175.0,
  "height_source": "user_input",
  "response_unit": "cm",
  "measurements": {
    "M01_chest": {
      "value_cm": 102.3,
      "unit": "cm",
      "confidence": "HIGH",
      "source": "smpl_anthro_full",
      "is_manual_override": false,
      "is_required_for_garment": true,
      "ease_cm": 8.0,
      "cutting_value_cm": 110.3
    }
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

`ease_cm` and `cutting_value_cm` are `null` when no `fit_style` is supplied. `is_required_for_garment` is `null` when no `garment_type` is supplied.

**Inches response (`?units=in`):** all `value_cm` fields contain inch values; `response_unit` and each field's `unit` are `"in"`.

---

### POST `/api/v1/scan/manual`

Submit all 32 measurements entered manually. All fields are optional; missing fields become `null` with `LOW` confidence.

**Query parameters:** same `units` param as submit.

**Request body:**

```json
{
  "height_cm": 175.0,
  "garment_type": "kameez",
  "fit_style": "regular",
  "M01_chest": 102.0,
  "M03_waist": 88.0,
  "M05_hips": 104.0
}
```

Supplied fields are returned with `is_manual_override: true` and `MEDIUM` confidence.

---

### GET `/api/v1/scan/result/{scan_id}`

Retrieve a completed scan result by ID. Returns 404 when not found.

**Query parameters:** `units=cm|in`

---

### GET `/api/v1/scan/status/{scan_id}`

Returns `{"scan_id": "...", "status": "QUEUED|PROCESSING|COMPLETE|FAILED"}`. Returns 404 when not found.

---

### GET `/api/v1/scan/validation-rules`

Returns all physiological range limits and cross-measurement consistency rules as JSON. The Flutter app fetches this once per launch for client-side pre-validation.

---

### GET `/api/v1/scan/health`

```json
{
  "pipeline": "ok",
  "models_loaded": true,
  "pose_model": true,
  "smpl_model": true,
  "jobs_queued": 0,
  "jobs_processing": 0,
  "jobs_complete": 4,
  "jobs_failed": 0
}
```

---

### Profiles API

Scan profiles are persisted to Firestore when `AUTH_ENABLED=true`. All profile endpoints require a valid Firebase ID token.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/profiles` | List profiles for the authenticated customer |
| `GET` | `/api/v1/profiles/{profile_id}` | Get a single profile by Firestore document ID |
| `GET` | `/api/v1/profiles/by-scan/{scan_id}` | Get a profile by scan UUID |
| `DELETE` | `/api/v1/profiles/{profile_id}` | Delete a profile |

**List query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `limit` | `20` | Max results per page (1–100) |
| `offset` | `0` | Pagination offset |

---

### GET `/health`

Basic liveness check. Returns `{"status": "ok"}`.

---

## Testing

The test suite lives in `backend/tests/` and covers the scan module logic without loading any ML models.

```bash
cd backend
python -m pytest tests/ -v
```

**Test modules:**

| File | Tests | Coverage |
|---|---|---|
| `tests/test_validator.py` | 34 | All 5 validator passes: hard limits, norms, cross-measurement rules, garment required fields, mesh quality gate |
| `tests/test_garments.py` | 13 | GARMENT_REQUIRED_FIELDS correctness; apply_garment_profile required flags and ease/cutting values |
| `tests/test_confidence.py` | 15 | score_field for all sources and edge cases; overall_confidence majority vote |
| `tests/test_schemas.py` | 18 | ManualMeasurementRequest boundary validation; ScanSubmitRequest fields; PoseID enum |
| `tests/test_api.py` | 22 | FastAPI integration: /health, /validation-rules, /manual, /status 404, /result 404 |

Run a single module:

```bash
python -m pytest tests/test_validator.py -v
```

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
│   │   │   ├── scan.py                  # Scan routes, unit conversion
│   │   │   └── profiles.py              # Profile CRUD routes
│   │   ├── db/
│   │   │   ├── firestore.py             # Firestore client init
│   │   │   ├── crud.py                  # save / list / get / delete profiles
│   │   │   └── models.py                # ScanProfile Pydantic model
│   │   ├── auth.py                      # Firebase token verification, FastAPI deps
│   │   ├── config.py                    # Pydantic settings (env / .env)
│   │   └── main.py                      # FastAPI app, lifespan model loading, CORS
│   │       measurement_engine/
│   │       ├── models/
│   │       │   ├── model_manager.py     # Loads pose + SMPL at startup
│   │       │   ├── pose.py              # MediaPipe PoseLandmarker wrapper
│   │       │   ├── segmentation.py      # DeepLabV3 body mask extraction
│   │       │   └── smpl.py              # SMPL loader + multi-view beta optimizer
│   │       ├── scan/
│   │       │   ├── schemas.py           # Pydantic request/response models
│   │       │   ├── pipeline.py          # Orchestrates all pipeline stages
│   │       │   ├── frame_scorer.py      # Blur / pose / lighting / occlusion scoring
│   │       │   ├── frame_selector.py    # Picks best frame per pose
│   │       │   ├── height_estimator.py  # User input / sensor fusion / population mean
│   │       │   ├── measurements.py      # 32-measurement extraction from SMPL mesh
│   │       │   ├── confidence.py        # Per-field confidence assignment
│   │       │   ├── garments.py          # Required fields + ease allowances per garment
│   │       │   ├── validator.py         # 5-pass validation (hard limits → mesh quality)
│   │       │   ├── job_store.py         # In-memory async job registry
│   │       │   └── norms.py             # ANSUR II population norms
│   │       └── smpl_anthropometry/      # SMPL-Anthropometry integration
│   ├── tests/
│   │   ├── conftest.py                  # Fixtures: typical_male_measurements, helpers
│   │   ├── test_validator.py            # 34 tests — all 5 validator passes
│   │   ├── test_garments.py             # 13 tests — required fields + ease
│   │   ├── test_confidence.py           # 15 tests — score_field + overall_confidence
│   │   ├── test_schemas.py              # 18 tests — request validation boundaries
│   │   └── test_api.py                  # 22 tests — FastAPI integration
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
