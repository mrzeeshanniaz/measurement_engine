"""
SMPL-Anthropometry adapter for TailorSync.

Bridges the DavidBoja/SMPL-Anthropometry library into our pipeline.

Two modes:
  FULL  — SMPL PKL model files are present in data/smpl/.
          Uses proper anatomical plane cuts + body-part segmentation.
          Requires: SMPL_NEUTRAL.pkl (free download below).

  TRIMESH — No PKL files. Falls back to trimesh plane-intersection on
            our parametric cylinder mesh. Better than vertex-proximity
            convex-hull, but still limited to our parametric mesh shape.

How to enable FULL mode:
  1. Register (free) at https://smpl.is.tue.mpg.de/
  2. Download  basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl
  3. Rename to SMPL_NEUTRAL.pkl
  4. Place in  backend/app/measurement_engine/smpl_anthropometry/data/smpl/

SMPL-Anthropometry measurements → TailorSync codes:
  chest circumference       → M01_chest
  waist circumference       → M03_waist
  hip circumference         → M05_hips
  neck circumference        → M06_neck
  bicep right circumference → M07_bicep
  wrist right circumference → M08_wrist
  thigh left circumference  → M09_thigh
  calf left circumference   → M12_calf
  ankle left circumference  → M13_ankle   ← fixes +20cm bias bug
  arm right length          → M19_sleeve_length
  inside leg height         → M21_inseam
  shoulder to crotch height → M25_torso_length
  shoulder breadth          → M26_shoulder_width  ← fixes +13cm bias bug
  height                    → (validation only)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DATA_DIR = _HERE / "data" / "smpl"
_PKL_PATH = _DATA_DIR / "SMPL_NEUTRAL.pkl"

# TailorSync measurement code → SMPL-Anthropometry name
_TS_TO_SMPL = {
    "M01_chest":        "chest circumference",
    "M03_waist":        "waist circumference",
    "M05_hips":         "hip circumference",
    "M06_neck":         "neck circumference",
    "M07_bicep":        "bicep right circumference",
    "M08_wrist":        "wrist right circumference",
    "M09_thigh":        "thigh left circumference",
    "M12_calf":         "calf left circumference",
    "M13_ankle":        "ankle left circumference",
    "M19_sleeve_length":"arm right length",
    "M21_inseam":       "inside leg height",
    "M25_torso_length": "shoulder to crotch height",
    "M26_shoulder_width":"shoulder breadth",
}


def smpl_pkl_available() -> bool:
    return _PKL_PATH.exists()


def extract_from_smpl_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    height_cm: float,
) -> dict[str, Optional[float]]:
    """
    Extract TailorSync measurements from an SMPL mesh.

    Args:
        vertices:  (N, 3) mesh vertices — must be 6890 for FULL mode.
        faces:     (F, 3) triangle indices.
        height_cm: known body height used for scale normalisation.

    Returns:
        dict of {ts_code: value_cm} for all mappable measurements.
        Returns empty dict if extraction fails.
    """
    if smpl_pkl_available() and len(vertices) == 6890:
        result = _extract_full(vertices, height_cm)
        if result:
            return result
        # FULL mode failed — fall through to trimesh
    return _extract_trimesh(vertices, faces, height_cm)


# ---------------------------------------------------------------------------
# FULL mode — uses MeasureSMPL with PKL model files
# ---------------------------------------------------------------------------

_CHUMPY_STUB = _HERE / "chumpy_stub"

def _inject_chumpy_stub() -> None:
    """Insert our chumpy stub into sys.modules so smplx can load PKL files."""
    if "chumpy" not in sys.modules:
        stub_path = str(_CHUMPY_STUB)
        if stub_path not in sys.path:
            sys.path.insert(0, stub_path)
        import importlib
        import chumpy as _c  # noqa: F401 — side-effect: registers in sys.modules


def _extract_full(vertices: np.ndarray, height_cm: float) -> dict[str, Optional[float]]:
    """Uses the proper SMPL-Anthropometry pipeline."""
    _orig_cwd = os.getcwd()
    try:
        import torch
        _inject_chumpy_stub()

        sys.path.insert(0, str(_HERE))
        # MeasureSMPL hardcodes body_model_root="data"; must run from _HERE
        os.chdir(_HERE)
        from measure import MeasureSMPL
        from measurement_definitions import MEASUREMENT_TYPES

        measurer = MeasureSMPL()
        verts_t = torch.from_numpy(vertices.astype(np.float32))
        measurer.from_verts(verts=verts_t)

        target_names = ["height"] + list(_TS_TO_SMPL.values())
        measurer.measure(target_names)
        measurer.height_normalize_measurements(height_cm)

        normed = measurer.height_normalized_measurements
        result = {}
        for ts_code, smpl_name in _TS_TO_SMPL.items():
            val = normed.get(smpl_name)
            result[ts_code] = round(float(val), 1) if val is not None else None

        logger.info("SMPL-Anthropometry FULL mode: extracted %d measurements", len(result))
        return result

    except Exception as exc:
        logger.warning("SMPL-Anthropometry FULL mode failed: %s — falling back to trimesh", exc)
        return {}
    finally:
        os.chdir(_orig_cwd)


# ---------------------------------------------------------------------------
# TRIMESH mode — uses trimesh plane intersection on any mesh
# Fixes the two geometry bugs from the benchmark:
#   M13 ankle: cuts at correct anatomical height (0.03), not wrong 0.06
#   M26 shoulder: cuts at correct anatomical height using plane normal
# ---------------------------------------------------------------------------

def _extract_trimesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    height_cm: float,
) -> dict[str, Optional[float]]:
    """
    Extracts circumferences via trimesh.intersections.mesh_plane.
    Better than the vertex-proximity convex-hull approach:
      - Uses actual triangle intersections (not just nearby vertices)
      - Correct height ratios per measurement
      - No ankle cross-section bug
    """
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        y_min = float(np.min(vertices[:, 1]))
        y_max = float(np.max(vertices[:, 1]))
        y_span = y_max - y_min

        def circ_at(ratio: float) -> Optional[float]:
            y_cut = y_min + ratio * y_span
            segs = trimesh.intersections.mesh_plane(
                mesh,
                plane_normal=[0, 1, 0],
                plane_origin=[0, y_cut, 0],
            )
            if segs is None or len(segs) == 0:
                return None
            # Scale from mesh units to cm
            scale = height_cm / y_span
            perimeter = float(np.sum(
                np.linalg.norm(segs[:, 1, :] - segs[:, 0, :], axis=1)
            )) * scale
            return round(perimeter, 1)

        def width_at(ratio: float, axis: int = 0) -> Optional[float]:
            """Frontal (axis=0/X) or depth (axis=2/Z) width at height ratio."""
            y_cut = y_min + ratio * y_span
            tol = 0.015 * y_span
            mask = np.abs(vertices[:, 1] - y_cut) < tol
            pts = vertices[mask]
            if len(pts) < 2:
                return None
            scale = height_cm / y_span
            return round(float(np.max(pts[:, axis]) - np.min(pts[:, axis])) * scale, 1)

        # Fixed height ratios based on SMPL-Anthropometry landmark definitions
        result: dict[str, Optional[float]] = {
            "M01_chest":         circ_at(0.76),
            "M03_waist":         circ_at(0.62),
            "M05_hips":          circ_at(0.52),
            "M06_neck":          circ_at(0.87),
            "M07_bicep":         circ_at(0.78),
            "M08_wrist":         None,           # not in parametric mesh arms
            "M09_thigh":         circ_at(0.46),
            "M12_calf":          circ_at(0.14),
            "M13_ankle":         circ_at(0.03),  # FIX: was 0.06 (lower calf)
            "M26_shoulder_width": width_at(0.82, axis=0),  # FIX: was landmark-only
        }

        logger.info(
            "SMPL-Anthropometry TRIMESH mode: extracted %d measurements",
            sum(1 for v in result.values() if v is not None),
        )
        return result

    except Exception as exc:
        logger.warning("SMPL-Anthropometry TRIMESH mode failed: %s", exc)
        return {}
