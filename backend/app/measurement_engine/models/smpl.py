"""
SMPL body mesh fitter.

Two modes (selected automatically at load time):

  REAL  — SMPL_NEUTRAL_clean.pkl is present.
          Generates a true 6890-vertex SMPL mesh by:
            1. Estimating shape betas from MediaPipe landmark proportions
            2. Applying SMPL shape blend shapes to the template mesh
            3. Scaling to known height
          This mesh goes into SMPL-Anthropometry for accurate measurements.

  PARAMETRIC — Fallback cylinder-stack mesh (original MVP placeholder).
               Used only when PKL is missing.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_PKL_PATH = (
    Path(__file__).parent.parent
    / "smpl_anthropometry" / "data" / "smpl" / "SMPL_NEUTRAL_clean.pkl"
)


@dataclass
class MeshResult:
    vertices: np.ndarray   # (N, 3) in cm
    faces: np.ndarray      # (F, 3) triangle indices


class SMPLFitter:

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.is_loaded = False
        self._smpl_data: Optional[dict] = None
        # Precomputed at load time so _estimate_betas() does no per-request einsums.
        # Shape: (n_measurement_fns, n_betas=10)
        self._jacobian: Optional[np.ndarray] = None
        # Template baseline ratios (measurement / unit-height) for each fn.
        self._template_ratios: Optional[np.ndarray] = None

    async def load(self) -> None:
        if _PKL_PATH.exists():
            with open(_PKL_PATH, "rb") as f:
                self._smpl_data = pickle.load(f)
            n_verts = np.array(self._smpl_data["v_template"]).shape[0]
            n_faces = np.array(self._smpl_data["f"]).shape[0]
            logger.info("SMPL fitter ready (REAL mode — %d verts, %d faces)", n_verts, n_faces)
            self._precompute_jacobian()
        else:
            logger.info("SMPL fitter ready (PARAMETRIC fallback — PKL not found at %s)", _PKL_PATH)
        self.is_loaded = True

    def _precompute_jacobian(self) -> None:
        """
        Precompute dM/d_beta Jacobian for the three measurement functions used in
        beta estimation.  This is constant for the template mesh and only needs
        to run once at startup (~0.5 s) rather than on every scan request (~1-3 s).
        """
        import time
        t0 = time.monotonic()
        d = self._smpl_data
        v_template = np.array(d["v_template"], dtype=np.float64)
        shapedirs  = np.array(d["shapedirs"],  dtype=np.float64)
        n_betas    = 10
        delta      = 2.0

        # The three measurement functions (must match _proportions_from_landmarks order)
        meas_fns = [
            self._make_shoulder_width_fn(),
            self._make_hip_width_fn(),
            self._make_torso_fn(),
        ]

        h0 = float(np.max(v_template[:, 1]) - np.min(v_template[:, 1]))
        jacobian_rows = []
        template_ratios = []

        for meas_fn in meas_fns:
            row = []
            for i in range(n_betas):
                b_p = np.zeros(n_betas); b_p[i] =  delta
                b_m = np.zeros(n_betas); b_m[i] = -delta
                vp = v_template + np.einsum("ijk,k->ij", shapedirs, b_p)
                vm = v_template + np.einsum("ijk,k->ij", shapedirs, b_m)
                hp = float(np.max(vp[:, 1]) - np.min(vp[:, 1]))
                hm = float(np.max(vm[:, 1]) - np.min(vm[:, 1]))
                mp_ = meas_fn(vp) / hp if hp > 0 else 0.0
                mm_ = meas_fn(vm) / hm if hm > 0 else 0.0
                row.append((mp_ - mm_) / (2 * delta))
            jacobian_rows.append(row)
            template_ratios.append(meas_fn(v_template) / h0 if h0 > 0 else 0.0)

        self._jacobian        = np.array(jacobian_rows)          # (3, 10)
        self._template_ratios = np.array(template_ratios)        # (3,)
        logger.info("SMPL Jacobian precomputed in %.2f s", time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Static measurement functions (used by precompute and estimate_betas)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_shoulder_width_fn():
        def fn(v):
            y_min, y_span = np.min(v[:, 1]), np.max(v[:, 1]) - np.min(v[:, 1])
            y_cut = y_min + 0.82 * y_span
            pts = v[np.abs(v[:, 1] - y_cut) < 0.02 * y_span]
            return float(np.max(pts[:, 0]) - np.min(pts[:, 0])) if len(pts) > 1 else 0.0
        return fn

    @staticmethod
    def _make_hip_width_fn():
        def fn(v):
            y_min, y_span = np.min(v[:, 1]), np.max(v[:, 1]) - np.min(v[:, 1])
            y_cut = y_min + 0.52 * y_span
            pts = v[np.abs(v[:, 1] - y_cut) < 0.02 * y_span]
            return float(np.max(pts[:, 0]) - np.min(pts[:, 0])) if len(pts) > 1 else 0.0
        return fn

    @staticmethod
    def _make_torso_fn():
        def fn(v):
            y_span = float(np.max(v[:, 1]) - np.min(v[:, 1]))
            return abs(0.82 - 0.52) * y_span
        return fn

    def fit(
        self,
        image: Image.Image,
        landmarks: Optional[dict] = None,
        height_cm: float = 170.0,
    ) -> Optional[MeshResult]:
        if not self.is_loaded:
            raise RuntimeError("SMPL fitter not loaded")
        try:
            if self._smpl_data is not None:
                return self._fit_real(landmarks, height_cm)
            return self._fit_parametric()
        except Exception as exc:
            logger.error("SMPL fitting failed: %s", exc)
            return None

    def unload(self) -> None:
        self.is_loaded = False
        self._smpl_data = None

    # ------------------------------------------------------------------
    # REAL mode — SMPL shape model from PKL
    # ------------------------------------------------------------------

    def _fit_real(
        self,
        landmarks: Optional[dict],
        height_cm: float,
    ) -> MeshResult:
        """
        Generate a person-specific SMPL mesh:
          1. Estimate shape betas from MediaPipe landmark proportions
          2. Apply SMPL shape blend shapes: v = v_template + shapedirs @ betas
          3. Scale mesh so its height matches height_cm
        """
        d = self._smpl_data
        v_template  = np.array(d["v_template"], dtype=np.float64)   # (6890, 3)
        shapedirs   = np.array(d["shapedirs"],  dtype=np.float64)   # (6890, 3, 10)
        faces       = np.array(d["f"],          dtype=np.int32)     # (13776, 3)
        J_regressor = d["J_regressor"]                               # sparse (24, 6890)

        betas = self._estimate_betas(landmarks, height_cm, v_template, shapedirs)

        # Apply shape blend shapes
        v_shaped = v_template + np.einsum("ijk,k->ij", shapedirs, betas)

        # Scale to known height (SMPL Y-axis = vertical)
        v_min = float(np.min(v_shaped[:, 1]))
        v_max = float(np.max(v_shaped[:, 1]))
        mesh_height = v_max - v_min
        if mesh_height < 1e-6:
            scale = 1.0
        else:
            scale = height_cm / mesh_height

        v_scaled = v_shaped * scale
        # Translate so feet sit at y=0
        v_scaled[:, 1] -= float(np.min(v_scaled[:, 1]))

        return MeshResult(vertices=v_scaled.astype(np.float32), faces=faces)

    def _estimate_betas(
        self,
        landmarks: Optional[dict],
        height_cm: float,
        v_template: np.ndarray,
        shapedirs: np.ndarray,
        n_betas: int = 10,
    ) -> np.ndarray:
        """
        Estimate SMPL shape parameters from MediaPipe landmark proportions.
        Uses the precomputed Jacobian (dM/d_beta) so no per-request einsums.
        """
        if landmarks is None:
            return np.zeros(n_betas)

        ratios = self._ratios_from_landmarks(landmarks)
        if ratios is None or self._jacobian is None or self._template_ratios is None:
            return np.zeros(n_betas)

        # Build target vector: desired_ratio - template_ratio
        # Only use rows where we have a valid landmark ratio.
        valid = [(i, r) for i, r in enumerate(ratios) if r is not None]
        if not valid:
            return np.zeros(n_betas)

        idx = [i for i, _ in valid]
        t   = np.array([r - self._template_ratios[i] for i, r in valid])
        J   = self._jacobian[idx]                               # (n_valid, 10)

        # Regularised least squares: minimise ||J @ betas - t||² + λ||betas||²
        lam = 0.1
        betas, _, _, _ = np.linalg.lstsq(
            J.T @ J + lam * np.eye(n_betas),
            J.T @ t,
            rcond=None,
        )
        return np.clip(betas, -3.0, 3.0)

    def _ratios_from_landmarks(
        self,
        landmarks: dict,
    ) -> Optional[list[Optional[float]]]:
        """
        Extract the three body proportion ratios (shoulder_width, hip_width,
        torso_length) that correspond to the three rows of _jacobian.
        Returns [ratio_or_None, ratio_or_None, ratio_or_None].
        Returns None if body span cannot be determined (no scale anchor).
        """
        NOSE, L_ANKLE       = 0, 27
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP        = 23, 24

        nose  = landmarks.get(NOSE)
        lankl = landmarks.get(L_ANKLE)
        if not (nose and lankl and nose.visibility > 0.3 and lankl.visibility > 0.3):
            return None
        body_span = abs(lankl.y - nose.y)
        if body_span < 0.05:
            return None

        def norm(dx: float) -> float:
            return dx / body_span

        ls, rs = landmarks.get(L_SHOULDER), landmarks.get(R_SHOULDER)
        lh, rh = landmarks.get(L_HIP),      landmarks.get(R_HIP)

        sw = norm(abs(ls.x - rs.x)) if (ls and rs and ls.visibility > 0.5 and rs.visibility > 0.5) else None
        hw = norm(abs(lh.x - rh.x)) if (lh and rh and lh.visibility > 0.5 and rh.visibility > 0.5) else None
        tl = norm(abs(lh.y - ls.y)) if (ls and lh and ls.visibility > 0.5 and lh.visibility > 0.5) else None

        return [sw, hw, tl]

    # ------------------------------------------------------------------
    # PARAMETRIC fallback — cylinder-stack (original MVP)
    # ------------------------------------------------------------------

    def _fit_parametric(
        self,
        n_rings: int = 60,
        pts_per_ring: int = 32,
    ) -> MeshResult:
        """Cylinder-stack body mesh in cm (height 0→170)."""
        HEIGHT = 170.0
        profile = [
            (0.00, 0.040, 0.035),
            (0.06, 0.045, 0.038),
            (0.14, 0.055, 0.045),
            (0.20, 0.050, 0.040),
            (0.28, 0.060, 0.050),
            (0.38, 0.075, 0.055),
            (0.46, 0.090, 0.065),
            (0.52, 0.095, 0.075),
            (0.58, 0.080, 0.065),
            (0.62, 0.070, 0.055),
            (0.67, 0.078, 0.060),
            (0.73, 0.090, 0.068),
            (0.76, 0.098, 0.072),
            (0.80, 0.100, 0.065),
            (0.82, 0.105, 0.058),
            (0.85, 0.042, 0.038),
            (0.88, 0.038, 0.034),
            (0.91, 0.055, 0.055),
            (0.96, 0.068, 0.068),
            (1.00, 0.058, 0.058),
        ]
        vertices, ring_counts = [], []
        for h_ratio, hw_x, hw_z in profile:
            y = h_ratio * HEIGHT
            for i in range(pts_per_ring):
                angle = 2.0 * np.pi * i / pts_per_ring
                vertices.append([hw_x * HEIGHT * np.cos(angle), y, hw_z * HEIGHT * np.sin(angle)])
            ring_counts.append(pts_per_ring)

        verts = np.array(vertices, dtype=np.float32)
        faces, offset = [], 0
        for r, n_pts in enumerate(ring_counts[:-1]):
            n_next = ring_counts[r + 1]
            n_min = min(n_pts, n_next)
            for i in range(n_min):
                a, b = offset + i, offset + (i + 1) % n_pts
                c, d = offset + n_pts + i % n_next, offset + n_pts + (i + 1) % n_next
                faces += [[a, b, c], [b, d, c]]
            offset += n_pts

        return MeshResult(vertices=verts, faces=np.array(faces, dtype=np.int32))
