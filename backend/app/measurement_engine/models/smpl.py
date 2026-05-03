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

    async def load(self) -> None:
        if _PKL_PATH.exists():
            with open(_PKL_PATH, "rb") as f:
                self._smpl_data = pickle.load(f)
            logger.info("SMPL fitter ready (REAL mode — %d verts, %d faces)",
                        np.array(self._smpl_data["v_template"]).shape[0],
                        np.array(self._smpl_data["f"]).shape[0])
        else:
            logger.info("SMPL fitter ready (PARAMETRIC fallback — PKL not found at %s)", _PKL_PATH)
        self.is_loaded = True

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

        Uses visible body-width ratios (shoulder width, hip width) to set
        the body size/shape betas via a least-squares fit against the SMPL
        shape principal components.
        """
        if landmarks is None:
            return np.zeros(n_betas)

        # Extract normalised proportions from front-view landmarks
        # MediaPipe indices: L_SHOULDER=11, R_SHOULDER=12, L_HIP=23, R_HIP=24
        # NOSE=0, L_ANKLE=27
        proportions = self._proportions_from_landmarks(landmarks, height_cm)
        if proportions is None:
            return np.zeros(n_betas)

        # For each beta, compute what measurement it affects by finite difference
        # on the template mesh (cheap, done at fit time)
        delta = 2.0  # beta perturbation magnitude
        target_measurements = []
        jacobian_rows = []

        for name, ratio, meas_fn in proportions:
            target_cm = ratio * height_cm
            row = []
            for i in range(n_betas):
                b_plus  = np.zeros(n_betas); b_plus[i]  =  delta
                b_minus = np.zeros(n_betas); b_minus[i] = -delta
                v_plus  = v_template + np.einsum("ijk,k->ij", shapedirs, b_plus)
                v_minus = v_template + np.einsum("ijk,k->ij", shapedirs, b_minus)
                # Normalise both to unit height
                h_plus  = np.max(v_plus[:, 1])  - np.min(v_plus[:, 1])
                h_minus = np.max(v_minus[:, 1]) - np.min(v_minus[:, 1])
                m_plus  = meas_fn(v_plus)  / h_plus  if h_plus  > 0 else 0.0
                m_minus = meas_fn(v_minus) / h_minus if h_minus > 0 else 0.0
                row.append((m_plus - m_minus) / (2 * delta))
            jacobian_rows.append(row)

            # Target = desired measurement ratio (measurement / height)
            v0 = v_template
            h0 = np.max(v0[:, 1]) - np.min(v0[:, 1])
            m0 = meas_fn(v0) / h0 if h0 > 0 else 0.0
            target_measurements.append(ratio - m0)

        J = np.array(jacobian_rows)
        t = np.array(target_measurements)

        # Regularised least squares: minimise ||J @ betas - t||² + λ||betas||²
        lam = 0.1
        betas, _, _, _ = np.linalg.lstsq(
            J.T @ J + lam * np.eye(n_betas),
            J.T @ t,
            rcond=None,
        )
        # Clamp betas to plausible SMPL range
        return np.clip(betas, -3.0, 3.0)

    def _proportions_from_landmarks(
        self,
        landmarks: dict,
        height_cm: float,
    ) -> Optional[list]:
        """
        Extract observable body proportion ratios from MediaPipe front-view landmarks.
        Returns list of (name, ratio, mesh_measurement_fn) tuples.
        """
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP = 23, 24
        NOSE, L_ANKLE = 0, 27

        lm = landmarks
        proportions = []

        # Body span in normalised coords (needed to convert to ratios)
        nose  = lm.get(NOSE)
        lankl = lm.get(L_ANKLE)
        if not (nose and lankl and nose.visibility > 0.3 and lankl.visibility > 0.3):
            return None
        body_span = abs(lankl.y - nose.y)
        if body_span < 0.05:
            return None

        def px_to_ratio(dx_norm):
            return dx_norm / body_span

        # Shoulder width
        ls, rs = lm.get(L_SHOULDER), lm.get(R_SHOULDER)
        if ls and rs and ls.visibility > 0.5 and rs.visibility > 0.5:
            sw_ratio = px_to_ratio(abs(ls.x - rs.x))
            def _shoulder_width_fn(v):
                # X-span at ~82% height (shoulder level)
                y_min, y_span = np.min(v[:, 1]), np.max(v[:, 1]) - np.min(v[:, 1])
                y_cut = y_min + 0.82 * y_span
                tol = 0.02 * y_span
                pts = v[np.abs(v[:, 1] - y_cut) < tol]
                return float(np.max(pts[:, 0]) - np.min(pts[:, 0])) if len(pts) > 1 else 0.0
            proportions.append(("shoulder_width", sw_ratio, _shoulder_width_fn))

        # Hip width
        lh, rh = lm.get(L_HIP), lm.get(R_HIP)
        if lh and rh and lh.visibility > 0.5 and rh.visibility > 0.5:
            hw_ratio = px_to_ratio(abs(lh.x - rh.x))
            def _hip_width_fn(v):
                y_min, y_span = np.min(v[:, 1]), np.max(v[:, 1]) - np.min(v[:, 1])
                y_cut = y_min + 0.52 * y_span
                tol = 0.02 * y_span
                pts = v[np.abs(v[:, 1] - y_cut) < tol]
                return float(np.max(pts[:, 0]) - np.min(pts[:, 0])) if len(pts) > 1 else 0.0
            proportions.append(("hip_width", hw_ratio, _hip_width_fn))

        # Torso length ratio (shoulder Y to hip Y)
        if ls and lh and ls.visibility > 0.5 and lh.visibility > 0.5:
            torso_ratio = px_to_ratio(abs(lh.y - ls.y))
            def _torso_fn(v):
                y_min, y_span = np.min(v[:, 1]), np.max(v[:, 1]) - np.min(v[:, 1])
                shoulder_y = y_min + 0.82 * y_span
                hip_y = y_min + 0.52 * y_span
                return abs(shoulder_y - hip_y)
            proportions.append(("torso_length", torso_ratio, _torso_fn))

        return proportions if proportions else None

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
