"""
SMPL body mesh fitter.

Two modes (selected automatically at load time):

  REAL  — SMPL_NEUTRAL_clean.pkl is present.
          Generates a true 6890-vertex SMPL mesh by:
            1. Estimating shape betas from MediaPipe landmark proportions
            2. Optionally refining betas via multi-view silhouette IoU
               optimization (scipy Powell) when body masks are supplied
            3. Applying SMPL shape blend shapes to the template mesh
            4. Scaling to known height
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

import cv2
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
        # Precomputed at load time
        self._v_template_np: Optional[np.ndarray] = None   # (6890, 3) float64
        self._faces_np:      Optional[np.ndarray] = None   # (13776, 3) int32
        self._shapedirs_flat: Optional[np.ndarray] = None  # (6890*3, 10) float64
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
        Precompute:
          - v_template_np, faces_np, shapedirs_flat  (used every scan)
          - dM/d_beta Jacobian for the three landmark-based measurement functions
            (shoulder width, hip width, torso height) for fast beta estimation
        """
        import time
        t0 = time.monotonic()
        d = self._smpl_data
        v_template = np.array(d["v_template"], dtype=np.float64)
        shapedirs  = np.array(d["shapedirs"],  dtype=np.float64)
        n_betas    = 10
        delta      = 2.0

        # Cache for fast per-scan mesh builds
        self._v_template_np  = v_template
        self._faces_np       = np.array(d["f"], dtype=np.int32)
        # (6890, 3, 10) → (6890*3, 10):  v = v_template + (shapedirs_flat @ betas).reshape(N,3)
        self._shapedirs_flat = shapedirs.reshape(-1, n_betas)

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
                vp = v_template + (self._shapedirs_flat @ b_p).reshape(-1, 3)
                vm = v_template + (self._shapedirs_flat @ b_m).reshape(-1, 3)
                hp = float(np.max(vp[:, 1]) - np.min(vp[:, 1]))
                hm = float(np.max(vm[:, 1]) - np.min(vm[:, 1]))
                mp_ = meas_fn(vp) / hp if hp > 0 else 0.0
                mm_ = meas_fn(vm) / hm if hm > 0 else 0.0
                row.append((mp_ - mm_) / (2 * delta))
            jacobian_rows.append(row)
            template_ratios.append(meas_fn(v_template) / h0 if h0 > 0 else 0.0)

        self._jacobian        = np.array(jacobian_rows)
        self._template_ratios = np.array(template_ratios)
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

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def fit(
        self,
        image: Image.Image,
        landmarks: Optional[dict] = None,
        height_cm: float = 170.0,
    ) -> Optional[MeshResult]:
        """Single-view fitting (backward compatible — delegates to fit_multiview)."""
        return self.fit_multiview(landmarks=landmarks, height_cm=height_cm)

    def fit_multiview(
        self,
        landmarks: Optional[dict] = None,
        height_cm: float = 170.0,
        front_mask: Optional[np.ndarray] = None,
        side_mask: Optional[np.ndarray] = None,
        back_mask: Optional[np.ndarray] = None,
    ) -> Optional[MeshResult]:
        """
        Multi-view SMPL fitting.

        1. Estimate initial betas from front-view MediaPipe landmark proportions.
        2. When one or more body masks are supplied (from the segmenter), refine
           betas via scipy Powell minimization of the per-view silhouette IoU loss.
        3. Build and return the final scaled mesh.

        Falls back to the cylinder-stack parametric mesh when the PKL is absent.
        """
        if not self.is_loaded:
            raise RuntimeError("SMPL fitter not loaded")
        try:
            if self._v_template_np is not None:
                return self._fit_real_multiview(
                    landmarks, height_cm, front_mask, side_mask, back_mask
                )
            return self._fit_parametric()
        except Exception as exc:
            logger.error("SMPL fitting failed: %s", exc)
            return None

    def unload(self) -> None:
        self.is_loaded = False
        self._smpl_data = None
        self._v_template_np = None
        self._faces_np = None
        self._shapedirs_flat = None

    # ------------------------------------------------------------------
    # REAL mode — landmark estimate + optional multi-view silhouette refinement
    # ------------------------------------------------------------------

    def _fit_real_multiview(
        self,
        landmarks: Optional[dict],
        height_cm: float,
        front_mask: Optional[np.ndarray],
        side_mask: Optional[np.ndarray],
        back_mask: Optional[np.ndarray],
    ) -> MeshResult:
        v_template    = self._v_template_np
        shapedirs_flat = self._shapedirs_flat

        betas = self._estimate_betas(landmarks, height_cm, v_template, shapedirs_flat)

        # Collect views that have a body mask for silhouette optimization
        views: list[tuple[np.ndarray, str]] = []
        if front_mask is not None:
            views.append((front_mask, "front"))
        if side_mask is not None:
            views.append((side_mask, "side"))
        if back_mask is not None:
            views.append((back_mask, "back"))

        if views:
            betas = self._optimize_betas_multiview(betas, height_cm, views)

        return self._build_mesh(betas, height_cm)

    def _build_mesh(self, betas: np.ndarray, height_cm: float) -> MeshResult:
        """Apply betas, scale to height_cm, translate feet to y=0."""
        v_shaped = self._v_template_np + (self._shapedirs_flat @ betas).reshape(-1, 3)
        v_min = float(np.min(v_shaped[:, 1]))
        v_max = float(np.max(v_shaped[:, 1]))
        mesh_height = v_max - v_min
        scale = height_cm / mesh_height if mesh_height > 1e-6 else 1.0
        v_scaled = v_shaped * scale
        v_scaled[:, 1] -= float(np.min(v_scaled[:, 1]))
        return MeshResult(vertices=v_scaled.astype(np.float32), faces=self._faces_np)

    # ------------------------------------------------------------------
    # Beta estimation — landmark-based (front view only)
    # ------------------------------------------------------------------

    def _estimate_betas(
        self,
        landmarks: Optional[dict],
        height_cm: float,
        v_template: np.ndarray,
        shapedirs_flat: np.ndarray,
        n_betas: int = 10,
    ) -> np.ndarray:
        if landmarks is None:
            return np.zeros(n_betas)

        ratios = self._ratios_from_landmarks(landmarks)
        if ratios is None or self._jacobian is None or self._template_ratios is None:
            return np.zeros(n_betas)

        valid = [(i, r) for i, r in enumerate(ratios) if r is not None]
        if not valid:
            return np.zeros(n_betas)

        idx = [i for i, _ in valid]
        t   = np.array([r - self._template_ratios[i] for i, r in valid])
        J   = self._jacobian[idx]

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
        NOSE, L_ANKLE          = 0, 27
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP           = 23, 24

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
    # B8: Multi-view silhouette optimization
    # ------------------------------------------------------------------

    def _optimize_betas_multiview(
        self,
        betas_init: np.ndarray,
        height_cm: float,
        views: list[tuple[np.ndarray, str]],
    ) -> np.ndarray:
        """
        Refine SMPL shape betas to minimise per-view silhouette IoU loss.

        Loss = Σ_view (1 - IoU(projected_hull(mesh(β)), body_mask_view))
             + λ · ‖β‖²

        Optimizer: scipy Powell (gradient-free, converges well on smooth
        functions, ~100–300 function evaluations for 10 betas).

        Body masks are downsampled to 128 px on the long edge so each IoU
        evaluation is ≈ 0.5 ms; the full optimization completes in < 0.5 s.
        """
        from scipy.optimize import minimize

        # Pre-downsample all masks once — avoids per-iteration resize
        small_views = [(self._downsample_mask(m), name) for m, name in views]

        v_template     = self._v_template_np
        shapedirs_flat = self._shapedirs_flat

        def _loss(betas: np.ndarray) -> float:
            betas = np.asarray(betas, dtype=np.float64)
            v = v_template + (shapedirs_flat @ betas).reshape(-1, 3)
            mesh_h = float(np.max(v[:, 1]) - np.min(v[:, 1]))
            if mesh_h < 1e-6:
                return 10.0
            scale = height_cm / mesh_h
            v = v * scale
            v[:, 1] -= float(np.min(v[:, 1]))

            total = sum(
                1.0 - self._silhouette_iou_fast(v, mask, view_name)
                for mask, view_name in small_views
            )
            total += 0.05 * float(np.dot(betas, betas))   # L2 regularisation
            return total

        loss_init = _loss(betas_init)
        result = minimize(
            _loss,
            betas_init,
            method="Powell",
            options={"maxiter": 300, "ftol": 5e-4, "xtol": 5e-3},
        )
        betas_opt = np.clip(result.x, -3.0, 3.0)

        # Only accept the refined betas when they genuinely improve the loss
        if result.fun < loss_init:
            logger.debug(
                "Multi-view beta refinement: loss %.4f → %.4f (%d iters)",
                loss_init, result.fun, result.nit,
            )
            return betas_opt

        logger.debug("Multi-view beta refinement did not improve loss (%.4f); keeping initial betas", loss_init)
        return betas_init

    @staticmethod
    def _downsample_mask(mask: np.ndarray, target_long_edge: int = 128) -> np.ndarray:
        """Downsample mask to ≤ target_long_edge on the longer side (INTER_NEAREST)."""
        H, W = mask.shape[:2]
        long_edge = max(H, W)
        if long_edge <= target_long_edge:
            return mask
        scale = target_long_edge / long_edge
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _silhouette_iou_fast(
        vertices: np.ndarray,
        mask: np.ndarray,
        view: str,
    ) -> float:
        """
        Convex-hull silhouette IoU for one view.

        View projections (SMPL coordinates: Y=up, X=lateral, Z=depth):
          front — project onto XY plane  (X-axis = width,  -Z toward camera)
          side  — project onto ZY plane  (Z-axis = depth → appears as width)
          back  — project onto XY plane, X mirrored
        """
        H, W = mask.shape[:2]
        ys_m, xs_m = np.where(mask > 0)
        if len(ys_m) == 0:
            return 1.0

        y_max_m = int(ys_m.max())
        y_min_m = int(ys_m.min())
        mask_height_px = y_max_m - y_min_m
        if mask_height_px < 4:
            return 1.0

        mesh_height = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
        if mesh_height < 1e-6:
            return 1.0

        px_per_cm = mask_height_px / mesh_height
        x_center  = float(xs_m.mean())

        if view == "front":
            proj_x = vertices[:, 0]
        elif view == "side":
            proj_x = vertices[:, 2]
        elif view == "back":
            proj_x = -vertices[:, 0]
        else:
            return 1.0

        px = np.clip((proj_x * px_per_cm + x_center).astype(np.int32), 0, W - 1)
        py = np.clip((y_max_m - vertices[:, 1] * px_per_cm).astype(np.int32), 0, H - 1)

        pts  = np.stack([px, py], axis=1)
        hull = cv2.convexHull(pts)

        silhouette = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(silhouette, hull, 255)

        inter = int(np.count_nonzero((silhouette > 0) & (mask > 0)))
        union = int(np.count_nonzero((silhouette > 0) | (mask > 0)))
        return float(inter / union) if union > 0 else 1.0

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
