"""
TailorSync 32-Measurement Extractor.

Implements the full M01–M32 body measurement taxonomy defined in the
TailorSync Complete Measurement Specification.

Extraction methods (in priority order per measurement):
  1. SMPL mesh circumference / geodesic distance  (most accurate)
  2. MediaPipe landmark geometry scaled by height  (good for lengths/widths)
  3. Height-ratio anthropometric estimate           (fallback, LOW confidence)

All output values are in centimetres.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


def _mesh_cross_section_ring(
    segs: np.ndarray,
) -> Optional[tuple[float, np.ndarray]]:
    """
    Given trimesh plane-intersection segments (N, 2, 3), isolate the largest
    connected ring and return (perimeter_mesh_units, xz_pts).

    Handles multiple disconnected rings produced by a horizontal cut through a
    body mesh:
      - Upper body (chest/waist):  torso ring is largest; arm rings are smaller.
      - Lower body (knee/calf):    each leg forms its own ring; returns the larger
                                   one rather than a convex hull spanning both.
    """
    if len(segs) == 0:
        return None

    # Project onto the XZ cutting plane (all Y values are ≈ y_target on the plane).
    pts2d = segs[:, :, [0, 2]]   # (N, 2, 2) — axis-0 = X, axis-1 = Z

    # Union-Find: group segments that share an endpoint.
    SNAP = 1e-4  # 0.001 mm — tight enough for trimesh float precision
    parent = list(range(len(segs)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        pa, pb = _find(a), _find(b)
        if pa != pb:
            parent[pa] = pb

    endpoint_map: dict = defaultdict(list)
    for i, seg in enumerate(pts2d):
        for pt in seg:
            key = (int(pt[0] / SNAP), int(pt[1] / SNAP))
            endpoint_map[key].append(i)

    for members in endpoint_map.values():
        for j in range(1, len(members)):
            _union(members[0], members[j])

    # Accumulate perimeter length and XZ points per component.
    comp_len: dict[int, float] = defaultdict(float)
    comp_pts: dict[int, list] = defaultdict(list)
    for i, seg in enumerate(pts2d):
        root = _find(i)
        comp_len[root] += float(np.linalg.norm(seg[1] - seg[0]))
        comp_pts[root].extend([seg[0], seg[1]])

    if not comp_len:
        return None

    best = max(comp_len, key=lambda r: comp_len[r])
    return comp_len[best], np.array(comp_pts[best])

# SMPL-Anthropometry integration (optional — degrades gracefully without PKL files)
try:
    from app.measurement_engine.smpl_anthropometry.adapter import (
        extract_from_smpl_mesh,
        smpl_pkl_available,
    )
    _SMPL_ANTHRO_AVAILABLE = True
except ImportError:
    _SMPL_ANTHRO_AVAILABLE = False
    def extract_from_smpl_mesh(*_a, **_kw):  # type: ignore[misc]
        return {}
    def smpl_pkl_available() -> bool:  # type: ignore[misc]
        return False


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass
class LandmarkPoint:
    x: float   # normalised 0-1
    y: float
    z: float
    visibility: float


@dataclass
class RawMeasurements:
    """Holds all extracted values before confidence scoring."""

    height_cm: float

    # Section A
    M01_chest:       Optional[float] = None
    M02_under_bust:  Optional[float] = None
    M03_waist:       Optional[float] = None
    M04_abdomen:     Optional[float] = None
    M05_hips:        Optional[float] = None
    M06_neck:        Optional[float] = None
    M07_bicep:       Optional[float] = None
    M08_wrist:       Optional[float] = None

    # Section B
    M09_thigh:       Optional[float] = None
    M10_mid_thigh:   Optional[float] = None
    M11_knee:        Optional[float] = None
    M12_calf:        Optional[float] = None
    M13_ankle:       Optional[float] = None

    # Section C
    M15_shoulder_to_waist_front: Optional[float] = None
    M16_shoulder_to_waist_back:  Optional[float] = None
    M17_kameez_length:           Optional[float] = None
    M18_dress_length:            Optional[float] = None
    M19_sleeve_length:           Optional[float] = None
    M20_sleeve_length_elbow:     Optional[float] = None
    M21_inseam:                  Optional[float] = None
    M22_outseam:                 Optional[float] = None
    M23_crotch_depth_front:      Optional[float] = None
    M24_crotch_depth_back:       Optional[float] = None
    M25_torso_length:            Optional[float] = None

    # Section D
    M26_shoulder_width: Optional[float] = None
    M27_chest_width:    Optional[float] = None
    M28_back_width:     Optional[float] = None
    M29_hip_width:      Optional[float] = None
    M30_chest_depth:    Optional[float] = None
    M31_waist_depth:    Optional[float] = None
    M32_armhole_depth:  Optional[float] = None

    # Source tracking per field
    sources: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MediaPipe landmark indices (BlazePose 33-point model)
# ---------------------------------------------------------------------------
class _L:
    NOSE          = 0
    L_SHOULDER    = 11
    R_SHOULDER    = 12
    L_ELBOW       = 13
    R_ELBOW       = 14
    L_WRIST       = 15
    R_WRIST       = 16
    L_HIP         = 23
    R_HIP         = 24
    L_KNEE        = 25
    R_KNEE        = 26
    L_ANKLE       = 27
    R_ANKLE       = 28


# ---------------------------------------------------------------------------
# Anthropometric ratios relative to height
# (derived from ANSUR II dataset averages — used as fallback estimates)
# ---------------------------------------------------------------------------
_RATIOS = {
    "chest_circ":          0.530,
    "under_bust_circ":     0.460,
    "waist_circ":          0.435,
    "abdomen_circ":        0.480,
    "hip_circ":            0.540,
    "neck_circ":           0.215,
    "bicep_circ":          0.175,
    "wrist_circ":          0.100,
    "thigh_circ":          0.320,
    "mid_thigh_circ":      0.270,
    "knee_circ":           0.230,
    "calf_circ":           0.215,
    "ankle_circ":          0.135,
    "shoulder_to_waist_f": 0.245,
    "shoulder_to_waist_b": 0.235,
    "kameez_length":       0.580,
    "dress_length":        0.680,
    "sleeve_length":       0.320,
    "sleeve_elbow":        0.185,
    "inseam":              0.465,
    "outseam":             0.590,
    "crotch_front":        0.170,
    "crotch_back":         0.175,
    "torso_length":        0.310,
    "shoulder_width":      0.236,
    "chest_width":         0.200,
    "back_width":          0.195,
    "hip_width":           0.175,
    "chest_depth":         0.135,
    "waist_depth":         0.110,
    "armhole_depth":       0.115,
}


# ---------------------------------------------------------------------------
# Clothing compensation offsets — SCAN-08
#
# The SMPL mesh is fitted on a clothed body.  These offsets subtract the
# estimated fabric thickness from circumference measurements so the tailor
# receives true body measurements, not clothed-body measurements.
#
# Values are means from the MTailor / NIST clothing thickness study (cm).
# A future version should make these configurable per clothing category
# submitted by the mobile app (e.g. "wearing_thick_hoodie=true").
# ---------------------------------------------------------------------------

_CLOTHING_OFFSETS_CM: dict[str, float] = {
    # Values from PRD Complete Measurement Specification §3.6 (clothing compensation)
    "chest":      0.8,
    "under_bust": 0.6,
    "waist":      0.4,
    "abdomen":    0.5,
    "hips":       0.6,
    "neck":       0.2,
    "bicep":      0.4,
    "wrist":      0.0,
    "thigh":      0.5,
    "mid_thigh":  0.4,
    "knee":       0.3,
    "calf":       0.3,
    "ankle":      0.0,
}


def apply_clothing_compensation(
    value_cm: Optional[float],
    key: str,
) -> Optional[float]:
    """
    Subtract clothing thickness offset from a mesh-derived circumference.
    Only applied when source is 'smpl_mesh' (landmark-based values are not
    affected by clothing thickness in the same way).
    """
    if value_cm is None:
        return None
    offset = _CLOTHING_OFFSETS_CM.get(key, 0.0)
    return round(max(value_cm - offset, value_cm * 0.85), 1)  # never reduce by more than 15%


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class MeasurementExtractor:
    """
    Extracts all 32 TailorSync measurements from available data.

    Args:
        height_cm:       User-provided height — the scale anchor.
        landmarks:       Dict of MediaPipe landmark index → LandmarkPoint.
                         May contain front-view and/or side-view landmarks.
        mesh_vertices:   (N, 3) SMPL mesh vertices in normalised mesh coords.
                         If provided, circumferences are extracted from the mesh.
        front_landmarks: Landmarks from the FRONT frame.
        side_landmarks:  Landmarks from the SIDE_LEFT frame (for depth measures).
        back_landmarks:  Landmarks from the BACK frame (for back widths).
        arms_landmarks:  Landmarks from the ARMS_OUT frame (for sleeve/bicep).
    """

    def __init__(
        self,
        height_cm: float,
        front_landmarks:  Optional[dict[int, LandmarkPoint]] = None,
        side_landmarks:   Optional[dict[int, LandmarkPoint]] = None,
        back_landmarks:   Optional[dict[int, LandmarkPoint]] = None,
        arms_landmarks:   Optional[dict[int, LandmarkPoint]] = None,
        mesh_vertices:    Optional[np.ndarray] = None,
        mesh_faces:       Optional[np.ndarray] = None,
        img_aspect_ratio: float = 1.0,
    ):
        self.height_cm = height_cm
        self.front = front_landmarks or {}
        self.side  = side_landmarks  or {}
        self.back  = back_landmarks  or {}
        self.arms  = arms_landmarks  or {}
        self.mesh  = mesh_vertices
        self.mesh_faces = mesh_faces

        # MediaPipe normalises x to [0, img_width] and y to [0, img_height]
        # independently.  _px_to_cm is calibrated on the Y axis (body height span).
        # Horizontal (X) measurements need the aspect ratio to convert correctly.
        # For a 1080×1920 portrait: aspect = 1080/1920 = 0.5625.
        self._aspect_ratio = img_aspect_ratio

        # Scale factor: maps 1 unit of normalised Y distance → cm
        self._px_to_cm = self._compute_pixel_scale()

        # Mesh scale: if mesh provided, scale mesh-unit → cm
        self._mesh_scale = self._compute_mesh_scale() if self.mesh is not None else None

        # SMPL-Anthropometry override cache — populated once in extract()
        self._smpl_anthro: dict[str, Optional[float]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self) -> RawMeasurements:
        r = RawMeasurements(height_cm=self.height_cm)

        # Pre-compute SMPL-Anthropometry measurements (overrides mesh-based where available)
        if _SMPL_ANTHRO_AVAILABLE and self.mesh is not None and self.mesh_faces is not None:
            self._smpl_anthro = extract_from_smpl_mesh(
                vertices=self.mesh,
                faces=self.mesh_faces,
                height_cm=self.height_cm,
            )
            mode = "FULL" if smpl_pkl_available() else "TRIMESH"
            logger.debug("SMPL-Anthropometry (%s): %d measurements", mode, len(self._smpl_anthro))

        # Section A — circumferences
        r.M01_chest,      r.sources["M01"] = self._chest_circ()
        r.M02_under_bust, r.sources["M02"] = self._under_bust_circ()
        r.M03_waist,      r.sources["M03"] = self._waist_circ()
        r.M04_abdomen,    r.sources["M04"] = self._abdomen_circ()
        r.M05_hips,       r.sources["M05"] = self._hip_circ()
        r.M06_neck,       r.sources["M06"] = self._neck_circ()
        r.M07_bicep,      r.sources["M07"] = self._bicep_circ()
        r.M08_wrist,      r.sources["M08"] = self._wrist_circ()

        # Section B — lower circumferences
        r.M09_thigh,     r.sources["M09"] = self._thigh_circ()
        r.M10_mid_thigh, r.sources["M10"] = self._mid_thigh_circ()
        r.M11_knee,      r.sources["M11"] = self._knee_circ()
        r.M12_calf,      r.sources["M12"] = self._calf_circ()
        r.M13_ankle,     r.sources["M13"] = self._ankle_circ()

        # Section C — lengths
        r.M15_shoulder_to_waist_front, r.sources["M15"] = self._shoulder_to_waist_front()
        r.M16_shoulder_to_waist_back,  r.sources["M16"] = self._shoulder_to_waist_back()
        r.M17_kameez_length,           r.sources["M17"] = self._kameez_length()
        r.M18_dress_length,            r.sources["M18"] = self._dress_length()
        r.M19_sleeve_length,           r.sources["M19"] = self._sleeve_length()
        r.M20_sleeve_length_elbow,     r.sources["M20"] = self._sleeve_length_elbow()
        r.M21_inseam,                  r.sources["M21"] = self._inseam()
        r.M22_outseam,                 r.sources["M22"] = self._outseam()
        r.M23_crotch_depth_front,      r.sources["M23"] = self._crotch_depth_front()
        r.M24_crotch_depth_back,       r.sources["M24"] = self._crotch_depth_back()
        r.M25_torso_length,            r.sources["M25"] = self._torso_length()

        # Section D — widths & depths
        r.M26_shoulder_width, r.sources["M26"] = self._shoulder_width()
        r.M27_chest_width,    r.sources["M27"] = self._chest_width()
        r.M28_back_width,     r.sources["M28"] = self._back_width()
        r.M29_hip_width,      r.sources["M29"] = self._hip_width()
        r.M30_chest_depth,    r.sources["M30"] = self._chest_depth()
        r.M31_waist_depth,    r.sources["M31"] = self._waist_depth()
        r.M32_armhole_depth,  r.sources["M32"] = self._armhole_depth()

        return r

    # ------------------------------------------------------------------
    # Scale helpers
    # ------------------------------------------------------------------

    def _compute_pixel_scale(self) -> float:
        """Returns cm per normalised-coordinate unit from the front frame."""
        nose  = self.front.get(_L.NOSE)
        lankl = self.front.get(_L.L_ANKLE)
        rankl = self.front.get(_L.R_ANKLE)

        if nose and (lankl or rankl):
            ankle = lankl or rankl
            body_span_norm = abs(ankle.y - nose.y)
            if body_span_norm > 0.05:
                return self.height_cm / body_span_norm

        # Fallback: assume body occupies ~90% of frame height
        return self.height_cm / 0.90

    def _compute_mesh_scale(self) -> float:
        """Returns cm per mesh-unit (Y axis = height)."""
        if self.mesh is None:
            return 1.0
        mesh_height = float(np.max(self.mesh[:, 1]) - np.min(self.mesh[:, 1]))
        if mesh_height < 1e-6:
            return self.height_cm
        return self.height_cm / mesh_height

    # ------------------------------------------------------------------
    # Landmark-based measurement helpers
    # ------------------------------------------------------------------

    def _lm_dist_y(self, lm_a: LandmarkPoint, lm_b: LandmarkPoint) -> float:
        """Vertical distance (Y axis) between two normalised landmarks → cm."""
        return abs(lm_a.y - lm_b.y) * self._px_to_cm

    def _lm_dist_x(self, lm_a: LandmarkPoint, lm_b: LandmarkPoint) -> float:
        """Horizontal distance (X axis) → cm, corrected for image aspect ratio."""
        return abs(lm_a.x - lm_b.x) * self._px_to_cm * self._aspect_ratio

    def _ratio_fallback(self, key: str) -> tuple[float, str]:
        return round(self.height_cm * _RATIOS[key], 1), "height_ratio"

    def _smpl_anthro_or(
        self,
        ts_code: str,
        fallback_fn,
    ) -> tuple[Optional[float], str]:
        """Return SMPL-Anthropometry value when available, else call fallback_fn."""
        val = self._smpl_anthro.get(ts_code)
        if val is not None:
            source = "smpl_anthro_full" if smpl_pkl_available() else "smpl_anthro_trimesh"
            return val, source
        return fallback_fn()

    # ------------------------------------------------------------------
    # SMPL mesh circumference extraction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Shared helper: trimesh plane cut → largest connected ring
    # ------------------------------------------------------------------

    def _ring_at_height_ratio(
        self,
        height_ratio: float,
    ) -> Optional[tuple[float, np.ndarray]]:
        """
        Cut the SMPL mesh at `height_ratio` (0=feet, 1=head) with a horizontal
        plane and return the largest connected ring as (perimeter_cm, xz_pts).

        Uses trimesh triangle-plane intersection instead of vertex proximity so
        the result is topologically correct:
          - Upper body: torso ring is returned; arm rings (smaller) are discarded.
          - Lower body: the larger of the two leg rings is returned; the convex
            hull across both legs is never computed.
        """
        if self.mesh is None or self.mesh_faces is None or self._mesh_scale is None:
            return None
        try:
            import trimesh
            mesh_tm = trimesh.Trimesh(
                vertices=self.mesh, faces=self.mesh_faces, process=False
            )
            y_min = float(np.min(self.mesh[:, 1]))
            y_span = float(np.max(self.mesh[:, 1])) - y_min
            y_target = y_min + height_ratio * y_span

            segs = trimesh.intersections.mesh_plane(
                mesh_tm,
                plane_normal=[0, 1, 0],
                plane_origin=[0, y_target, 0],
            )
            if segs is None or len(segs) == 0:
                return None
            return _mesh_cross_section_ring(segs)
        except Exception:
            return None

    def _mesh_circumference_at_height_ratio(
        self,
        height_ratio: float,
        tolerance_ratio: float = 0.012,
    ) -> Optional[float]:
        """
        Circumference of the largest body-part ring at `height_ratio`.
        Prefers trimesh plane intersection (topologically correct) and falls
        back to vertex-proximity convex hull only when trimesh is unavailable.
        """
        ring = self._ring_at_height_ratio(height_ratio)
        if ring is not None:
            perimeter, _ = ring
            return round(perimeter * self._mesh_scale, 1)

        # Fallback: vertex-proximity convex hull (no body-part isolation)
        if self.mesh is None or self._mesh_scale is None:
            return None
        y_min = float(np.min(self.mesh[:, 1]))
        y_span = float(np.max(self.mesh[:, 1])) - y_min
        y_target = y_min + height_ratio * y_span
        mask = np.abs(self.mesh[:, 1] - y_target) < tolerance_ratio * y_span
        pts = self.mesh[mask]
        if len(pts) < 4:
            return None
        pts_xz = pts[:, [0, 2]]
        try:
            hull = ConvexHull(pts_xz)
            verts = pts_xz[hull.vertices]
            perimeter = float(np.sum(
                np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1)
            ))
            return round(perimeter * self._mesh_scale, 1)
        except Exception:
            return None

    def _mesh_width_at_height_ratio(self, height_ratio: float) -> Optional[float]:
        """Frontal (X-axis) width of the largest ring at a given height ratio."""
        ring = self._ring_at_height_ratio(height_ratio)
        if ring is not None:
            _, xz_pts = ring
            return round(float(np.max(xz_pts[:, 0]) - np.min(xz_pts[:, 0])) * self._mesh_scale, 1)

        # Fallback: vertex proximity (includes arms / both legs)
        if self.mesh is None:
            return None
        y_min = float(np.min(self.mesh[:, 1]))
        y_span = float(np.max(self.mesh[:, 1])) - y_min
        y_target = y_min + height_ratio * y_span
        mask = np.abs(self.mesh[:, 1] - y_target) < 0.015 * y_span
        pts = self.mesh[mask]
        if len(pts) < 2:
            return None
        return round(float(np.max(pts[:, 0]) - np.min(pts[:, 0])) * self._mesh_scale, 1)

    def _mesh_depth_at_height_ratio(self, height_ratio: float) -> Optional[float]:
        """Anteroposterior (Z-axis) depth of the largest ring at a given height ratio."""
        ring = self._ring_at_height_ratio(height_ratio)
        if ring is not None:
            _, xz_pts = ring
            # xz_pts columns: 0=X, 1=Z
            return round(float(np.max(xz_pts[:, 1]) - np.min(xz_pts[:, 1])) * self._mesh_scale, 1)

        # Fallback: vertex proximity
        if self.mesh is None:
            return None
        y_min = float(np.min(self.mesh[:, 1]))
        y_span = float(np.max(self.mesh[:, 1])) - y_min
        y_target = y_min + height_ratio * y_span
        mask = np.abs(self.mesh[:, 1] - y_target) < 0.015 * y_span
        pts = self.mesh[mask]
        if len(pts) < 2:
            return None
        return round(float(np.max(pts[:, 2]) - np.min(pts[:, 2])) * self._mesh_scale, 1)

    # ------------------------------------------------------------------
    # Section A — Upper body circumferences
    # ------------------------------------------------------------------

    def _chest_circ(self) -> tuple[Optional[float], str]:
        def _fallback():
            v = self._mesh_circumference_at_height_ratio(0.76)
            if v:
                return apply_clothing_compensation(v, "chest"), "smpl_mesh"
            return self._ratio_fallback("chest_circ")
        return self._smpl_anthro_or("M01_chest", _fallback)

    def _under_bust_circ(self) -> tuple[Optional[float], str]:
        # The under-bust band sits below the breast at ~70% of standing height,
        # noticeably below the chest line (0.76). Sampling at 0.73 caught the
        # bust itself and produced |Z|≈3.3 against the norm in the proof run.
        v = self._mesh_circumference_at_height_ratio(0.70)
        if v:
            return apply_clothing_compensation(v, "under_bust"), "smpl_mesh"
        return self._ratio_fallback("under_bust_circ")

    def _waist_circ(self) -> tuple[Optional[float], str]:
        def _fallback():
            v = self._mesh_circumference_at_height_ratio(0.62)
            if v:
                return apply_clothing_compensation(v, "waist"), "smpl_mesh"
            return self._ratio_fallback("waist_circ")
        return self._smpl_anthro_or("M03_waist", _fallback)

    def _abdomen_circ(self) -> tuple[Optional[float], str]:
        v = self._mesh_circumference_at_height_ratio(0.58)
        if v:
            return apply_clothing_compensation(v, "abdomen"), "smpl_mesh"
        return self._ratio_fallback("abdomen_circ")

    def _hip_circ(self) -> tuple[Optional[float], str]:
        def _fallback():
            v = self._mesh_circumference_at_height_ratio(0.52)
            if v:
                return apply_clothing_compensation(v, "hips"), "smpl_mesh"
            return self._ratio_fallback("hip_circ")
        return self._smpl_anthro_or("M05_hips", _fallback)

    def _neck_circ(self) -> tuple[Optional[float], str]:
        def _fallback():
            v = self._mesh_circumference_at_height_ratio(0.87)
            if v:
                return apply_clothing_compensation(v, "neck"), "smpl_mesh"
            return self._ratio_fallback("neck_circ")
        return self._smpl_anthro_or("M06_neck", _fallback)

    def _bicep_circ(self) -> tuple[Optional[float], str]:
        def _fallback():
            return self._ratio_fallback("bicep_circ")
        return self._smpl_anthro_or("M07_bicep", _fallback)

    def _wrist_circ(self) -> tuple[Optional[float], str]:
        return self._smpl_anthro_or("M08_wrist", lambda: self._ratio_fallback("wrist_circ"))

    # ------------------------------------------------------------------
    # Section B — Lower body circumferences
    # ------------------------------------------------------------------

    def _thigh_circ(self) -> tuple[Optional[float], str]:
        def _fallback():
            return self._ratio_fallback("thigh_circ")
        return self._smpl_anthro_or("M09_thigh", _fallback)

    def _mid_thigh_circ(self) -> tuple[Optional[float], str]:
        v = self._mesh_circumference_at_height_ratio(0.38)
        if v:
            return apply_clothing_compensation(v, "mid_thigh"), "smpl_mesh"
        return self._ratio_fallback("mid_thigh_circ")

    def _knee_circ(self) -> tuple[Optional[float], str]:
        v = self._mesh_circumference_at_height_ratio(0.28)
        if v:
            return apply_clothing_compensation(v, "knee"), "smpl_mesh"
        return self._ratio_fallback("knee_circ")

    def _calf_circ(self) -> tuple[Optional[float], str]:
        def _fallback():
            return self._ratio_fallback("calf_circ")
        return self._smpl_anthro_or("M12_calf", _fallback)

    def _ankle_circ(self) -> tuple[Optional[float], str]:
        def _fallback():
            # 0.03 targets the ankle ring; 0.06 was the lower-calf ring (benchmark bug)
            v = self._mesh_circumference_at_height_ratio(0.03)
            if v:
                return apply_clothing_compensation(v, "ankle"), "smpl_mesh"
            return self._ratio_fallback("ankle_circ")
        return self._smpl_anthro_or("M13_ankle", _fallback)

    # ------------------------------------------------------------------
    # Section C — Lengths & heights
    # ------------------------------------------------------------------

    def _shoulder_to_waist_front(self) -> tuple[Optional[float], str]:
        ls = self.front.get(_L.L_SHOULDER)
        lh = self.front.get(_L.L_HIP)
        if ls and lh and ls.visibility > 0.5 and lh.visibility > 0.5:
            # Waist is approximately 60% of the way from shoulder to hip
            dist = self._lm_dist_y(ls, lh) * 0.65
            return round(dist, 1), "landmark"
        return self._ratio_fallback("shoulder_to_waist_f")

    def _shoulder_to_waist_back(self) -> tuple[Optional[float], str]:
        ls = self.back.get(_L.L_SHOULDER) or self.front.get(_L.L_SHOULDER)
        lh = self.back.get(_L.L_HIP) or self.front.get(_L.L_HIP)
        if ls and lh and ls.visibility > 0.5 and lh.visibility > 0.5:
            dist = self._lm_dist_y(ls, lh) * 0.60
            return round(dist, 1), "landmark"
        return self._ratio_fallback("shoulder_to_waist_b")

    def _kameez_length(self) -> tuple[Optional[float], str]:
        # Back neck to mid-thigh — typical kameez hem
        ls = self.front.get(_L.L_SHOULDER)
        lk = self.front.get(_L.L_KNEE)
        if ls and lk and ls.visibility > 0.5 and lk.visibility > 0.5:
            dist = self._lm_dist_y(ls, lk) * 0.90
            return round(dist, 1), "landmark"
        return self._ratio_fallback("kameez_length")

    def _dress_length(self) -> tuple[Optional[float], str]:
        ls = self.front.get(_L.L_SHOULDER)
        la = self.front.get(_L.L_ANKLE)
        if ls and la and ls.visibility > 0.5 and la.visibility > 0.5:
            # Shoulder→ankle spans ≈ 86% of height; ANSUR-II "dress length" lands
            # near mid-calf (≈ 67% of height). 0.78 of shoulder→ankle ≈ 67% of height,
            # matching both the norm and the height-ratio fallback (0.680).
            dist = self._lm_dist_y(ls, la) * 0.78
            return round(dist, 1), "landmark"
        return self._ratio_fallback("dress_length")

    def _sleeve_length(self) -> tuple[Optional[float], str]:
        def _fallback():
            # Only use landmark distance when arms are extended (arms_out pose).
            # In a standard standing photo arms hang at sides, making shoulder→wrist
            # distance ~17 cm longer than actual sleeve (benchmark bias confirmed).
            src = self.arms  # arms-at-sides (self.front) excluded intentionally
            if src:
                ls = src.get(_L.L_SHOULDER)
                lw = src.get(_L.L_WRIST)
                if ls and lw and ls.visibility > 0.5 and lw.visibility > 0.5:
                    dx = (ls.x - lw.x) * self._px_to_cm
                    dy = (ls.y - lw.y) * self._px_to_cm
                    dist = round(float(np.sqrt(dx ** 2 + dy ** 2)), 1)
                    return dist, "landmark"
            return self._ratio_fallback("sleeve_length")
        return self._smpl_anthro_or("M19_sleeve_length", _fallback)

    def _sleeve_length_elbow(self) -> tuple[Optional[float], str]:
        # Mirror the M19 fix: only use the arms-out pose. With arms at sides,
        # shoulder→elbow projects onto a vertical line and overstates length.
        src = self.arms
        if src:
            ls = src.get(_L.L_SHOULDER)
            le = src.get(_L.L_ELBOW)
            if ls and le and ls.visibility > 0.5 and le.visibility > 0.5:
                dx = (ls.x - le.x) * self._px_to_cm
                dy = (ls.y - le.y) * self._px_to_cm
                dist = round(float(np.sqrt(dx ** 2 + dy ** 2)), 1)
                return dist, "landmark"
        return self._ratio_fallback("sleeve_elbow")

    def _inseam(self) -> tuple[Optional[float], str]:
        lh = self.front.get(_L.L_HIP)
        la = self.front.get(_L.L_ANKLE)
        if lh and la and lh.visibility > 0.5 and la.visibility > 0.5:
            dist = self._lm_dist_y(lh, la)
            return round(dist, 1), "landmark"
        return self._ratio_fallback("inseam")

    def _outseam(self) -> tuple[Optional[float], str]:
        # Outseam = waist to floor along outer leg. MediaPipe has no waist landmark
        # so we cannot compute this directly. The previous approach (inseam + 10 cm)
        # was hardcoding a rise that varies 18–24 cm in practice.
        # Height-ratio fallback (0.590 × height) is more accurate than a fixed offset.
        return self._ratio_fallback("outseam")

    def _crotch_depth_front(self) -> tuple[Optional[float], str]:
        # PRD M23: front rise (waist to crotch along body). No crotch landmark in
        # MediaPipe and the body mesh has no annotated crotch vertex either.
        # SMPL-Anthropometry exposes a calibrated rise; otherwise fall back to
        # the height-ratio estimate.
        return self._smpl_anthro_or(
            "M23_crotch_depth_front",
            lambda: self._ratio_fallback("crotch_front"),
        )

    def _crotch_depth_back(self) -> tuple[Optional[float], str]:
        # Back rise typically 4% longer than front rise.
        def _fallback() -> tuple[Optional[float], str]:
            front, source = self._crotch_depth_front()
            if front is not None and source != "height_ratio":
                return round(front * 1.04, 1), source
            return self._ratio_fallback("crotch_back")
        return self._smpl_anthro_or("M24_crotch_depth_back", _fallback)

    def _torso_length(self) -> tuple[Optional[float], str]:
        def _fallback():
            # PRD M25: neck to crotch. MediaPipe has no crotch landmark; hip joint
            # is the closest proxy (~2-4 cm above the perineum in standing pose).
            ls = self.front.get(_L.L_SHOULDER)
            lh = self.front.get(_L.L_HIP)
            if ls and lh and ls.visibility > 0.5 and lh.visibility > 0.5:
                dist = self._lm_dist_y(ls, lh)
                return round(dist, 1), "landmark"
            return self._ratio_fallback("torso_length")
        return self._smpl_anthro_or("M25_torso_length", _fallback)

    # ------------------------------------------------------------------
    # Section D — Widths & depths
    # ------------------------------------------------------------------

    def _shoulder_width(self) -> tuple[Optional[float], str]:
        def _fallback():
            # Landmark-only shoulder width is +13 cm biased on standard photos because
            # MediaPipe places shoulder points at the silhouette edge, not the joint.
            # Prefer mesh width; fall back to landmark only when mesh unavailable.
            v = self._mesh_width_at_height_ratio(0.82)
            if v:
                return v, "smpl_mesh"
            ls = self.front.get(_L.L_SHOULDER)
            rs = self.front.get(_L.R_SHOULDER)
            if ls and rs and ls.visibility > 0.5 and rs.visibility > 0.5:
                # Apply empirical correction factor from benchmark (-13.5 cm / ~50 cm = ~27%)
                dist = self._lm_dist_x(ls, rs) * 0.73
                return round(dist, 1), "landmark"
            return self._ratio_fallback("shoulder_width")
        return self._smpl_anthro_or("M26_shoulder_width", _fallback)

    def _chest_width(self) -> tuple[Optional[float], str]:
        # Standing-body proportions place shoulders at ratio ~0.82 and the chest
        # at ~0.72. Sampling at 0.76 (chest *circ* height) catches the broader
        # shoulder span; 0.72 gives a chest-only frontal width.
        v = self._mesh_width_at_height_ratio(0.72)
        if v:
            return v, "smpl_mesh"
        sw, _ = self._shoulder_width()
        if sw:
            return round(sw * 0.85, 1), "derived"
        return self._ratio_fallback("chest_width")

    def _back_width(self) -> tuple[Optional[float], str]:
        ls = self.back.get(_L.L_SHOULDER)
        rs = self.back.get(_L.R_SHOULDER)
        if ls and rs and ls.visibility > 0.5 and rs.visibility > 0.5:
            dist = self._lm_dist_x(ls, rs)
            return round(dist * 0.78, 1), "landmark"  # armhole-to-armhole
        v = self._mesh_width_at_height_ratio(0.79)
        if v:
            return round(v * 0.78, 1), "smpl_mesh"
        return self._ratio_fallback("back_width")

    def _hip_width(self) -> tuple[Optional[float], str]:
        lh = self.front.get(_L.L_HIP)
        rh = self.front.get(_L.R_HIP)
        if lh and rh and lh.visibility > 0.5 and rh.visibility > 0.5:
            dist = self._lm_dist_x(lh, rh)
            return round(dist, 1), "landmark"
        v = self._mesh_width_at_height_ratio(0.52)
        if v:
            return v, "smpl_mesh"
        return self._ratio_fallback("hip_width")

    def _chest_depth(self) -> tuple[Optional[float], str]:
        v = self._mesh_depth_at_height_ratio(0.76)
        if v:
            return v, "smpl_mesh"
        # In a strict side profile, L_SHOULDER and R_SHOULDER collapse to roughly
        # the same image point, so neither X nor Y distance approximates depth.
        # There is no reliable single-landmark proxy — go straight to height_ratio.
        return self._ratio_fallback("chest_depth")

    def _waist_depth(self) -> tuple[Optional[float], str]:
        v = self._mesh_depth_at_height_ratio(0.62)
        if v:
            return v, "smpl_mesh"
        return self._ratio_fallback("waist_depth")

    def _armhole_depth(self) -> tuple[Optional[float], str]:
        ls = self.front.get(_L.L_SHOULDER)
        le = self.front.get(_L.L_ELBOW)
        if ls and le and ls.visibility > 0.5 and le.visibility > 0.5:
            # Armhole depth ≈ vertical distance from shoulder point to armhole bottom
            dist = self._lm_dist_y(ls, le) * 0.45
            return round(dist, 1), "landmark"
        return self._ratio_fallback("armhole_depth")
