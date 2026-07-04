"""
Scan pipeline — orchestrates the full flow from submitted frames to
a scored 32-measurement ScanMeasurements result.

Steps:
  1. Decode each base64 frame → PIL image
  2. Run MediaPipe pose detection per frame
  3. Score each frame (blur, angle, occlusion, lighting)
  4. Fit SMPL mesh using the best multi-view frames
  5. Extract all 32 measurements
  6. Score confidence per measurement
  7. Return ScanMeasurements + overall confidence
"""

from __future__ import annotations

import base64
import io
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.measurement_engine.scan.schemas import (
    CameraMetadata,
    Confidence,
    FitStyle,
    FrameScore,
    GarmentType,
    PoseFrame,
    PoseID,
    ScanMeasurements,
    ScanResponse,
    ScanStatus,
)
from app.measurement_engine.scan.frame_scorer import FrameScorer
from app.measurement_engine.scan.frame_selector import select_and_resize
from app.measurement_engine.scan.measurements import (
    LandmarkPoint,
    MeasurementExtractor,
    RawMeasurements,
)
from app.measurement_engine.scan.confidence import (
    build_scan_measurements,
    overall_confidence,
)
from app.measurement_engine.scan.height_estimator import HeightEstimator
from app.measurement_engine.scan.validator import validate

logger = logging.getLogger(__name__)


@dataclass
class _ProcessedFrame:
    pose_id: PoseID
    image: Image.Image
    landmarks: Optional[dict[int, LandmarkPoint]]
    score: FrameScore
    body_mask: Optional[np.ndarray] = None   # uint8 (H,W): 255=person, 0=bg


class ScanPipeline:
    """
    Stateless pipeline.  One instance is created per request; model
    wrappers are injected from app.state to avoid reloading between requests.
    """

    def __init__(self, pose_model, smpl_model, segmenter_model=None):
        """
        Args:
            pose_model:       MediaPipePoseWrapper instance (already loaded).
            smpl_model:       SMPLFitter instance (already loaded).
            segmenter_model:  MediaPipeSegmenter instance (optional; may be None).
        """
        self._pose       = pose_model
        self._smpl       = smpl_model
        self._segmenter  = segmenter_model
        self._scorer     = FrameScorer()
        self._height_est = HeightEstimator()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        frames: list[PoseFrame],
        height_cm: Optional[float] = None,
        camera_metadata: Optional[CameraMetadata] = None,
        garment_type: Optional[GarmentType] = None,
        fit_style: Optional[FitStyle] = None,
    ) -> ScanResponse:
        scan_id = str(uuid.uuid4())

        try:
            processed = self._process_frames(frames)

            # SCAN-06: keep top-N frames per pose, resize to ≤ 1024px
            processed = select_and_resize(processed)

            # Resolve height anchor before any downstream step
            front_lm  = self._landmarks_for(processed, PoseID.FRONT)
            front_img = next((p.image for p in processed if p.pose_id == PoseID.FRONT), None)
            img_h_px  = front_img.height if front_img else 1080
            height_est = self._height_est.estimate(
                height_cm, camera_metadata, front_lm, img_h_px
            )
            logger.info(
                "Height resolved: %.1f cm (source=%s, confidence=%s)",
                height_est.value_cm, height_est.source, height_est.confidence,
            )

            mesh = self._fit_mesh(processed, height_est.value_cm)  # (verts, faces) or None

            # F6: silhouette IoU between projected mesh and per-view body masks
            mesh_fit_score = self._compute_mesh_fit_score(
                mesh[0] if mesh else None,
                processed,
                height_est.value_cm,
            )
            logger.info("Mesh fit score (silhouette IoU): %.3f", mesh_fit_score)

            # Surface to Prometheus when metrics are enabled (no-op if absent).
            try:
                from app.metrics import mesh_fit_iou
                mesh_fit_iou.observe(mesh_fit_score)
            except Exception:
                pass

            raw  = self._extract_measurements(height_est.value_cm, processed, mesh)
            measurements, conf = self._score(
                raw, processed, height_est.source, height_est.confidence, mesh_fit_score
            )

            # F8/F9: mark required fields and compute ease / cutting dimensions
            from app.measurement_engine.scan.garments import apply_garment_profile
            measurements = apply_garment_profile(measurements, garment_type, fit_style)

            validation = validate(
                measurements, height_est.value_cm, garment_type,
                mesh_fit_score=mesh_fit_score,
            )

            return ScanResponse(
                scan_id=scan_id,
                status=ScanStatus.COMPLETE,
                overall_confidence=conf,
                frames_received=len(frames),
                height_cm=height_est.value_cm,
                height_source=height_est.source,
                measurements=measurements,
                validation=validation,
                garment_type=garment_type,
                fit_style=fit_style,
            )

        except Exception as exc:
            logger.exception("Scan pipeline failed")
            return ScanResponse(
                scan_id=scan_id,
                status=ScanStatus.FAILED,
                overall_confidence=Confidence.LOW,
                frames_received=len(frames),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Step 1 — decode + pose detect + score each frame
    # ------------------------------------------------------------------

    def _process_frames(self, frames: list[PoseFrame]) -> list[_ProcessedFrame]:
        processed: list[_ProcessedFrame] = []
        for pf in frames:
            img = self._decode_image(pf.image_b64)

            # Body segmentation (optional — None when segmenter is unavailable)
            mask = self._segmenter.segment(img) if self._segmenter else None

            raw_lm = self._pose.detect_landmarks(img)
            lm = self._convert_landmarks(raw_lm)
            score = self._scorer.score(img, pf.pose_id, lm, body_mask=mask)
            processed.append(_ProcessedFrame(
                pose_id=pf.pose_id,
                image=img,
                landmarks=lm,
                score=score,
                body_mask=mask,
            ))
            logger.debug(
                "Frame %s scored %.2f (blur=%.2f pose=%.2f angle=%.2f occ=%.2f)",
                pf.pose_id, score.composite,
                score.blur_score, score.pose_confidence, score.angle_match,
                score.occlusion_score,
            )
        return processed

    # ------------------------------------------------------------------
    # Step 2 — SMPL mesh fitting
    # ------------------------------------------------------------------

    def _fit_mesh(
        self,
        processed: list[_ProcessedFrame],
        height_cm: float,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        B8: Multi-view SMPL mesh fitting.

        Collects body masks from all available poses (front, side, back) and
        passes them to fit_multiview() for silhouette-IoU-guided beta refinement.
        Falls back gracefully when masks or extra views are absent.
        """
        usable = [p for p in processed if p.score.is_usable]
        if not usable:
            logger.warning("No usable frames for mesh fitting — using landmark-only path")
            return None

        front_lm = self._landmarks_for(processed, PoseID.FRONT)

        # Gather body masks per view for multi-view silhouette optimization
        def _mask_for(pose: PoseID) -> Optional[np.ndarray]:
            frame = self._get_frame(processed, pose)
            return frame.body_mask if frame else None

        front_mask = _mask_for(PoseID.FRONT)
        side_mask  = _mask_for(PoseID.SIDE_LEFT)
        back_mask  = _mask_for(PoseID.BACK)

        n_views = sum(m is not None for m in (front_mask, side_mask, back_mask))
        logger.info(
            "SMPL fit: height=%.1f cm, landmark_views=1, mask_views=%d",
            height_cm, n_views,
        )

        try:
            mesh_result = self._smpl.fit_multiview(
                landmarks=front_lm,
                height_cm=height_cm,
                front_mask=front_mask,
                side_mask=side_mask,
                back_mask=back_mask,
            )
            if mesh_result is None:
                return None

            return mesh_result.vertices, mesh_result.faces
        except Exception as e:
            logger.warning("SMPL fitting failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Step 3 — extract raw measurements
    # ------------------------------------------------------------------

    def _extract_measurements(
        self,
        height_cm: float,
        processed: list[_ProcessedFrame],
        mesh: Optional[tuple[np.ndarray, np.ndarray]],
    ) -> RawMeasurements:
        verts, faces = mesh if mesh is not None else (None, None)

        # Compute aspect ratio from the front frame so horizontal landmark
        # distances are scaled correctly for non-square (portrait) images.
        front_img = next((p.image for p in processed if p.pose_id == PoseID.FRONT), None)
        aspect = (front_img.width / front_img.height) if front_img else 1.0

        extractor = MeasurementExtractor(
            height_cm=height_cm,
            front_landmarks=self._landmarks_for(processed, PoseID.FRONT),
            side_landmarks=self._landmarks_for(processed, PoseID.SIDE_LEFT),
            back_landmarks=self._landmarks_for(processed, PoseID.BACK),
            arms_landmarks=self._landmarks_for(processed, PoseID.ARMS_OUT),
            mesh_vertices=verts,
            mesh_faces=faces,
            img_aspect_ratio=aspect,
        )
        return extractor.extract()

    # ------------------------------------------------------------------
    # Step 4 — confidence scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        raw: RawMeasurements,
        processed: list[_ProcessedFrame],
        height_source: str,
        height_confidence: Confidence,
        mesh_fit_score: float = 1.0,
    ) -> tuple[ScanMeasurements, Confidence]:
        frame_composites = {p.pose_id.value: p.score.composite for p in processed}
        lm_vis = self._landmark_visibilities(processed)

        measurements = build_scan_measurements(
            raw, frame_composites, lm_vis, height_source, height_confidence,
            mesh_fit_score=mesh_fit_score,
        )
        conf = overall_confidence(measurements)
        return measurements, conf

    # ------------------------------------------------------------------
    # Mesh fit score (F6) — silhouette IoU
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mesh_fit_score(
        vertices: Optional[np.ndarray],
        processed: list[_ProcessedFrame],
        height_cm: float,
    ) -> float:
        """
        IoU between the SMPL mesh's convex-hull silhouette and the segmenter
        body masks across available views.  Returns the mean IoU over (front,
        side_left, back), using only views whose mask is present.

        Returns 1.0 (no penalty) when vertices or all masks are absent.

        We use a convex-hull approximation rather than full triangle
        rasterization; it's fast (~1 ms per view) and sufficient to detect
        gross mesh mis-scale / mis-placement that would invalidate
        depth-derived measurements.
        """
        if vertices is None:
            return 1.0

        # (pose, projection axes) — front/back share (x,y); side uses (z,y)
        views = [
            (PoseID.FRONT,     (0, 1)),
            (PoseID.SIDE_LEFT, (2, 1)),
            (PoseID.BACK,      (0, 1)),
        ]

        ious: list[float] = []
        for pose, (h_axis, v_axis) in views:
            frame = next((p for p in processed if p.pose_id == pose), None)
            if frame is None or frame.body_mask is None:
                continue
            iou = ScanPipeline._silhouette_iou(
                vertices, frame.body_mask, height_cm, h_axis, v_axis
            )
            if iou is not None:
                ious.append(iou)

        return float(np.mean(ious)) if ious else 1.0

    @staticmethod
    def _silhouette_iou(
        vertices: np.ndarray,
        body_mask: np.ndarray,
        height_cm: float,
        h_axis: int,
        v_axis: int,
    ) -> Optional[float]:
        """One-view silhouette IoU between the mesh convex hull and a body mask."""
        # Some segmenters emit (H, W, 1) — collapse to 2D so np.where unpacks cleanly.
        if body_mask.ndim == 3:
            body_mask = body_mask.squeeze(-1)
        if body_mask.ndim != 2:
            return None
        H, W = body_mask.shape
        ys, xs = np.where(body_mask > 0)
        if len(ys) == 0:
            return None

        y_max = int(ys.max())
        x_center = float(xs.mean())
        mask_height_px = y_max - int(ys.min())
        if mask_height_px < 10:
            return None

        px_per_cm = mask_height_px / height_cm

        vx = np.clip(
            (vertices[:, h_axis] * px_per_cm + x_center).astype(np.int32), 0, W - 1
        )
        vy = np.clip(
            (y_max - vertices[:, v_axis] * px_per_cm).astype(np.int32), 0, H - 1
        )

        pts = np.stack([vx, vy], axis=1)
        hull = cv2.convexHull(pts)

        silhouette = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(silhouette, hull, 255)

        inter = int(np.count_nonzero((silhouette > 0) & (body_mask > 0)))
        union = int(np.count_nonzero((silhouette > 0) | (body_mask > 0)))
        return float(inter / union) if union > 0 else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_image(b64: str) -> Image.Image:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # correct phone rotation before any processing
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @staticmethod
    def _convert_landmarks(
        raw: Optional[dict],
    ) -> Optional[dict[int, LandmarkPoint]]:
        if raw is None:
            return None
        out: dict[int, LandmarkPoint] = {}
        for idx, lm in raw.items():
            out[int(idx)] = LandmarkPoint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
        return out

    @staticmethod
    def _get_frame(
        frames: list[_ProcessedFrame],
        pose_id: PoseID,
    ) -> Optional[_ProcessedFrame]:
        for f in frames:
            if f.pose_id == pose_id:
                return f
        return None

    def _landmarks_for(
        self,
        processed: list[_ProcessedFrame],
        pose_id: PoseID,
    ) -> Optional[dict[int, LandmarkPoint]]:
        f = self._get_frame(processed, pose_id)
        return f.landmarks if f else None

    @staticmethod
    def _landmark_visibilities(
        processed: list[_ProcessedFrame],
    ) -> dict[str, float]:
        """
        Return mean visibility for the key landmarks used per measurement.
        Keyed by measurement code (M01, M26, …).
        Only the FRONT frame is used as the primary reference.
        """
        front = next(
            (p for p in processed if p.pose_id == PoseID.FRONT), None
        )
        if not front or not front.landmarks:
            return {}

        lm = front.landmarks

        def vis(*indices: int) -> float:
            scores = [lm[i].visibility for i in indices if i in lm]
            return float(np.mean(scores)) if scores else 0.0

        return {
            "M01": vis(11, 12, 23, 24),  # shoulders + hips (chest level)
            "M03": vis(23, 24),
            "M05": vis(23, 24),
            "M06": vis(0, 11, 12),
            "M07": vis(13, 14),
            "M08": vis(15, 16),
            "M09": vis(23, 24, 25, 26),
            "M11": vis(25, 26),
            "M15": vis(11, 23),
            "M16": vis(12, 24),
            "M19": vis(11, 15),
            "M21": vis(23, 27),
            "M22": vis(23, 27),
            "M25": vis(11, 23),
            "M26": vis(11, 12),
            "M27": vis(11, 12),
            "M29": vis(23, 24),
            "M32": vis(11, 13),
        }
