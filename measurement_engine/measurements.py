"""
Body measurement calculation from detected pose landmarks.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .pose_detector import PoseDetector, PoseResult
from .utils import distance_2d, estimate_circumference, midpoint, pixel_to_cm


@dataclass
class MeasurementResult:
    """Body measurements extracted from a single image.

    All numeric fields are in the unit indicated by :attr:`units`
    (either ``"cm"`` when a reference height is provided, or ``"pixels"``).
    ``None`` means the measurement could not be computed (e.g. the relevant
    body part was not visible).
    """

    shoulder_width: Optional[float] = None
    chest_circumference: Optional[float] = None
    waist_circumference: Optional[float] = None
    hip_circumference: Optional[float] = None
    inseam_length: Optional[float] = None
    sleeve_length: Optional[float] = None
    torso_length: Optional[float] = None
    back_length: Optional[float] = None
    total_height: Optional[float] = None
    neck_circumference: Optional[float] = None

    # Metadata
    units: str = "pixels"
    confidence: float = 0.0
    scale_factor: Optional[float] = None  # cm per pixel (set when units == "cm")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, object]:
        """Return all measurements as a plain dictionary."""
        return {
            "shoulder_width": self.shoulder_width,
            "chest_circumference": self.chest_circumference,
            "waist_circumference": self.waist_circumference,
            "hip_circumference": self.hip_circumference,
            "inseam_length": self.inseam_length,
            "sleeve_length": self.sleeve_length,
            "torso_length": self.torso_length,
            "back_length": self.back_length,
            "total_height": self.total_height,
            "neck_circumference": self.neck_circumference,
            "units": self.units,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        unit = self.units
        lines = [
            f"MeasurementResult (units={unit}, confidence={self.confidence:.2f}):"
        ]
        for key, value in self.to_dict().items():
            if key not in ("units", "confidence") and value is not None:
                lines.append(f"  {key}: {value:.1f} {unit}")
        return "\n".join(lines)


class MeasurementCalculator:
    """Derives body measurements from a :class:`~.pose_detector.PoseResult`.

    All intermediate distances are computed in pixels; the optional
    *person_height_cm* parameter is used to compute a pixel-to-cm scale
    factor so that results are returned in real-world centimetres.
    """

    # Typical depth-to-width ratios for body cross-sections (ellipse model)
    CHEST_DEPTH_RATIO: float = 0.65
    WAIST_DEPTH_RATIO: float = 0.75
    HIP_DEPTH_RATIO: float = 0.70
    NECK_DEPTH_RATIO: float = 0.85

    # Waist is typically ~85 % of hip width for an average body
    WAIST_TO_HIP_RATIO: float = 0.85

    # Neck width as a fraction of shoulder width
    NECK_TO_SHOULDER_RATIO: float = 0.22

    def __init__(self, pose_result: PoseResult) -> None:
        """Initialise with pose-detection results.

        Args:
            pose_result: Output of :meth:`~.pose_detector.PoseDetector.detect`.
        """
        self.pose = pose_result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _px(self, idx: int):
        """Return pixel coordinates for landmark *idx*."""
        return self.pose.get_pixel_coords(idx)

    def _visible(self, *indices: int, threshold: float = 0.5) -> bool:
        """Return ``True`` if every listed landmark is visible above *threshold*."""
        return all(self.pose.is_visible(i, threshold) for i in indices)

    # ------------------------------------------------------------------ #
    # Individual measurement helpers (all return pixel distances)          #
    # ------------------------------------------------------------------ #

    def _shoulder_width_px(self) -> Optional[float]:
        LS, RS = PoseDetector.LEFT_SHOULDER, PoseDetector.RIGHT_SHOULDER
        if not self._visible(LS, RS):
            return None
        return distance_2d(self._px(LS), self._px(RS))

    def _hip_width_px(self) -> Optional[float]:
        LH, RH = PoseDetector.LEFT_HIP, PoseDetector.RIGHT_HIP
        if not self._visible(LH, RH):
            return None
        return distance_2d(self._px(LH), self._px(RH))

    def _torso_length_px(self) -> Optional[float]:
        LS, RS = PoseDetector.LEFT_SHOULDER, PoseDetector.RIGHT_SHOULDER
        LH, RH = PoseDetector.LEFT_HIP, PoseDetector.RIGHT_HIP
        if not self._visible(LS, RS, LH, RH):
            return None
        shoulder_mid = midpoint(self._px(LS), self._px(RS))
        hip_mid = midpoint(self._px(LH), self._px(RH))
        return distance_2d(shoulder_mid, hip_mid)

    def _inseam_length_px(self) -> Optional[float]:
        LH, RH = PoseDetector.LEFT_HIP, PoseDetector.RIGHT_HIP
        LA, RA = PoseDetector.LEFT_ANKLE, PoseDetector.RIGHT_ANKLE
        if not self._visible(LH, RH, LA, RA):
            return None
        hip_mid = midpoint(self._px(LH), self._px(RH))
        ankle_mid = midpoint(self._px(LA), self._px(RA))
        return distance_2d(hip_mid, ankle_mid)

    def _sleeve_length_px(self) -> Optional[float]:
        """Sleeve length measured along the arm: shoulder → elbow → wrist."""
        for shoulder, elbow, wrist in [
            (
                PoseDetector.LEFT_SHOULDER,
                PoseDetector.LEFT_ELBOW,
                PoseDetector.LEFT_WRIST,
            ),
            (
                PoseDetector.RIGHT_SHOULDER,
                PoseDetector.RIGHT_ELBOW,
                PoseDetector.RIGHT_WRIST,
            ),
        ]:
            if self._visible(shoulder, elbow, wrist):
                return distance_2d(self._px(shoulder), self._px(elbow)) + distance_2d(
                    self._px(elbow), self._px(wrist)
                )
        return None

    def _total_height_px(self) -> Optional[float]:
        """Estimate full body height from estimated head top to ankle level."""
        if not self._visible(PoseDetector.NOSE):
            return None

        # Determine the lowest ankle point (largest y in image coordinates)
        foot_y: Optional[float] = None
        for ankle_idx in (PoseDetector.LEFT_ANKLE, PoseDetector.RIGHT_ANKLE):
            if self._visible(ankle_idx):
                ay = self._px(ankle_idx)[1]
                foot_y = ay if foot_y is None else max(foot_y, ay)

        if foot_y is None:
            return None

        nose = self._px(PoseDetector.NOSE)
        nose_y = nose[1]

        # Estimate the top of the head from the nose-to-hip proportion.
        # On average the nose is at roughly 10 % of total height from the top
        # and the hip at roughly 52 % – so nose-to-hip spans ~42 % of height.
        # head_top ≈ nose_y - (nose_to_hip_px * 0.10 / 0.42)
        LH, RH = PoseDetector.LEFT_HIP, PoseDetector.RIGHT_HIP
        if self._visible(LH, RH):
            hip_mid_y = midpoint(self._px(LH), self._px(RH))[1]
            nose_to_hip_px = abs(hip_mid_y - nose_y)
            head_top_y = nose_y - nose_to_hip_px * (0.10 / 0.42)
        else:
            # Fallback: assume head top is 15 % above the nose
            head_top_y = nose_y - abs(foot_y - nose_y) * 0.15

        return foot_y - head_top_y

    def _confidence(self) -> float:
        """Fraction of key landmarks that are visible."""
        key = [
            PoseDetector.LEFT_SHOULDER,
            PoseDetector.RIGHT_SHOULDER,
            PoseDetector.LEFT_HIP,
            PoseDetector.RIGHT_HIP,
            PoseDetector.LEFT_KNEE,
            PoseDetector.RIGHT_KNEE,
            PoseDetector.LEFT_ANKLE,
            PoseDetector.RIGHT_ANKLE,
        ]
        return sum(1 for i in key if self.pose.is_visible(i)) / len(key)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def calculate(
        self, person_height_cm: Optional[float] = None
    ) -> MeasurementResult:
        """Compute all body measurements.

        Args:
            person_height_cm: Known height of the person in cm.  When
                supplied, all measurements are returned in centimetres;
                otherwise they are returned in pixels.

        Returns:
            :class:`MeasurementResult` with every available measurement.
        """
        result = MeasurementResult()
        result.confidence = self._confidence()

        # ---- Raw pixel distances ---------------------------------------- #
        shoulder_px = self._shoulder_width_px()
        hip_px = self._hip_width_px()
        torso_px = self._torso_length_px()
        inseam_px = self._inseam_length_px()
        sleeve_px = self._sleeve_length_px()
        height_px = self._total_height_px()

        # ---- Determine scale factor -------------------------------------- #
        scale: Optional[float] = None
        if person_height_cm and height_px and height_px > 0:
            scale = person_height_cm / height_px
            result.units = "cm"
            result.scale_factor = scale
        else:
            result.units = "pixels"

        def to_unit(px: Optional[float]) -> Optional[float]:
            """Convert *px* to the output unit and round to 1 decimal place."""
            if px is None:
                return None
            value = pixel_to_cm(px, scale) if scale is not None else px
            return round(value, 1)

        # ---- Linear measurements ---------------------------------------- #
        result.shoulder_width = to_unit(shoulder_px)
        result.torso_length = to_unit(torso_px)
        # back_length mirrors torso_length: both measure the vertical distance
        # from the shoulder midpoint to the hip midpoint in a frontal, standing
        # pose, which is the closest approximation available from a single image.
        result.back_length = to_unit(torso_px)
        result.inseam_length = to_unit(inseam_px)
        result.sleeve_length = to_unit(sleeve_px)
        result.total_height = to_unit(height_px)

        # ---- Circumference estimates from widths (ellipse model) --------- #
        if shoulder_px is not None:
            chest_width = to_unit(shoulder_px)
            if chest_width is not None:
                result.chest_circumference = round(
                    estimate_circumference(chest_width, self.CHEST_DEPTH_RATIO), 1
                )
                neck_width = chest_width * self.NECK_TO_SHOULDER_RATIO
                result.neck_circumference = round(
                    estimate_circumference(neck_width, self.NECK_DEPTH_RATIO), 1
                )

        if hip_px is not None:
            hip_width = to_unit(hip_px)
            if hip_width is not None:
                result.hip_circumference = round(
                    estimate_circumference(hip_width, self.HIP_DEPTH_RATIO), 1
                )
                waist_width = hip_width * self.WAIST_TO_HIP_RATIO
                result.waist_circumference = round(
                    estimate_circumference(waist_width, self.WAIST_DEPTH_RATIO), 1
                )

        return result
