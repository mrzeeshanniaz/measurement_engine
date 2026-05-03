"""
Main measurement engine – public entry point.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .measurements import MeasurementCalculator, MeasurementResult
from .pose_detector import PoseDetector
from .utils import load_image


class MeasurementEngine:
    """Extract body measurements from images for tailoring purposes.

    The engine uses MediaPipe Pose to detect 33 body landmarks in a
    front-view photograph and derives common tailoring measurements
    (shoulder width, chest/waist/hip circumferences, inseam, sleeve length,
    and more) from those landmarks.

    If the person's height is known, pass it as *person_height_cm* to
    :meth:`analyze` so that results are returned in centimetres; otherwise
    measurements are returned in pixels.

    Example::

        from measurement_engine import MeasurementEngine

        engine = MeasurementEngine()
        result = engine.analyze("front_view.jpg", person_height_cm=175.0)
        print(result)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialise the engine.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
                (passed to MediaPipe; range 0–1).
            min_tracking_confidence: Minimum confidence for landmark tracking
                (passed to MediaPipe; range 0–1).
        """
        self._detector = PoseDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        person_height_cm: Optional[float] = None,
    ) -> MeasurementResult:
        """Analyse *image* and return body measurements.

        For best results the image should show a **full-body, front-view**
        photograph of a person standing upright with arms slightly away from
        the body in good, even lighting.

        Args:
            image: Path to an image file **or** a NumPy array in BGR format
                   (as returned by ``cv2.imread``).
            person_height_cm: Optional known height of the person in cm.
                              When provided all measurements are converted to
                              centimetres; otherwise they are in pixels.

        Returns:
            :class:`~.measurements.MeasurementResult` with all available
            measurements.

        Raises:
            FileNotFoundError: If *image* is a path that does not exist.
            ValueError: If no person is detected in the image.
        """
        img_array = self._load(image)
        pose_result = self._detector.detect(img_array)

        if pose_result is None:
            raise ValueError(
                "No person detected in the image. "
                "Please ensure the image shows a full-body, front-view "
                "photograph of a person in good lighting."
            )

        calculator = MeasurementCalculator(pose_result)
        return calculator.calculate(person_height_cm=person_height_cm)

    def analyze_with_visualization(
        self,
        image: Union[str, Path, np.ndarray],
        person_height_cm: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[MeasurementResult, np.ndarray]:
        """Analyse *image*, draw landmarks and measurements, and return both.

        Args:
            image: Path to an image file or a NumPy array (BGR).
            person_height_cm: Optional known height for cm calibration.
            output_path: If given, the annotated image is saved to this path.

        Returns:
            A ``(MeasurementResult, annotated_image)`` tuple where
            *annotated_image* is a BGR NumPy array.

        Raises:
            FileNotFoundError: If *image* is a path that does not exist.
            ValueError: If no person is detected in the image.
        """
        import cv2  # noqa: PLC0415
        import mediapipe as mp  # noqa: PLC0415

        img_array = self._load(image)
        pose_result = self._detector.detect(img_array)

        if pose_result is None:
            raise ValueError(
                "No person detected in the image. "
                "Please ensure the image shows a full-body, front-view "
                "photograph of a person in good lighting."
            )

        calculator = MeasurementCalculator(pose_result)
        measurements = calculator.calculate(person_height_cm=person_height_cm)

        # Draw pose skeleton
        annotated = img_array.copy()
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=self._detector.min_detection_confidence,
        ) as pose:
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

        # Overlay measurement text
        unit = measurements.units
        y = 30
        for key, value in measurements.to_dict().items():
            if key in ("units", "confidence") or value is None:
                continue
            cv2.putText(
                annotated,
                f"{key}: {value:.1f} {unit}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 0),
                2,
            )
            y += 24

        if output_path:
            cv2.imwrite(output_path, annotated)

        return measurements, annotated

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Return *image* as a BGR NumPy array."""
        if isinstance(image, np.ndarray):
            return image
        return load_image(str(image))
