"""
Pose detection module using MediaPipe Pose.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Landmark:
    """A single body landmark detected by the pose model."""

    x: float           # Normalised x coordinate [0, 1]
    y: float           # Normalised y coordinate [0, 1]
    z: float           # Depth relative to the hip midpoint (same scale as x)
    visibility: float  # Visibility / presence score [0, 1]


@dataclass
class PoseResult:
    """Result of a single-image pose-detection pass."""

    landmarks: Dict[int, Landmark]
    image_width: int
    image_height: int

    def get_pixel_coords(self, landmark_idx: int) -> Tuple[float, float]:
        """Return pixel coordinates ``(x, y)`` for the given landmark index."""
        lm = self.landmarks[landmark_idx]
        return (lm.x * self.image_width, lm.y * self.image_height)

    def is_visible(self, landmark_idx: int, threshold: float = 0.5) -> bool:
        """Return ``True`` if the landmark is present and visible above *threshold*."""
        if landmark_idx not in self.landmarks:
            return False
        return self.landmarks[landmark_idx].visibility >= threshold


class PoseDetector:
    """Detects body pose landmarks in images using MediaPipe Pose.

    Attributes mirror the MediaPipe landmark index constants for convenient
    access (e.g. ``PoseDetector.LEFT_SHOULDER == 11``).
    """

    # ------------------------------------------------------------------ #
    # MediaPipe Pose landmark indices                                      #
    # ------------------------------------------------------------------ #
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialise the pose detector.

        Args:
            min_detection_confidence: Minimum confidence required to consider
                a detection successful (passed directly to MediaPipe).
            min_tracking_confidence: Minimum confidence required for landmark
                tracking (passed directly to MediaPipe).
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _initialize(self) -> None:
        """Lazy-initialise the MediaPipe Pose model."""
        if self._pose is None:
            import mediapipe as mp  # noqa: PLC0415

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, image: np.ndarray) -> Optional[PoseResult]:
        """Detect pose landmarks in *image*.

        Args:
            image: Input image as a NumPy array in **BGR** format (as returned
                   by ``cv2.imread``).

        Returns:
            :class:`PoseResult` containing all detected landmarks, or ``None``
            if no person could be detected.
        """
        self._initialize()

        import cv2  # noqa: PLC0415

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        results = self._pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        landmarks: Dict[int, Landmark] = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )

        return PoseResult(
            landmarks=landmarks,
            image_width=width,
            image_height=height,
        )

    def __del__(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
