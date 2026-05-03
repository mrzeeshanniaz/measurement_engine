"""
Utility functions for the measurement engine.
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from the given file path.

    Args:
        path: Path to the image file.

    Returns:
        Image as a NumPy array in BGR format.

    Raises:
        FileNotFoundError: If the file does not exist or cannot be read.
    """
    import cv2

    if not Path(path).exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from: {path}")
    return image


def pixel_to_cm(pixels: float, scale_factor: float) -> float:
    """Convert a pixel measurement to centimetres.

    Args:
        pixels: Measurement in pixels.
        scale_factor: Conversion factor (cm per pixel).  Must be a positive
                      non-zero value, typically derived from a known reference
                      length such as the person's real height.

    Returns:
        Measurement in centimetres.
    """
    return pixels * scale_factor


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two 2-D points.

    Args:
        p1: First point as ``(x, y)``.
        p2: Second point as ``(x, y)``.

    Returns:
        Euclidean distance between the two points.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def midpoint(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> Tuple[float, float]:
    """Calculate the midpoint between two 2-D points.

    Args:
        p1: First point as ``(x, y)``.
        p2: Second point as ``(x, y)``.

    Returns:
        Midpoint as ``(x, y)``.
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def estimate_circumference(width_cm: float, depth_ratio: float = 0.70) -> float:
    """Estimate body circumference from its visible width using an ellipse model.

    A body cross-section is approximated as an ellipse.  Given the visible
    width (from a front-view image), the depth is estimated as a fraction of
    the width based on typical human body proportions.  The perimeter is
    then computed with Ramanujan's approximation.

    Args:
        width_cm: Full visible width in centimetres.
        depth_ratio: Ratio of depth to width.  Defaults to 0.70, a reasonable
                     average across common body cross-sections.

    Returns:
        Estimated circumference in centimetres.
    """
    a = width_cm / 2               # semi-major axis (half width)
    b = a * depth_ratio            # semi-minor axis (estimated half depth)

    # Ramanujan's approximation for ellipse perimeter
    h = ((a - b) / (a + b)) ** 2
    circumference = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    return circumference
