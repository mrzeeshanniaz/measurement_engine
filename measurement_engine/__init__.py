"""
Measurement Engine
==================

A Python package that extracts body measurements from photographs for use in
tailoring.  It uses MediaPipe Pose to detect body landmarks and derives
standard tailoring measurements (shoulder width, chest/waist/hip
circumferences, inseam, sleeve length, etc.) from those landmarks.
"""

from .engine import MeasurementEngine
from .measurements import MeasurementResult

__version__ = "0.1.0"
__all__ = ["MeasurementEngine", "MeasurementResult"]
