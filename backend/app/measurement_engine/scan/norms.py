"""
Anthropometric population norms for body measurement validation.

Source: ANSUR II (US Army, n=6,000) adjusted for South Asian populations
using WHO Asian BMI study corrections (−3% on girths, −1% on lengths).

Each norm is expressed as a ratio relative to total body height so the
plausible range scales correctly for any height input.

Structure per measurement:
  mean_ratio  — population mean / height
  sd_ratio    — population SD / height
  z_warn      — Z-score at which a WARNING is triggered (default 2.0)
  z_error     — Z-score at which an ERROR is triggered (default 3.0)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Norm:
    mean_ratio: float   # mean / height
    sd_ratio:   float   # SD / height
    z_warn:     float = 2.0
    z_error:    float = 3.0

    def plausible_range(self, height_cm: float, z: float | None = None) -> tuple[float, float]:
        z = z or self.z_error
        lo = (self.mean_ratio - z * self.sd_ratio) * height_cm
        hi = (self.mean_ratio + z * self.sd_ratio) * height_cm
        return round(lo, 1), round(hi, 1)

    def z_score(self, value_cm: float, height_cm: float) -> float:
        """Signed Z-score: positive = above mean, negative = below."""
        expected = self.mean_ratio * height_cm
        sd = self.sd_ratio * height_cm
        if sd < 0.1:
            return 0.0
        return (value_cm - expected) / sd


# M01–M32 norms (all ratios relative to height)
NORMS: dict[str, Norm] = {
    # Section A — Upper body circumferences
    "M01": Norm(mean_ratio=0.542, sd_ratio=0.038),  # Chest / bust
    "M02": Norm(mean_ratio=0.462, sd_ratio=0.036),  # Under-bust
    "M03": Norm(mean_ratio=0.455, sd_ratio=0.048),  # Waist
    "M04": Norm(mean_ratio=0.492, sd_ratio=0.046),  # Abdomen / navel
    "M05": Norm(mean_ratio=0.548, sd_ratio=0.040),  # Hips / seat
    "M06": Norm(mean_ratio=0.212, sd_ratio=0.014),  # Neck
    "M07": Norm(mean_ratio=0.173, sd_ratio=0.017),  # Bicep
    "M08": Norm(mean_ratio=0.098, sd_ratio=0.009),  # Wrist

    # Section B — Lower body circumferences
    "M09": Norm(mean_ratio=0.316, sd_ratio=0.028),  # Thigh
    "M10": Norm(mean_ratio=0.263, sd_ratio=0.024),  # Mid-thigh
    "M11": Norm(mean_ratio=0.224, sd_ratio=0.018),  # Knee
    "M12": Norm(mean_ratio=0.211, sd_ratio=0.017),  # Calf
    "M13": Norm(mean_ratio=0.132, sd_ratio=0.011),  # Ankle

    # Section C — Lengths & heights
    "M15": Norm(mean_ratio=0.242, sd_ratio=0.018),  # Shoulder to waist (front)
    "M16": Norm(mean_ratio=0.232, sd_ratio=0.016),  # Shoulder to waist (back)
    "M17": Norm(mean_ratio=0.573, sd_ratio=0.030),  # Kameez length
    "M18": Norm(mean_ratio=0.672, sd_ratio=0.032),  # Dress / suit length
    "M19": Norm(mean_ratio=0.318, sd_ratio=0.022),  # Sleeve length (full)
    "M20": Norm(mean_ratio=0.182, sd_ratio=0.016),  # Sleeve length (elbow)
    "M21": Norm(mean_ratio=0.458, sd_ratio=0.024),  # Inseam
    "M22": Norm(mean_ratio=0.582, sd_ratio=0.026),  # Outseam
    "M23": Norm(mean_ratio=0.168, sd_ratio=0.014),  # Crotch depth (front)
    "M24": Norm(mean_ratio=0.174, sd_ratio=0.015),  # Crotch depth (back)
    "M25": Norm(mean_ratio=0.308, sd_ratio=0.020),  # Torso length

    # Section D — Widths & depths
    "M26": Norm(mean_ratio=0.234, sd_ratio=0.015),  # Shoulder width
    "M27": Norm(mean_ratio=0.197, sd_ratio=0.014),  # Chest width
    "M28": Norm(mean_ratio=0.192, sd_ratio=0.013),  # Back width
    "M29": Norm(mean_ratio=0.173, sd_ratio=0.013),  # Hip width
    "M30": Norm(mean_ratio=0.132, sd_ratio=0.012),  # Chest depth
    "M31": Norm(mean_ratio=0.108, sd_ratio=0.011),  # Waist depth
    "M32": Norm(mean_ratio=0.112, sd_ratio=0.010),  # Armhole depth
}


# Absolute physiological hard limits (cm) — values outside these are
# almost certainly scan errors regardless of height
HARD_LIMITS: dict[str, tuple[float, float]] = {
    "M01": (50.0,  200.0),
    "M02": (40.0,  180.0),
    "M03": (40.0,  200.0),
    "M04": (40.0,  220.0),
    "M05": (50.0,  220.0),
    "M06": (20.0,   70.0),
    "M07": (15.0,   80.0),
    "M08": (10.0,   40.0),
    "M09": (25.0,  120.0),
    "M10": (20.0,  100.0),
    "M11": (18.0,   80.0),
    "M12": (15.0,   70.0),
    "M13": (12.0,   50.0),
    "M15": (20.0,   60.0),
    "M16": (18.0,   58.0),
    "M17": (50.0,  150.0),
    "M18": (60.0,  180.0),
    "M19": (30.0,  100.0),
    "M20": (15.0,   55.0),
    "M21": (50.0,  120.0),
    "M22": (60.0,  140.0),
    "M23": (15.0,   45.0),
    "M24": (18.0,   50.0),
    "M25": (40.0,  100.0),
    "M26": (25.0,   70.0),
    "M27": (20.0,   65.0),
    "M28": (20.0,   65.0),
    "M29": (20.0,   70.0),
    "M30": (10.0,   45.0),
    "M31": ( 8.0,   40.0),
    "M32": ( 8.0,   30.0),
}
