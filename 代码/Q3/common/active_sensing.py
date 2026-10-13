from __future__ import annotations

import math

from .geometry import clip_bearing_wedge, distance, minimum_enclosing_circle, polygon_diameter
from .models import Point, TargetState


def rotated_step(origin: Point, bearing_deg: float, offset_deg: float, length: float) -> Point:
    angle = math.radians(bearing_deg + offset_deg)
    return Point(origin.x + length * math.cos(angle), origin.y + length * math.sin(angle))


def q2_analytic_candidates(
    first_position: Point,
    first_bearing_deg: float,
    min_reception_radius: float = 1000.0,
    error_deg: float = 1.0,
) -> tuple[Point, Point]:
    """Robust candidates S1 + rho_min R(+/-(60-error)) u1."""
    gamma_deg = 60.0 - error_deg
    return (
        rotated_step(first_position, first_bearing_deg, gamma_deg, min_reception_radius),
        rotated_step(first_position, first_bearing_deg, -gamma_deg, min_reception_radius),
    )


def in_q2_robust_region(a: float, b: float, rho_min: float = 1000.0, delta_deg: float = 1.0) -> bool:
    delta = math.radians(delta_deg)
    r2 = a * a + b * b
    return (
        r2 <= rho_min * rho_min + 1e-7
        and r2 <= 2 * rho_min * (a * math.cos(delta) - abs(b) * math.sin(delta)) + 1e-7
    )


class ActiveSensor:
    """Q2 candidate generator with a Q1-based worst-case geometric scorer."""

    def __init__(self, min_reception_radius: float = 1000.0, error_deg: float = 1.01) -> None:
        self.min_reception_radius = min_reception_radius
        self.error_deg = error_deg

    def choose_next(self, target: TargetState, current: Point) -> Point:
        if not target.observations:
            raise ValueError("target has no direction observation")
        first = target.observations[0]
        if len(target.observations) == 1:
            candidates = q2_analytic_candidates(
                first.position,
                first.bearing_deg,
                self.min_reception_radius,
                self.error_deg,
            )
            return min(candidates, key=lambda p: distance(current, p))

        mec = minimum_enclosing_circle(target.polygon)
        if len(target.observations) % 2 == 0 and distance(current, mec.center) > 5.0:
            return mec.center

        latest = target.observations[-1]
        diameter = polygon_diameter(target.polygon)[0]
        step = min(self.min_reception_radius, max(40.0, diameter))
        offset_deg = 60.0 - self.error_deg
        candidates = (
            rotated_step(latest.position, latest.bearing_deg, offset_deg, step),
            rotated_step(latest.position, latest.bearing_deg, -offset_deg, step),
        )
        # Predict each candidate's bearing toward the current feasible-region center,
        # then use the exact Q1 clipping evaluator and prefer the smaller diameter.
        def score(p: Point) -> tuple[float, float]:
            angle = math.degrees(math.atan2(mec.center.y - p.y, mec.center.x - p.x)) % 360.0
            predicted = clip_bearing_wedge(target.polygon, p, angle, self.error_deg)
            d = polygon_diameter(predicted)[0] if predicted else float("inf")
            return d, distance(current, p)

        return min(candidates, key=score)
