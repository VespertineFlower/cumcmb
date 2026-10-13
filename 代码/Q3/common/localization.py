from __future__ import annotations

import math

from .geometry import (
    clip_bearing_wedge,
    clip_to_outer_disk,
    diameter_circle_covers,
    exclusion_aware_convex_outer,
    minimum_enclosing_circle,
    polygon_diameter,
    regular_outer_polygon,
)
from .models import Circle, Observation, Point, TargetState, TargetStatus


class InconsistentObservations(RuntimeError):
    pass


class BearingLocalizer:
    """Q1 incremental convex feasible-region estimator."""

    def __init__(
        self,
        arena_radius: float = 1800.0,
        bearing_error_deg: float = 1.01,
        max_reception_radius: float = 1500.0,
        polygon_sides: int = 128,
    ) -> None:
        self.arena_radius = arena_radius
        self.bearing_error_deg = bearing_error_deg
        self.max_reception_radius = max_reception_radius
        self.polygon_sides = polygon_sides

    def new_target(self, channel: int) -> TargetState:
        initial = regular_outer_polygon(
            Point(0.0, 0.0), self.arena_radius, self.polygon_sides
        )
        return TargetState(
            channel=channel,
            polygon=list(initial),
            positive_polygon=list(initial),
        )

    def _planning_outer(
        self, positive_polygon: list[Point], exclusion_centers: list[Point]
    ) -> tuple[list[Point], float]:
        """Rebuild with progressively finer boundary cells near clearability."""
        polygon = exclusion_aware_convex_outer(
            positive_polygon,
            exclusion_centers,
            resolution=25.0,
            max_depth=10,
        )
        if not polygon or not exclusion_centers:
            return polygon, 25.0

        resolution = 25.0
        # Each finer pass starts from the raw positive polygon and all original
        # exclusion disks, so coarse approximations never become constraints.
        radius = minimum_enclosing_circle(polygon).radius
        if radius <= 120.0:
            polygon = exclusion_aware_convex_outer(
                positive_polygon,
                exclusion_centers,
                resolution=10.0,
                max_depth=11,
            )
            resolution = 10.0
            radius = minimum_enclosing_circle(polygon).radius
        if radius <= 55.0:
            polygon = exclusion_aware_convex_outer(
                positive_polygon,
                exclusion_centers,
                resolution=2.0,
                max_depth=13,
            )
            resolution = 2.0
            radius = minimum_enclosing_circle(polygon).radius
        if radius <= 28.0:
            polygon = exclusion_aware_convex_outer(
                positive_polygon,
                exclusion_centers,
                resolution=0.5,
                max_depth=15,
            )
            resolution = 0.5
        return polygon, resolution

    def add_direction(self, target: TargetState, position: Point, bearing_deg: float) -> None:
        base = target.positive_polygon or target.polygon
        polygon = clip_bearing_wedge(
            base, position, bearing_deg, self.bearing_error_deg
        )
        polygon = clip_to_outer_disk(
            polygon, position, self.max_reception_radius, max(32, self.polygon_sides // 2)
        )
        if not polygon:
            raise InconsistentObservations(
                f"channel {target.channel}: bearing observations have empty intersection"
            )
        plan_poly, resolution = self._planning_outer(
            polygon, target.exclusion_centers
        )
        if not plan_poly:
            raise InconsistentObservations(
                f"channel {target.channel}: exclusions leave no feasible region"
            )
        target.positive_polygon = polygon
        target.polygon = plan_poly
        target.exclusion_resolution_m = resolution
        target.observations.append(Observation(position, bearing_deg % 360.0))
        target.status = TargetStatus.LOCALIZING
        if self.is_clearable(target):
            target.status = TargetStatus.CLEARABLE

    def add_no_signal(
        self, target: TargetState, position: Point, exclusion_radius: float = 1000.0
    ) -> None:
        """Keep a raw exclusion constraint and rebuild a safe planning hull."""
        appended = all(
            math.dist((position.x, position.y), (old.x, old.y)) > 1e-7
            for old in target.exclusion_centers
        )
        if appended:
            target.exclusion_centers.append(position)
        positive = target.positive_polygon or target.polygon
        if abs(exclusion_radius - 1000.0) <= 1e-9:
            polygon, resolution = self._planning_outer(
                positive, target.exclusion_centers
            )
        else:
            polygon = exclusion_aware_convex_outer(
                positive,
                target.exclusion_centers,
                exclusion_radius=exclusion_radius,
                resolution=2.0,
                max_depth=13,
            )
            resolution = 2.0
        if not polygon:
            if appended:
                target.exclusion_centers.pop()
            raise InconsistentObservations(
                f"channel {target.channel}: no-signal exclusions leave no feasible region"
            )
        target.polygon = polygon
        target.exclusion_resolution_m = resolution

    @staticmethod
    def diameter(target: TargetState) -> float:
        return polygon_diameter(target.polygon)[0]

    @staticmethod
    def covering_circle(target: TargetState) -> Circle:
        return minimum_enclosing_circle(target.polygon)

    def is_clearable(self, target: TargetState, clear_radius: float = 20.0) -> bool:
        return bool(target.polygon) and self.covering_circle(target).radius <= clear_radius + 1e-6

    @staticmethod
    def q1_report(target: TargetState) -> dict[str, object]:
        diameter, pairs = polygon_diameter(target.polygon)
        covered, diameter_circle = diameter_circle_covers(target.polygon)
        mec = minimum_enclosing_circle(target.polygon)
        return {
            "vertex_count": len(target.polygon),
            "diameter": diameter,
            "diameter_pair_count": len(pairs),
            "diameter_circle_covers": covered,
            "diameter_circle": diameter_circle,
            "minimum_enclosing_circle": mec,
        }
