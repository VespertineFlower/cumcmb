"""Problem 4 geometry and strict 25-point coverage certificate.

This simulator-independent module contains only mathematical primitives:
the certified triangular mesh, convex clipping, and minimum enclosing circle.
The online state and route policy are implemented in
``common.planner``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import cos, hypot, pi, sin, sqrt
from typing import Sequence


# ---------------------------------------------------------------------------
# Model parameters and certified 25-point construction
# ---------------------------------------------------------------------------

ARENA_RADIUS = 1800.0
GUARANTEED_RANGE = 1000.0
MAX_RECEPTION_RANGE = 1500.0
BEARING_ERROR_DEG = 1.01
CLEAR_RADIUS = 20.0
EPS = 1e-9
MESH_EDGE_TOLERANCE = 1e-7
GOLDEN_ANGLE_DEG = 137.507764

TRIANGLE25_LATTICE_INDICES = (
    (-3, 1),
    (-3, 2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (-1, 2),
    (0, -2),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 2),
    (1, -3),
    (1, -2),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, -3),
    (2, -2),
    (2, -1),
    (2, 0),
)
TRIANGLE25_FACES = (
    (9, 8, 13),
    (8, 12, 13),
    (12, 8, 7),
    (9, 10, 5),
    (1, 4, 5),
    (4, 9, 5),
    (4, 8, 9),
    (0, 4, 1),
    (2, 6, 7),
    (14, 10, 9),
    (14, 9, 13),
    (10, 14, 15),
    (14, 20, 15),
    (3, 2, 7),
    (8, 3, 7),
    (4, 3, 8),
    (0, 3, 4),
    (11, 12, 7),
    (6, 11, 7),
    (11, 17, 12),
    (17, 11, 16),
    (12, 18, 13),
    (23, 18, 22),
    (17, 18, 12),
    (18, 17, 22),
    (17, 21, 22),
    (21, 17, 16),
    (19, 14, 13),
    (19, 23, 24),
    (20, 19, 24),
    (14, 19, 20),
    (19, 18, 23),
    (18, 19, 13),
)
TRIANGLE25_ROUTE_INDICES = (
    13,
    8,
    3,
    0,
    1,
    4,
    5,
    9,
    10,
    15,
    14,
    20,
    24,
    19,
    23,
    18,
    22,
    21,
    16,
    17,
    12,
    11,
    6,
    2,
    7,
)


@dataclass(frozen=True, order=True)
class Point:
    x: float
    y: float


TRIANGLE25_POINTS = tuple(
    Point(1000.0 * i + 500.0 * j + 150.0, 500.0 * sqrt(3.0) * j + 50.0 * sqrt(3.0))
    for i, j in TRIANGLE25_LATTICE_INDICES
)
STRICT_SEARCH_ROUTE = tuple(TRIANGLE25_POINTS[i] for i in TRIANGLE25_ROUTE_INDICES)


# ---------------------------------------------------------------------------
# Basic geometry and strict-search certificate
# ---------------------------------------------------------------------------


def distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def cross(a: Point, b: Point, c: Point) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


@dataclass(frozen=True)
class TriangleCell:
    cell_id: str
    vertices: tuple[Point, Point, Point]

    @property
    def certificate_points(self) -> tuple[Point, ...]:
        return self.vertices

    def contains(self, point: Point) -> bool:
        a, b, c = self.vertices
        signs = (cross(a, b, point), cross(b, c, point), cross(c, a, point))
        return all(value >= -EPS for value in signs) or all(
            value <= EPS for value in signs
        )

    def certifiable(self, guaranteed_range: float = GUARANTEED_RANGE) -> bool:
        return (
            max(
                distance(a, b)
                for a, b in (
                    (self.vertices[0], self.vertices[1]),
                    (self.vertices[1], self.vertices[2]),
                    (self.vertices[2], self.vertices[0]),
                )
            )
            <= guaranteed_range + EPS
        )


def build_triangle_cells(
    arena_radius: float = ARENA_RADIUS,
    guaranteed_range: float = GUARANTEED_RANGE,
) -> list[TriangleCell]:
    if (
        abs(arena_radius - ARENA_RADIUS) > EPS
        or abs(guaranteed_range - GUARANTEED_RANGE) > EPS
    ):
        raise ValueError("the certified 25-point mesh is fixed for radii 1800/1000")
    cells = [
        TriangleCell(f"t{index:02d}", tuple(TRIANGLE25_POINTS[i] for i in face))
        for index, face in enumerate(TRIANGLE25_FACES)
    ]
    if not all(cell.certifiable(guaranteed_range) for cell in cells):
        raise RuntimeError("invalid triangular certificate")
    return cells


class CellStatus(str, Enum):
    UNVISITED = "unvisited"
    PARTIAL = "partial"
    EXCLUDED = "excluded"
    DETECTED = "detected"


@dataclass
class CellEvidence:
    cell: TriangleCell
    no_signal_points: set[Point] = field(default_factory=set)
    detected_at: Point | None = None

    @property
    def status(self) -> CellStatus:
        if self.detected_at is not None:
            return CellStatus.DETECTED
        if all(
            point in self.no_signal_points for point in self.cell.certificate_points
        ):
            return CellStatus.EXCLUDED
        if self.no_signal_points:
            return CellStatus.PARTIAL
        return CellStatus.UNVISITED

    def record(self, point: Point, result: str) -> None:
        if point not in self.cell.certificate_points:
            raise ValueError(
                "cell evidence may only be recorded at a certificate point"
            )
        if result == "no_signal":
            self.no_signal_points.add(point)
        elif result in {"direction", "near"}:
            self.detected_at = point
        else:
            raise ValueError(f"unsupported result: {result}")


# ---------------------------------------------------------------------------
# Convex feasible-region update and minimum enclosing circle
# ---------------------------------------------------------------------------


def regular_disk_polygon(radius: float, vertices: int = 360) -> list[Point]:
    """Return a regular polygon circumscribing the requested disk."""
    if radius <= 0 or vertices < 3:
        raise ValueError("radius must be positive and vertices must be at least 3")
    vertex_radius = radius / cos(pi / vertices)
    return [
        Point(
            vertex_radius * cos((2 * k + 1) * pi / vertices),
            vertex_radius * sin((2 * k + 1) * pi / vertices),
        )
        for k in range(vertices)
    ]


def clip_halfplane(
    polygon: Sequence[Point], normal: Point, bound: float
) -> list[Point]:
    """Clip a convex polygon by normal dot x <= bound."""
    if not polygon:
        return []
    output: list[Point] = []
    previous = polygon[-1]
    prev_val = normal.x * previous.x + normal.y * previous.y - bound
    for current in polygon:
        current_value = normal.x * current.x + normal.y * current.y - bound
        previous_inside = prev_val <= EPS
        current_inside = current_value <= EPS
        if previous_inside != current_inside:
            denominator = prev_val - current_value
            ratio = 0.0 if abs(denominator) <= EPS else prev_val / denominator
            output.append(
                Point(
                    previous.x + ratio * (current.x - previous.x),
                    previous.y + ratio * (current.y - previous.y),
                )
            )
        if current_inside:
            output.append(current)
        previous, prev_val = current, current_value
    return output


def clip_bearing_wedge(
    polygon: Sequence[Point],
    sensor: Point,
    bearing_deg: float,
    error_deg: float = BEARING_ERROR_DEG,
) -> list[Point]:
    """Intersect a convex polygon with a forward bearing wedge."""
    low = (bearing_deg - error_deg) * pi / 180.0
    high = (bearing_deg + error_deg) * pi / 180.0
    u_low = Point(cos(low), sin(low))
    u_high = Point(cos(high), sin(high))
    # cross(u_low, x-sensor) >= 0
    first_normal = Point(u_low.y, -u_low.x)
    first_bound = first_normal.x * sensor.x + first_normal.y * sensor.y
    clipped = clip_halfplane(polygon, first_normal, first_bound)
    # cross(u_high, x-sensor) <= 0
    second_normal = Point(-u_high.y, u_high.x)
    second_bound = second_normal.x * sensor.x + second_normal.y * sensor.y
    return clip_halfplane(clipped, second_normal, second_bound)


def clip_reception_disk(
    polygon: Sequence[Point],
    sensor: Point,
    radius: float = MAX_RECEPTION_RANGE,
    sides: int = 180,
) -> list[Point]:
    """Conservatively intersect with an outer polygon around a reception disk."""
    result = list(polygon)
    # Tangent half-planes form a circumscribed polygon, so no feasible point is
    # incorrectly removed. It is the conservative disk representation used by
    # the online feasible-region update.
    for k in range(sides):
        angle = 2 * pi * k / sides
        normal = Point(cos(angle), sin(angle))
        bound = normal.x * sensor.x + normal.y * sensor.y + radius
        result = clip_halfplane(result, normal, bound)
        if not result:
            break
    return result


def _circle_from_two(a: Point, b: Point) -> tuple[Point, float]:
    center = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
    return center, distance(a, b) / 2


def _circle_from_three(a: Point, b: Point, c: Point) -> tuple[Point, float] | None:
    denominator = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
    if abs(denominator) <= EPS:
        return None
    aa = a.x * a.x + a.y * a.y
    bb = b.x * b.x + b.y * b.y
    cc = c.x * c.x + c.y * c.y
    center = Point(
        (aa * (b.y - c.y) + bb * (c.y - a.y) + cc * (a.y - b.y)) / denominator,
        (aa * (c.x - b.x) + bb * (a.x - c.x) + cc * (b.x - a.x)) / denominator,
    )
    return center, distance(center, a)


def minimum_enclosing_circle(points: Sequence[Point]) -> tuple[Point, float]:
    """Compute the exact MEC by exhaustive candidates for a small polygon."""
    if not points:
        raise ValueError("cannot enclose an empty point set")
    candidates: list[tuple[Point, float]] = [(points[0], 0.0)]
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            candidates.append(_circle_from_two(a, b))
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                circle = _circle_from_three(a, points[j], points[k])
                if circle is not None:
                    candidates.append(circle)
    feasible = [
        (center, radius)
        for center, radius in candidates
        if all(distance(center, point) <= radius + 1e-7 for point in points)
    ]
    return min(feasible, key=lambda item: item[1])
