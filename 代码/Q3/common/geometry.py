from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable

from .models import Circle, Point

EPS = 1e-9


def add(a: Point, b: Point) -> Point:
    return Point(a.x + b.x, a.y + b.y)


def sub(a: Point, b: Point) -> Point:
    return Point(a.x - b.x, a.y - b.y)


def scale(a: Point, k: float) -> Point:
    return Point(a.x * k, a.y * k)


def dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def cross(a: Point, b: Point) -> float:
    return a.x * b.y - a.y * b.x


def distance2(a: Point, b: Point) -> float:
    dx, dy = a.x - b.x, a.y - b.y
    return dx * dx + dy * dy


def distance(a: Point, b: Point) -> float:
    return math.sqrt(distance2(a, b))


def regular_outer_polygon(center: Point, radius: float, sides: int = 128) -> list[Point]:
    """A circumscribed regular polygon, so the physical disk is never excluded."""
    if sides < 4:
        raise ValueError("sides must be at least 4")
    outer_r = radius / math.cos(math.pi / sides)
    return [
        Point(center.x + outer_r * math.cos(2 * math.pi * i / sides + math.pi / sides),
              center.y + outer_r * math.sin(2 * math.pi * i / sides + math.pi / sides))
        for i in range(sides)
    ]


def _deduplicate_polygon(points: list[Point], eps: float = 1e-7) -> list[Point]:
    if not points:
        return []
    result: list[Point] = []
    for p in points:
        if not result or distance2(p, result[-1]) > eps * eps:
            result.append(p)
    if len(result) > 1 and distance2(result[0], result[-1]) <= eps * eps:
        result.pop()
    if len(result) < 3:
        return result
    cleaned: list[Point] = []
    for i, p in enumerate(result):
        prev, nxt = result[i - 1], result[(i + 1) % len(result)]
        if abs(cross(sub(p, prev), sub(nxt, p))) > eps:
            cleaned.append(p)
    return cleaned or result


def clip_halfplane(
    polygon: list[Point], signed_value: Callable[[Point], float], eps: float = EPS
) -> list[Point]:
    """Sutherland-Hodgman clipping, keeping signed_value(point) >= 0."""
    if not polygon:
        return []
    output: list[Point] = []
    a = polygon[-1]
    fa = signed_value(a)
    for b in polygon:
        fb = signed_value(b)
        a_in, b_in = fa >= -eps, fb >= -eps
        if a_in != b_in:
            denom = fa - fb
            if abs(denom) > eps:
                t = fa / denom
                output.append(Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)))
        if b_in:
            output.append(b)
        a, fa = b, fb
    return _deduplicate_polygon(output)


def clip_bearing_wedge(
    polygon: list[Point], station: Point, bearing_deg: float, error_deg: float = 1.0
) -> list[Point]:
    lo = math.radians((bearing_deg - error_deg) % 360.0)
    hi = math.radians((bearing_deg + error_deg) % 360.0)
    u_lo = Point(math.cos(lo), math.sin(lo))
    u_hi = Point(math.cos(hi), math.sin(hi))
    result = clip_halfplane(polygon, lambda p: cross(u_lo, sub(p, station)))
    return clip_halfplane(result, lambda p: -cross(u_hi, sub(p, station)))


def clip_to_outer_disk(
    polygon: list[Point], center: Point, radius: float, sides: int = 64
) -> list[Point]:
    """Conservatively clip to a circumscribed polygon around a known disk."""
    result = polygon
    apothem = radius
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        normal = Point(math.cos(angle), math.sin(angle))
        result = clip_halfplane(
            result, lambda p, n=normal: apothem - dot(sub(p, center), n)
        )
        if not result:
            break
    return result


def convex_hull(points: Iterable[Point]) -> list[Point]:
    """Monotone-chain convex hull in counter-clockwise order."""
    unique = sorted({(p.x, p.y) for p in points})
    pts = [Point(x, y) for x, y in unique]
    if len(pts) <= 1:
        return pts

    def build(sequence: list[Point]) -> list[Point]:
        half: list[Point] = []
        for point in sequence:
            while len(half) >= 2 and cross(
                sub(half[-1], half[-2]), sub(point, half[-1])
            ) <= EPS:
                half.pop()
            half.append(point)
        return half

    lower = build(pts)
    upper = build(list(reversed(pts)))
    return lower[:-1] + upper[:-1]


def exclusion_aware_convex_outer(
    positive_polygon: list[Point],
    exclusion_centers: Iterable[Point],
    exclusion_radius: float = 1000.0,
    resolution: float = 25.0,
    max_depth: int = 10,
) -> list[Point]:
    """Certified outer convex hull after subtracting no-signal disks.

    A quadtree cell is discarded only when its entire clipped rectangle lies
    inside an exclusion disk.  Boundary cells are retained, so the returned
    hull may be conservative but can never exclude a feasible target.
    """
    centers = list(exclusion_centers)
    if not positive_polygon or not centers:
        return list(positive_polygon)

    min_x = min(p.x for p in positive_polygon)
    max_x = max(p.x for p in positive_polygon)
    min_y = min(p.y for p in positive_polygon)
    max_y = max(p.y for p in positive_polygon)
    kept: list[Point] = []

    def clip_box(x0: float, x1: float, y0: float, y1: float) -> list[Point]:
        clipped = clip_halfplane(positive_polygon, lambda p: p.x - x0)
        clipped = clip_halfplane(clipped, lambda p: x1 - p.x)
        clipped = clip_halfplane(clipped, lambda p: p.y - y0)
        return clip_halfplane(clipped, lambda p: y1 - p.y)

    def min_distance_to_box(center: Point, x0: float, x1: float, y0: float, y1: float) -> float:
        dx = max(x0 - center.x, 0.0, center.x - x1)
        dy = max(y0 - center.y, 0.0, center.y - y1)
        return math.hypot(dx, dy)

    def visit(x0: float, x1: float, y0: float, y1: float, depth: int) -> None:
        clipped = clip_box(x0, x1, y0, y1)
        if not clipped:
            return
        # A convex function reaches its maximum over a polygon at a vertex.
        if any(
            max(distance(center, point) for point in clipped)
            <= exclusion_radius + EPS
            for center in centers
        ):
            return
        touches_exclusion = any(
            min_distance_to_box(center, x0, x1, y0, y1)
            < exclusion_radius
            for center in centers
        )
        if (
            not touches_exclusion
            or depth >= max_depth
            or max(x1 - x0, y1 - y0) <= resolution
        ):
            kept.extend(clipped)
            return
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2
        visit(x0, xm, y0, ym, depth + 1)
        visit(xm, x1, y0, ym, depth + 1)
        visit(x0, xm, ym, y1, depth + 1)
        visit(xm, x1, ym, y1, depth + 1)

    visit(min_x, max_x, min_y, max_y, 0)
    return convex_hull(kept)


def polygon_diameter(polygon: list[Point]) -> tuple[float, list[tuple[Point, Point]]]:
    """Rotating-calipers diameter for a counter-clockwise convex polygon."""
    n = len(polygon)
    if n < 2:
        return 0.0, []
    if n == 2:
        return distance(polygon[0], polygon[1]), [(polygon[0], polygon[1])]
    best2 = -1.0
    pairs: list[tuple[Point, Point]] = []

    def consider(a: Point, b: Point) -> None:
        nonlocal best2, pairs
        d2 = distance2(a, b)
        tol = 1e-8 * max(1.0, best2, d2)
        if d2 > best2 + tol:
            best2, pairs = d2, [(a, b)]
        elif abs(d2 - best2) <= tol and all({a, b} != {x, y} for x, y in pairs):
            pairs.append((a, b))

    j = 1
    for i in range(n):
        ni = (i + 1) % n
        area = lambda k: abs(cross(sub(polygon[ni], polygon[i]), sub(polygon[k], polygon[i])))
        while area((j + 1) % n) > area(j) + EPS:
            j = (j + 1) % n
        consider(polygon[i], polygon[j])
        consider(polygon[ni], polygon[j])
        if abs(area((j + 1) % n) - area(j)) <= EPS:
            consider(polygon[i], polygon[(j + 1) % n])
            consider(polygon[ni], polygon[(j + 1) % n])
    return math.sqrt(max(0.0, best2)), pairs


def _contains(circle: Circle, p: Point, eps: float = 1e-7) -> bool:
    return distance2(circle.center, p) <= (circle.radius + eps) ** 2


def _circle2(a: Point, b: Point) -> Circle:
    center = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
    return Circle(center, distance(a, b) / 2)


def _circle3(a: Point, b: Point, c: Point) -> Circle | None:
    d = 2 * cross(sub(b, a), sub(c, a))
    if abs(d) <= EPS:
        return None
    a2, b2, c2 = dot(a, a), dot(b, b), dot(c, c)
    ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d
    uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d
    center = Point(ux, uy)
    return Circle(center, distance(center, a))


def minimum_enclosing_circle(points: Iterable[Point]) -> Circle:
    pts = list(points)
    if not pts:
        raise ValueError("minimum enclosing circle requires at least one point")
    random.Random(2026).shuffle(pts)
    circle = Circle(pts[0], 0.0)
    for i, p in enumerate(pts):
        if _contains(circle, p):
            continue
        circle = Circle(p, 0.0)
        for j in range(i):
            q = pts[j]
            if _contains(circle, q):
                continue
            circle = _circle2(p, q)
            for k in range(j):
                r = pts[k]
                if _contains(circle, r):
                    continue
                candidate = _circle3(p, q, r)
                if candidate is not None:
                    circle = candidate
                else:
                    choices = (_circle2(p, q), _circle2(p, r), _circle2(q, r))
                    circle = min(
                        (c0 for c0 in choices if all(_contains(c0, x) for x in (p, q, r))),
                        key=lambda c0: c0.radius,
                    )
    return circle


def diameter_circle_covers(polygon: list[Point]) -> tuple[bool, Circle | None]:
    diameter, pairs = polygon_diameter(polygon)
    for a, b in pairs:
        circle = _circle2(a, b)
        if all(_contains(circle, p) for p in polygon):
            return True, circle
    return False, None if not pairs else _circle2(*pairs[0])


def project_to_convex_polygon(point: Point, polygon: list[Point]) -> Point:
    """Return point itself when inside, otherwise its nearest boundary projection."""
    if not polygon:
        raise ValueError("cannot project onto an empty polygon")
    if len(polygon) == 1:
        return polygon[0]
    # The localization polygons are counter-clockwise.
    if all(cross(sub(polygon[(i + 1) % len(polygon)], polygon[i]), sub(point, polygon[i])) >= -EPS
           for i in range(len(polygon))):
        return point
    best, best_d2 = polygon[0], float("inf")
    edge_count = len(polygon) if len(polygon) > 2 else 1
    for i in range(edge_count):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        ab = sub(b, a)
        denom = dot(ab, ab)
        t = 0.0 if denom <= EPS else max(0.0, min(1.0, dot(sub(point, a), ab) / denom))
        candidate = add(a, scale(ab, t))
        d2 = distance2(point, candidate)
        if d2 < best_d2:
            best, best_d2 = candidate, d2
    return best
