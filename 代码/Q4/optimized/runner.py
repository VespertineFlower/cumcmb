"""Independent official-simulator runner for the 25-point concentric certificate.

This experiment keeps the Q4 bearing, Lazy, localization, clear and logging
logic unchanged. It replaces only the strict certificate and fixed search
route. The original triangular-grid runner remains untouched.

Construction: origin + 12 inner vertices at radius 925 m and phase 15 degrees
+ 12 outer vertices at radius 1800/cos(15 degrees). The 36 stored certificate
triangles tile the outer regular polygon, which circumscribes the arena disk.
Every certificate edge is <=1000 m, so absence certification remains strict.
"""

from __future__ import annotations

import argparse
import json
import logging
from math import cos, hypot, pi, sin
from pathlib import Path
from typing import Iterable, Sequence

from common.geometry import (
    ARENA_RADIUS,
    EPS,
    GUARANTEED_RANGE,
    CellEvidence,
    Point,
    TriangleCell,
    distance,
    regular_disk_polygon,
)
from common.planner import (
    ChannelState,
    Q4Diagnostics,
    Q4Triangle25SurroundPlanner,
)
from baseline.runner import OfficialQ4Controller, OfficialSimulatorClient


CANDIDATE_ID = "ring-o12-i12-v000"
INNER_RADIUS_M = 925.0
RING_SIZE = 12
OUTER_RADIUS_M = ARENA_RADIUS / cos(pi / RING_SIZE)
RING25_POINTS = (
    Point(0.0, 0.0),
    *(
        Point(
            INNER_RADIUS_M * cos(pi / RING_SIZE + 2 * pi * index / RING_SIZE),
            INNER_RADIUS_M * sin(pi / RING_SIZE + 2 * pi * index / RING_SIZE),
        )
        for index in range(RING_SIZE)
    ),
    *(
        Point(
            OUTER_RADIUS_M * cos(2 * pi * index / RING_SIZE),
            OUTER_RADIUS_M * sin(2 * pi * index / RING_SIZE),
        )
        for index in range(RING_SIZE)
    ),
)

# Delaunay faces of the fixed concentric construction. Their edge lengths and
# complete polygon tiling are verified at import time below. No SciPy is needed
# for an official run.
RING25_FACES = (
    (10, 11, 0),
    (6, 18, 19),
    (18, 6, 5),
    (5, 6, 0),
    (1, 2, 0),
    (4, 5, 0),
    (20, 21, 8),
    (8, 9, 0),
    (9, 10, 0),
    (21, 9, 8),
    (9, 21, 22),
    (10, 9, 22),
    (23, 10, 22),
    (23, 24, 11),
    (10, 23, 11),
    (11, 12, 0),
    (12, 1, 0),
    (24, 12, 11),
    (12, 24, 13),
    (1, 12, 13),
    (14, 1, 13),
    (14, 15, 2),
    (1, 14, 2),
    (17, 18, 5),
    (4, 17, 5),
    (17, 4, 16),
    (15, 3, 2),
    (4, 3, 16),
    (3, 15, 16),
    (2, 3, 0),
    (3, 4, 0),
    (7, 6, 19),
    (7, 20, 8),
    (20, 7, 19),
    (7, 8, 0),
    (6, 7, 0),
)

# Fixed open route from the Pareto experiment. It starts with a measurement at
# the initial position (origin), then visits every other certificate vertex.
RING25_ROUTE_INDICES = (
    0,
    10,
    11,
    12,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    21,
    22,
    23,
    24,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
)
RING25_ROUTE = tuple(RING25_POINTS[index] for index in RING25_ROUTE_INDICES)


def _triangle_area(cell: TriangleCell) -> float:
    a, b, c = cell.vertices
    return abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2


def build_ring25_cells() -> list[TriangleCell]:
    """Build and verify the fixed 36-triangle strict coverage certificate."""
    cells = [
        TriangleCell(f"r{index:02d}", tuple(RING25_POINTS[i] for i in face))
        for index, face in enumerate(RING25_FACES)
    ]
    if len(RING25_POINTS) != 25 or len(cells) != 36:
        raise RuntimeError("ring25 point or triangle count changed")
    if set(RING25_ROUTE) != set(RING25_POINTS) or len(set(RING25_ROUTE)) != 25:
        raise RuntimeError("ring25 route does not visit every certificate point once")
    if not all(cell.certifiable(GUARANTEED_RANGE) for cell in cells):
        raise RuntimeError("ring25 triangle exceeds guaranteed reception range")

    # A planar triangulation with n=25 and h=12 has 2n-h-2=36 faces. Check
    # that all non-boundary edges are shared twice, all outer edges once, and
    # the summed face area equals the circumscribed regular polygon area.
    edge_counts: dict[tuple[int, int], int] = {}
    for face in RING25_FACES:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(a, b), max(a, b))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    outer_edges = {
        (
            min(13 + index, 13 + (index + 1) % RING_SIZE),
            max(13 + index, 13 + (index + 1) % RING_SIZE),
        )
        for index in range(RING_SIZE)
    }
    if (
        any(
            count != (1 if edge in outer_edges else 2)
            for edge, count in edge_counts.items()
        )
        or set(edge for edge, count in edge_counts.items() if count == 1) != outer_edges
    ):
        raise RuntimeError("ring25 triangles do not form the expected mesh")
    polygon_area = RING_SIZE * OUTER_RADIUS_M**2 * sin(2 * pi / RING_SIZE) / 2
    if abs(sum(_triangle_area(cell) for cell in cells) - polygon_area) > 1e-5:
        raise RuntimeError("ring25 triangles do not tile the outer polygon")
    if abs(OUTER_RADIUS_M * cos(pi / RING_SIZE) - ARENA_RADIUS) > EPS:
        raise RuntimeError("ring25 outer ring does not circumscribe the arena")
    return cells


RING25_CELLS = build_ring25_cells()
RING25_ROUTE_LENGTH_M = sum(
    distance(a, b) for a, b in zip((Point(0.0, 0.0), *RING25_ROUTE), RING25_ROUTE)
)


class Q4Ring25Planner(Q4Triangle25SurroundPlanner):
    """The existing Q4 state machine with the concentric strict certificate."""

    def __init__(
        self,
        channels: Iterable[int] = range(1, 21),
        total_miss_risk: float = 0.0,
        miss_probability_per_probe: float = 0.5,
    ) -> None:
        ch_list = list(channels)
        super().__init__(
            channels=ch_list,
            total_miss_risk=total_miss_risk,
            miss_probability_per_probe=miss_probability_per_probe,
            adaptive_route=False,
        )
        self.certificate_cells = RING25_CELLS
        self.search_route = RING25_ROUTE
        self.channels = {
            channel: ChannelState(
                channel=channel,
                cells={
                    cell.cell_id: CellEvidence(cell) for cell in self.certificate_cells
                },
                polygon=regular_disk_polygon(ARENA_RADIUS),
            )
            for channel in ch_list
        }

    @property
    def search_route_strategy(self) -> str:
        return "fixed_concentric25_2opt"


class Ring25Diagnostics(Q4Diagnostics):
    """Keep the baseline log schema but label the ring route correctly."""

    def _svg(self) -> str:
        return (
            super()
            ._svg()
            .replace(
                "第四问：二十五点三角网格检测路径",
                "第四问：二十五点同心环检测路径",
            )
        )


class OfficialRing25Controller(OfficialQ4Controller):
    """Run the ring25 certificate through the unchanged official controller."""

    def __init__(
        self,
        client: OfficialSimulatorClient,
        total_miss_risk: float = 0.0,
        miss_probability_per_probe: float = 0.5,
        write_artifacts: bool = True,
    ) -> None:
        super().__init__(
            client,
            write_artifacts=write_artifacts,
            total_miss_risk=total_miss_risk,
            miss_probability_per_probe=miss_probability_per_probe,
            adaptive_route=False,
        )
        self.planner = Q4Ring25Planner(
            total_miss_risk=total_miss_risk,
            miss_probability_per_probe=miss_probability_per_probe,
        )
        self.diag_ = Ring25Diagnostics()
        self.diagnostics = self.diag_

    def run(self) -> dict[str, object]:
        result = super().run()
        result["mode"] = (
            "q4_ring25_surround_strict"
            if self.planner.probabilistic_negative_limit is None
            else "q4_ring25_surround_probabilistic"
        )
        result["certificate_triangles"] = len(self.planner.certificate_cells)
        result["strict_route_upper_bound_m"] = round(RING25_ROUTE_LENGTH_M, 3)
        result["max_search_radius_m"] = round(
            max(hypot(point.x, point.y) for point in RING25_POINTS), 3
        )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official Q4 runner: strict 25-point concentric certificate"
    )
    parser.add_argument("--robot-id")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--arena-id", default="default")
    parser.add_argument("--log", type=Path, default=Path("logs/q4_ring25.jsonl"))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--demo", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo:
        print(f"candidate: {CANDIDATE_ID}")
        print(f"strict points: {len(RING25_POINTS)}")
        print(f"certificate triangles: {len(RING25_CELLS)}")
        print(
            f"maximum certificate edge: {max(max(distance(a,b) for a,b in ((c.vertices[0],c.vertices[1]),(c.vertices[1],c.vertices[2]),(c.vertices[2],c.vertices[0]))) for c in RING25_CELLS):.3f} m"
        )
        print(f"fixed route length: {RING25_ROUTE_LENGTH_M:.3f} m")
        print(f"maximum exterior reach: {OUTER_RADIUS_M - ARENA_RADIUS:.3f} m")
        return 0
    robot_id = args.robot_id or input("请输入当前模拟器登录的参赛队号：").strip()
    if not robot_id:
        raise SystemExit("参赛队号不能为空")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    client = OfficialSimulatorClient(
        robot_id=robot_id,
        base_url=args.base_url,
        arena_id=args.arena_id,
        timeout_s=args.timeout,
        attempts=args.retries,
        log_path=args.log,
    )
    print(
        json.dumps(OfficialRing25Controller(client).run(), ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
