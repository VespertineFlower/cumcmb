"""Problem 4 rolling planner built on the strict triangular geometry layer.

This module owns per-channel states, Lazy scoring, equal-length Hamilton route
selection, localization probes, diagnostic artifacts, and offline verification.
It has no dependency on the official simulator HTTP interface.
"""

from __future__ import annotations

import csv
import html
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import atan2, ceil, cos, degrees, log, pi, sin, sqrt
from pathlib import Path
from random import Random
from typing import Any, Iterable

from common.geometry import (
    ARENA_RADIUS,
    CLEAR_RADIUS,
    EPS,
    GOLDEN_ANGLE_DEG,
    GUARANTEED_RANGE,
    MAX_RECEPTION_RANGE,
    MESH_EDGE_TOLERANCE,
    STRICT_SEARCH_ROUTE,
    TRIANGLE25_POINTS,
    CellEvidence,
    CellStatus,
    Point,
    build_triangle_cells,
    clip_bearing_wedge,
    clip_reception_disk,
    distance,
    minimum_enclosing_circle,
    regular_disk_polygon,
)


# ---------------------------------------------------------------------------
# Per-channel state machine
# ---------------------------------------------------------------------------


class ChannelStatus(str, Enum):
    UNKNOWN = "unknown"
    LOCALIZING = "localizing"
    CLEARABLE = "clearable"
    CLEARED = "cleared"
    ABSENT = "absent"


@dataclass
class ChannelState:
    channel: int
    cells: dict[str, CellEvidence]
    polygon: list[Point]
    status: ChannelStatus = ChannelStatus.UNKNOWN
    positive_observations: int = 0
    mec_center: Point = Point(0.0, 0.0)
    mec_radius: float = ARENA_RADIUS
    measured_points: set[Point] = field(default_factory=set)
    localization_probe_index: int = 0
    bearing_lines_deg: list[float] = field(default_factory=list)
    probabilistic_absence: bool = False
    modeled_miss_risk: float = 1.0

    def record(
        self, point: Point, result: str, bearing_deg: float | None = None
    ) -> None:
        if self.status in {ChannelStatus.CLEARED, ChannelStatus.ABSENT}:
            return
        self.measured_points.add(point)
        touching = [
            evidence
            for evidence in self.cells.values()
            if point in evidence.cell.certificate_points
        ]
        if result == "no_signal":
            for evidence in touching:
                evidence.record(point, result)
        elif result in {"direction", "near"}:
            for evidence in touching:
                evidence.record(point, result)
            if result == "near":
                self.status = ChannelStatus.CLEARABLE
                self.mec_center, self.mec_radius = point, CLEAR_RADIUS
                return
            if bearing_deg is None:
                raise ValueError("direction result requires bearing_deg")
            self.status = ChannelStatus.LOCALIZING
            self.positive_observations += 1
            self.bearing_lines_deg.append(bearing_deg % 180.0)
            self.polygon = clip_bearing_wedge(self.polygon, point, bearing_deg)
            self.polygon = clip_reception_disk(
                self.polygon, point, radius=MAX_RECEPTION_RANGE
            )
            if not self.polygon:
                raise ValueError("bearing observation made the feasible region empty")
            self.mec_center, self.mec_radius = minimum_enclosing_circle(self.polygon)
            if self.mec_radius <= CLEAR_RADIUS:
                self.status = ChannelStatus.CLEARABLE
        else:
            raise ValueError(f"unsupported result: {result}")

        if self.status == ChannelStatus.UNKNOWN and all(
            evidence.status == CellStatus.EXCLUDED for evidence in self.cells.values()
        ):
            self.status = ChannelStatus.ABSENT

    def mark_cleared(self) -> None:
        self.status = ChannelStatus.CLEARED

    def mark_probabilistically_absent(self, modeled_miss_risk: float) -> None:
        if self.status != ChannelStatus.UNKNOWN:
            return
        self.status = ChannelStatus.ABSENT
        self.probabilistic_absence = True
        self.modeled_miss_risk = modeled_miss_risk


# ---------------------------------------------------------------------------
# Search actions and diagnostic artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stop:
    point: Point
    channels: tuple[int, ...]
    gain: int


@dataclass
class Q4ActionRecord:
    sequence: int
    action: str
    role: str
    point: Point
    channel: int
    result: str
    virtual_time_s: float
    movement_m: float


@dataclass
class Q4Diagnostics:
    actions: list[Q4ActionRecord] = field(default_factory=list)

    def record(
        self,
        action: str,
        role: str,
        point: Point,
        channel: int,
        result: str,
        virtual_time_s: float,
        movement_m: float,
    ) -> None:
        self.actions.append(
            Q4ActionRecord(
                len(self.actions) + 1,
                action,
                role,
                point,
                channel,
                result,
                virtual_time_s,
                movement_m,
            )
        )

    def report(self, planner: "Q4Triangle25SurroundPlanner") -> dict[str, Any]:
        role_counts: dict[str, int] = defaultdict(int)
        ch_rows: list[dict[str, Any]] = []
        for action in self.actions:
            role_counts[action.role] += 1
        for channel in sorted(planner.channels):
            state = planner.channels[channel]
            actions = [item for item in self.actions if item.channel == channel]
            measures = [item for item in actions if item.action == "measure"]
            clears = [item for item in actions if item.action == "clear"]
            first_hit = next(
                (
                    item.virtual_time_s
                    for item in measures
                    if item.result in {"direction", "near"}
                ),
                None,
            )
            cleared_time = next(
                (item.virtual_time_s for item in clears if item.result == "success"),
                None,
            )
            ch_rows.append(
                {
                    "channel": channel,
                    "status": state.status.value,
                    "measure_count": len(measures),
                    "direction_count": sum(
                        item.result == "direction" for item in measures
                    ),
                    "near_count": sum(item.result == "near" for item in measures),
                    "no_signal_count": sum(
                        item.result == "no_signal" for item in measures
                    ),
                    "search_count": sum(item.role == "search" for item in measures),
                    "lazy_count": sum(item.role == "lazy" for item in measures),
                    "localize_count": sum(item.role == "localize" for item in measures),
                    "clear_attempt_count": len(clears),
                    "clear_failure_count": sum(
                        item.result != "success" for item in clears
                    ),
                    "first_discovery_time_s": first_hit,
                    "cleared_time_s": cleared_time,
                    "final_mec_radius_m": state.mec_radius,
                    "probabilistic_absence": state.probabilistic_absence,
                    "modeled_miss_risk": state.modeled_miss_risk,
                }
            )
        measures = [item for item in self.actions if item.action == "measure"]
        clears = [item for item in self.actions if item.action == "clear"]
        batch_sizes: list[int] = []
        last_point: Point | None = None
        batch = 0
        for item in measures:
            if last_point is not None and distance(last_point, item.point) <= EPS:
                batch += 1
            else:
                if batch:
                    batch_sizes.append(batch)
                last_point, batch = item.point, 1
        if batch:
            batch_sizes.append(batch)
        return {
            "summary": {
                "total_movement_m": sum(item.movement_m for item in self.actions),
                "measure_count": len(measures),
                "direction_count": sum(item.result == "direction" for item in measures),
                "near_count": sum(item.result == "near" for item in measures),
                "no_signal_count": sum(item.result == "no_signal" for item in measures),
                "search_measure_count": role_counts.get("search", 0),
                "lazy_measure_count": role_counts.get("lazy", 0),
                "localize_measure_count": role_counts.get("localize", 0),
                "clear_attempt_count": len(clears),
                "clear_failure_count": sum(item.result != "success" for item in clears),
                "cleared_channels": sum(
                    s.status == ChannelStatus.CLEARED for s in planner.channels.values()
                ),
                "absent_channels": sum(
                    s.status == ChannelStatus.ABSENT for s in planner.channels.values()
                ),
                "measure_batch_count": len(batch_sizes),
                "multi_channel_batch_count": sum(size > 1 for size in batch_sizes),
                "max_channels_measured_at_one_point": max(batch_sizes, default=0),
                "final_virtual_time_s": self.actions[-1].virtual_time_s
                if self.actions
                else 0.0,
                "probabilistically_absent_channels": sum(
                    s.probabilistic_absence for s in planner.channels.values()
                ),
                "modeled_total_miss_risk_upper_bound": sum(
                    s.modeled_miss_risk
                    for s in planner.channels.values()
                    if s.probabilistic_absence
                ),
                "risk_model": planner.risk_model,
                "search_route_strategy": planner.search_route_strategy,
            },
            "channels": ch_rows,
            "actions": [asdict(item) for item in self.actions],
        }

    def write(
        self, planner: "Q4Triangle25SurroundPlanner", raw_log: Path
    ) -> dict[str, str]:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        metrics = raw_log.with_name(f"{raw_log.stem}_metrics_{stamp}.json")
        svg = raw_log.with_name(f"{raw_log.stem}_path_{stamp}.svg")
        csv_path = raw_log.with_name(f"{raw_log.stem}_path_{stamp}_points.csv")
        report = self.report(planner)
        metrics.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "sequence",
                    "action",
                    "role",
                    "x",
                    "y",
                    "channel",
                    "result",
                    "virtual_time_s",
                    "movement_m",
                ]
            )
            for item in self.actions:
                writer.writerow(
                    [
                        item.sequence,
                        item.action,
                        item.role,
                        item.point.x,
                        item.point.y,
                        item.channel,
                        item.result,
                        item.virtual_time_s,
                        item.movement_m,
                    ]
                )
        svg.write_text(self._svg(), encoding="utf-8")
        return {
            "metrics_file": os.path.relpath(metrics),
            "path_plot": os.path.relpath(svg),
            "detection_points_file": os.path.relpath(csv_path),
        }

    def _svg(self) -> str:
        width, height, margin = 1000, 900, 75
        extent = max(
            [ARENA_RADIUS * 1.08]
            + [max(abs(a.point.x), abs(a.point.y)) * 1.08 for a in self.actions]
        )
        scale = min(width - 2 * margin, height - 2 * margin) / (2 * extent)
        cx, cy = width / 2, height / 2

        def xy(point: Point) -> tuple[float, float]:
            return cx + point.x * scale, cy - point.y * scale

        colors = {
            "search": "#0891b2",
            "lazy": "#9333ea",
            "localize": "#f59e0b",
            "clear": "#dc2626",
        }
        role_labels = {
            "search": "覆盖检测",
            "lazy": "顺路检测",
            "localize": "定位观测",
            "clear": "清除",
        }
        result_labels = {
            "direction": "示向",
            "near": "近距离",
            "no_signal": "无信号",
            "success": "成功",
            "failure": "失败",
            "failed": "失败",
        }
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            f'<text x="{cx:.0f}" y="35" text-anchor="middle" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="22" font-weight="700">第四问：二十五点三角网格检测路径</text>',
            f'<circle cx="{cx}" cy="{cy}" r="{ARENA_RADIUS * scale:.2f}" fill="none" stroke="#334155" stroke-width="2" stroke-dasharray="7 5"/>',
        ]
        axis_limit = extent * 0.96
        left, _ = xy(Point(-axis_limit, 0.0))
        right, _ = xy(Point(axis_limit, 0.0))
        _, bottom = xy(Point(0.0, -axis_limit))
        _, top = xy(Point(0.0, axis_limit))
        lines.extend([
            f'<line id="x-axis" x1="{left:.2f}" y1="{cy:.2f}" x2="{right:.2f}" y2="{cy:.2f}" stroke="#1e293b" stroke-width="1.8"/>',
            f'<polygon points="{right:.2f},{cy:.2f} {right - 10:.2f},{cy - 5:.2f} {right - 10:.2f},{cy + 5:.2f}" fill="#1e293b"/>',
            f'<line id="y-axis" x1="{cx:.2f}" y1="{bottom:.2f}" x2="{cx:.2f}" y2="{top:.2f}" stroke="#1e293b" stroke-width="1.8"/>',
            f'<polygon points="{cx:.2f},{top:.2f} {cx - 5:.2f},{top + 10:.2f} {cx + 5:.2f},{top + 10:.2f}" fill="#1e293b"/>',
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3" fill="#1e293b"/>',
            f'<text x="{cx + 8:.2f}" y="{cy + 18:.2f}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12" fill="#1e293b">原点（0，0）</text>',
            f'<text x="{right - 5:.2f}" y="{cy - 12:.2f}" text-anchor="end" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12" fill="#1e293b">正东（横轴，米）</text>',
            f'<text x="{cx + 13:.2f}" y="{top + 14:.2f}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12" fill="#1e293b">正北（纵轴，米）</text>',
        ])
        path = [Point(0.0, 0.0)]
        for item in self.actions:
            if distance(path[-1], item.point) > EPS:
                path.append(item.point)
        coords = " ".join(f"{xy(p)[0]:.2f},{xy(p)[1]:.2f}" for p in path)
        lines.append(
            f'<polyline points="{coords}" fill="none" stroke="#64748b" stroke-width="2" stroke-opacity="0.7"/>'
        )
        grouped: dict[tuple[float, float, str], list[Q4ActionRecord]] = defaultdict(
            list
        )
        for item in self.actions:
            grouped[(round(item.point.x, 6), round(item.point.y, 6), item.role)].append(
                item
            )
        for (x, y, role), items in grouped.items():
            sx, sy = xy(Point(x, y))
            title = html.escape(
                f"{role_labels.get(role, '检测')}（横坐标 {x:.2f}，纵坐标 {y:.2f}）；"
                + "，".join(f"第{a.sequence}次、频道{a.channel}、{result_labels.get(a.result, '其他结果')}" for a in items)
            )
            color = colors.get(role, "#475569")
            radius = 8 if role == "clear" else min(10.0, 4.0 + sqrt(len(items)))
            lines.append(
                f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{radius:.2f}" fill="{color}"><title>{title}</title></circle>'
            )
        for index, (role, color) in enumerate(colors.items()):
            y = 65 + index * 22
            lines.append(
                f'<circle cx="820" cy="{y}" r="6" fill="{color}"/><text x="833" y="{y + 4}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12">{role_labels[role]}</text>'
            )
        lines.append(
            f'<text x="35" y="875" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12">动作 {len(self.actions)}；检测 {sum(a.action == "measure" for a in self.actions)}；清除 {sum(a.action == "clear" for a in self.actions)}</text>'
        )
        lines.append("</svg>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rolling search, Lazy observation, and localization planner
# ---------------------------------------------------------------------------


class Q4Triangle25SurroundPlanner:
    """Rolling planner using 25 vertices of a strict triangular certificate."""

    def __init__(
        self,
        channels: Iterable[int] = range(1, 21),
        total_miss_risk: float = 0.0,
        miss_probability_per_probe: float = 0.5,
        adaptive_route: bool = True,
    ) -> None:
        ch_list = list(channels)
        if total_miss_risk < 0.0 or total_miss_risk >= 1.0:
            raise ValueError("total_miss_risk must be in [0, 1)")
        if not 0.0 < miss_probability_per_probe < 1.0:
            raise ValueError("miss_probability_per_probe must be in (0, 1)")
        self.certificate_cells = build_triangle_cells()
        certificate_points = {
            point
            for cell in self.certificate_cells
            for point in cell.certificate_points
        }
        if set(STRICT_SEARCH_ROUTE) != certificate_points:
            raise RuntimeError(
                "strict search route does not match certificate point set"
            )
        self.search_route = STRICT_SEARCH_ROUTE
        self.adaptive_route = adaptive_route
        self.search_history: list[Point] = []
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
        self.total_miss_risk = total_miss_risk
        self.miss_probability_per_probe = miss_probability_per_probe
        self.per_channel_miss_risk = (
            total_miss_risk / len(ch_list)
            if ch_list and total_miss_risk
            else 0.0
        )
        self.probabilistic_negative_limit = (
            ceil(log(self.per_channel_miss_risk) / log(miss_probability_per_probe))
            if self.per_channel_miss_risk
            else None
        )
        self.risk_model = (
            "strict deterministic surround certificate"
            if self.probabilistic_negative_limit is None
            else (
                "model-based: distinct negative search probes are treated as independent; "
                f"P(miss per probe)<={miss_probability_per_probe:g}; union bound over "
                f"{len(ch_list)} channels"
            )
        )

    def pending_stops(self) -> dict[Point, set[int]]:
        """Return strict-search points still required by each unknown channel."""
        stops: dict[Point, set[int]] = {}
        for channel, state in self.channels.items():
            if state.status != ChannelStatus.UNKNOWN:
                continue
            for evidence in state.cells.values():
                if evidence.status in {CellStatus.EXCLUDED, CellStatus.DETECTED}:
                    continue
                for point in evidence.cell.certificate_points:
                    if point not in evidence.no_signal_points:
                        stops.setdefault(point, set()).add(channel)
        return stops

    def next_stop(
        self, current: Point, current_channel: int | None = None
    ) -> Stop | None:
        """Choose the next stop on an exact-length Hamilton route.

        ``current`` is retained in the public interface because the controller
        and route experiments pass the robot position explicitly.  The strict
        continuation itself is determined by ``search_history``.
        """
        candidates = self.pending_stops()
        if not candidates:
            return None
        if not self.adaptive_route:
            point = next(
                (point for point in self.search_route if point in candidates), None
            )
        else:
            point = self._adaptive_hamilton_stop(candidates)
        if point is None:
            raise RuntimeError("pending certificate point is missing from search route")
        self.search_history.append(point)
        channels = candidates[point]
        ordered = sorted(channels, key=lambda c: c != current_channel)
        return Stop(point, tuple(ordered), len(channels))

    def _lazy_information_score(self, point: Point) -> tuple[float, int]:
        values: list[float] = []
        for state in self.channels.values():
            if (
                state.status != ChannelStatus.LOCALIZING
                or point in state.measured_points
            ):
                continue
            center_dist = distance(point, state.mec_center)
            if center_dist - state.mec_radius > MAX_RECEPTION_RANGE + EPS:
                continue
            pred_line = (
                degrees(
                    atan2(state.mec_center.y - point.y, state.mec_center.x - point.x)
                )
                % 180.0
            )
            separation = max(
                (
                    min(
                        abs((pred_line - old) % 180.0),
                        180.0 - abs((pred_line - old) % 180.0),
                    )
                    for old in state.bearing_lines_deg
                ),
                default=90.0,
            )
            proximity = 1.0 / (1.0 + center_dist / GUARANTEED_RANGE)
            values.append(2.0 * separation / 90.0 + proximity)
        values.sort(reverse=True)
        useful = values[:4]
        return sum(useful), len(useful)

    @staticmethod
    def _has_hamilton_completion(start: Point, remaining: frozenset[Point]) -> bool:
        """Check that 1000 m mesh edges can still visit every remaining point."""
        memo: set[tuple[Point, frozenset[Point]]] = set()

        def search(node: Point, left: frozenset[Point]) -> bool:
            if not left:
                return True
            key = (node, left)
            if key in memo:
                return False
            options = [
                point
                for point in left
                if distance(node, point) <= GUARANTEED_RANGE + MESH_EDGE_TOLERANCE
            ]
            options.sort(
                key=lambda point: sum(
                    distance(point, other) <= GUARANTEED_RANGE + MESH_EDGE_TOLERANCE
                    for other in left
                    if other != point
                )
            )
            for point in options:
                if search(point, left - {point}):
                    return True
            memo.add(key)
            return False

        return search(start, remaining)

    def _adaptive_hamilton_stop(self, pending: dict[Point, set[int]]) -> Point | None:
        visited = set(self.search_history)
        remaining = frozenset(set(TRIANGLE25_POINTS) - visited)
        if not remaining:
            return None
        if not self.search_history:
            options = sorted(
                remaining,
                key=lambda point: (distance(Point(0.0, 0.0), point), point.x, point.y),
            )
            minimum = distance(Point(0.0, 0.0), options[0])
            options = [
                point
                for point in options
                if distance(Point(0.0, 0.0), point) <= minimum + EPS
            ]
        else:
            last = self.search_history[-1]
            options = [
                point
                for point in remaining
                if distance(last, point) <= GUARANTEED_RANGE + MESH_EDGE_TOLERANCE
            ]
        feasible = [
            point
            for point in options
            if self._has_hamilton_completion(point, remaining - {point})
        ]
        if not feasible:
            return None

        def score(point: Point) -> tuple[float, int, int, int, float, float]:
            information, useful = self._lazy_information_score(point)
            route_pref = -self.search_route.index(point)
            return (
                information,
                useful,
                len(pending.get(point, ())),
                route_pref,
                -point.x,
                -point.y,
            )

        return max(feasible, key=score)

    def record(
        self,
        channel: int,
        point: Point,
        result: str,
        bearing_deg: float | None = None,
    ) -> None:
        state = self.channels[channel]
        state.record(point, result, bearing_deg)
        if (
            result == "no_signal"
            and state.status == ChannelStatus.UNKNOWN
            and self.probabilistic_negative_limit is not None
        ):
            neg_n = len(state.measured_points)
            modeled_risk = self.miss_probability_per_probe**neg_n
            state.modeled_miss_risk = modeled_risk
            if neg_n >= self.probabilistic_negative_limit:
                state.mark_probabilistically_absent(modeled_risk)

    def localization_probe(self, channel: int) -> Point:
        """Return the next orbit point around the current feasible region.

        A compact four-direction pattern is used after the MEC has contracted;
        otherwise a wider exploratory orbit is replanned after every response.
        The persistent phase prevents repeated near-parallel bearings when the
        MEC center moves.
        """
        state = self.channels[channel]
        if state.status != ChannelStatus.LOCALIZING:
            raise ValueError("localization probes require a localizing channel")
        radius = state.mec_radius
        if radius <= GUARANTEED_RANGE / (1.0 + sqrt(2.0)):
            orbit_r = max(30.0, sqrt(2.0) * radius + 1.0)
            angles = (0, 90, 180, 270)
        else:
            # A first bearing often leaves a long wedge.  A tiny orbit around
            # its MEC can remain entirely on the emitter's dark side.  Use a
            # baseline comparable to the current uncertainty so the path
            # actually crosses to both sides of the source hypothesis.
            orbit_r = min(900.0, max(500.0, radius))
            angles = (0, 180, 90, 270, 45, 225, 135, 315, 30, 210, 120, 300)
        candidates = [
            Point(
                state.mec_center.x + orbit_r * cos(angle * pi / 180.0),
                state.mec_center.y + orbit_r * sin(angle * pi / 180.0),
            )
            for angle in angles
        ]
        # Do not restart at the first angle after every MEC update.  The center
        # moves after a positive bearing, so coordinate-based "unused" checks
        # alone would repeatedly select the same side and create near-parallel
        # bearings.  Persist the angular phase per channel instead.
        index = state.localization_probe_index
        state.localization_probe_index += 1
        if index < len(candidates):
            return candidates[index]
        # Continue with a low-discrepancy angular sequence after one full orbit.
        phase = index * GOLDEN_ANGLE_DEG
        return Point(
            state.mec_center.x + orbit_r * cos(phase * pi / 180.0),
            state.mec_center.y + orbit_r * sin(phase * pi / 180.0),
        )

    @property
    def complete(self) -> bool:
        return all(
            state.status in {ChannelStatus.ABSENT, ChannelStatus.CLEARED}
            for state in self.channels.values()
        )

    @property
    def search_route_strategy(self) -> str:
        """Stable identifier used by result summaries and diagnostic logs."""
        return (
            "adaptive_equal_length_hamilton"
            if self.adaptive_route
            else "fixed_optimal_hamilton"
        )


# ---------------------------------------------------------------------------
# Offline verification of the strict geometric certificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionalSource:
    channel: int
    point: Point
    heading_deg: float
    radius: float = GUARANTEED_RANGE

    def measure(self, sensor: Point) -> tuple[str, float | None]:
        separation = distance(sensor, self.point)
        if separation <= CLEAR_RADIUS:
            return "near", None
        heading = self.heading_deg * pi / 180.0
        illuminated = (
            cos(heading) * (sensor.x - self.point.x)
            + sin(heading) * (sensor.y - self.point.y)
            >= -EPS
        )
        if separation <= self.radius + EPS and illuminated:
            bearing = (
                degrees(atan2(self.point.y - sensor.y, self.point.x - sensor.x)) % 360
            )
            return "direction", bearing
        return "no_signal", None


def verify_surround_property(
    trials_per_cell: int = 200, seed: int = 20260911
) -> tuple[int, int]:
    """Monte Carlo check of the geometric certificate (not its proof)."""
    rng = Random(seed)
    cells = build_triangle_cells()
    checked = 0
    failures = 0
    for cell in cells:
        for _ in range(trials_per_cell):
            a, b, c = cell.vertices
            u, v = rng.random(), rng.random()
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            source = DirectionalSource(
                1,
                Point(
                    a.x + u * (b.x - a.x) + v * (c.x - a.x),
                    a.y + u * (b.y - a.y) + v * (c.y - a.y),
                ),
                rng.uniform(0.0, 360.0),
            )
            checked += 1
            if not any(
                source.measure(point)[0] != "no_signal"
                for point in cell.certificate_points
            ):
                failures += 1
    return checked, failures


def demo() -> None:
    cells = build_triangle_cells()
    vertices = {point for cell in cells for point in cell.certificate_points}
    checked, failures = verify_surround_property()
    planner = Q4Triangle25SurroundPlanner(channels=[1])
    first = planner.next_stop(Point(0.0, 0.0))
    print(f"certificate triangles: {len(cells)}")
    print(f"shared vertices: {len(vertices)}")
    print(
        f"strict open-route length: {sum(distance(a, b) for a, b in zip((Point(0.0, 0.0), *STRICT_SEARCH_ROUTE), STRICT_SEARCH_ROUTE)):.3f} m"
    )
    print(f"surround Monte Carlo: {checked} cases, {failures} failures")
    print(f"first rolling stop: {first}")
