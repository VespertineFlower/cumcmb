"""Problem 3 rolling path planner, independent of simulator communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from collections.abc import Iterable, Sequence

from .geometry import distance
from .models import Point

ARENA_RADIUS = 1800.0
GUARANTEE_RADIUS = 1000.0
DEFAULT_COVER_RING_RADIUS = 1130.0


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    point: Point
    kind: str = "cover"
    channel: int | None = None
    service_time_s: float = 0.0
    priority: float = 0.0
    mandatory: bool = True


def route_length(start: Point, route: Sequence[Task]) -> float:
    total = 0.0
    current = start
    for task in route:
        total += distance(current, task.point)
        current = task.point
    return total


def choose_route_aware_candidate(
    current: Point, candidates: Sequence[Point], continuation_anchors: Sequence[Point] = ()
) -> Point:
    if not candidates:
        raise ValueError("at least one candidate is required")

    def score(point: Point) -> float:
        continuation = min(
            (distance(point, anchor) for anchor in continuation_anchors), default=0.0
        )
        return distance(current, point) + continuation

    return min(candidates, key=score)


def cheapest_insertion_index(start: Point, route: Sequence[Task], task: Task) -> tuple[int, float]:
    best_index, best_delta = 0, float("inf")
    for index in range(len(route) + 1):
        before = start if index == 0 else route[index - 1].point
        if index == len(route):
            delta = distance(before, task.point)
        else:
            after = route[index].point
            delta = distance(before, task.point) + distance(task.point, after) - distance(before, after)
        if delta < best_delta:
            best_index, best_delta = index, delta
    return best_index, best_delta


def nearest_neighbor_route(start: Point, tasks: Iterable[Task]) -> list[Task]:
    remaining = list(tasks)
    route: list[Task] = []
    current = start
    while remaining:
        index = min(range(len(remaining)), key=lambda i: distance(current, remaining[i].point))
        task = remaining.pop(index)
        route.append(task)
        current = task.point
    return route


def two_opt_open(start: Point, route: Sequence[Task]) -> list[Task]:
    best = list(route)
    if len(best) < 3:
        return best
    while True:
        best_length = route_length(start, best)
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                if route_length(start, candidate) + 1e-9 < best_length:
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            return best


def seven_point_cover(radius: float = DEFAULT_COVER_RING_RADIUS) -> list[Task]:
    tasks = [Task("cover-center", Point(0.0, 0.0), service_time_s=119.0)]
    for k in range(6):
        angle = k * pi / 3.0
        tasks.append(
            Task(
                f"cover-{k}",
                Point(radius * cos(angle), radius * sin(angle)),
                service_time_s=120.0,
            )
        )
    return tasks


@dataclass(slots=True)
class ChannelCoverage:
    """Per-channel no-signal history and strict seven-point certification."""

    channel: int
    status: str = "unknown"
    no_signal_points: list[Point] = field(default_factory=list)
    fallback_scanned: set[str] = field(default_factory=set)

    def record(self, point: Point, result: str, fallback_id: str | None = None) -> None:
        if result == "no_signal":
            if all(distance(point, old) > 1e-7 for old in self.no_signal_points):
                self.no_signal_points.append(point)
            if fallback_id is not None:
                self.fallback_scanned.add(fallback_id)
            if len(self.fallback_scanned) == 7:
                self.status = "absent"
        elif result == "direction":
            self.status = "localizing"
        elif result == "near":
            self.status = "clearable"
        else:
            raise ValueError(f"unknown measure result: {result}")

    def mark_cleared(self) -> None:
        self.status = "cleared"

    @property
    def needs_search(self) -> bool:
        return self.status == "unknown"

    @property
    def certified_complete(self) -> bool:
        return self.status in {"absent", "cleared"}

    def history_key(self, digits: int = 3) -> tuple[tuple[float, float], ...]:
        return tuple((round(p.x, digits), round(p.y, digits)) for p in self.no_signal_points)


def polar_witness_grid(radial_steps: int = 18, angular_steps: int = 72) -> list[Point]:
    """Planning-only approximation; never used to certify absence."""
    points = [Point(0.0, 0.0)]
    for ring_idx in range(1, radial_steps + 1):
        radius = ARENA_RADIUS * ring_idx / radial_steps
        count = max(6, round(angular_steps * ring_idx / radial_steps))
        for angle_index in range(count):
            angle = 2.0 * pi * angle_index / count
            points.append(Point(radius * cos(angle), radius * sin(angle)))
    return points


@dataclass(slots=True)
class CoverageManager:
    witnesses: list[Point] = field(default_factory=polar_witness_grid)
    channels: dict[int, ChannelCoverage] = field(
        default_factory=lambda: {channel: ChannelCoverage(channel) for channel in range(1, 21)}
    )

    @staticmethod
    def _covered(point: Point, centers: Sequence[Point]) -> bool:
        return any(distance(point, center) <= GUARANTEE_RADIUS for center in centers)

    def marginal_witness_count(self, channel: int, candidate: Point) -> tuple[int, int]:
        state = self.channels[channel]
        if not state.needs_search:
            return 0, 0
        uncovered = [p for p in self.witnesses if not self._covered(p, state.no_signal_points)]
        new_cover = sum(distance(p, candidate) <= GUARANTEE_RADIUS for p in uncovered)
        return new_cover, len(uncovered)

    def channels_worth_scanning(
        self,
        candidate: Point,
        min_new_fraction: float = 0.02,
        force_channels: Iterable[int] = (),
        current_channel: int | None = None,
    ) -> list[int]:
        forced = list(dict.fromkeys(c for c in force_channels if self.channels[c].needs_search))
        selected: list[tuple[float, int]] = []
        for channel, state in self.channels.items():
            if not state.needs_search or channel in forced:
                continue
            new_count, uncovered_n = self.marginal_witness_count(channel, candidate)
            fraction = new_count / uncovered_n if uncovered_n else 0.0
            if fraction >= min_new_fraction:
                selected.append((fraction, channel))
        selected.sort(reverse=True)
        result = forced + [channel for _, channel in selected]
        if current_channel in result:
            result.remove(current_channel)
            result.insert(0, current_channel)
        return result

    def history_groups(self) -> list[list[int]]:
        groups: dict[tuple[tuple[float, float], ...], list[int]] = {}
        for channel, state in self.channels.items():
            if state.needs_search:
                groups.setdefault(state.history_key(), []).append(channel)
        return list(groups.values())

    def all_channels_certified(self) -> bool:
        return all(state.certified_complete for state in self.channels.values())


@dataclass(slots=True)
class RollingPlanner:
    speed_mps: float = 5.0
    priority_distance: float = 200.0
    tasks: dict[str, Task] = field(default_factory=dict)

    def upsert(self, task: Task) -> None:
        self.tasks[task.task_id] = task

    def remove(self, task_id: str) -> None:
        self.tasks.pop(task_id, None)

    def remove_channel(self, channel: int) -> None:
        for task_id, task in list(self.tasks.items()):
            if task.channel == channel:
                self.remove(task_id)

    def planned_route(self, current: Point) -> list[Task]:
        route = two_opt_open(current, nearest_neighbor_route(current, self.tasks.values()))
        for task in sorted(route, key=lambda item: item.priority, reverse=True):
            old_index = route.index(task)
            without = route[:old_index] + route[old_index + 1 :]
            new_index, added = cheapest_insertion_index(current, without, task)
            if new_index < old_index and added <= task.priority * self.priority_distance:
                route = without[:new_index] + [task] + without[new_index:]
        return two_opt_open(current, route)

    def next_task(self, current: Point) -> Task | None:
        route = self.planned_route(current)
        return route[0] if route else None

    def continuation_anchors(self, current: Point, exclude: str | None = None) -> list[Point]:
        tasks = [task for task_id, task in self.tasks.items() if task_id != exclude]
        return [task.point for task in two_opt_open(current, nearest_neighbor_route(current, tasks))]

    def estimated_time_s(
        self,
        current: Point,
        current_channel: int | None = None,
        coverage: CoverageManager | None = None,
    ) -> float:
        route = self.planned_route(current)
        total = route_length(current, route) / self.speed_mps
        channel = current_channel
        for task in route:
            if task.kind == "cover" and coverage is not None:
                channels = [
                    number
                    for number, state in coverage.channels.items()
                    if state.needs_search and task.task_id not in state.fallback_scanned
                ]
                if channel in channels:
                    channels.remove(channel)
                    channels.insert(0, channel)
                for number in channels:
                    total += 5.0
                    if channel is not None and number != channel:
                        total += 1.0
                    channel = number
            elif task.kind == "localize" and task.channel is not None:
                total += 5.0
                if channel is not None and task.channel != channel:
                    total += 1.0
                channel = task.channel
            elif task.kind == "clear":
                # Clear neither switches nor changes the measurement channel.
                total += 5.0
            else:
                total += task.service_time_s
        return total
