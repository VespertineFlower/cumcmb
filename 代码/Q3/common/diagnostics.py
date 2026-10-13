"""Runtime metrics for evaluating the Q3 online scheduler."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .geometry import distance
from .models import Point


@dataclass(slots=True)
class ChannelMetrics:
    channel: int
    direction_count: int = 0
    near_count: int = 0
    no_signal_count: int = 0
    first_discovery_time_s: float | None = None
    cleared_time_s: float | None = None
    movement_first_discovery_to_clear_m: float = 0.0
    clear_attempt_count: int = 0
    clear_failure_count: int = 0


@dataclass(slots=True)
class ExcursionMetrics:
    start: Point
    return_cover: Point
    actual_distance_m: float
    direct_distance_m: float
    extra_distance_m: float


@dataclass(slots=True)
class ReversalEvent:
    at: Point
    previous_move_m: float
    current_move_m: float
    turn_angle_deg: float


@dataclass(slots=True)
class ActionRecord:
    sequence: int
    action: str
    role: str
    point: Point
    channel: int
    result: str
    virtual_time_s: float
    movement_m: float


@dataclass(slots=True)
class _OpenExcursion:
    start: Point
    distance_m: float = 0.0


@dataclass(slots=True)
class RuntimeDiagnostics:
    channels: dict[int, ChannelMetrics] = field(default_factory=dict)
    total_movement_m: float = 0.0
    excursions: list[ExcursionMetrics] = field(default_factory=list)
    reversals: list[ReversalEvent] = field(default_factory=list)
    same_point_measure_batches: list[list[int]] = field(default_factory=list)
    piggyback_measure_count: int = 0
    actions: list[ActionRecord] = field(default_factory=list)
    _open_excursion: _OpenExcursion | None = None
    _last_move_vector: tuple[float, float] | None = None
    _last_move_length: float = 0.0
    _batch_point: Point | None = None
    _batch_channels: list[int] = field(default_factory=list)

    def channel(self, channel: int) -> ChannelMetrics:
        return self.channels.setdefault(channel, ChannelMetrics(channel))

    def record_move(self, start: Point, end: Point) -> None:
        length = distance(start, end)
        if length <= 1e-8:
            return
        self.total_movement_m += length
        for metrics in self.channels.values():
            if metrics.first_discovery_time_s is not None and metrics.cleared_time_s is None:
                metrics.movement_first_discovery_to_clear_m += length
        if self._open_excursion is not None:
            self._open_excursion.distance_m += length

        vector = (end.x - start.x, end.y - start.y)
        if self._last_move_vector is not None and self._last_move_length >= 800 and length >= 800:
            ax, ay = self._last_move_vector
            bx, by = vector
            cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (self._last_move_length * length)))
            angle = math.degrees(math.acos(cosine))
            if angle >= 150:
                self.reversals.append(ReversalEvent(start, self._last_move_length, length, angle))
        self._last_move_vector = vector
        self._last_move_length = length

    def record_measure(self, point: Point, channel: int, result: str, virtual_time_s: float) -> None:
        metrics = self.channel(channel)
        if result == "direction":
            metrics.direction_count += 1
        elif result == "near":
            metrics.near_count += 1
        elif result == "no_signal":
            metrics.no_signal_count += 1
        if result in {"direction", "near"} and metrics.first_discovery_time_s is None:
            metrics.first_discovery_time_s = virtual_time_s
        self._record_measure_batch(point, channel)

    def record_action(
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
            ActionRecord(
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

    def record_clear(self, channel: int, success: bool, virtual_time_s: float) -> None:
        self._flush_measure_batch()
        metrics = self.channel(channel)
        metrics.clear_attempt_count += 1
        if success:
            metrics.cleared_time_s = virtual_time_s
        else:
            metrics.clear_failure_count += 1

    def record_piggyback(self, count: int) -> None:
        self.piggyback_measure_count += count

    def begin_non_cover_excursion(self, start: Point) -> None:
        if self._open_excursion is None:
            self._open_excursion = _OpenExcursion(start)

    def end_at_cover(self, cover_point: Point) -> None:
        if self._open_excursion is None:
            return
        direct = distance(self._open_excursion.start, cover_point)
        actual = self._open_excursion.distance_m
        self.excursions.append(
            ExcursionMetrics(
                self._open_excursion.start,
                cover_point,
                actual,
                direct,
                max(0.0, actual - direct),
            )
        )
        self._open_excursion = None

    def _record_measure_batch(self, point: Point, channel: int) -> None:
        if self._batch_point is None or distance(self._batch_point, point) <= 1e-7:
            self._batch_point = point
            self._batch_channels.append(channel)
            return
        self._flush_measure_batch()
        self._batch_point = point
        self._batch_channels = [channel]

    def _flush_measure_batch(self) -> None:
        if self._batch_channels:
            self.same_point_measure_batches.append(self._batch_channels)
        self._batch_point = None
        self._batch_channels = []

    def report(self) -> dict[str, object]:
        self._flush_measure_batch()
        discovered = [m for m in self.channels.values() if m.first_discovery_time_s is not None]
        cleared = [m for m in discovered if m.cleared_time_s is not None]
        batch_sizes = [len(batch) for batch in self.same_point_measure_batches]
        return {
            "summary": {
                "total_movement_m": self.total_movement_m,
                "discovered_channels": len(discovered),
                "cleared_channels": len(cleared),
                "effective_direction_count": sum(m.direction_count for m in discovered),
                "cover_excursion_count": len(self.excursions),
                "cover_return_extra_distance_m": sum(e.extra_distance_m for e in self.excursions),
                "long_reversal_count": len(self.reversals),
                "measure_batch_count": len(batch_sizes),
                "multi_channel_batch_count": sum(size > 1 for size in batch_sizes),
                "max_channels_measured_at_one_point": max(batch_sizes, default=0),
                "piggyback_measure_count": self.piggyback_measure_count,
                "clear_attempt_count": sum(m.clear_attempt_count for m in self.channels.values()),
                "clear_failure_count": sum(m.clear_failure_count for m in self.channels.values()),
            },
            "channels": [asdict(self.channels[c]) for c in sorted(self.channels)],
            "cover_excursions": [asdict(item) for item in self.excursions],
            "long_reversals": [asdict(item) for item in self.reversals],
            "same_point_measure_batches": self.same_point_measure_batches,
            "actions": [asdict(item) for item in self.actions],
        }

    def write(self, path: str | Path) -> dict[str, object]:
        report = self.report()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
