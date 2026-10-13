"""Problem 3 anytime controller with two explicitly separated observation roles.

Task observations preserve the Q2/Q3 guarantee chain.  Opportunistic observations
are allowed only at a point the robot must already visit, so they add no movement.
They are budgeted and filtered to prevent uncontrolled channel scanning.
"""

from __future__ import annotations

import math
from collections import defaultdict

from baseline.controller import RobotController
from common.geometry import distance, minimum_enclosing_circle
from common.models import Point, TargetStatus
from common.q3_path_planner import CoverageManager, RollingPlanner, seven_point_cover


class AnytimeRobotController(RobotController):
    """A platform-ready Q3 controller using task and opportunistic observations."""

    def __init__(
        self,
        client,
        safety_margin_s: float = 15.0,
        max_opportunistic_per_stop: int = 4,
        max_unknown_per_stop: int = 2,
        min_unknown_gain: float = 0.12,
        min_angular_information: float = 0.35,
        no_signal_cooldown_stops: int = 2,
    ) -> None:
        super().__init__(client, mode="q3", safety_margin_s=safety_margin_s)
        self.opp_cap = max_opportunistic_per_stop
        self.unk_cap = max_unknown_per_stop
        self.min_unknown_gain = min_unknown_gain
        self.min_angle_info = min_angular_information
        self.cooldown_stops = no_signal_cooldown_stops
        self._stop_index = 0
        self._opp_block_until: dict[int, int] = defaultdict(int)
        self._measured_points: dict[int, list[Point]] = defaultdict(list)

    def _run_q3_rolling(self) -> None:
        planner = RollingPlanner()
        self.q3_planner = planner
        self.q3_coverage = CoverageManager()
        for task in seven_point_cover():
            planner.upsert(task)

        while True:
            task = planner.next_task(self.client.position)
            if task is None:
                if self.q3_coverage.all_channels_certified():
                    break
                self._restore_missing_q3_tasks()
                task = planner.next_task(self.client.position)
                if task is None:
                    raise RuntimeError(
                        "Q3 anytime task queue is empty before every channel is cleared or absent"
                    )
            self._check_deadline()
            planner.remove(task.task_id)
            if task.kind != "cover" and any(item.kind == "cover" for item in planner.tasks.values()):
                self.diag_.begin_non_cover_excursion(self.client.position)

            if task.kind == "cover":
                self._scan_task_search_point(task.point, task.task_id)
                self.diag_.end_at_cover(task.point)
            elif task.kind == "localize":
                channel = int(task.channel)
                response = self._measure(task.point, channel, role="task_localize")
                self._remember_measurement(channel, task.point)
                self._consume_measure(channel, task.point, response, defer_near_clear=True)
                self._take_opportunistic_observations(task.point, exclude={channel})
            elif task.kind == "clear":
                self._execute_planned_clear(task)
                self._take_opportunistic_observations(task.point, exclude={int(task.channel)})
            else:
                raise RuntimeError(f"unknown Q3 anytime task kind: {task.kind}")

        if not self.q3_coverage.all_channels_certified():
            raise RuntimeError("Q3 anytime attempted to exit with uncertified channels")

    def _scan_task_search_point(self, point: Point, fallback_id: str) -> None:
        """Mandatory seven-point coverage; it is never charged to opportunity budget."""
        coverage = self.q3_coverage
        assert coverage is not None
        channels = [
            channel
            for channel, state in coverage.channels.items()
            if state.needs_search and fallback_id not in state.fallback_scanned
        ]
        if self.client.current_channel in channels:
            channels.remove(self.client.current_channel)
            channels.insert(0, self.client.current_channel)
        for channel in channels:
            self._check_deadline()
            response = self._measure(point, channel, role="task_cover")
            self._remember_measurement(channel, point)
            result = str(response["measure_result"])
            coverage.channels[channel].record(point, result, fallback_id)
            self._consume_measure(channel, point, response, defer_near_clear=True)

        # Active targets can still benefit from this already mandatory stop.
        self._take_opportunistic_observations(point)

    def _take_opportunistic_observations(self, point: Point, exclude: set[int] | None = None) -> None:
        """Use a zero-detour stop, subject to reception, information and count budgets."""
        self._stop_index += 1
        excluded = exclude or set()
        candidates: list[tuple[float, int, str]] = []

        # First preference: improve a target already being localized.  The MEC
        # containment test guarantees the whole feasible region is within 1000 m.
        for channel, target in self.targets.items():
            if channel in excluded or channel in self.removed:
                continue
            if target.status not in {TargetStatus.DETECTED, TargetStatus.LOCALIZING}:
                continue
            if self._seen_here(channel, point) or self._cooling_down(channel):
                continue
            mec = minimum_enclosing_circle(target.polygon)
            if distance(point, mec.center) + mec.radius > 1000.0 + 1e-7:
                continue
            score = self._angular_information(target, point, mec.center)
            if score >= self.min_angle_info:
                candidates.append((2.0 + score, channel, "active target"))

        # Second preference: discover an unknown channel only when this stop adds
        # substantial uncovered area.  Cap this class independently.
        coverage = self.q3_coverage
        if coverage is not None:
            unknown: list[tuple[float, int, str]] = []
            for channel, state in coverage.channels.items():
                if channel in excluded or not state.needs_search:
                    continue
                if self._seen_here(channel, point) or self._cooling_down(channel):
                    continue
                new_count, uncovered_n = coverage.marginal_witness_count(channel, point)
                gain = new_count / uncovered_n if uncovered_n else 0.0
                if gain >= self.min_unknown_gain:
                    unknown.append((gain, channel, "coverage gain"))
            unknown.sort(reverse=True)
            candidates.extend(unknown[: self.unk_cap])

        candidates.sort(reverse=True)
        selected = candidates[: self.opp_cap]
        if not selected:
            return
        self.diag_.record_piggyback(len(selected))
        for _, channel, _ in selected:
            self._check_deadline()
            response = self._measure(point, channel, role="opportunistic_observation")
            self._remember_measurement(channel, point)
            result = str(response["measure_result"])
            if result == "no_signal":
                self._opp_block_until[channel] = (
                    self._stop_index + self.cooldown_stops
                )
            if coverage is not None and coverage.channels[channel].needs_search:
                # Opportunity samples help planning but do not count as one of the
                # seven mandatory absence certificates.
                coverage.channels[channel].record(point, result, None)
            self._consume_measure(
                channel,
                point,
                response,
                defer_near_clear=True,
                advances_guarantee=False,
            )

    def _remember_measurement(self, channel: int, point: Point) -> None:
        if not self._seen_here(channel, point):
            self._measured_points[channel].append(point)

    def _seen_here(self, channel: int, point: Point) -> bool:
        return any(distance(point, old) <= 1e-6 for old in self._measured_points[channel])

    def _cooling_down(self, channel: int) -> bool:
        return self._stop_index < self._opp_block_until[channel]

    @staticmethod
    def _angular_information(target, point: Point, center: Point) -> float:
        if not target.observations:
            return 1.0
        predicted = math.degrees(math.atan2(center.y - point.y, center.x - point.x)) % 180.0
        best = 0.0
        for observation in target.observations:
            old = observation.bearing_deg % 180.0
            delta = abs(predicted - old)
            separation = min(delta, 180.0 - delta)
            best = max(best, separation)
        return min(1.0, best / 90.0)
