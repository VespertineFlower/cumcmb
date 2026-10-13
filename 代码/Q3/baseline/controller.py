from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from common.active_sensing import ActiveSensor, q2_analytic_candidates, rotated_step
from common.communication import CommunicationError, SimulatorClient
from common.diagnostics import RuntimeDiagnostics
from common.geometry import distance, polygon_diameter, project_to_convex_polygon
from common.localization import BearingLocalizer, InconsistentObservations
from common.models import Point, TargetState, TargetStatus
from common.plotting import RunPlotter
from common.q3_path_planner import (
    CoverageManager,
    RollingPlanner,
    Task,
    choose_route_aware_candidate,
    seven_point_cover,
)


class RobotController:
    def __init__(
        self,
        client: SimulatorClient,
        mode: str = "q3",
        safety_margin_s: float = 15.0,
        max_clear_failures_per_channel: int = 3,
    ) -> None:
        if mode != "q3":
            raise ValueError("Q3 controller only supports q3")
        self.client = client
        self.mode = mode
        self.safety_margin_s = safety_margin_s
        self.localizer = BearingLocalizer()
        self.active_sensor = ActiveSensor()
        self.targets: dict[int, TargetState] = {}
        self.removed: set[int] = set()
        self.deadline = float("inf")
        self.logger = logging.getLogger("robot_dog.controller")
        self.q3_planner: RollingPlanner | None = None
        self.q3_coverage: CoverageManager | None = None
        self.diag_ = RuntimeDiagnostics()
        self.diagnostics = self.diag_
        self._bad_meas_pts: dict[int, list[Point]] = defaultdict(list)
        self._bad_clear_pts: dict[int, list[Point]] = defaultdict(list)
        self._clear_fails: dict[int, int] = defaultdict(int)
        self._needs_new_obs: set[int] = set()
        self._held_exclusions: dict[int, list[Point]] = defaultdict(list)
        self.max_clear_failures_per_channel = max_clear_failures_per_channel

    def run(self) -> dict[str, float | int | str]:
        enter = self.client.enter()
        remaining = float(enter["remaining_real_duration_s"])
        self.deadline = time.monotonic() + max(0.0, remaining - self.safety_margin_s)
        try:
            self._run_q3_rolling()
        finally:
            try:
                self.client.exit()
            except CommunicationError:
                self.logger.exception("could not send /exit (the simulator may already be closed)")
        report_path: str | None = None
        plot_path: str | None = None
        points_path: str | None = None
        if hasattr(self.client, "log_path"):
            raw_log = Path(self.client.log_path)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            report_file = raw_log.with_name(f"{raw_log.stem}_metrics_{stamp}.json")
            self.diag_.write(report_file)
            report_path = os.path.relpath(report_file)
            svg_file = raw_log.with_name(f"{raw_log.stem}_path_{stamp}.svg")
            svg_output, csv_output = RunPlotter().write(self.diag_.actions, svg_file)
            plot_path = os.path.relpath(svg_output)
            points_path = os.path.relpath(csv_output)
        summary = self.diag_.report()["summary"]
        return {
            "mode": self.mode,
            "cleared": len(self.removed),
            "virtual_time_s": self.client.virtual_time_s,
            "diagnostics": summary,
            "diagnostics_file": report_path,
            "metrics_csv": os.path.relpath(self.client.summary_csv_path) if hasattr(self.client, "summary_csv_path") else None,
            "path_plot": plot_path,
            "detection_points_file": points_path,
        }

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
                        "Q3 task queue is empty before every channel is cleared or absent"
                    )
            self._check_deadline()
            planner.remove(task.task_id)
            if task.kind != "cover" and any(item.kind == "cover" for item in planner.tasks.values()):
                self.diag_.begin_non_cover_excursion(self.client.position)
            if task.kind == "cover":
                self._scan_q3_search_point(task.point, fallback_id=task.task_id, force_fallback=True)
                self.diag_.end_at_cover(task.point)
            elif task.kind == "localize":
                response = self._measure(task.point, int(task.channel), role="localize")
                self._consume_measure(int(task.channel), task.point, response, defer_near_clear=True)
                self._piggyback_q3_search(task.point)
                if response["measure_result"] == "no_signal":
                    # This should not occur at the first Q2 robust point for an omnidirectional
                    # source. Keep the target alive so later coverage observations can repair it.
                    self.logger.warning("channel %s: no signal at planned Q3 localization point", task.channel)
            elif task.kind == "clear":
                self._execute_planned_clear(task)
                self._piggyback_q3_search(task.point)
            else:
                raise RuntimeError(f"unknown Q3 task kind: {task.kind}")

        if not self.q3_coverage.all_channels_certified():
            raise RuntimeError("Q3 attempted to exit with uncertified channels")

    def _restore_missing_q3_tasks(self) -> None:
        """Rebuild mandatory work if feedback handling ever leaves the queue empty."""
        planner = self.q3_planner
        coverage = self.q3_coverage
        if planner is None or coverage is None:
            return

        cover_tasks = {task.task_id: task for task in seven_point_cover()}
        for state in coverage.channels.values():
            if not state.needs_search:
                continue
            for task_id, task in cover_tasks.items():
                if task_id not in state.fallback_scanned:
                    planner.upsert(task)

        for channel, target in self.targets.items():
            if channel not in self.removed and not coverage.channels[channel].certified_complete:
                self._sync_q3_target_task(target)

    def _execute_planned_clear(self, task: Task) -> None:
        channel = int(task.channel)
        response = self._clear(task.point, channel, role="clear")
        target = self.targets[channel]
        if response["clear_result"] == "success":
            target.status = TargetStatus.REMOVED
            self.removed.add(channel)
            if self.q3_planner:
                self.q3_planner.remove_channel(channel)
            if self.q3_coverage:
                self.q3_coverage.channels[channel].mark_cleared()
        else:
            self._clear_fails[channel] += 1
            if all(
                distance(task.point, old) > 1e-7
                for old in self._bad_clear_pts[channel]
            ):
                self._bad_clear_pts[channel].append(task.point)
            if self._clear_fails[channel] >= self.max_clear_failures_per_channel:
                raise RuntimeError(
                    f"channel {channel}: clear failed "
                    f"{self._clear_fails[channel]} times"
                )
            # The failed point supplies no new bearing.  Require a distinct
            # measurement before another clear attempt can be scheduled.
            self._needs_new_obs.add(channel)
            target.status = TargetStatus.LOCALIZING
            self._sync_q3_target_task(target)

    def _scan_q3_search_point(
        self,
        point: Point,
        fallback_id: str | None = None,
        force_fallback: bool = False,
        selected_channels: list[int] | None = None,
    ) -> None:
        coverage = self.q3_coverage
        if coverage is None:
            return
        if force_fallback:
            forced = [
                channel
                for channel, state in coverage.channels.items()
                if state.needs_search and fallback_id not in state.fallback_scanned
            ]
        else:
            forced = []
        channels = selected_channels or coverage.channels_worth_scanning(
            point, force_channels=forced, current_channel=self.client.current_channel
        )
        for channel in channels:
            self._check_deadline()
            role = "cover" if force_fallback else "piggyback"
            response = self._measure(point, channel, role=role)
            result = str(response["measure_result"])
            coverage.channels[channel].record(point, result, fallback_id if force_fallback else None)
            self._consume_measure(channel, point, response, defer_near_clear=True)

    def _piggyback_q3_search(self, point: Point) -> None:
        coverage = self.q3_coverage
        if coverage is None:
            return
        channels = coverage.channels_worth_scanning(
            point,
            min_new_fraction=0.08,
            current_channel=self.client.current_channel,
        )
        if channels:
            self.diag_.record_piggyback(len(channels))
            self._scan_q3_search_point(point, selected_channels=channels)

    def _consume_measure(
        self,
        channel: int,
        point: Point,
        response: dict[str, object],
        defer_near_clear: bool = False,
        advances_guarantee: bool = True,
    ) -> None:
        result = str(response["measure_result"])
        target = self.targets.get(channel)
        if result == "no_signal":
            if target is not None:
                target.last_measure_result = result
                try:
                    self.localizer.add_no_signal(target, point)
                except InconsistentObservations:
                    # A planned robust point returning no signal contradicts
                    # the ideal model.  Preserve the previous safe hull.
                    self.logger.exception(
                        "channel %d: inconsistent no-signal exclusion", channel
                    )
                if defer_near_clear and target.observations and self.q3_planner:
                    # Defer regeneration until another coverage observation changes the state;
                    # regenerating the same deterministic point here would loop forever.
                    cover_left = any(t.kind == "cover" for t in self.q3_planner.tasks.values())
                    if not cover_left:
                        if len(target.guarantee_observations) == 1:
                            first = target.guarantee_observations[0]
                            alternatives = q2_analytic_candidates(
                                first.position,
                                first.bearing_deg,
                                error_deg=self.active_sensor.error_deg,
                            )
                            fallback = max(
                                alternatives,
                                key=lambda p: ((p.x - self.client.position.x) ** 2 + (p.y - self.client.position.y) ** 2),
                            )
                        else:
                            fallback = self.active_sensor.choose_next(target, self.client.position)
                        self.q3_planner.upsert(
                            Task(
                                f"localize-{channel}", fallback, "localize", channel, 6.0,
                                priority=0.5,
                            )
                        )
            return
        if target is None:
            target = self.localizer.new_target(channel)
            if self.q3_coverage is not None:
                # Reuse every no-signal exclusion collected before discovery.
                target.exclusion_centers.extend(
                    self.q3_coverage.channels[channel].no_signal_points
                )
            target.status = TargetStatus.DETECTED
            self.targets[channel] = target
        target.last_measure_result = result
        if result == "near":
            self._needs_new_obs.discard(channel)
            if defer_near_clear and self.q3_planner:
                target.status = TargetStatus.CLEARABLE
                self.q3_planner.remove(f"localize-{channel}")
                self.q3_planner.upsert(
                    Task(f"clear-{channel}", point, "clear", channel, 5.0, priority=4.0)
                )
                return
            clear = self._clear(point, channel, role="clear")
            if clear["clear_result"] == "success":
                target.status = TargetStatus.REMOVED
                self.removed.add(channel)
            return
        is_new = not target.observations
        accepted_dir = False
        try:
            self.localizer.add_direction(target, point, float(response["svd_deg"]))
        except InconsistentObservations:
            self.logger.exception("inconsistent direction for channel %d", channel)
            if all(distance(point, old) > 1e-7 for old in self._bad_meas_pts[channel]):
                self._bad_meas_pts[channel].append(point)
            # A first bearing can only conflict with inherited no-signal
            # exclusions or numerical approximations.  Quarantine those
            # exclusions and retry from the arena rather than leaving a target
            # with no observations.
            if is_new and target.exclusion_centers:
                inherited = list(target.exclusion_centers)
                target.exclusion_centers.clear()
                try:
                    self.localizer.add_direction(
                        target, point, float(response["svd_deg"])
                    )
                except InconsistentObservations:
                    target.exclusion_centers.extend(inherited)
                else:
                    self._held_exclusions[channel].extend(inherited)
                    self.logger.warning(
                        "channel %d: quarantined %d inherited no-signal exclusions",
                        channel,
                        len(inherited),
                    )
                    accepted_dir = True
            if is_new and not accepted_dir:
                self.targets.pop(channel, None)
                raise RuntimeError(
                    f"channel {channel}: first direction is inconsistent even "
                    "after removing inherited exclusions"
                )
        else:
            accepted_dir = True

        if accepted_dir:
            observation = target.observations[-1]
            # An arbitrary first bearing is a valid S1 even when discovered at
            # an opportunity stop.  Later opportunity bearings update geometry
            # but cannot silently advance the certified construction.
            # The Q2 certificate contains at most four deliberately constructed
            # bearings.  Recovery/task observations taken after that point still
            # refine the full feasible set, but must not silently extend the
            # four-observation certificate.
            if (
                not target.guarantee_observations
                or advances_guarantee
                and len(target.guarantee_observations) < 4
            ):
                target.guarantee_observations.append(observation)
            self._needs_new_obs.discard(channel)
        if defer_near_clear:
            self._sync_q3_target_task(target)

    def _sync_q3_target_task(self, target: TargetState) -> None:
        planner = self.q3_planner
        if planner is None or target.channel in self.removed:
            return
        planner.remove(f"localize-{target.channel}")
        planner.remove(f"clear-{target.channel}")
        if (
            self.localizer.is_clearable(target)
            and target.channel not in self._needs_new_obs
        ):
            center = self.localizer.covering_circle(target).center
            target.status = TargetStatus.CLEARABLE
            planner.upsert(
                Task(f"clear-{target.channel}", center, "clear", target.channel, 5.0, priority=3.0)
            )
            return
        next_point = self._q3_next_detection_point(target)
        planner.upsert(
            Task(f"localize-{target.channel}", next_point, "localize", target.channel, 6.0, priority=1.0)
        )

    def _q3_next_detection_point(self, target: TargetState) -> Point:
        planner = self.q3_planner
        assert planner is not None
        anchors = planner.continuation_anchors(self.client.position, f"localize-{target.channel}")
        guarantee = target.guarantee_observations
        if len(guarantee) == 1:
            first = guarantee[0]
            candidates = self._unblocked_candidates(
                target.channel,
                q2_analytic_candidates(
                    first.position,
                    first.bearing_deg,
                    error_deg=self.active_sensor.error_deg,
                ),
            )
            return choose_route_aware_candidate(self.client.position, candidates, anchors)
        if len(guarantee) == 2:
            desired = min(anchors, key=lambda p: self._point_distance_to_polygon(p, target.polygon), default=None)
            if desired is None:
                desired = self.localizer.covering_circle(target).center
            projected = project_to_convex_polygon(desired, target.polygon)
            choices = self._unblocked_candidates(
                target.channel,
                (projected, self.localizer.covering_circle(target).center, *target.polygon),
            )
            return min(choices, key=lambda p: distance(self.client.position, p))
        if len(guarantee) == 3:
            latest = guarantee[-1]
            length = max(40.0, min(1000.0, polygon_diameter(target.polygon)[0]))
            offset_deg = 60.0 - self.active_sensor.error_deg
            candidates = (
                rotated_step(latest.position, latest.bearing_deg, offset_deg, length),
                rotated_step(latest.position, latest.bearing_deg, -offset_deg, length),
            )
            candidates = self._unblocked_candidates(target.channel, candidates)
            return choose_route_aware_candidate(self.client.position, candidates, anchors)
        proposed = self.active_sensor.choose_next(target, self.client.position)
        choices = self._unblocked_candidates(
            target.channel,
            (proposed, self.localizer.covering_circle(target).center, *target.polygon),
        )
        return min(choices, key=lambda p: distance(self.client.position, p))

    def _unblocked_candidates(self, channel: int, candidates) -> tuple[Point, ...]:
        blocked = (
            self._bad_meas_pts[channel]
            + self._bad_clear_pts[channel]
        )
        usable = tuple(
            point
            for point in candidates
            if all(distance(point, old) > 1e-6 for old in blocked)
        )
        if not usable:
            raise RuntimeError(
                f"channel {channel}: no unused recovery point after inconsistent observations"
            )
        return usable

    @staticmethod
    def _point_distance_to_polygon(point: Point, polygon: list[Point]) -> float:
        projected = project_to_convex_polygon(point, polygon)
        return ((point.x - projected.x) ** 2 + (point.y - projected.y) ** 2) ** 0.5

    def _clear_ready_targets(self) -> None:
        for channel, target in list(self.targets.items()):
            if channel not in self.removed and target.status == TargetStatus.CLEARABLE:
                self._attempt_clear(target)

    def _attempt_clear(self, target: TargetState) -> bool:
        center = self.localizer.covering_circle(target).center
        response = self._clear(center, target.channel, role="clear")
        if response["clear_result"] == "success":
            target.status = TargetStatus.REMOVED
            self.removed.add(target.channel)
            return True
        # Conservative polygon approximation can very rarely put the center just outside 20 m.
        target.status = TargetStatus.LOCALIZING
        return False

    def _check_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("real-time safety deadline reached")

    def _measure(self, point: Point, channel: int, role: str = "measure") -> dict[str, object]:
        start = self.client.position
        response = self.client.measure(point, channel)
        movement = distance(start, point)
        self.diag_.record_move(start, point)
        self.diag_.record_measure(
            point,
            channel,
            str(response["measure_result"]),
            float(response["virtual_time_s"]),
        )
        self.diag_.record_action(
            "measure", role, point, channel, str(response["measure_result"]),
            float(response["virtual_time_s"]), movement,
        )
        return response

    def _clear(self, point: Point, channel: int, role: str = "clear") -> dict[str, object]:
        start = self.client.position
        response = self.client.clear(point, channel)
        movement = distance(start, point)
        self.diag_.record_move(start, point)
        self.diag_.record_clear(
            channel,
            response["clear_result"] == "success",
            float(response["virtual_time_s"]),
        )
        self.diag_.record_action(
            "clear", role, point, channel, str(response["clear_result"]),
            float(response["virtual_time_s"]), movement,
        )
        return response
