"""Problem 4 official runner for the modular 25-point solver.

The mathematical geometry, channel state, route selection, diagnostics, and
offline verification live in ``common.planner``. This module contains
only the official HTTP adapter, end-to-end controller, and command-line entry
point. Public model names are re-exported for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from math import atan2, degrees
from pathlib import Path
from typing import Any, Sequence

import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.geometry import (
    ARENA_RADIUS,
    BEARING_ERROR_DEG,
    CLEAR_RADIUS,
    EPS,
    GOLDEN_ANGLE_DEG,
    GUARANTEED_RANGE,
    MAX_RECEPTION_RANGE,
    MESH_EDGE_TOLERANCE,
    STRICT_SEARCH_ROUTE,
    TRIANGLE25_FACES,
    TRIANGLE25_LATTICE_INDICES,
    TRIANGLE25_POINTS,
    CellEvidence,
    CellStatus,
    Point,
    TriangleCell,
    build_triangle_cells,
    clip_bearing_wedge,
    clip_halfplane,
    clip_reception_disk,
    cross,
    distance,
    minimum_enclosing_circle,
    regular_disk_polygon,
)
from common.planner import (
    ChannelState,
    ChannelStatus,
    DirectionalSource,
    Q4ActionRecord,
    Q4Diagnostics,
    Q4Triangle25SurroundPlanner,
    Stop,
    demo,
    verify_surround_property,
)

__all__ = [
    "ARENA_RADIUS",
    "BEARING_ERROR_DEG",
    "CLEAR_RADIUS",
    "EPS",
    "GOLDEN_ANGLE_DEG",
    "GUARANTEED_RANGE",
    "MAX_RECEPTION_RANGE",
    "MESH_EDGE_TOLERANCE",
    "STRICT_SEARCH_ROUTE",
    "TRIANGLE25_FACES",
    "TRIANGLE25_LATTICE_INDICES",
    "TRIANGLE25_POINTS",
    "CellEvidence",
    "CellStatus",
    "ChannelState",
    "ChannelStatus",
    "CommunicationError",
    "DirectionalSource",
    "OfficialQ4Controller",
    "OfficialSimulatorClient",
    "Point",
    "Q4ActionRecord",
    "Q4Diagnostics",
    "Q4Triangle25SurroundPlanner",
    "Stop",
    "TriangleCell",
    "build_argument_parser",
    "build_triangle_cells",
    "clip_bearing_wedge",
    "clip_halfplane",
    "clip_reception_disk",
    "cross",
    "demo",
    "distance",
    "minimum_enclosing_circle",
    "official_main",
    "regular_disk_polygon",
    "verify_surround_property",
]


# ---------------------------------------------------------------------------
# Official simulator HTTP client
# ---------------------------------------------------------------------------

from shared.official_client import CommunicationError, RetryPolicy
from shared.official_client import SimulatorClient as _SharedSimulatorClient


class OfficialSimulatorClient(_SharedSimulatorClient):
    """Q4 defaults over the shared official four-endpoint client."""

    def __init__(
        self,
        robot_id: str,
        base_url: str = "http://127.0.0.1:2026",
        arena_id: str = "default",
        timeout_s: float = 5.0,
        attempts: int = 5,
        log_path: str | Path = "logs/q4_triangle25.jsonl",
    ) -> None:
        log_file = Path(log_path)
        if log_file.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = log_file.with_name(
                f"{log_file.stem}_{stamp}{log_file.suffix}"
            )
        super().__init__(
            robot_id=robot_id,
            base_url=base_url,
            arena_id=arena_id,
            retry=RetryPolicy(attempts=attempts, timeout_s=timeout_s),
            log_path=log_file,
            summary_stem=Path(log_path).stem,
        )
        self.position = Point(0.0, 0.0)
        self.timeout_s = timeout_s
        self.attempts = attempts


# ---------------------------------------------------------------------------
# End-to-end controller
# ---------------------------------------------------------------------------


class OfficialQ4Controller:
    """Official-interface controller around the standalone 25-point planner."""

    def __init__(
        self,
        client: OfficialSimulatorClient,
        safety_margin_s: float = 15.0,
        max_localization_attempts: int = 40,
        max_lazy_opportunities_per_stop: int = 4,
        write_artifacts: bool = True,
        total_miss_risk: float = 0.0,
        miss_probability_per_probe: float = 0.5,
        adaptive_route: bool = True,
    ) -> None:
        self.client = client
        self.planner = Q4Triangle25SurroundPlanner(
            total_miss_risk=total_miss_risk,
            miss_probability_per_probe=miss_probability_per_probe,
            adaptive_route=adaptive_route,
        )
        self.safety_margin_s = safety_margin_s
        self.loc_attempts = max_localization_attempts
        self.lazy_cap = max_lazy_opportunities_per_stop
        self.write_artifacts = write_artifacts
        self.deadline = float("inf")
        self.measurements = 0
        self.clear_attempts = 0
        self.travel_m = 0.0
        self.logger = logging.getLogger("q4_triangle25")
        self.diag_ = Q4Diagnostics()
        self.diagnostics = self.diag_
        self.artifact_paths: dict[str, str] = {}

    @staticmethod
    def _line_separation(a: float, b: float) -> float:
        delta = abs((a - b) % 180.0)
        return min(delta, 180.0 - delta)

    def _lazy_channels_at_stop(self, point: Point) -> list[int]:
        """Select active channels worth measuring at an already required stop."""
        ranked: list[tuple[float, int]] = []
        for channel, state in self.planner.channels.items():
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
            geometry = (
                max(
                    (
                        self._line_separation(pred_line, old)
                        for old in state.bearing_lines_deg
                    ),
                    default=90.0,
                )
                / 90.0
            )
            proximity = 1.0 / (1.0 + center_dist / GUARANTEED_RANGE)
            ranked.append((2.0 * geometry + proximity, channel))
        ranked.sort(reverse=True)
        return [
            channel for _, channel in ranked[: self.lazy_cap]
        ]

    def _take_lazy_observations(self, point: Point) -> None:
        """Gather information at a triangular-route stop with zero extra travel."""
        for channel in self._lazy_channels_at_stop(point):
            if self.planner.channels[channel].status == ChannelStatus.LOCALIZING:
                self._measure(point, channel, role="lazy")

    def _nearest_active_channel(self) -> int | None:
        active = [
            (distance(self.client.position, state.mec_center), channel)
            for channel, state in self.planner.channels.items()
            if state.status in {ChannelStatus.LOCALIZING, ChannelStatus.CLEARABLE}
        ]
        return min(active, default=(0.0, None))[1]

    def _check_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("real-time safety deadline reached")

    def _measure(
        self, point: Point, channel: int, role: str = "localize"
    ) -> dict[str, Any]:
        self._check_deadline()
        movement = distance(self.client.position, point)
        self.travel_m += movement
        response = self.client.measure(point, channel)
        self.measurements += 1
        result = str(response["measure_result"])
        bearing = float(response["svd_deg"]) if result == "direction" else None
        self.planner.record(channel, point, result, bearing)
        self.diag_.record(
            "measure",
            role,
            point,
            channel,
            result,
            float(response["virtual_time_s"]),
            movement,
        )
        return response

    def _try_clear(self, channel: int) -> bool:
        state = self.planner.channels[channel]
        if state.status != ChannelStatus.CLEARABLE:
            return False
        self._check_deadline()
        movement = distance(self.client.position, state.mec_center)
        self.travel_m += movement
        response = self.client.clear(state.mec_center, channel)
        self.clear_attempts += 1
        self.diag_.record(
            "clear",
            "clear",
            state.mec_center,
            channel,
            str(response["clear_result"]),
            float(response["virtual_time_s"]),
            movement,
        )
        if response["clear_result"] == "success":
            state.mark_cleared()
            return True
        state.status = ChannelStatus.LOCALIZING
        return False

    def _localize_channel(self, channel: int) -> None:
        state = self.planner.channels[channel]
        for _ in range(self.loc_attempts):
            if state.status == ChannelStatus.CLEARABLE and self._try_clear(channel):
                return
            point = self.planner.localization_probe(channel)
            self._measure(point, channel, role="localize")
        if state.status == ChannelStatus.CLEARABLE and self._try_clear(channel):
            return
        raise RuntimeError(
            f"channel {channel}: not clearable after {self.loc_attempts} probes"
        )

    def _run_strict_search_phase(self) -> None:
        """Visit the 25-point certificate and use mandatory stops for Lazy probes."""
        while True:
            stop = self.planner.next_stop(
                self.client.position, current_channel=self.client.current_channel
            )
            if stop is None:
                return
            for channel in stop.channels:
                if self.planner.channels[channel].status == ChannelStatus.UNKNOWN:
                    self._measure(stop.point, channel, role="search")
            self._take_lazy_observations(stop.point)

    def _run_localization_phase(self) -> None:
        """Finish discovered channels in nearest-next order."""
        while True:
            ch = self._nearest_active_channel()
            if ch is None:
                return
            self._localize_channel(ch)

    def _unfinished_channels(self) -> dict[int, str]:
        return {
            channel: state.status.value
            for channel, state in self.planner.channels.items()
            if state.status not in {ChannelStatus.ABSENT, ChannelStatus.CLEARED}
        }

    def run(self) -> dict[str, Any]:
        entered = self.client.enter()
        remaining = float(entered["remaining_real_duration_s"])
        self.deadline = time.monotonic() + max(0.0, remaining - self.safety_margin_s)
        error: BaseException | None = None
        try:
            self._run_strict_search_phase()
            self._run_localization_phase()

            if not self.planner.complete:
                raise RuntimeError(
                    f"unfinished channels: {self._unfinished_channels()}"
                )
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                self.client.exit()
            except CommunicationError:
                if error is None:
                    raise
                self.logger.exception(
                    "could not send /exit while handling another failure"
                )
            if self.write_artifacts:
                try:
                    self.artifact_paths = self.diag_.write(
                        self.planner, self.client.log_path
                    )
                except Exception:
                    self.logger.exception("could not write Q4 diagnostic artifacts")

        cleared = sum(
            state.status == ChannelStatus.CLEARED
            for state in self.planner.channels.values()
        )
        absent = sum(
            state.status == ChannelStatus.ABSENT
            for state in self.planner.channels.values()
        )
        return {
            "mode": (
                "q4_triangle25_surround_strict"
                if self.planner.probabilistic_negative_limit is None
                else "q4_triangle25_surround_probabilistic"
            ),
            "cleared": cleared,
            "absent": absent,
            "measurements": self.measurements,
            "clear_attempts": self.clear_attempts,
            "travel_m": round(self.travel_m, 3),
            "virtual_time_s": self.client.virtual_time_s,
            "total_miss_risk_target": self.planner.total_miss_risk,
            "per_channel_miss_risk_target": self.planner.per_channel_miss_risk,
            "negative_limit": self.planner.probabilistic_negative_limit,
            "risk_model": self.planner.risk_model,
            "search_route_strategy": self.planner.search_route_strategy,
            "log": os.path.relpath(self.client.log_path),
            "metrics_csv": os.path.relpath(self.client.summary_csv_path) if hasattr(self.client, "summary_csv_path") else None,
        } | self.artifact_paths


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without starting the simulator run."""
    parser = argparse.ArgumentParser(
        description="Problem 4 strict 25-point triangular-surround client"
    )
    parser.add_argument("--robot-id")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--arena-id", default="default")
    parser.add_argument("--log", type=Path, default=Path("logs/q4_triangle25.jsonl"))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--total-miss-risk",
        type=float,
        default=0.0,
        help="model-based total miss-risk target; use 0 for strict deterministic mode",
    )
    parser.add_argument(
        "--miss-probability-per-probe",
        type=float,
        default=0.5,
        help="assumed independent miss probability of one distinct search probe",
    )
    parser.add_argument(
        "--fixed-route",
        action="store_true",
        help="disable Lazy-information route adaptation and use the fixed 25-point optimum",
    )
    parser.add_argument("--demo", action="store_true")
    return parser


def official_main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.demo:
        demo()
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
    result = OfficialQ4Controller(
        client,
        total_miss_risk=args.total_miss_risk,
        miss_probability_per_probe=args.miss_probability_per_probe,
        adaptive_route=not args.fixed_route,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(official_main())
