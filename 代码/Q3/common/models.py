from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Observation:
    position: Point
    bearing_deg: float


class TargetStatus(str, Enum):
    UNKNOWN = "unknown"
    DETECTED = "detected"
    LOCALIZING = "localizing"
    CLEARABLE = "clearable"
    REMOVED = "removed"


@dataclass(slots=True)
class TargetState:
    channel: int
    polygon: list[Point]
    # Convex constraints from the arena, received bearings and positive
    # reception disks.  polygon is a conservative convex hull after applying
    # the non-convex no-signal exclusions kept below.
    positive_polygon: list[Point] = field(default_factory=list)
    exclusion_centers: list[Point] = field(default_factory=list)
    exclusion_resolution_m: float = 25.0
    # Every valid bearing clips the feasible region.  Only this separate list
    # advances the certified Q2/Q3 observation chain.
    observations: list[Observation] = field(default_factory=list)
    guarantee_observations: list[Observation] = field(default_factory=list)
    status: TargetStatus = TargetStatus.UNKNOWN
    last_measure_result: str | None = None


@dataclass(frozen=True, slots=True)
class Circle:
    center: Point
    radius: float
