"""Core data structures for the Israel Hiking Transit Planner."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import LineString


@dataclass
class TrailAccessPoint:
    """A bus stop near a trail entry point."""
    stop_id: str
    stop_name: str
    walk_distance_m: float
    trail_entry_lat: float
    trail_entry_lon: float
    trail_km_from_start: float


@dataclass
class Trail:
    """A hiking trail with geometry and metadata."""
    id: str
    name: str
    source: str  # "osm"
    geometry: LineString
    distance_km: float
    elevation_gain_m: float
    difficulty: str
    colors: list[str]  # ITC trail color markings
    is_loop: bool
    access_points: list[TrailAccessPoint] = field(default_factory=list)
    # ── v0.2 season & elevation ──
    recommended_seasons: list[str] = field(default_factory=list)
    season_warnings: list[str] = field(default_factory=list)
    elevation_loss_m: float = 0.0
    max_elevation_m: float = 0.0
    min_elevation_m: float = 0.0
    elevation_profile: list[float] = field(default_factory=list)


@dataclass
class BusLeg:
    """One leg of a transit journey."""
    line: str
    operator: str
    from_stop_id: str
    from_stop_name: str
    to_stop_id: str
    to_stop_name: str
    departure: datetime.datetime
    arrival: datetime.datetime


@dataclass
class HikeSegment:
    """The hiking portion of a trip."""
    trail_name: str
    entry_stop_name: str
    walk_to_trail_m: float
    hike_start: datetime.datetime
    hike_end: datetime.datetime
    hiking_hours: float
    estimated_distance_km: float
    is_loop: bool
    colors: list[str]
    # ── v0.2 elevation & through-hike ──
    elevation_gain_m: float = 0.0
    elevation_loss_m: float = 0.0
    is_through_hike: bool = False
    exit_stop_name: Optional[str] = None
    walk_from_trail_m: float = 0.0


@dataclass
class HikePlan:
    """A complete plan: transit out + hike + transit back."""
    trail: Trail
    access_point: TrailAccessPoint
    outbound_legs: list[BusLeg]
    hike_segment: HikeSegment
    return_legs: list[BusLeg]
    departure_from_origin: datetime.datetime
    arrival_at_origin: datetime.datetime
    hiking_ratio: float  # hiking_time / total_trip_time
    deadline: datetime.datetime
    total_hours: float
    score: float = 0.0
    # ── v0.2 ──
    warnings: list[str] = field(default_factory=list)
    exit_access_point: Optional[TrailAccessPoint] = None


@dataclass
class HikeQuery:
    """User query parameters."""
    origin: str
    date: datetime.date
    max_transfers: int = 1
    safety_margin_hours: float = 2.0
    max_walk_to_trail_m: int = 1000
    min_hiking_hours: float = 1.0
    max_results: int = 20
    earliest_departure: Optional[datetime.time] = None  # default 06:00
    sort_by: str = "hiking_ratio"
    # ── v0.2 filters ──
    colors: Optional[list[str]] = None
    min_distance_km: Optional[float] = None
    max_distance_km: Optional[float] = None
    loop_only: bool = False
    linear_only: bool = False
    max_elevation_gain_m: Optional[float] = None
    difficulty: Optional[str] = None
