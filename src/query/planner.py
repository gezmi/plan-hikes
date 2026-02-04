"""Main planning logic — orchestrates transit routing, trail lookup, and scoring.

Given a HikeQuery, returns a ranked list of HikePlan objects.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import LineString

from src.config import (
    CITY_COORDINATES,
    DATA_DIR,
    GTFS_ZIP_PATH,
    MAX_TRAIL_DISTANCE_KM,
    MAX_WALK_TO_TRAIL_M,
    MIN_HIKING_HOURS,
    NAISMITH_CLIMB_FACTOR,
    NAISMITH_SPEED_KMH,
    STOP_SEARCH_RADIUS_M,
    THROUGH_HIKE_MAX_DISTANCE_KM,
    THROUGH_HIKE_MIN_DISTANCE_KM,
    WALK_SPEED_KMH,
)
from src.ingest.gtfs import (
    build_transit_db,
    download_gtfs,
    find_origin_stops,
    find_origin_stops_db,
    load_feed_for_date,
)
from src.ingest.osm_trails import fetch_hiking_trails
from src.ingest.shabbat import get_deadline
from src.index.spatial_join import build_trail_access_points
from src.models import HikePlan, HikeQuery, HikeSegment, Trail, TrailAccessPoint
from src.query.transit_router import TransitRouter, TransitRouterDB

TRAIL_INDEX_PATH = DATA_DIR / "processed" / "trail_index.json"

logger = logging.getLogger(__name__)


@dataclass
class PlannerContext:
    """Pre-loaded, origin-independent data for planning hikes."""
    feed: object  # _FilteredFeed or None (when using SQLite path)
    trails: list[Trail]
    deadline: datetime.datetime
    deadline_secs: int
    router: TransitRouter  # or TransitRouterDB
    db_path: Path | None = None  # set when using SQLite low-memory path


def load_trail_index(path: Path | None = None) -> list[Trail]:
    """Load pre-processed trails from the JSON index.

    Parameters
    ----------
    path : Path, optional
        Path to trail_index.json. Defaults to data/processed/trail_index.json.

    Returns
    -------
    list[Trail]
        Trail objects with access points and geometry reconstructed.
    """
    index_path = path or TRAIL_INDEX_PATH
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trails: list[Trail] = []
    for entry in data["trails"]:
        # Reconstruct geometry: index stores [[lat, lon], ...], Shapely wants [(lon, lat), ...]
        geom_coords = [(lon, lat) for lat, lon in entry["geometry"]]
        geometry = LineString(geom_coords) if len(geom_coords) >= 2 else LineString()

        access_points = [
            TrailAccessPoint(
                stop_id=ap["stop_id"],
                stop_name=ap["stop_name"],
                walk_distance_m=ap["walk_distance_m"],
                trail_entry_lat=ap["trail_entry_lat"],
                trail_entry_lon=ap["trail_entry_lon"],
                trail_km_from_start=ap["trail_km_from_start"],
            )
            for ap in entry["access_points"]
        ]

        trail = Trail(
            id=entry["id"],
            name=entry["name"],
            source=entry["source"],
            geometry=geometry,
            distance_km=entry["distance_km"],
            elevation_gain_m=entry["elevation_gain_m"],
            difficulty=entry["difficulty"],
            colors=entry["colors"],
            is_loop=entry["is_loop"],
            access_points=access_points,
            recommended_seasons=entry.get("recommended_seasons", []),
            season_warnings=entry.get("season_warnings", []),
            elevation_loss_m=entry.get("elevation_loss_m", 0.0),
            max_elevation_m=entry.get("max_elevation_m", 0.0),
            min_elevation_m=entry.get("min_elevation_m", 0.0),
            elevation_profile=entry.get("elevation_profile", []),
        )
        trails.append(trail)

    logger.info("Loaded %d trails from pre-processed index %s", len(trails), index_path)
    return trails


def prepare_data_from_index(
    query: HikeQuery,
    *,
    low_memory: bool = True,
) -> PlannerContext:
    """Like prepare_data, but loads trails from the pre-processed index.

    Skips the Overpass query, elevation enrichment, and spatial join.

    Parameters
    ----------
    query : HikeQuery
        The user query (used for date, filters, etc.).
    low_memory : bool
        If *True* (default), stream the GTFS zip into a SQLite database
        and use :class:`TransitRouterDB` (~20 MB RAM).
        If *False*, load the full GTFS feed into pandas/partridge
        (~1-2 GB RAM) for faster routing.
    """
    gtfs_path = download_gtfs()

    if low_memory:
        # ── Low-memory path: GTFS CSV → SQLite on disk ───────────────
        db_path = build_transit_db(gtfs_path, query.date)
        router = TransitRouterDB(db_path, query.date)
        feed = None
    else:
        # ── High-memory path: partridge → pandas DataFrames ──────────
        feed = load_feed_for_date(gtfs_path, query.date)
        router = TransitRouter(feed, query.date)
        db_path = None

    # ── Trails from pre-processed index ───────────────────────────────
    trails = load_trail_index()

    # Apply user-specified filters
    trails = _filter_trails(trails, query)
    logger.info("%d trails after user filters", len(trails))

    # ── Deadline ──────────────────────────────────────────────────────
    deadline = get_deadline(query.date, query.safety_margin_hours)
    deadline_secs = _datetime_to_seconds(deadline)
    logger.info("Deadline: %s", deadline.strftime("%H:%M"))

    return PlannerContext(
        feed=feed,
        trails=trails,
        deadline=deadline,
        deadline_secs=deadline_secs,
        router=router,
        db_path=db_path,
    )


def _resolve_origin(origin: str) -> tuple[float, float]:
    """Resolve an origin city name to (lat, lon) coordinates."""
    key = origin.strip().lower()
    if key not in CITY_COORDINATES:
        known = ", ".join(sorted(CITY_COORDINATES.keys()))
        raise ValueError(
            f"Unknown origin city '{origin}'. Known cities: {known}"
        )
    return CITY_COORDINATES[key]


def _time_to_seconds(t: datetime.time) -> int:
    """Convert a time object to seconds since midnight."""
    return t.hour * 3600 + t.minute * 60 + t.second


def _datetime_to_seconds(dt: datetime.datetime) -> int:
    """Convert a datetime to seconds since midnight of its date."""
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _estimate_hike_time_hours(
    distance_km: float, elevation_gain_m: float
) -> float:
    """Estimate hiking time using Naismith's rule."""
    return distance_km / NAISMITH_SPEED_KMH + elevation_gain_m / NAISMITH_CLIMB_FACTOR


def _walk_time_hours(distance_m: float) -> float:
    """Time to walk a given distance at WALK_SPEED_KMH."""
    return (distance_m / 1000.0) / WALK_SPEED_KMH


# Months considered rainy season for flash flood warnings
_RAINY_MONTHS = {11, 12, 1, 2, 3}


def _date_to_season(date: datetime.date) -> str:
    """Map a date to a season name."""
    month = date.month
    if month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "autumn"
    else:
        return "winter"


def _filter_trails(trails: list[Trail], query: HikeQuery) -> list[Trail]:
    """Apply user-specified filters to the trail list."""
    result = trails

    if query.colors:
        query_colors = {c.lower() for c in query.colors}
        result = [t for t in result if query_colors & set(c.lower() for c in t.colors)]

    if query.min_distance_km is not None:
        result = [t for t in result if t.distance_km >= query.min_distance_km]

    if query.max_distance_km is not None:
        result = [t for t in result if t.distance_km <= query.max_distance_km]

    if query.loop_only:
        result = [t for t in result if t.is_loop]

    if query.linear_only:
        result = [t for t in result if not t.is_loop]

    if query.max_elevation_gain_m is not None:
        result = [t for t in result if t.elevation_gain_m <= query.max_elevation_gain_m]

    if query.difficulty is not None:
        result = [t for t in result if t.difficulty.lower() == query.difficulty.lower()]

    return result


def prepare_data(query: HikeQuery) -> PlannerContext:
    """Load GTFS, fetch/enrich/filter trails, compute deadline — origin-independent.

    Returns a PlannerContext that can be reused across multiple origins.
    """
    # ── Load GTFS ─────────────────────────────────────────────────────
    gtfs_path = download_gtfs()
    feed = load_feed_for_date(gtfs_path, query.date)
    router = TransitRouter(feed, query.date)

    # ── Trails + spatial join ─────────────────────────────────────────
    trails = fetch_hiking_trails()
    logger.info("Loaded %d hiking trails from OSM", len(trails))

    # Enrich trails with SRTM elevation data (optional, degrades gracefully)
    try:
        from src.ingest.osm_trails import enrich_trails_with_elevation
        enrich_trails_with_elevation(trails)
    except Exception as e:
        logger.info("Elevation enrichment unavailable: %s", e)

    # Filter out mega-trails (Israel Trail, Golan Trail, etc.)
    trails = [t for t in trails if t.distance_km <= MAX_TRAIL_DISTANCE_KM]
    logger.info("%d trails after distance filter (<= %d km)", len(trails), MAX_TRAIL_DISTANCE_KM)

    # Apply user-specified filters (v0.2)
    trails = _filter_trails(trails, query)
    logger.info("%d trails after user filters", len(trails))

    trails = build_trail_access_points(
        trails, feed.stops, max_distance_m=query.max_walk_to_trail_m
    )
    logger.info("%d trails have bus-accessible entry points", len(trails))

    # ── Deadline ──────────────────────────────────────────────────────
    deadline = get_deadline(query.date, query.safety_margin_hours)
    deadline_secs = _datetime_to_seconds(deadline)
    logger.info("Deadline: %s", deadline.strftime("%H:%M"))

    return PlannerContext(
        feed=feed,
        trails=trails,
        deadline=deadline,
        deadline_secs=deadline_secs,
        router=router,
    )


def plan_hikes_for_origin(query: HikeQuery, ctx: PlannerContext) -> list[HikePlan]:
    """Plan hikes for a single origin using pre-loaded data.

    Parameters
    ----------
    query : HikeQuery
        Must have a valid ``origin`` field.
    ctx : PlannerContext
        Pre-loaded context from ``prepare_data()``.
    """
    lat, lon = _resolve_origin(query.origin)
    logger.info("Origin: %s → (%.4f, %.4f)", query.origin, lat, lon)

    if ctx.db_path is not None:
        origin_stop_ids = find_origin_stops_db(ctx.db_path, lat, lon, STOP_SEARCH_RADIUS_M)
    else:
        origin_stop_ids = find_origin_stops(ctx.feed, lat, lon, STOP_SEARCH_RADIUS_M)
    if not origin_stop_ids:
        logger.warning("No bus stops found within %dm of %s", STOP_SEARCH_RADIUS_M, query.origin)
        return []
    logger.info("Found %d origin stops near %s", len(origin_stop_ids), query.origin)

    earliest_dep = query.earliest_departure or datetime.time(6, 0)
    earliest_dep_secs = _time_to_seconds(earliest_dep)

    origin_stop_set = set(origin_stop_ids)
    plans: list[HikePlan] = []

    for trail in ctx.trails:
        trail_plans = _plan_single_trail(
            trail=trail,
            router=ctx.router,
            origin_stop_ids=origin_stop_ids,
            origin_stop_set=origin_stop_set,
            earliest_dep_secs=earliest_dep_secs,
            deadline=ctx.deadline,
            deadline_secs=ctx.deadline_secs,
            query=query,
        )
        plans.extend(trail_plans)

    plans.sort(key=lambda p: p.hiking_ratio, reverse=True)
    logger.info("Found %d viable hiking plans for %s", len(plans), query.origin)
    return plans[: query.max_results]


def plan_hikes(query: HikeQuery) -> list[HikePlan]:
    """Plan hikes for a single origin — backward-compatible entry point.

    Equivalent to ``prepare_data(query)`` + ``plan_hikes_for_origin(query, ctx)``.
    """
    ctx = prepare_data(query)
    return plan_hikes_for_origin(query, ctx)


def _plan_single_trail(
    trail: Trail,
    router: TransitRouter,
    origin_stop_ids: list[str],
    origin_stop_set: set[str],
    earliest_dep_secs: int,
    deadline: datetime.datetime,
    deadline_secs: int,
    query: HikeQuery,
) -> list[HikePlan]:
    """Try to build HikePlans for a single trail.

    Returns up to 2 plans: best out-and-back + best through-hike.
    """
    results: list[HikePlan] = []

    # ── Best out-and-back (or loop) plan ──
    best_oab: HikePlan | None = None
    for ap in trail.access_points:
        plan = _plan_access_point(
            trail=trail,
            ap=ap,
            router=router,
            origin_stop_ids=origin_stop_ids,
            origin_stop_set=origin_stop_set,
            earliest_dep_secs=earliest_dep_secs,
            deadline=deadline,
            deadline_secs=deadline_secs,
            min_hiking_hours=query.min_hiking_hours,
        )
        if plan is None:
            continue
        if best_oab is None or plan.hiking_ratio > best_oab.hiking_ratio:
            best_oab = plan

    if best_oab is not None:
        results.append(best_oab)

    # ── Through-hike plans (non-loop trails with 2+ access points) ──
    if not trail.is_loop and len(trail.access_points) >= 2:
        best_through: HikePlan | None = None
        aps = trail.access_points

        for i in range(len(aps)):
            for j in range(len(aps)):
                if i == j:
                    continue
                entry_ap, exit_ap = aps[i], aps[j]
                segment_km = abs(exit_ap.trail_km_from_start - entry_ap.trail_km_from_start)
                if segment_km < THROUGH_HIKE_MIN_DISTANCE_KM:
                    continue
                if segment_km > THROUGH_HIKE_MAX_DISTANCE_KM:
                    continue

                plan = _plan_through_hike(
                    trail=trail,
                    entry_ap=entry_ap,
                    exit_ap=exit_ap,
                    segment_km=segment_km,
                    router=router,
                    origin_stop_ids=origin_stop_ids,
                    origin_stop_set=origin_stop_set,
                    earliest_dep_secs=earliest_dep_secs,
                    deadline=deadline,
                    deadline_secs=deadline_secs,
                    min_hiking_hours=query.min_hiking_hours,
                )
                if plan is None:
                    continue
                if best_through is None or plan.hiking_ratio > best_through.hiking_ratio:
                    best_through = plan

        if best_through is not None:
            results.append(best_through)

    return results


def _plan_access_point(
    trail: Trail,
    ap: TrailAccessPoint,
    router: TransitRouter,
    origin_stop_ids: list[str],
    origin_stop_set: set[str],
    earliest_dep_secs: int,
    deadline: datetime.datetime,
    deadline_secs: int,
    min_hiking_hours: float,
) -> HikePlan | None:
    """Try to build a HikePlan for a single trail access point."""
    trail_stop_ids = [ap.stop_id]
    trail_stop_set = {ap.stop_id}

    # ── Return route (work backwards from deadline) ───────────────────
    return_legs = router.find_return(
        trail_stops=trail_stop_ids,
        origin_stops=origin_stop_set,
        deadline_secs=deadline_secs,
    )
    if return_legs is None:
        return None

    # The hiker must finish walking back to the stop before the return departure
    return_departure = return_legs[0].departure
    return_dep_secs = _datetime_to_seconds(return_departure)
    walk_back_secs = _walk_time_hours(ap.walk_distance_m) * 3600
    hike_end_secs = return_dep_secs - walk_back_secs
    if hike_end_secs <= earliest_dep_secs:
        return None

    # ── Outbound route (work forwards from morning) ───────────────────
    outbound_legs = router.find_outbound(
        origin_stops=origin_stop_ids,
        dest_stops=trail_stop_set,
        earliest_departure_secs=earliest_dep_secs,
    )
    if outbound_legs is None:
        return None

    # The hiker starts walking to trail after arriving at the stop
    outbound_arrival = outbound_legs[-1].arrival
    outbound_arr_secs = _datetime_to_seconds(outbound_arrival)
    walk_to_secs = _walk_time_hours(ap.walk_distance_m) * 3600
    hike_start_secs = outbound_arr_secs + walk_to_secs

    if hike_start_secs >= hike_end_secs:
        return None

    # ── Hiking window ─────────────────────────────────────────────────
    hiking_window_hours = (hike_end_secs - hike_start_secs) / 3600.0

    # Estimate required time for the trail
    estimated_time = _estimate_hike_time_hours(
        trail.distance_km, trail.elevation_gain_m
    )

    if trail.is_loop:
        # Must complete the full loop
        if hiking_window_hours < estimated_time:
            return None
        actual_hiking_hours = estimated_time
        estimated_distance = trail.distance_km
    else:
        # Out-and-back: hiker walks as far as they can in half the window
        half_window = hiking_window_hours / 2.0
        # Speed considering elevation: distance / time ratio
        if estimated_time > 0:
            effective_speed = trail.distance_km / estimated_time
        else:
            effective_speed = NAISMITH_SPEED_KMH
        max_one_way_km = half_window * effective_speed
        one_way_km = min(max_one_way_km, trail.distance_km)
        estimated_distance = one_way_km * 2
        actual_hiking_hours = estimated_distance / effective_speed if effective_speed > 0 else 0

    if actual_hiking_hours < min_hiking_hours:
        return None

    # ── Build datetimes ───────────────────────────────────────────────
    date = deadline.date() if hasattr(deadline, 'date') else deadline
    base = datetime.datetime.combine(date, datetime.time())

    hike_start_dt = base + datetime.timedelta(seconds=hike_start_secs)
    hike_end_dt = base + datetime.timedelta(seconds=hike_end_secs)

    departure_from_origin = outbound_legs[0].departure
    arrival_at_origin = return_legs[-1].arrival

    total_hours = (arrival_at_origin - departure_from_origin).total_seconds() / 3600.0
    hiking_ratio = actual_hiking_hours / total_hours if total_hours > 0 else 0

    hike_segment = HikeSegment(
        trail_name=trail.name,
        entry_stop_name=ap.stop_name,
        walk_to_trail_m=ap.walk_distance_m,
        hike_start=hike_start_dt,
        hike_end=hike_end_dt,
        hiking_hours=actual_hiking_hours,
        estimated_distance_km=estimated_distance,
        is_loop=trail.is_loop,
        colors=trail.colors,
    )

    # ── Season warnings ──────────────────────────────────────────────
    warnings: list[str] = []
    if trail.season_warnings and date.month in _RAINY_MONTHS:
        warnings.extend(trail.season_warnings)

    return HikePlan(
        trail=trail,
        access_point=ap,
        outbound_legs=outbound_legs,
        hike_segment=hike_segment,
        return_legs=return_legs,
        departure_from_origin=departure_from_origin,
        arrival_at_origin=arrival_at_origin,
        hiking_ratio=hiking_ratio,
        deadline=deadline,
        total_hours=total_hours,
        score=hiking_ratio,
        warnings=warnings,
    )


def _plan_through_hike(
    trail: Trail,
    entry_ap: TrailAccessPoint,
    exit_ap: TrailAccessPoint,
    segment_km: float,
    router: TransitRouter,
    origin_stop_ids: list[str],
    origin_stop_set: set[str],
    earliest_dep_secs: int,
    deadline: datetime.datetime,
    deadline_secs: int,
    min_hiking_hours: float,
) -> HikePlan | None:
    """Try to build a through-hike plan: enter at entry_ap, exit at exit_ap."""
    # ── Return route from EXIT stop ────────────────────────────────────
    return_legs = router.find_return(
        trail_stops=[exit_ap.stop_id],
        origin_stops=origin_stop_set,
        deadline_secs=deadline_secs,
    )
    if return_legs is None:
        return None

    return_departure = return_legs[0].departure
    return_dep_secs = _datetime_to_seconds(return_departure)
    walk_from_trail_secs = _walk_time_hours(exit_ap.walk_distance_m) * 3600
    hike_end_secs = return_dep_secs - walk_from_trail_secs
    if hike_end_secs <= earliest_dep_secs:
        return None

    # ── Outbound route to ENTRY stop ───────────────────────────────────
    outbound_legs = router.find_outbound(
        origin_stops=origin_stop_ids,
        dest_stops={entry_ap.stop_id},
        earliest_departure_secs=earliest_dep_secs,
    )
    if outbound_legs is None:
        return None

    outbound_arrival = outbound_legs[-1].arrival
    outbound_arr_secs = _datetime_to_seconds(outbound_arrival)
    walk_to_trail_secs = _walk_time_hours(entry_ap.walk_distance_m) * 3600
    hike_start_secs = outbound_arr_secs + walk_to_trail_secs

    if hike_start_secs >= hike_end_secs:
        return None

    # ── Hiking time (Naismith on segment) ──────────────────────────────
    # Approximate segment elevation proportionally
    if trail.distance_km > 0:
        seg_elevation_gain = trail.elevation_gain_m * (segment_km / trail.distance_km)
    else:
        seg_elevation_gain = 0.0
    estimated_time = _estimate_hike_time_hours(segment_km, seg_elevation_gain)

    hiking_window_hours = (hike_end_secs - hike_start_secs) / 3600.0
    if hiking_window_hours < estimated_time:
        return None

    actual_hiking_hours = estimated_time
    if actual_hiking_hours < min_hiking_hours:
        return None

    # ── Build datetimes ────────────────────────────────────────────────
    date = deadline.date() if hasattr(deadline, 'date') else deadline
    base = datetime.datetime.combine(date, datetime.time())

    hike_start_dt = base + datetime.timedelta(seconds=hike_start_secs)
    hike_end_dt = base + datetime.timedelta(seconds=hike_end_secs)

    departure_from_origin = outbound_legs[0].departure
    arrival_at_origin = return_legs[-1].arrival

    total_hours = (arrival_at_origin - departure_from_origin).total_seconds() / 3600.0
    hiking_ratio = actual_hiking_hours / total_hours if total_hours > 0 else 0

    # Approximate segment elevation loss proportionally
    if trail.distance_km > 0:
        seg_elevation_loss = trail.elevation_loss_m * (segment_km / trail.distance_km)
    else:
        seg_elevation_loss = 0.0

    hike_segment = HikeSegment(
        trail_name=trail.name,
        entry_stop_name=entry_ap.stop_name,
        walk_to_trail_m=entry_ap.walk_distance_m,
        hike_start=hike_start_dt,
        hike_end=hike_end_dt,
        hiking_hours=actual_hiking_hours,
        estimated_distance_km=segment_km,
        is_loop=False,
        colors=trail.colors,
        elevation_gain_m=round(seg_elevation_gain, 1),
        elevation_loss_m=round(seg_elevation_loss, 1),
        is_through_hike=True,
        exit_stop_name=exit_ap.stop_name,
        walk_from_trail_m=exit_ap.walk_distance_m,
    )

    # ── Season warnings ────────────────────────────────────────────────
    warnings: list[str] = []
    if trail.season_warnings and date.month in _RAINY_MONTHS:
        warnings.extend(trail.season_warnings)

    return HikePlan(
        trail=trail,
        access_point=entry_ap,
        outbound_legs=outbound_legs,
        hike_segment=hike_segment,
        return_legs=return_legs,
        departure_from_origin=departure_from_origin,
        arrival_at_origin=arrival_at_origin,
        hiking_ratio=hiking_ratio,
        deadline=deadline,
        total_hours=total_hours,
        score=hiking_ratio,
        warnings=warnings,
        exit_access_point=exit_ap,
    )
