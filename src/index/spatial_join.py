"""Spatial join: find bus stops near hiking trails.

Builds an STRtree from GTFS stop locations, queries each trail geometry
to find nearby stops, and populates Trail.access_points with
TrailAccessPoint objects.  Handles coordinate-order conversion
(trail geometries are already stored as Shapely (lon, lat) LineStrings)
and uses haversine for precise distance filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from src.config import DEDUP_TRAIL_DISTANCE_M, MAX_WALK_TO_TRAIL_M
from src.ingest.gtfs import haversine
from src.models import Trail, TrailAccessPoint

logger = logging.getLogger(__name__)


@dataclass
class _StopRecord:
    """Lightweight container for a single GTFS stop."""

    stop_id: str
    stop_name: str
    lat: float
    lon: float


def build_trail_access_points(
    trails: list[Trail],
    stops_df: pd.DataFrame,
    max_distance_m: int = MAX_WALK_TO_TRAIL_M,
) -> list[Trail]:
    """Find bus stops near each trail and populate access_points.

    Parameters
    ----------
    trails:
        List of Trail objects whose ``geometry`` is a Shapely LineString
        in (lon, lat) coordinate order.
    stops_df:
        DataFrame with columns ``stop_id``, ``stop_name``,
        ``stop_lat``, ``stop_lon``.
    max_distance_m:
        Maximum walk distance (in meters) from a bus stop to the trail
        for it to be considered an access point.

    Returns
    -------
    list[Trail]
        Only trails that have at least one access point.  Each trail's
        ``access_points`` list is populated and sorted by
        ``trail_km_from_start``.
    """

    if stops_df.empty:
        logger.warning("stops_df is empty — no access points can be built")
        return []

    # ------------------------------------------------------------------
    # 1. Build stop records and STRtree from stop Point(lon, lat)
    # ------------------------------------------------------------------
    stop_records: list[_StopRecord] = []
    stop_points: list[Point] = []

    for row in stops_df.itertuples(index=False):
        stop_records.append(
            _StopRecord(
                stop_id=str(row.stop_id),
                stop_name=str(row.stop_name),
                lat=float(row.stop_lat),
                lon=float(row.stop_lon),
            )
        )
        stop_points.append(Point(float(row.stop_lon), float(row.stop_lat)))

    tree = STRtree(stop_points)
    logger.info("Built STRtree from %d stops", len(stop_points))

    # Degree buffer — rough conversion: 1 degree ≈ 111 km at equator.
    # At Israel's latitude (~31°N) the longitudinal degree is shorter,
    # but using the equatorial value gives a conservative (wider) buffer.
    buffer_deg = max_distance_m / 111_000

    # ------------------------------------------------------------------
    # 2. For each trail, find candidate stops and compute distances
    # ------------------------------------------------------------------
    trails_with_access: list[Trail] = []
    total = len(trails)

    for trail in trails:
        if trail.geometry is None or trail.geometry.is_empty:
            continue

        trail_geom: LineString = trail.geometry

        # Buffer the trail geometry to create a search envelope
        buffered = trail_geom.buffer(buffer_deg)

        # Query STRtree for candidate stop indices within the buffer
        candidate_indices = tree.query(buffered)

        access_points: list[TrailAccessPoint] = []

        for idx in candidate_indices:
            stop = stop_records[idx]
            stop_point = stop_points[idx]

            # Find the nearest point on the trail to this stop
            nearest_on_trail, _ = nearest_points(trail_geom, stop_point)

            # nearest_on_trail is in (lon, lat) order
            entry_lon = nearest_on_trail.x
            entry_lat = nearest_on_trail.y

            # Compute precise haversine distance
            walk_dist_m = haversine(stop.lat, stop.lon, entry_lat, entry_lon)

            if walk_dist_m > max_distance_m:
                continue

            # Compute how far along the trail (in km) this entry point is.
            # shapely.project gives the distance along the geometry in the
            # geometry's coordinate units (degrees here).  We normalise it
            # to a 0-1 fraction and multiply by the trail's real distance.
            fraction = trail_geom.project(stop_point, normalized=True)
            trail_km = fraction * trail.distance_km

            access_points.append(
                TrailAccessPoint(
                    stop_id=stop.stop_id,
                    stop_name=stop.stop_name,
                    walk_distance_m=round(walk_dist_m, 1),
                    trail_entry_lat=entry_lat,
                    trail_entry_lon=entry_lon,
                    trail_km_from_start=round(trail_km, 2),
                )
            )

        # --------------------------------------------------------------
        # 3. Deduplicate access points that are very close along the trail
        # --------------------------------------------------------------
        access_points = _deduplicate_access_points(
            access_points, DEDUP_TRAIL_DISTANCE_M
        )

        # Sort by position along the trail
        access_points.sort(key=lambda ap: ap.trail_km_from_start)

        if access_points:
            trail.access_points = access_points
            trails_with_access.append(trail)

    logger.info(
        "Spatial join complete: %d / %d trails have access points",
        len(trails_with_access),
        total,
    )

    return trails_with_access


def _deduplicate_access_points(
    access_points: list[TrailAccessPoint],
    min_trail_distance_m: float,
) -> list[TrailAccessPoint]:
    """Remove access points that are too close together along the trail.

    When two access points are within *min_trail_distance_m* metres of
    each other (measured along the trail), keep only the one with the
    shorter walk from the bus stop.

    Parameters
    ----------
    access_points:
        Unsorted list of candidate access points for a single trail.
    min_trail_distance_m:
        Minimum separation along the trail (in metres) between two
        retained access points.

    Returns
    -------
    list[TrailAccessPoint]
        Deduplicated list, sorted by ``trail_km_from_start``.
    """

    if len(access_points) <= 1:
        return access_points

    # Sort by position along trail
    sorted_pts = sorted(access_points, key=lambda ap: ap.trail_km_from_start)

    kept: list[TrailAccessPoint] = [sorted_pts[0]]

    for ap in sorted_pts[1:]:
        trail_separation_m = (
            abs(ap.trail_km_from_start - kept[-1].trail_km_from_start) * 1000
        )

        if trail_separation_m < min_trail_distance_m:
            # Too close — keep the one with the shorter walk distance
            if ap.walk_distance_m < kept[-1].walk_distance_m:
                kept[-1] = ap
        else:
            kept.append(ap)

    return kept
