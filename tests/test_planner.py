"""Tests for the Israel Hiking Transit Planner.

Unit tests for individual components that can run without external data
(GTFS, Overpass, Hebcal).  Uses mock data and fixtures.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from shapely.geometry import LineString

from src.config import CITY_COORDINATES
from src.ingest.gtfs import haversine, find_origin_stops, get_active_service_ids
from src.ingest.osm_trails import _parse_colors, _parse_season_info, _stitch_ways
from src.ingest.shabbat import get_deadline, _conservative_candle_estimate
from src.index.spatial_join import build_trail_access_points, _deduplicate_access_points
from src.models import (
    BusLeg,
    HikePlan,
    HikeQuery,
    HikeSegment,
    Trail,
    TrailAccessPoint,
)
from src.query.transit_router import TransitRouter, TransitRouterDB, _time_to_seconds
from src.ingest.gtfs import (
    build_transit_db,
    find_origin_stops_db,
    _gtfs_time_to_seconds,
    _get_active_service_ids_from_csv,
)
from src.query.planner import (
    PlannerContext,
    _resolve_origin,
    _estimate_hike_time_hours,
    _walk_time_hours,
    _filter_trails,
    _date_to_season,
    _plan_through_hike,
    _plan_single_trail,
    load_trail_index,
    plan_hikes,
    plan_hikes_for_origin,
    prepare_data,
)


# ═══════════════════════════════════════════════════════════════════════
# Haversine
# ═══════════════════════════════════════════════════════════════════════


class TestHaversine:
    def test_same_point(self):
        assert haversine(31.89, 34.81, 31.89, 34.81) == 0.0

    def test_known_distance(self):
        # Rehovot to Jerusalem: ~40 km
        dist = haversine(31.8928, 34.8113, 31.7683, 35.2137)
        assert 35_000 < dist < 45_000

    def test_symmetry(self):
        d1 = haversine(31.0, 34.0, 32.0, 35.0)
        d2 = haversine(32.0, 35.0, 31.0, 34.0)
        assert abs(d1 - d2) < 0.01

    def test_short_distance(self):
        # Two points ~111 m apart (0.001 degree latitude)
        dist = haversine(31.000, 34.000, 31.001, 34.000)
        assert 100 < dist < 120


# ═══════════════════════════════════════════════════════════════════════
# GTFS time parsing
# ═══════════════════════════════════════════════════════════════════════


class TestTimeToSeconds:
    def test_normal_time(self):
        assert _time_to_seconds("07:30:00") == 7 * 3600 + 30 * 60

    def test_midnight(self):
        assert _time_to_seconds("00:00:00") == 0

    def test_past_midnight(self):
        # GTFS allows hours >= 24
        assert _time_to_seconds("25:30:00") == 25 * 3600 + 30 * 60

    def test_noon(self):
        assert _time_to_seconds("12:00:00") == 43200


# ═══════════════════════════════════════════════════════════════════════
# Origin resolution
# ═══════════════════════════════════════════════════════════════════════


class TestResolveOrigin:
    def test_known_city(self):
        lat, lon = _resolve_origin("Rehovot")
        assert abs(lat - 31.8928) < 0.01
        assert abs(lon - 34.8113) < 0.01

    def test_case_insensitive(self):
        lat1, lon1 = _resolve_origin("REHOVOT")
        lat2, lon2 = _resolve_origin("rehovot")
        assert lat1 == lat2 and lon1 == lon2

    def test_unknown_city(self):
        with pytest.raises(ValueError, match="Unknown origin"):
            _resolve_origin("Narnia")

    def test_all_cities_defined(self):
        for city in CITY_COORDINATES:
            lat, lon = _resolve_origin(city)
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180


# ═══════════════════════════════════════════════════════════════════════
# Naismith's rule / walk time
# ═══════════════════════════════════════════════════════════════════════


class TestEstimations:
    def test_flat_trail(self):
        hours = _estimate_hike_time_hours(8.0, 0)
        assert abs(hours - 2.0) < 0.01  # 8 km / 4 km/h

    def test_hilly_trail(self):
        hours = _estimate_hike_time_hours(8.0, 600)
        # 8/4 + 600/600 = 2 + 1 = 3 hours
        assert abs(hours - 3.0) < 0.01

    def test_walk_time(self):
        hours = _walk_time_hours(1000)
        # 1 km / 4.5 km/h ≈ 0.222 hours
        assert abs(hours - 1.0 / 4.5) < 0.001


# ═══════════════════════════════════════════════════════════════════════
# OSM color parsing
# ═══════════════════════════════════════════════════════════════════════


class TestParseColors:
    def test_osmc_symbol(self):
        assert _parse_colors({"osmc:symbol": "red:white:red_bar"}) == ["red"]

    def test_colour_tag(self):
        assert _parse_colors({"colour": "blue"}) == ["blue"]

    def test_unknown_color_ignored(self):
        assert _parse_colors({"osmc:symbol": "yellow:white:yellow_bar"}) == []

    def test_multiple_sources(self):
        tags = {"osmc:symbol": "red:white:red_bar", "colour": "blue"}
        result = _parse_colors(tags)
        assert sorted(result) == ["blue", "red"]

    def test_empty_tags(self):
        assert _parse_colors({}) == []


# ═══════════════════════════════════════════════════════════════════════
# Way stitching
# ═══════════════════════════════════════════════════════════════════════


class TestStitchWays:
    def test_two_connected_ways(self):
        node_coords = {1: (31.0, 34.0), 2: (31.1, 34.1), 3: (31.2, 34.2)}
        way_nodes = {10: [1, 2], 20: [2, 3]}
        result = _stitch_ways([10, 20], way_nodes, node_coords)
        assert len(result) == 3
        assert result[0] == (31.0, 34.0)
        assert result[-1] == (31.2, 34.2)

    def test_reversed_way(self):
        node_coords = {1: (31.0, 34.0), 2: (31.1, 34.1), 3: (31.2, 34.2)}
        way_nodes = {10: [1, 2], 20: [3, 2]}  # second way is reversed
        result = _stitch_ways([10, 20], way_nodes, node_coords)
        assert len(result) == 3

    def test_single_way(self):
        node_coords = {1: (31.0, 34.0), 2: (31.1, 34.1)}
        way_nodes = {10: [1, 2]}
        result = _stitch_ways([10], way_nodes, node_coords)
        assert len(result) == 2

    def test_empty_ways(self):
        result = _stitch_ways([], {}, {})
        assert result == []

    def test_missing_way_ref(self):
        node_coords = {1: (31.0, 34.0), 2: (31.1, 34.1)}
        way_nodes = {10: [1, 2]}
        # way_ref 99 doesn't exist
        result = _stitch_ways([10, 99], way_nodes, node_coords)
        assert len(result) == 2

    def test_disconnected_takes_longest(self):
        node_coords = {
            1: (31.0, 34.0), 2: (31.1, 34.1), 3: (31.2, 34.2),
            4: (32.0, 35.0), 5: (32.1, 35.1),
        }
        way_nodes = {10: [1, 2, 3], 20: [4, 5]}
        result = _stitch_ways([10, 20], way_nodes, node_coords)
        assert len(result) == 3  # longest chain is [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════
# Shabbat deadline
# ═══════════════════════════════════════════════════════════════════════


class TestDeadline:
    def test_weekday_deadline(self):
        # 2026-02-03 is a Tuesday
        deadline = get_deadline(datetime.date(2026, 2, 3))
        assert deadline.hour == 18
        assert deadline.minute == 0

    def test_saturday_raises(self):
        # 2026-02-07 is a Saturday
        with pytest.raises(ValueError, match="Saturday"):
            get_deadline(datetime.date(2026, 2, 7))

    @patch("src.ingest.shabbat.fetch_candle_lighting")
    def test_friday_deadline(self, mock_candle):
        # 2026-02-06 is a Friday
        mock_candle.return_value = datetime.datetime(2026, 2, 6, 16, 53)
        deadline = get_deadline(datetime.date(2026, 2, 6), safety_margin_hours=2.0)
        assert deadline == datetime.datetime(2026, 2, 6, 14, 53)

    def test_conservative_winter(self):
        dt = _conservative_candle_estimate(datetime.date(2026, 1, 15))
        assert dt.hour == 16
        assert dt.minute == 30

    def test_conservative_summer(self):
        dt = _conservative_candle_estimate(datetime.date(2026, 6, 15))
        assert dt.hour == 19
        assert dt.minute == 0


# ═══════════════════════════════════════════════════════════════════════
# Find origin stops
# ═══════════════════════════════════════════════════════════════════════


class TestFindOriginStops:
    def _make_feed(self, stops_data):
        feed = MagicMock()
        feed.stops = pd.DataFrame(stops_data)
        return feed

    def test_finds_nearby_stops(self):
        feed = self._make_feed({
            "stop_id": ["s1", "s2", "s3"],
            "stop_name": ["Stop 1", "Stop 2", "Stop 3"],
            "stop_lat": [31.8930, 31.8925, 32.0000],
            "stop_lon": [34.8115, 34.8110, 34.8000],
        })
        result = find_origin_stops(feed, 31.8928, 34.8113, radius_m=500)
        assert "s1" in result
        assert "s2" in result
        assert "s3" not in result  # too far

    def test_sorted_by_distance(self):
        feed = self._make_feed({
            "stop_id": ["far", "near"],
            "stop_name": ["Far", "Near"],
            "stop_lat": [31.8940, 31.8929],
            "stop_lon": [34.8120, 34.8114],
        })
        result = find_origin_stops(feed, 31.8928, 34.8113, radius_m=500)
        assert result[0] == "near"

    def test_empty_when_none_nearby(self):
        feed = self._make_feed({
            "stop_id": ["s1"],
            "stop_name": ["Stop 1"],
            "stop_lat": [32.0],
            "stop_lon": [35.0],
        })
        result = find_origin_stops(feed, 31.8928, 34.8113, radius_m=500)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════
# Get active service IDs
# ═══════════════════════════════════════════════════════════════════════


class TestGetActiveServiceIds:
    def test_calendar_basic(self):
        feed = MagicMock()
        feed.calendar = pd.DataFrame({
            "service_id": ["svc1", "svc2"],
            "monday": [1, 0],
            "tuesday": [1, 1],
            "wednesday": [1, 1],
            "thursday": [1, 1],
            "friday": [1, 0],
            "saturday": [0, 0],
            "sunday": [1, 1],
            "start_date": [20260101, 20260101],
            "end_date": [20261231, 20261231],
        })
        feed.calendar_dates = pd.DataFrame(columns=["service_id", "date", "exception_type"])

        # 2026-02-03 is a Tuesday
        result = get_active_service_ids(feed, datetime.date(2026, 2, 3))
        assert "svc1" in result
        assert "svc2" in result

    def test_calendar_dates_exception(self):
        feed = MagicMock()
        feed.calendar = pd.DataFrame({
            "service_id": ["svc1"],
            "tuesday": [1],
            "monday": [0], "wednesday": [0], "thursday": [0],
            "friday": [0], "saturday": [0], "sunday": [0],
            "start_date": [20260101],
            "end_date": [20261231],
        })
        feed.calendar_dates = pd.DataFrame({
            "service_id": ["svc1", "svc_extra"],
            "date": [20260203, 20260203],
            "exception_type": [2, 1],  # remove svc1, add svc_extra
        })

        result = get_active_service_ids(feed, datetime.date(2026, 2, 3))
        assert "svc1" not in result  # removed by exception
        assert "svc_extra" in result  # added by exception


# ═══════════════════════════════════════════════════════════════════════
# Spatial join
# ═══════════════════════════════════════════════════════════════════════


class TestSpatialJoin:
    def _make_trail(self, coords_latlon, name="Test Trail", distance_km=5.0):
        """Create a Trail with a LineString in (lon, lat) Shapely order."""
        geom = LineString([(lon, lat) for lat, lon in coords_latlon])
        return Trail(
            id="test:1",
            name=name,
            source="osm",
            geometry=geom,
            distance_km=distance_km,
            elevation_gain_m=0,
            difficulty="unknown",
            colors=["red"],
            is_loop=False,
        )

    def _make_stops_df(self, stops):
        return pd.DataFrame(stops, columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])

    def test_finds_nearby_stop(self):
        # Trail running north-south
        trail = self._make_trail(
            [(31.80, 34.80), (31.81, 34.80), (31.82, 34.80)],
            distance_km=2.0,
        )
        stops_df = self._make_stops_df([
            ("s1", "Nearby Stop", 31.81, 34.8005),  # ~55m from trail
        ])
        result = build_trail_access_points([trail], stops_df, max_distance_m=500)
        assert len(result) == 1
        assert len(result[0].access_points) == 1
        assert result[0].access_points[0].stop_id == "s1"

    def test_filters_far_stop(self):
        trail = self._make_trail(
            [(31.80, 34.80), (31.81, 34.80), (31.82, 34.80)],
            distance_km=2.0,
        )
        stops_df = self._make_stops_df([
            ("s1", "Far Stop", 31.81, 34.82),  # ~2km from trail
        ])
        result = build_trail_access_points([trail], stops_df, max_distance_m=500)
        assert len(result) == 0

    def test_empty_stops(self):
        trail = self._make_trail([(31.80, 34.80), (31.82, 34.80)])
        stops_df = pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])
        result = build_trail_access_points([trail], stops_df)
        assert result == []


class TestDeduplicateAccessPoints:
    def test_keeps_closer_walk(self):
        ap1 = TrailAccessPoint("s1", "Stop 1", 200, 31.0, 34.0, 1.0)
        ap2 = TrailAccessPoint("s2", "Stop 2", 100, 31.0, 34.0, 1.05)  # 50m apart on trail
        result = _deduplicate_access_points([ap1, ap2], 200)
        assert len(result) == 1
        assert result[0].stop_id == "s2"  # shorter walk

    def test_keeps_both_when_far_apart(self):
        ap1 = TrailAccessPoint("s1", "Stop 1", 200, 31.0, 34.0, 0.0)
        ap2 = TrailAccessPoint("s2", "Stop 2", 200, 31.0, 34.0, 5.0)  # 5km apart
        result = _deduplicate_access_points([ap1, ap2], 200)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════
# Transit router
# ═══════════════════════════════════════════════════════════════════════


class TestTransitRouter:
    @pytest.fixture
    def simple_feed(self):
        """Create a minimal GTFS feed with two trips."""
        feed = MagicMock()
        feed.stops = pd.DataFrame({
            "stop_id": ["A", "B", "C", "D"],
            "stop_name": ["Origin", "Mid", "Trail", "Other"],
        })
        feed.agency = pd.DataFrame({
            "agency_id": ["ag1"],
            "agency_name": ["Egged"],
        })
        feed.routes = pd.DataFrame({
            "route_id": ["r1", "r2"],
            "agency_id": ["ag1", "ag1"],
            "route_short_name": ["100", "200"],
        })
        feed.trips = pd.DataFrame({
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r2"],
            "service_id": ["svc1", "svc1"],
        })
        feed.stop_times = pd.DataFrame({
            "trip_id":        ["t1", "t1", "t1", "t2", "t2", "t2"],
            "stop_id":        ["A",  "B",  "C",  "C",  "B",  "A"],
            "stop_sequence":  [1,    2,    3,    1,    2,    3],
            "arrival_time":   ["07:00:00", "07:30:00", "08:00:00",
                               "15:00:00", "15:30:00", "16:00:00"],
            "departure_time": ["07:00:00", "07:30:00", "08:00:00",
                               "15:00:00", "15:30:00", "16:00:00"],
        })
        return feed

    def test_build_router(self, simple_feed):
        router = TransitRouter(simple_feed, datetime.date(2026, 2, 3))
        assert "A" in router.stop_departures
        assert "t1" in router.trip_stop_sequence
        assert len(router.trip_stop_sequence["t1"]) == 3

    def test_find_outbound_direct(self, simple_feed):
        router = TransitRouter(simple_feed, datetime.date(2026, 2, 3))
        legs = router.find_outbound(
            origin_stops=["A"],
            dest_stops={"C"},
            earliest_departure_secs=6 * 3600,  # 06:00
        )
        assert legs is not None
        assert len(legs) == 1
        assert legs[0].line == "100"
        assert legs[0].from_stop_id == "A"
        assert legs[0].to_stop_id == "C"

    def test_find_return_direct(self, simple_feed):
        router = TransitRouter(simple_feed, datetime.date(2026, 2, 3))
        legs = router.find_return(
            trail_stops=["C"],
            origin_stops={"A"},
            deadline_secs=18 * 3600,  # 18:00
        )
        assert legs is not None
        assert len(legs) == 1
        assert legs[0].line == "200"
        assert legs[0].from_stop_id == "C"
        assert legs[0].to_stop_id == "A"

    def test_no_route_found(self, simple_feed):
        router = TransitRouter(simple_feed, datetime.date(2026, 2, 3))
        # No route from D to C
        legs = router.find_outbound(
            origin_stops=["D"],
            dest_stops={"C"},
            earliest_departure_secs=6 * 3600,
        )
        assert legs is None

    def test_return_respects_deadline(self, simple_feed):
        router = TransitRouter(simple_feed, datetime.date(2026, 2, 3))
        # Deadline before the return trip departs
        legs = router.find_return(
            trail_stops=["C"],
            origin_stops={"A"},
            deadline_secs=14 * 3600,  # 14:00, before the 15:00 departure
        )
        assert legs is None


class TestTransitRouterWithTransfer:
    @pytest.fixture
    def transfer_feed(self):
        """Feed where origin→trail requires a transfer at B."""
        feed = MagicMock()
        feed.stops = pd.DataFrame({
            "stop_id": ["A", "B", "C"],
            "stop_name": ["Origin", "Transfer", "Trail"],
        })
        feed.agency = pd.DataFrame({
            "agency_id": ["ag1"],
            "agency_name": ["Egged"],
        })
        feed.routes = pd.DataFrame({
            "route_id": ["r1", "r2"],
            "agency_id": ["ag1", "ag1"],
            "route_short_name": ["100", "200"],
        })
        feed.trips = pd.DataFrame({
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r2"],
            "service_id": ["svc1", "svc1"],
        })
        # t1: A→B, t2: B→C (no direct A→C)
        feed.stop_times = pd.DataFrame({
            "trip_id":        ["t1", "t1", "t2", "t2"],
            "stop_id":        ["A",  "B",  "B",  "C"],
            "stop_sequence":  [1,    2,    1,    2],
            "arrival_time":   ["07:00:00", "07:30:00", "07:35:00", "08:00:00"],
            "departure_time": ["07:00:00", "07:30:00", "07:35:00", "08:00:00"],
        })
        return feed

    def test_one_transfer_outbound(self, transfer_feed):
        router = TransitRouter(transfer_feed, datetime.date(2026, 2, 3))
        legs = router.find_outbound(
            origin_stops=["A"],
            dest_stops={"C"},
            earliest_departure_secs=6 * 3600,
        )
        assert legs is not None
        assert len(legs) == 2
        assert legs[0].from_stop_id == "A"
        assert legs[0].to_stop_id == "B"
        assert legs[1].from_stop_id == "B"
        assert legs[1].to_stop_id == "C"


# ═══════════════════════════════════════════════════════════════════════
# Model dataclasses
# ═══════════════════════════════════════════════════════════════════════


class TestModels:
    def test_hike_query_defaults(self):
        q = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 6))
        assert q.max_transfers == 1
        assert q.safety_margin_hours == 2.0
        assert q.max_walk_to_trail_m == 1000
        assert q.min_hiking_hours == 1.0
        assert q.max_results == 20

    def test_trail_access_points_default_empty(self):
        trail = Trail(
            id="t1", name="Test", source="osm",
            geometry=LineString([(34.0, 31.0), (34.1, 31.1)]),
            distance_km=10, elevation_gain_m=0,
            difficulty="unknown", colors=[], is_loop=False,
        )
        assert trail.access_points == []

    def test_bus_leg_creation(self):
        leg = BusLeg(
            line="100", operator="Egged",
            from_stop_id="s1", from_stop_name="Origin",
            to_stop_id="s2", to_stop_name="Dest",
            departure=datetime.datetime(2026, 2, 6, 7, 0),
            arrival=datetime.datetime(2026, 2, 6, 8, 0),
        )
        assert (leg.arrival - leg.departure).total_seconds() == 3600


# ═══════════════════════════════════════════════════════════════════════
# Trail filters (v0.2)
# ═══════════════════════════════════════════════════════════════════════


class TestFilterTrails:
    def _make_trail(self, name="Trail", distance_km=5.0, elevation_gain_m=100,
                    colors=None, is_loop=False, difficulty="unknown"):
        return Trail(
            id=f"osm:{name}",
            name=name,
            source="osm",
            geometry=LineString([(34.0, 31.0), (34.1, 31.1)]),
            distance_km=distance_km,
            elevation_gain_m=elevation_gain_m,
            difficulty=difficulty,
            colors=colors or [],
            is_loop=is_loop,
        )

    def _make_query(self, **kwargs):
        defaults = {"origin": "Rehovot", "date": datetime.date(2026, 2, 6)}
        defaults.update(kwargs)
        return HikeQuery(**defaults)

    def test_no_filters_returns_all(self):
        trails = [self._make_trail("A"), self._make_trail("B")]
        result = _filter_trails(trails, self._make_query())
        assert len(result) == 2

    def test_filter_by_color(self):
        trails = [
            self._make_trail("Red", colors=["red"]),
            self._make_trail("Blue", colors=["blue"]),
            self._make_trail("Both", colors=["red", "blue"]),
        ]
        result = _filter_trails(trails, self._make_query(colors=["red"]))
        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"Red", "Both"}

    def test_filter_min_distance(self):
        trails = [
            self._make_trail("Short", distance_km=2.0),
            self._make_trail("Long", distance_km=10.0),
        ]
        result = _filter_trails(trails, self._make_query(min_distance_km=5.0))
        assert len(result) == 1
        assert result[0].name == "Long"

    def test_filter_max_distance(self):
        trails = [
            self._make_trail("Short", distance_km=2.0),
            self._make_trail("Long", distance_km=10.0),
        ]
        result = _filter_trails(trails, self._make_query(max_distance_km=5.0))
        assert len(result) == 1
        assert result[0].name == "Short"

    def test_filter_loop_only(self):
        trails = [
            self._make_trail("Loop", is_loop=True),
            self._make_trail("Linear", is_loop=False),
        ]
        result = _filter_trails(trails, self._make_query(loop_only=True))
        assert len(result) == 1
        assert result[0].name == "Loop"

    def test_filter_linear_only(self):
        trails = [
            self._make_trail("Loop", is_loop=True),
            self._make_trail("Linear", is_loop=False),
        ]
        result = _filter_trails(trails, self._make_query(linear_only=True))
        assert len(result) == 1
        assert result[0].name == "Linear"

    def test_filter_max_elevation_gain(self):
        trails = [
            self._make_trail("Flat", elevation_gain_m=50),
            self._make_trail("Steep", elevation_gain_m=500),
        ]
        result = _filter_trails(trails, self._make_query(max_elevation_gain_m=200))
        assert len(result) == 1
        assert result[0].name == "Flat"

    def test_filter_difficulty(self):
        trails = [
            self._make_trail("Easy", difficulty="easy"),
            self._make_trail("Hard", difficulty="hard"),
        ]
        result = _filter_trails(trails, self._make_query(difficulty="easy"))
        assert len(result) == 1
        assert result[0].name == "Easy"

    def test_filter_difficulty_case_insensitive(self):
        trails = [self._make_trail("Easy", difficulty="Easy")]
        result = _filter_trails(trails, self._make_query(difficulty="easy"))
        assert len(result) == 1

    def test_combined_filters(self):
        trails = [
            self._make_trail("A", distance_km=3.0, colors=["red"], is_loop=True),
            self._make_trail("B", distance_km=8.0, colors=["red"], is_loop=False),
            self._make_trail("C", distance_km=3.0, colors=["blue"], is_loop=True),
        ]
        result = _filter_trails(trails, self._make_query(
            colors=["red"], loop_only=True, min_distance_km=2.0,
        ))
        assert len(result) == 1
        assert result[0].name == "A"


# ═══════════════════════════════════════════════════════════════════════
# Season awareness (v0.2)
# ═══════════════════════════════════════════════════════════════════════


class TestSeasonParsing:
    def test_wadi_name_detected(self):
        seasons, warnings = _parse_season_info("Wadi Qelt Trail", {}, [(31.5, 35.0)])
        assert "summer" in seasons
        assert len(warnings) == 1
        assert "flood" in warnings[0].lower()

    def test_nahal_name_detected(self):
        seasons, warnings = _parse_season_info("Nahal Arugot", {}, [(31.5, 35.0)])
        assert len(warnings) == 1

    def test_negev_name_detected(self):
        seasons, warnings = _parse_season_info("Negev Highland Trail", {}, [(31.5, 35.0)])
        assert len(warnings) == 1

    def test_deep_negev_by_latitude(self):
        # Trail in deep Negev (lat < 31.0)
        seasons, warnings = _parse_season_info("Some Trail", {}, [(30.5, 34.8)])
        assert len(warnings) == 1

    def test_osm_tag_detection(self):
        tags = {"description": "Dry riverbed, flood danger in winter"}
        seasons, warnings = _parse_season_info("Trail X", tags, [(32.0, 35.0)])
        assert len(warnings) == 1

    def test_non_desert_trail(self):
        # Galilee trail, no keywords, high latitude
        seasons, warnings = _parse_season_info("Mount Meron Loop", {}, [(33.0, 35.4)])
        assert seasons == []
        assert warnings == []

    def test_ein_keyword_detected(self):
        seasons, warnings = _parse_season_info("Ein Gedi Trail", {}, [(31.5, 35.3)])
        assert len(warnings) == 1

    def test_case_insensitive_keywords(self):
        seasons, warnings = _parse_season_info("WADI ARUGOT", {}, [(31.5, 35.0)])
        assert len(warnings) == 1

    def test_recommended_seasons_for_desert(self):
        seasons, _ = _parse_season_info("Wadi Og", {}, [(31.5, 35.0)])
        assert set(seasons) == {"spring", "autumn", "summer"}


class TestDateToSeason:
    def test_winter(self):
        assert _date_to_season(datetime.date(2026, 1, 15)) == "winter"
        assert _date_to_season(datetime.date(2026, 2, 15)) == "winter"

    def test_spring(self):
        assert _date_to_season(datetime.date(2026, 3, 15)) == "spring"
        assert _date_to_season(datetime.date(2026, 5, 15)) == "spring"

    def test_summer(self):
        assert _date_to_season(datetime.date(2026, 6, 15)) == "summer"
        assert _date_to_season(datetime.date(2026, 8, 15)) == "summer"

    def test_autumn(self):
        assert _date_to_season(datetime.date(2026, 9, 15)) == "autumn"
        assert _date_to_season(datetime.date(2026, 11, 15)) == "autumn"


# ═══════════════════════════════════════════════════════════════════════
# Elevation sampling (v0.2)
# ═══════════════════════════════════════════════════════════════════════


class TestElevationSampler:
    def test_tile_name(self):
        from src.ingest.elevation import ElevationSampler
        sampler = ElevationSampler()
        assert sampler._tile_name(31.5, 34.8) == "N31E034"
        assert sampler._tile_name(-33.9, -18.4) == "S34W019"
        assert sampler._tile_name(0.5, 0.5) == "N00E000"

    def test_gain_loss_calculation(self):
        from src.ingest.elevation import ElevationSampler
        sampler = ElevationSampler()

        # Mock sample_point to return a known elevation profile
        elevations = iter([100, 150, 200, 180, 220])
        sampler.sample_point = lambda lat, lon: next(elevations)

        geom = LineString([(34.0, 31.0), (34.01, 31.0), (34.02, 31.0),
                           (34.03, 31.0), (34.04, 31.0)])
        # Force exactly 4 intervals (5 samples)
        result = sampler.sample_trail(geom, distance_km=0.2)

        assert result["elevation_gain_m"] > 0
        assert result["elevation_loss_m"] > 0
        assert result["max_elevation_m"] == 220.0
        assert result["min_elevation_m"] == 100.0

    @patch("src.ingest.elevation.ElevationSampler._get_dataset")
    def test_no_srtm_data_returns_zeros(self, mock_get_ds):
        from src.ingest.elevation import ElevationSampler
        mock_get_ds.return_value = None
        sampler = ElevationSampler()
        geom = LineString([(34.0, 31.0), (34.1, 31.1)])
        result = sampler.sample_trail(geom, distance_km=5.0)
        assert result["elevation_gain_m"] == 0.0
        assert result["elevation_loss_m"] == 0.0

    def test_monotonic_ascent(self):
        from src.ingest.elevation import ElevationSampler
        sampler = ElevationSampler()

        elevations = iter([100, 200, 300])
        sampler.sample_point = lambda lat, lon: next(elevations)

        geom = LineString([(34.0, 31.0), (34.01, 31.0), (34.02, 31.0)])
        result = sampler.sample_trail(geom, distance_km=0.1)

        assert result["elevation_gain_m"] == 200.0
        assert result["elevation_loss_m"] == 0.0

    def test_monotonic_descent(self):
        from src.ingest.elevation import ElevationSampler
        sampler = ElevationSampler()

        elevations = iter([300, 200, 100])
        sampler.sample_point = lambda lat, lon: next(elevations)

        geom = LineString([(34.0, 31.0), (34.01, 31.0), (34.02, 31.0)])
        result = sampler.sample_trail(geom, distance_km=0.1)

        assert result["elevation_gain_m"] == 0.0
        assert result["elevation_loss_m"] == 200.0

    def test_enrichment_graceful_degradation(self):
        """enrich_trails_with_elevation should not crash when no SRTM data."""
        from src.ingest.osm_trails import enrich_trails_with_elevation
        trail = Trail(
            id="test:1", name="Test", source="osm",
            geometry=LineString([(34.0, 31.0), (34.1, 31.1)]),
            distance_km=5.0, elevation_gain_m=0,
            difficulty="unknown", colors=[], is_loop=False,
        )
        # Should not raise even without SRTM files
        enrich_trails_with_elevation([trail])


# ═══════════════════════════════════════════════════════════════════════
# Through-hikes (v0.2)
# ═══════════════════════════════════════════════════════════════════════


class TestThroughHike:
    """Tests for through-hike planning (enter at one stop, exit at another)."""

    @pytest.fixture
    def through_hike_feed(self):
        """Feed with separate outbound and return routes to different stops."""
        feed = MagicMock()
        feed.stops = pd.DataFrame({
            "stop_id": ["O", "E1", "E2"],
            "stop_name": ["Origin", "Entry Stop", "Exit Stop"],
        })
        feed.agency = pd.DataFrame({
            "agency_id": ["ag1"],
            "agency_name": ["Egged"],
        })
        feed.routes = pd.DataFrame({
            "route_id": ["r1", "r2", "r3"],
            "agency_id": ["ag1", "ag1", "ag1"],
            "route_short_name": ["100", "200", "300"],
        })
        feed.trips = pd.DataFrame({
            "trip_id": ["t1", "t2", "t3"],
            "route_id": ["r1", "r2", "r3"],
            "service_id": ["svc1", "svc1", "svc1"],
        })
        # t1: O→E1 (outbound to entry), t2: E2→O (return from exit), t3: O→E2 (outbound to exit too)
        feed.stop_times = pd.DataFrame({
            "trip_id":        ["t1", "t1", "t2", "t2", "t3", "t3"],
            "stop_id":        ["O",  "E1", "E2", "O",  "O",  "E2"],
            "stop_sequence":  [1,    2,    1,    2,    1,    2],
            "arrival_time":   ["07:00:00", "08:00:00", "14:00:00", "15:00:00",
                               "07:30:00", "08:30:00"],
            "departure_time": ["07:00:00", "08:00:00", "14:00:00", "15:00:00",
                               "07:30:00", "08:30:00"],
        })
        return feed

    def _make_linear_trail(self, access_points):
        return Trail(
            id="osm:999",
            name="Linear Trail",
            source="osm",
            geometry=LineString([(34.0, 31.0), (34.1, 31.1)]),
            distance_km=10.0,
            elevation_gain_m=200,
            difficulty="moderate",
            colors=["red"],
            is_loop=False,
            access_points=access_points,
        )

    def _make_loop_trail(self, access_points):
        return Trail(
            id="osm:888",
            name="Loop Trail",
            source="osm",
            geometry=LineString([(34.0, 31.0), (34.05, 31.05), (34.0, 31.0)]),
            distance_km=6.0,
            elevation_gain_m=100,
            difficulty="easy",
            colors=["blue"],
            is_loop=True,
            access_points=access_points,
        )

    def test_through_hike_basic(self, through_hike_feed):
        """Through-hike plan with entry and exit at different stops."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        entry_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        exit_ap = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 5.0)
        trail = self._make_linear_trail([entry_ap, exit_ap])

        deadline = datetime.datetime(2026, 2, 3, 18, 0)
        plan = _plan_through_hike(
            trail=trail,
            entry_ap=entry_ap,
            exit_ap=exit_ap,
            segment_km=5.0,
            router=router,
            origin_stop_ids=["O"],
            origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline,
            deadline_secs=18 * 3600,
            min_hiking_hours=1.0,
        )
        assert plan is not None
        assert plan.hike_segment.is_through_hike is True
        assert plan.hike_segment.exit_stop_name == "Exit Stop"
        assert plan.exit_access_point is exit_ap

    def test_through_hike_no_return_route(self, through_hike_feed):
        """Should return None when no return bus from exit stop."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        entry_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        exit_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 5.0)  # same stop, but test route

        # Use a stop with no return route
        bad_exit = TrailAccessPoint("NONE", "Nowhere", 200, 31.0, 34.0, 5.0)
        trail = self._make_linear_trail([entry_ap, bad_exit])

        deadline = datetime.datetime(2026, 2, 3, 18, 0)
        plan = _plan_through_hike(
            trail=trail,
            entry_ap=entry_ap,
            exit_ap=bad_exit,
            segment_km=5.0,
            router=router,
            origin_stop_ids=["O"],
            origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline,
            deadline_secs=18 * 3600,
            min_hiking_hours=1.0,
        )
        assert plan is None

    def test_through_hike_segment_too_short(self, through_hike_feed):
        """Skip through-hike when segment distance < minimum."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        entry_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        exit_ap = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 1.0)  # only 1km apart
        trail = self._make_linear_trail([entry_ap, exit_ap])

        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        deadline = datetime.datetime(2026, 2, 3, 18, 0)

        plans = _plan_single_trail(
            trail=trail, router=router,
            origin_stop_ids=["O"], origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline, deadline_secs=18 * 3600,
            query=query,
        )
        # Should not have through-hike (segment < 3km)
        through_plans = [p for p in plans if p.hike_segment.is_through_hike]
        assert len(through_plans) == 0

    def test_through_hike_segment_too_long(self, through_hike_feed):
        """Skip through-hike when segment distance > maximum."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        entry_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        exit_ap = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 25.0)  # 25km apart
        trail = Trail(
            id="osm:777", name="Very Long Trail", source="osm",
            geometry=LineString([(34.0, 31.0), (34.3, 31.3)]),
            distance_km=30.0, elevation_gain_m=500,
            difficulty="hard", colors=["red"], is_loop=False,
            access_points=[entry_ap, exit_ap],
        )

        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        deadline = datetime.datetime(2026, 2, 3, 18, 0)

        plans = _plan_single_trail(
            trail=trail, router=router,
            origin_stop_ids=["O"], origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline, deadline_secs=18 * 3600,
            query=query,
        )
        through_plans = [p for p in plans if p.hike_segment.is_through_hike]
        assert len(through_plans) == 0

    def test_loop_trail_skips_through_hike(self, through_hike_feed):
        """Loop trails should not generate through-hike plans."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        ap1 = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        ap2 = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 5.0)
        trail = self._make_loop_trail([ap1, ap2])

        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        deadline = datetime.datetime(2026, 2, 3, 18, 0)

        plans = _plan_single_trail(
            trail=trail, router=router,
            origin_stop_ids=["O"], origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline, deadline_secs=18 * 3600,
            query=query,
        )
        through_plans = [p for p in plans if p.hike_segment.is_through_hike]
        assert len(through_plans) == 0

    def test_single_access_point_no_through_hike(self, through_hike_feed):
        """Trails with only one access point cannot have through-hikes."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        ap1 = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        trail = self._make_linear_trail([ap1])

        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        deadline = datetime.datetime(2026, 2, 3, 18, 0)

        plans = _plan_single_trail(
            trail=trail, router=router,
            origin_stop_ids=["O"], origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline, deadline_secs=18 * 3600,
            query=query,
        )
        through_plans = [p for p in plans if p.hike_segment.is_through_hike]
        assert len(through_plans) == 0

    def test_through_hike_both_directions(self, through_hike_feed):
        """Through-hike should try both A->B and B->A."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        ap1 = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        ap2 = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 5.0)
        trail = self._make_linear_trail([ap1, ap2])

        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        deadline = datetime.datetime(2026, 2, 3, 18, 0)

        plans = _plan_single_trail(
            trail=trail, router=router,
            origin_stop_ids=["O"], origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline, deadline_secs=18 * 3600,
            query=query,
        )
        # Should have at most 2 plans: 1 out-and-back + 1 best through-hike
        assert len(plans) <= 2

    def test_through_hike_estimated_distance(self, through_hike_feed):
        """Through-hike should have correct segment distance."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        entry_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 1.0)
        exit_ap = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 8.0)
        trail = self._make_linear_trail([entry_ap, exit_ap])

        deadline = datetime.datetime(2026, 2, 3, 18, 0)
        plan = _plan_through_hike(
            trail=trail,
            entry_ap=entry_ap,
            exit_ap=exit_ap,
            segment_km=7.0,
            router=router,
            origin_stop_ids=["O"],
            origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline,
            deadline_secs=18 * 3600,
            min_hiking_hours=1.0,
        )
        assert plan is not None
        assert plan.hike_segment.estimated_distance_km == 7.0

    def test_through_hike_window_too_small(self, through_hike_feed):
        """Return None if hiking window is insufficient for the segment."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        entry_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        exit_ap = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 10.0)

        # Large trail with high elevation — estimated time will be long
        trail = Trail(
            id="osm:111", name="Hard Trail", source="osm",
            geometry=LineString([(34.0, 31.0), (34.1, 31.1)]),
            distance_km=10.0, elevation_gain_m=3000,  # very steep
            difficulty="hard", colors=["red"], is_loop=False,
            access_points=[entry_ap, exit_ap],
        )

        # Tight deadline: only 2h after earliest bus arrives (08:00) -> 10:00
        deadline = datetime.datetime(2026, 2, 3, 10, 30)
        plan = _plan_through_hike(
            trail=trail,
            entry_ap=entry_ap,
            exit_ap=exit_ap,
            segment_km=10.0,
            router=router,
            origin_stop_ids=["O"],
            origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline,
            deadline_secs=int(10.5 * 3600),
            min_hiking_hours=1.0,
        )
        assert plan is None

    def test_through_hike_proportional_elevation(self, through_hike_feed):
        """Through-hike should estimate elevation gain proportionally."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        entry_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        exit_ap = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 5.0)
        trail = self._make_linear_trail([entry_ap, exit_ap])
        trail.elevation_gain_m = 200  # 200m gain over 10km

        deadline = datetime.datetime(2026, 2, 3, 18, 0)
        plan = _plan_through_hike(
            trail=trail,
            entry_ap=entry_ap,
            exit_ap=exit_ap,
            segment_km=5.0,  # half the trail
            router=router,
            origin_stop_ids=["O"],
            origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline,
            deadline_secs=18 * 3600,
            min_hiking_hours=1.0,
        )
        assert plan is not None
        # 5km out of 10km -> 100m of 200m gain
        assert plan.hike_segment.elevation_gain_m == 100.0

    def test_plan_single_trail_returns_list(self, through_hike_feed):
        """_plan_single_trail should return a list of plans."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        ap1 = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        trail = self._make_linear_trail([ap1])

        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        deadline = datetime.datetime(2026, 2, 3, 18, 0)

        result = _plan_single_trail(
            trail=trail, router=router,
            origin_stop_ids=["O"], origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline, deadline_secs=18 * 3600,
            query=query,
        )
        assert isinstance(result, list)

    def test_through_hike_min_hiking_hours(self, through_hike_feed):
        """Through-hike should fail if segment takes less than min_hiking_hours."""
        router = TransitRouter(through_hike_feed, datetime.date(2026, 2, 3))
        entry_ap = TrailAccessPoint("E1", "Entry Stop", 200, 31.0, 34.0, 0.0)
        exit_ap = TrailAccessPoint("E2", "Exit Stop", 300, 31.1, 34.1, 3.5)

        # Short flat segment: 3.5km at 4km/h = 0.875h
        trail = Trail(
            id="osm:555", name="Short Trail", source="osm",
            geometry=LineString([(34.0, 31.0), (34.1, 31.1)]),
            distance_km=10.0, elevation_gain_m=0,
            difficulty="easy", colors=[], is_loop=False,
            access_points=[entry_ap, exit_ap],
        )

        deadline = datetime.datetime(2026, 2, 3, 18, 0)
        plan = _plan_through_hike(
            trail=trail,
            entry_ap=entry_ap,
            exit_ap=exit_ap,
            segment_km=3.5,
            router=router,
            origin_stop_ids=["O"],
            origin_stop_set={"O"},
            earliest_dep_secs=6 * 3600,
            deadline=deadline,
            deadline_secs=18 * 3600,
            min_hiking_hours=2.0,  # requires 2h minimum
        )
        assert plan is None


# ═══════════════════════════════════════════════════════════════════════
# Elevation profiles (v0.2)
# ═══════════════════════════════════════════════════════════════════════


class TestSparkline:
    def test_flat_profile(self):
        from src.output.cli_formatter import _sparkline
        result = _sparkline([100.0] * 10, width=10)
        assert len(result) == 10
        # All same value -> all same char
        assert len(set(result)) == 1

    def test_ascending_profile(self):
        from src.output.cli_formatter import _sparkline
        result = _sparkline([0.0, 50.0, 100.0], width=3)
        assert len(result) == 3
        assert result[0] < result[-1]  # char ordering

    def test_empty_profile(self):
        from src.output.cli_formatter import _sparkline
        assert _sparkline([]) == ""
        assert _sparkline([42.0]) == ""

    def test_width_resampling_down(self):
        from src.output.cli_formatter import _sparkline
        result = _sparkline(list(range(200)), width=40)
        assert len(result) == 40

    def test_width_resampling_up(self):
        from src.output.cli_formatter import _sparkline
        result = _sparkline([0.0, 100.0, 50.0], width=10)
        assert len(result) == 10


class TestElevationProfile:
    def test_sample_trail_returns_profile(self):
        from src.ingest.elevation import ElevationSampler
        sampler = ElevationSampler()
        elevations = iter([100, 200, 300])
        sampler.sample_point = lambda lat, lon: next(elevations)
        geom = LineString([(34.0, 31.0), (34.01, 31.0), (34.02, 31.0)])
        result = sampler.sample_trail(geom, distance_km=0.1)
        assert "elevation_profile" in result
        assert len(result["elevation_profile"]) == 3
        assert result["elevation_profile"] == [100, 200, 300]

    @patch("src.ingest.elevation.ElevationSampler._get_dataset")
    def test_no_data_returns_empty_profile(self, mock_get_ds):
        from src.ingest.elevation import ElevationSampler
        mock_get_ds.return_value = None
        sampler = ElevationSampler()
        geom = LineString([(34.0, 31.0), (34.1, 31.1)])
        result = sampler.sample_trail(geom, distance_km=5.0)
        assert result["elevation_profile"] == []

    def test_trail_model_has_elevation_profile(self):
        trail = Trail(
            id="t1", name="Test", source="osm",
            geometry=LineString([(34.0, 31.0), (34.1, 31.1)]),
            distance_km=10, elevation_gain_m=0,
            difficulty="unknown", colors=[], is_loop=False,
        )
        assert trail.elevation_profile == []


# ═══════════════════════════════════════════════════════════════════════
# Multiple origins (v0.2)
# ═══════════════════════════════════════════════════════════════════════


class TestMultipleOrigins:
    """Tests for the prepare_data / plan_hikes_for_origin refactor."""

    def test_planner_context_fields(self):
        """PlannerContext should have the expected fields."""
        ctx = PlannerContext(
            feed="fake_feed",
            trails=[],
            deadline=datetime.datetime(2026, 2, 3, 18, 0),
            deadline_secs=18 * 3600,
            router="fake_router",
        )
        assert ctx.feed == "fake_feed"
        assert ctx.trails == []
        assert ctx.deadline_secs == 64800
        assert ctx.router == "fake_router"

    @patch("src.query.planner.get_deadline")
    @patch("src.query.planner.build_trail_access_points")
    @patch("src.query.planner.fetch_hiking_trails")
    @patch("src.query.planner.load_feed_for_date")
    @patch("src.query.planner.download_gtfs")
    def test_prepare_data_returns_context(
        self, mock_dl, mock_load, mock_fetch, mock_join, mock_deadline
    ):
        """prepare_data should return a PlannerContext."""
        mock_dl.return_value = "/fake/gtfs.zip"
        mock_feed = MagicMock()
        mock_feed.stops = pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])
        mock_load.return_value = mock_feed
        mock_fetch.return_value = []
        mock_join.return_value = []
        mock_deadline.return_value = datetime.datetime(2026, 2, 3, 18, 0)

        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        ctx = prepare_data(query)

        assert isinstance(ctx, PlannerContext)
        assert ctx.feed is mock_feed
        assert isinstance(ctx.trails, list)
        assert ctx.deadline == datetime.datetime(2026, 2, 3, 18, 0)

    @patch("src.query.planner.find_origin_stops")
    def test_plan_hikes_for_origin_no_stops(self, mock_find):
        """Should return [] when no origin stops found."""
        mock_find.return_value = []
        ctx = PlannerContext(
            feed=MagicMock(),
            trails=[],
            deadline=datetime.datetime(2026, 2, 3, 18, 0),
            deadline_secs=18 * 3600,
            router=MagicMock(),
        )
        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        result = plan_hikes_for_origin(query, ctx)
        assert result == []

    def test_plan_hikes_for_origin_unknown_origin(self):
        """Should raise ValueError for unknown origin."""
        ctx = PlannerContext(
            feed=MagicMock(),
            trails=[],
            deadline=datetime.datetime(2026, 2, 3, 18, 0),
            deadline_secs=18 * 3600,
            router=MagicMock(),
        )
        query = HikeQuery(origin="Narnia", date=datetime.date(2026, 2, 3))
        with pytest.raises(ValueError, match="Unknown origin"):
            plan_hikes_for_origin(query, ctx)

    @patch("src.query.planner.get_deadline")
    @patch("src.query.planner.build_trail_access_points")
    @patch("src.query.planner.fetch_hiking_trails")
    @patch("src.query.planner.load_feed_for_date")
    @patch("src.query.planner.download_gtfs")
    @patch("src.query.planner.find_origin_stops")
    def test_plan_hikes_backward_compat(
        self, mock_find, mock_dl, mock_load, mock_fetch, mock_join, mock_deadline
    ):
        """plan_hikes() should still work as a single-origin wrapper."""
        mock_dl.return_value = "/fake/gtfs.zip"
        mock_feed = MagicMock()
        mock_feed.stops = pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])
        mock_load.return_value = mock_feed
        mock_fetch.return_value = []
        mock_join.return_value = []
        mock_deadline.return_value = datetime.datetime(2026, 2, 3, 18, 0)
        mock_find.return_value = []

        query = HikeQuery(origin="Rehovot", date=datetime.date(2026, 2, 3))
        result = plan_hikes(query)
        assert isinstance(result, list)

    def test_origin_header_formatter(self):
        """print_origin_header should not crash."""
        from src.output.cli_formatter import print_origin_header
        # Just ensure it doesn't raise
        print_origin_header("Rehovot", 5)
        print_origin_header("Jerusalem", 0)


# ═══════════════════════════════════════════════════════════════════════
# Pre-processed trail index (v0.3)
# ═══════════════════════════════════════════════════════════════════════


class TestTrailIndex:
    """Tests for loading trails from pre-processed JSON index."""

    def _make_index_file(self, tmp_path):
        """Create a minimal trail_index.json in tmp_path and return its path."""
        index = {
            "generated_at": "2026-02-03T12:00:00",
            "n_trails": 1,
            "trails": [
                {
                    "id": "osm:99999",
                    "name": "Test Trail",
                    "source": "osm",
                    "distance_km": 5.0,
                    "elevation_gain_m": 150.0,
                    "elevation_loss_m": 120.0,
                    "min_elevation_m": 200.0,
                    "max_elevation_m": 400.0,
                    "difficulty": "moderate",
                    "colors": ["red", "blue"],
                    "is_loop": False,
                    "recommended_seasons": ["spring", "autumn"],
                    "season_warnings": ["Flash flood danger"],
                    "elevation_profile": [200.0, 300.0, 400.0, 350.0, 250.0],
                    "geometry": [
                        [31.0, 34.8],
                        [31.05, 34.85],
                        [31.1, 34.9],
                    ],
                    "access_points": [
                        {
                            "stop_id": "s1",
                            "stop_name": "Bus Stop A",
                            "walk_distance_m": 500.0,
                            "trail_entry_lat": 31.0,
                            "trail_entry_lon": 34.8,
                            "trail_km_from_start": 0.0,
                        },
                        {
                            "stop_id": "s2",
                            "stop_name": "Bus Stop B",
                            "walk_distance_m": 300.0,
                            "trail_entry_lat": 31.1,
                            "trail_entry_lon": 34.9,
                            "trail_km_from_start": 4.5,
                        },
                    ],
                }
            ],
        }
        import json
        path = tmp_path / "trail_index.json"
        path.write_text(json.dumps(index), encoding="utf-8")
        return path

    def test_load_basic(self, tmp_path):
        """load_trail_index returns Trail objects from JSON."""
        path = self._make_index_file(tmp_path)
        trails = load_trail_index(path)
        assert len(trails) == 1
        t = trails[0]
        assert t.id == "osm:99999"
        assert t.name == "Test Trail"
        assert t.distance_km == 5.0
        assert t.is_loop is False

    def test_load_geometry(self, tmp_path):
        """Geometry should be reconstructed as a Shapely LineString."""
        path = self._make_index_file(tmp_path)
        trails = load_trail_index(path)
        geom = trails[0].geometry
        assert geom.geom_type == "LineString"
        coords = list(geom.coords)
        # Index stores [lat, lon]; Shapely stores (lon, lat)
        assert abs(coords[0][0] - 34.8) < 0.001  # lon
        assert abs(coords[0][1] - 31.0) < 0.001  # lat

    def test_load_access_points(self, tmp_path):
        """Access points should be reconstructed."""
        path = self._make_index_file(tmp_path)
        trails = load_trail_index(path)
        aps = trails[0].access_points
        assert len(aps) == 2
        assert aps[0].stop_id == "s1"
        assert aps[0].stop_name == "Bus Stop A"
        assert aps[0].walk_distance_m == 500.0
        assert aps[1].trail_km_from_start == 4.5

    def test_load_elevation_data(self, tmp_path):
        """Elevation fields should be populated from the index."""
        path = self._make_index_file(tmp_path)
        trails = load_trail_index(path)
        t = trails[0]
        assert t.elevation_gain_m == 150.0
        assert t.elevation_loss_m == 120.0
        assert t.min_elevation_m == 200.0
        assert t.max_elevation_m == 400.0
        assert len(t.elevation_profile) == 5

    def test_load_season_data(self, tmp_path):
        """Season fields should be populated."""
        path = self._make_index_file(tmp_path)
        trails = load_trail_index(path)
        t = trails[0]
        assert "spring" in t.recommended_seasons
        assert len(t.season_warnings) == 1

    def test_load_colors(self, tmp_path):
        """Trail colors should be preserved."""
        path = self._make_index_file(tmp_path)
        trails = load_trail_index(path)
        assert trails[0].colors == ["red", "blue"]


# ═══════════════════════════════════════════════════════════════════════
# SQLite transit database
# ═══════════════════════════════════════════════════════════════════════


def _make_transit_db(tmp_path):
    """Create a minimal transit SQLite database for testing."""
    import sqlite3

    db_path = tmp_path / "transit_test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE stops (
            stop_id TEXT PRIMARY KEY,
            stop_name TEXT,
            stop_lat REAL,
            stop_lon REAL
        );
        CREATE TABLE routes (
            route_id TEXT PRIMARY KEY,
            short_name TEXT,
            agency_name TEXT
        );
        CREATE TABLE trips (
            trip_id TEXT PRIMARY KEY,
            route_id TEXT
        );
        CREATE TABLE stop_times (
            trip_id TEXT,
            stop_id TEXT,
            stop_sequence INTEGER,
            arrival_secs INTEGER,
            departure_secs INTEGER
        );
        CREATE INDEX idx_st_stop_dep ON stop_times(stop_id, departure_secs);
        CREATE INDEX idx_st_trip_seq ON stop_times(trip_id, stop_sequence);
    """)

    # Same test data as the in-memory simple_feed fixture
    conn.executemany("INSERT INTO stops VALUES (?,?,?,?)", [
        ("A", "Origin", 31.8928, 34.8113),
        ("B", "Mid", 31.85, 34.82),
        ("C", "Trail", 31.80, 34.85),
        ("D", "Other", 32.00, 35.00),
    ])
    conn.executemany("INSERT INTO routes VALUES (?,?,?)", [
        ("r1", "100", "Egged"),
        ("r2", "200", "Egged"),
    ])
    conn.executemany("INSERT INTO trips VALUES (?,?)", [
        ("t1", "r1"),
        ("t2", "r2"),
    ])
    # Trip t1: A→B→C (outbound to trail), Trip t2: C→B→A (return)
    conn.executemany("INSERT INTO stop_times VALUES (?,?,?,?,?)", [
        ("t1", "A", 1, 25200, 25200),    # 07:00
        ("t1", "B", 2, 27000, 27000),    # 07:30
        ("t1", "C", 3, 28800, 28800),    # 08:00
        ("t2", "C", 1, 54000, 54000),    # 15:00
        ("t2", "B", 2, 55800, 55800),    # 15:30
        ("t2", "A", 3, 57600, 57600),    # 16:00
    ])
    conn.commit()
    conn.close()
    return db_path


class TestTransitRouterDB:
    """Tests for the SQLite-backed transit router."""

    def test_open_db(self, tmp_path):
        db_path = _make_transit_db(tmp_path)
        router = TransitRouterDB(db_path, datetime.date(2026, 2, 3))
        assert router.db_path == db_path
        router.close()

    def test_find_outbound_direct(self, tmp_path):
        db_path = _make_transit_db(tmp_path)
        router = TransitRouterDB(db_path, datetime.date(2026, 2, 3))
        legs = router.find_outbound(
            origin_stops=["A"],
            dest_stops={"C"},
            earliest_departure_secs=6 * 3600,
        )
        assert legs is not None
        assert len(legs) == 1
        assert legs[0].line == "100"
        assert legs[0].from_stop_id == "A"
        assert legs[0].to_stop_id == "C"
        router.close()

    def test_find_return_direct(self, tmp_path):
        db_path = _make_transit_db(tmp_path)
        router = TransitRouterDB(db_path, datetime.date(2026, 2, 3))
        legs = router.find_return(
            trail_stops=["C"],
            origin_stops={"A"},
            deadline_secs=18 * 3600,
        )
        assert legs is not None
        assert len(legs) == 1
        assert legs[0].line == "200"
        assert legs[0].from_stop_id == "C"
        assert legs[0].to_stop_id == "A"
        router.close()

    def test_no_route_found(self, tmp_path):
        db_path = _make_transit_db(tmp_path)
        router = TransitRouterDB(db_path, datetime.date(2026, 2, 3))
        legs = router.find_outbound(
            origin_stops=["D"],
            dest_stops={"C"},
            earliest_departure_secs=6 * 3600,
        )
        assert legs is None
        router.close()

    def test_return_respects_deadline(self, tmp_path):
        db_path = _make_transit_db(tmp_path)
        router = TransitRouterDB(db_path, datetime.date(2026, 2, 3))
        legs = router.find_return(
            trail_stops=["C"],
            origin_stops={"A"},
            deadline_secs=14 * 3600,
        )
        assert legs is None
        router.close()


class TestTransitRouterDBWithTransfer:
    """Test 1-transfer routing via SQLite."""

    def test_outbound_with_transfer(self, tmp_path):
        """Route A→B (trip t1) + B→C (trip t3) with transfer at B."""
        import sqlite3

        db_path = tmp_path / "transfer.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE stops (stop_id TEXT PRIMARY KEY, stop_name TEXT, stop_lat REAL, stop_lon REAL);
            CREATE TABLE routes (route_id TEXT PRIMARY KEY, short_name TEXT, agency_name TEXT);
            CREATE TABLE trips (trip_id TEXT PRIMARY KEY, route_id TEXT);
            CREATE TABLE stop_times (trip_id TEXT, stop_id TEXT, stop_sequence INTEGER,
                                     arrival_secs INTEGER, departure_secs INTEGER);
            CREATE INDEX idx_st_stop_dep ON stop_times(stop_id, departure_secs);
            CREATE INDEX idx_st_trip_seq ON stop_times(trip_id, stop_sequence);
        """)
        conn.executemany("INSERT INTO stops VALUES (?,?,?,?)", [
            ("A", "Origin", 0, 0), ("B", "Transfer", 0, 0), ("C", "Trail", 0, 0),
        ])
        conn.executemany("INSERT INTO routes VALUES (?,?,?)", [
            ("r1", "100", "Egged"), ("r3", "300", "Dan"),
        ])
        conn.executemany("INSERT INTO trips VALUES (?,?)", [
            ("t1", "r1"), ("t3", "r3"),
        ])
        conn.executemany("INSERT INTO stop_times VALUES (?,?,?,?,?)", [
            ("t1", "A", 1, 25200, 25200),   # 07:00
            ("t1", "B", 2, 27000, 27000),   # 07:30
            ("t3", "B", 1, 27120, 27120),   # 07:32 (transfer at B)
            ("t3", "C", 2, 28800, 28800),   # 08:00
        ])
        conn.commit()
        conn.close()

        router = TransitRouterDB(db_path, datetime.date(2026, 2, 3))
        legs = router.find_outbound(
            origin_stops=["A"],
            dest_stops={"C"},
            earliest_departure_secs=6 * 3600,
        )
        assert legs is not None
        assert len(legs) == 2
        assert legs[0].line == "100"
        assert legs[1].line == "300"
        router.close()


class TestBuildTransitDB:
    """Tests for the GTFS→SQLite streaming builder."""

    def test_gtfs_time_to_seconds(self):
        assert _gtfs_time_to_seconds("07:30:00") == 27000
        assert _gtfs_time_to_seconds("25:00:00") == 90000  # past midnight

    def test_build_from_zip(self, tmp_path):
        """Build a transit DB from a minimal GTFS zip."""
        import zipfile

        # Create a minimal GTFS zip
        gtfs_zip = tmp_path / "test.zip"
        with zipfile.ZipFile(gtfs_zip, "w") as zf:
            zf.writestr("agency.txt", "agency_id,agency_name\nag1,Egged\n")
            zf.writestr(
                "calendar.txt",
                "service_id,start_date,end_date,"
                "monday,tuesday,wednesday,thursday,friday,saturday,sunday\n"
                "svc1,20260101,20261231,1,1,1,1,1,0,0\n",
            )
            zf.writestr("calendar_dates.txt", "service_id,date,exception_type\n")
            zf.writestr(
                "routes.txt",
                "route_id,agency_id,route_short_name\nr1,ag1,100\n",
            )
            zf.writestr(
                "trips.txt",
                "trip_id,route_id,service_id\nt1,r1,svc1\n",
            )
            zf.writestr(
                "stops.txt",
                "stop_id,stop_name,stop_lat,stop_lon\nA,Origin,31.89,34.81\n"
                "B,Trail,31.80,34.85\n",
            )
            zf.writestr(
                "stop_times.txt",
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "t1,A,1,07:00:00,07:00:00\n"
                "t1,B,2,08:00:00,08:00:00\n",
            )

        # Point GTFS_DIR to tmp_path so the DB lands there
        from unittest.mock import patch as mpatch

        with mpatch("src.ingest.gtfs.GTFS_DIR", tmp_path):
            db_path = build_transit_db(gtfs_zip, datetime.date(2026, 2, 3))

        assert db_path.exists()

        # Verify contents
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        trips = conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
        stops = conn.execute("SELECT COUNT(*) FROM stops").fetchone()[0]
        st = conn.execute("SELECT COUNT(*) FROM stop_times").fetchone()[0]
        conn.close()

        assert trips == 1
        assert stops == 2
        assert st == 2

    def test_inactive_trips_excluded(self, tmp_path):
        """Trips for inactive services should not appear in the DB."""
        import zipfile

        gtfs_zip = tmp_path / "test.zip"
        with zipfile.ZipFile(gtfs_zip, "w") as zf:
            zf.writestr("agency.txt", "agency_id,agency_name\nag1,Egged\n")
            # svc1 runs Mon-Fri, svc_inactive runs Sat only
            zf.writestr(
                "calendar.txt",
                "service_id,start_date,end_date,"
                "monday,tuesday,wednesday,thursday,friday,saturday,sunday\n"
                "svc1,20260101,20261231,1,1,1,1,1,0,0\n"
                "svc_sat,20260101,20261231,0,0,0,0,0,1,0\n",
            )
            zf.writestr("calendar_dates.txt", "service_id,date,exception_type\n")
            zf.writestr(
                "routes.txt",
                "route_id,agency_id,route_short_name\nr1,ag1,100\nr2,ag1,200\n",
            )
            zf.writestr(
                "trips.txt",
                "trip_id,route_id,service_id\n"
                "t_active,r1,svc1\n"
                "t_sat,r2,svc_sat\n",
            )
            zf.writestr(
                "stops.txt",
                "stop_id,stop_name,stop_lat,stop_lon\nA,Stop A,31.89,34.81\n",
            )
            zf.writestr(
                "stop_times.txt",
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "t_active,A,1,07:00:00,07:00:00\n"
                "t_sat,A,1,09:00:00,09:00:00\n",
            )

        # Query for a Tuesday — only svc1 active
        from unittest.mock import patch as mpatch

        with mpatch("src.ingest.gtfs.GTFS_DIR", tmp_path):
            db_path = build_transit_db(gtfs_zip, datetime.date(2026, 2, 3))

        import sqlite3

        conn = sqlite3.connect(str(db_path))
        trips = conn.execute("SELECT trip_id FROM trips").fetchall()
        conn.close()

        trip_ids = [r[0] for r in trips]
        assert "t_active" in trip_ids
        assert "t_sat" not in trip_ids


class TestFindOriginStopsDB:
    """Tests for SQLite-based origin stop search."""

    def test_finds_nearby_stop(self, tmp_path):
        db_path = _make_transit_db(tmp_path)
        # Rehovot coordinates — stop A is at (31.8928, 34.8113)
        stops = find_origin_stops_db(db_path, 31.8928, 34.8113, radius_m=500)
        assert "A" in stops

    def test_excludes_distant_stop(self, tmp_path):
        db_path = _make_transit_db(tmp_path)
        # Query far from all stops
        stops = find_origin_stops_db(db_path, 33.0, 36.0, radius_m=500)
        assert len(stops) == 0

    def test_sorted_by_distance(self, tmp_path):
        db_path = _make_transit_db(tmp_path)
        # Use a wide radius to get multiple stops
        stops = find_origin_stops_db(db_path, 31.87, 34.82, radius_m=10000)
        assert len(stops) >= 2
        # First stop should be closest
        assert stops[0] in ("A", "B")
