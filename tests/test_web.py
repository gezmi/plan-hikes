"""Tests for the FastAPI web app."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from web.app import app, _serialize_plan
from src.models import (
    BusLeg,
    HikePlan,
    HikeSegment,
    Trail,
    TrailAccessPoint,
)
from src.query.planner import PlannerContext
from shapely.geometry import LineString


client = TestClient(app)


class TestCitiesEndpoint:
    def test_get_cities(self):
        res = client.get("/api/cities")
        assert res.status_code == 200
        data = res.json()
        assert "cities" in data
        assert len(data["cities"]) > 0
        # Check structure
        city = data["cities"][0]
        assert "name" in city
        assert "lat" in city
        assert "lon" in city

    def test_cities_sorted(self):
        res = client.get("/api/cities")
        names = [c["name"] for c in res.json()["cities"]]
        assert names == sorted(names)


class TestPlanEndpoint:
    def test_invalid_date(self):
        res = client.post("/api/plan", json={
            "origins": ["Rehovot"],
            "date": "not-a-date",
        })
        assert res.status_code == 400

    def test_unknown_origin(self):
        res = client.post("/api/plan", json={
            "origins": ["Narnia"],
            "date": "2026-02-06",
        })
        assert res.status_code == 400
        assert "Unknown origin" in res.json()["detail"]

    @patch("web.app._get_context")
    @patch("web.app.plan_hikes_for_origin")
    def test_plan_returns_results(self, mock_plan, mock_ctx):
        """Plan endpoint should return structured results."""
        mock_ctx.return_value = PlannerContext(
            feed=MagicMock(),
            trails=[],
            deadline=datetime.datetime(2026, 2, 3, 18, 0),
            deadline_secs=18 * 3600,
            router=MagicMock(),
        )
        mock_plan.return_value = []

        res = client.post("/api/plan", json={
            "origins": ["Rehovot"],
            "date": "2026-02-03",
        })
        assert res.status_code == 200
        data = res.json()
        assert "deadline" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["origin"] == "Rehovot"

    @patch("web.app._get_context")
    @patch("web.app.plan_hikes_for_origin")
    def test_multiple_origins(self, mock_plan, mock_ctx):
        """Should return results for each origin."""
        mock_ctx.return_value = PlannerContext(
            feed=MagicMock(),
            trails=[],
            deadline=datetime.datetime(2026, 2, 3, 18, 0),
            deadline_secs=18 * 3600,
            router=MagicMock(),
        )
        mock_plan.return_value = []

        res = client.post("/api/plan", json={
            "origins": ["Rehovot", "Jerusalem"],
            "date": "2026-02-03",
        })
        assert res.status_code == 200
        data = res.json()
        assert len(data["results"]) == 2
        origins = [r["origin"] for r in data["results"]]
        assert "Rehovot" in origins
        assert "Jerusalem" in origins


class TestSerializePlan:
    def _make_plan(self):
        trail = Trail(
            id="osm:12345",
            name="Test Trail",
            source="osm",
            geometry=LineString([(34.8, 31.0), (34.9, 31.1)]),
            distance_km=8.0,
            elevation_gain_m=200,
            difficulty="moderate",
            colors=["red"],
            is_loop=False,
            elevation_loss_m=150,
            min_elevation_m=100,
            max_elevation_m=400,
            elevation_profile=[100, 200, 300, 400, 300],
        )
        ap = TrailAccessPoint("s1", "Entry Stop", 500, 31.0, 34.8, 0.0)
        seg = HikeSegment(
            trail_name="Test Trail",
            entry_stop_name="Entry Stop",
            walk_to_trail_m=500,
            hike_start=datetime.datetime(2026, 2, 3, 9, 0),
            hike_end=datetime.datetime(2026, 2, 3, 14, 0),
            hiking_hours=5.0,
            estimated_distance_km=8.0,
            is_loop=False,
            colors=["red"],
        )
        outbound = BusLeg(
            line="100", operator="Egged",
            from_stop_id="o1", from_stop_name="Origin",
            to_stop_id="s1", to_stop_name="Entry Stop",
            departure=datetime.datetime(2026, 2, 3, 7, 0),
            arrival=datetime.datetime(2026, 2, 3, 8, 30),
        )
        return_leg = BusLeg(
            line="200", operator="Egged",
            from_stop_id="s1", from_stop_name="Entry Stop",
            to_stop_id="o1", to_stop_name="Origin",
            departure=datetime.datetime(2026, 2, 3, 14, 30),
            arrival=datetime.datetime(2026, 2, 3, 16, 0),
        )
        return HikePlan(
            trail=trail,
            access_point=ap,
            outbound_legs=[outbound],
            hike_segment=seg,
            return_legs=[return_leg],
            departure_from_origin=datetime.datetime(2026, 2, 3, 7, 0),
            arrival_at_origin=datetime.datetime(2026, 2, 3, 16, 0),
            hiking_ratio=0.556,
            deadline=datetime.datetime(2026, 2, 3, 18, 0),
            total_hours=9.0,
        )

    def test_serialize_basic(self):
        plan = self._make_plan()
        out = _serialize_plan(1, plan, 31.89, 34.81)
        assert out.rank == 1
        assert out.trail.name == "Test Trail"
        assert out.trail.id == "osm:12345"
        assert out.hiking_ratio == 0.556
        assert len(out.outbound) == 1
        assert len(out.return_legs) == 1

    def test_serialize_links(self):
        plan = self._make_plan()
        out = _serialize_plan(1, plan, 31.89, 34.81)
        assert out.links.osm is not None
        assert "12345" in out.links.osm
        assert out.links.directions is not None
        assert out.links.google_maps is not None

    def test_serialize_geometry(self):
        plan = self._make_plan()
        out = _serialize_plan(1, plan, None, None)
        # Geometry should be [[lat, lon], ...]
        assert len(out.trail.geometry) == 2
        # First coord: lon=34.8, lat=31.0 -> [31.0, 34.8]
        assert out.trail.geometry[0] == [31.0, 34.8]

    def test_serialize_elevation_profile(self):
        plan = self._make_plan()
        out = _serialize_plan(1, plan, None, None)
        assert out.elevation_profile == [100, 200, 300, 400, 300]


class TestStaticFiles:
    def test_index_html(self):
        res = client.get("/")
        assert res.status_code == 200
        assert "Israel Hiking" in res.text

    def test_css(self):
        res = client.get("/static/style.css")
        assert res.status_code == 200

    def test_js(self):
        res = client.get("/static/app.js")
        assert res.status_code == 200
