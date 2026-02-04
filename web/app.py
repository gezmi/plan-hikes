"""FastAPI web app for the Israel Hiking Transit Planner."""

from __future__ import annotations

import datetime
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import CITY_COORDINATES
from src.models import HikePlan, HikeQuery
from src.query.planner import (
    TRAIL_INDEX_PATH,
    PlannerContext,
    _resolve_origin,
    plan_hikes_for_origin,
    prepare_data,
    prepare_data_from_index,
)

logger = logging.getLogger(__name__)

# ── Shared planner context (loaded once at startup) ─────────────────
_ctx: PlannerContext | None = None
_ctx_date: datetime.date | None = None


def _get_context(date: datetime.date) -> PlannerContext:
    """Return (or reload) the shared PlannerContext for the given date.

    Prefers the pre-processed trail index when available (fast startup).
    Falls back to full prepare_data() if the index doesn't exist.
    """
    global _ctx, _ctx_date
    if _ctx is not None and _ctx_date == date:
        return _ctx
    query = HikeQuery(origin="rehovot", date=date)
    if TRAIL_INDEX_PATH.exists():
        logger.info("Loading from pre-processed index")
        _ctx = prepare_data_from_index(query)
    else:
        logger.info("No pre-processed index; running full data pipeline")
        _ctx = prepare_data(query)
    _ctx_date = date
    logger.info("PlannerContext loaded for %s", date)
    return _ctx


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load data on startup."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger.info("Starting Israel Hiking Transit Planner web app")
    yield


app = FastAPI(title="Israel Hiking Transit Planner", version="0.3.0", lifespan=lifespan)

# Serve static files (HTML/JS/CSS)
app.mount("/static", StaticFiles(directory="web/static"), name="static")


# ── Pydantic request/response models ────────────────────────────────


class PlanRequest(BaseModel):
    origins: list[str]
    date: str  # YYYY-MM-DD
    max_results: int = 20
    min_hike_hours: float = 1.0
    max_walk_m: int = 1000
    colors: Optional[list[str]] = None
    min_distance_km: Optional[float] = None
    max_distance_km: Optional[float] = None
    loop_only: bool = False
    linear_only: bool = False
    max_elevation_gain_m: Optional[float] = None


class BusLegOut(BaseModel):
    line: str
    operator: str
    from_stop: str
    to_stop: str
    departure: str
    arrival: str
    duration_min: int


class HikingOut(BaseModel):
    start: str
    end: str
    hours: float
    distance_km: float
    is_through_hike: bool
    entry_stop: str
    exit_stop: Optional[str]
    walk_to_trail_m: float
    walk_from_trail_m: float


class TrailOut(BaseModel):
    name: str
    id: str
    distance_km: float
    elevation_gain_m: float
    elevation_loss_m: float
    min_elevation_m: float
    max_elevation_m: float
    colors: list[str]
    is_loop: bool
    geometry: list[list[float]]  # [[lat, lon], ...]


class LinksOut(BaseModel):
    osm: Optional[str] = None
    google_maps: str
    israel_hiking: str
    directions: Optional[str] = None
    exit_point: Optional[str] = None


class HikePlanOut(BaseModel):
    rank: int
    trail: TrailOut
    hiking_ratio: float
    total_hours: float
    outbound: list[BusLegOut]
    hiking: HikingOut
    return_legs: list[BusLegOut]
    links: LinksOut
    warnings: list[str]
    deadline: str
    departure: str
    arrival: str
    access_point_lat: float
    access_point_lon: float
    elevation_profile: list[float]


class OriginResultOut(BaseModel):
    origin: str
    plans: list[HikePlanOut]


class PlanResponse(BaseModel):
    deadline: str
    results: list[OriginResultOut]


# ── Serialization helpers ────────────────────────────────────────────


def _serialize_leg(leg) -> BusLegOut:
    duration = int((leg.arrival - leg.departure).total_seconds() / 60)
    return BusLegOut(
        line=leg.line,
        operator=leg.operator,
        from_stop=leg.from_stop_name,
        to_stop=leg.to_stop_name,
        departure=leg.departure.strftime("%H:%M"),
        arrival=leg.arrival.strftime("%H:%M"),
        duration_min=duration,
    )


def _serialize_plan(rank: int, plan: HikePlan, origin_lat: float | None, origin_lon: float | None) -> HikePlanOut:
    trail = plan.trail
    seg = plan.hike_segment
    ap = plan.access_point

    # Trail geometry as [[lat, lon], ...]
    coords = list(trail.geometry.coords)
    geometry = [[lat, lon] for lon, lat in coords]

    # Links
    osm_url = None
    if trail.id.startswith("osm:"):
        rel_id = trail.id.split(":", 1)[1]
        osm_url = f"https://www.openstreetmap.org/relation/{rel_id}"

    google_maps = f"https://www.google.com/maps?q={ap.trail_entry_lat:.6f},{ap.trail_entry_lon:.6f}"
    israel_hiking = f"https://israelhiking.osm.org.il/#/map/15/{ap.trail_entry_lat:.5f}/{ap.trail_entry_lon:.5f}"

    directions = None
    if origin_lat is not None and origin_lon is not None:
        directions = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&origin={origin_lat:.6f},{origin_lon:.6f}"
            f"&destination={ap.trail_entry_lat:.6f},{ap.trail_entry_lon:.6f}"
            f"&travelmode=transit"
        )

    exit_point = None
    if seg.is_through_hike and plan.exit_access_point:
        eap = plan.exit_access_point
        exit_point = f"https://www.google.com/maps?q={eap.trail_entry_lat:.6f},{eap.trail_entry_lon:.6f}"

    return HikePlanOut(
        rank=rank,
        trail=TrailOut(
            name=trail.name,
            id=trail.id,
            distance_km=trail.distance_km,
            elevation_gain_m=trail.elevation_gain_m,
            elevation_loss_m=trail.elevation_loss_m,
            min_elevation_m=trail.min_elevation_m,
            max_elevation_m=trail.max_elevation_m,
            colors=trail.colors,
            is_loop=trail.is_loop,
            geometry=geometry,
        ),
        hiking_ratio=round(plan.hiking_ratio, 3),
        total_hours=round(plan.total_hours, 1),
        outbound=[_serialize_leg(l) for l in plan.outbound_legs],
        hiking=HikingOut(
            start=seg.hike_start.strftime("%H:%M"),
            end=seg.hike_end.strftime("%H:%M"),
            hours=round(seg.hiking_hours, 1),
            distance_km=round(seg.estimated_distance_km, 1),
            is_through_hike=seg.is_through_hike,
            entry_stop=seg.entry_stop_name,
            exit_stop=seg.exit_stop_name,
            walk_to_trail_m=round(seg.walk_to_trail_m),
            walk_from_trail_m=round(seg.walk_from_trail_m),
        ),
        return_legs=[_serialize_leg(l) for l in plan.return_legs],
        links=LinksOut(
            osm=osm_url,
            google_maps=google_maps,
            israel_hiking=israel_hiking,
            directions=directions,
            exit_point=exit_point,
        ),
        warnings=plan.warnings,
        deadline=plan.deadline.strftime("%H:%M"),
        departure=plan.departure_from_origin.strftime("%H:%M"),
        arrival=plan.arrival_at_origin.strftime("%H:%M"),
        access_point_lat=ap.trail_entry_lat,
        access_point_lon=ap.trail_entry_lon,
        elevation_profile=trail.elevation_profile,
    )


# ── API endpoints ────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse("web/static/index.html")


@app.get("/api/cities")
async def get_cities():
    """Return list of known origin cities."""
    return {
        "cities": [
            {"name": city.title(), "lat": coords[0], "lon": coords[1]}
            for city, coords in sorted(CITY_COORDINATES.items())
        ]
    }


@app.post("/api/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):
    """Plan hikes for one or more origins."""
    # Parse date
    try:
        date = datetime.date.fromisoformat(req.date)
    except ValueError:
        raise HTTPException(400, f"Invalid date: {req.date}")

    # Validate origins
    for o in req.origins:
        try:
            _resolve_origin(o)
        except ValueError as e:
            raise HTTPException(400, str(e))

    # Load/reuse context
    try:
        ctx = _get_context(date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Plan for each origin
    results: list[OriginResultOut] = []
    deadline_str = ctx.deadline.strftime("%H:%M")

    for origin in req.origins:
        query = HikeQuery(
            origin=origin,
            date=date,
            max_results=req.max_results,
            min_hiking_hours=req.min_hike_hours,
            max_walk_to_trail_m=req.max_walk_m,
            colors=req.colors,
            min_distance_km=req.min_distance_km,
            max_distance_km=req.max_distance_km,
            loop_only=req.loop_only,
            linear_only=req.linear_only,
            max_elevation_gain_m=req.max_elevation_gain_m,
        )

        plans = plan_hikes_for_origin(query, ctx)

        origin_coords = CITY_COORDINATES.get(origin.strip().lower())
        o_lat = origin_coords[0] if origin_coords else None
        o_lon = origin_coords[1] if origin_coords else None

        plan_outs = [
            _serialize_plan(i, p, o_lat, o_lon)
            for i, p in enumerate(plans, 1)
        ]

        results.append(OriginResultOut(origin=origin, plans=plan_outs))

    return PlanResponse(deadline=deadline_str, results=results)
