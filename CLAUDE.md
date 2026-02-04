# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Israel Hiking Transit Planner — a CLI tool (eventually web app) that finds transit-accessible hiking trails in Israel. Given an origin, date, and preferences, it returns ranked hiking options with complete bus plans (outbound + return), trail info, and time budget breakdowns. Accounts for Shabbat restrictions and safety margins.

Python project. CLI entry point: `src/cli.py`. Web API (v0.3+): `web/app.py`.

## Build & Run Commands

```bash
pip install -r requirements.txt
pip install -e .

# CLI usage
python -m src.cli --origin Rehovot --date 2026-02-06

# Run tests
pytest tests/
pytest tests/test_planner.py          # single test file
pytest tests/test_planner.py::test_x  # single test function

# Web app (v0.3+)
uvicorn web.app:app --reload
```

## Data Pipeline (Weekly/Monthly)

Before the query engine works, pre-processed data must exist in `data/`. The pipeline:

1. **GTFS download** (weekly) — FTP from `ftp://gtfs.mot.gov.il/israel-public-transportation.zip` → parse into `data/gtfs/`. ~30K stops, 60-day rolling window. Libraries: `gtfs-kit` or `partridge`.
2. **OSM Overpass query** (monthly) — fetch all `relation["route"="hiking"]` in Israel → `data/trails/osm_hiking_routes.geojson`. Library: `osmnx` or direct `requests`.
3. **hike-israel.com scrape** (monthly) — curated hikes + free GPX files → `data/trails/hike_israel/`. Rate limit: 1 req/2 sec. Library: `beautifulsoup4`, `gpxpy`.
4. **Hebcal Shabbat times** (cache 60 days) → `data/shabbat_times.json`.
5. **Spatial join** — R-tree index of all bus stops, buffer each trail by ~1km, find nearby stops. Result: each trail gets a list of `TrailAccessPoint`s (stop + walk distance + trail entry point).

Storage: SQLite for the pre-processed index; GeoJSON for trail geometries; flat files for cached GTFS/GPX.

## Architecture

### Source Layout

```
src/
├── cli.py                      # CLI entry point (click/typer + rich output)
├── config.py                   # Constants, defaults
├── ingest/                     # Data download & parsing
│   ├── gtfs.py                 # GTFS download + parse
│   ├── osm_trails.py           # Overpass query + parse
│   ├── hike_israel_scraper.py  # Scrape hike-israel.com
│   ├── shabbat.py              # Hebcal API client
│   └── elevation.py            # SRTM elevation data
├── index/                      # Pre-processing / spatial indexing
│   ├── spatial_join.py         # Trail ↔ stop proximity (R-tree)
│   ├── schedule_index.py       # Pre-index schedules by day/stop
│   └── trail_merge.py          # Merge OSM + hike-israel trail data
├── query/                      # Query engine
│   ├── planner.py              # Main planning logic (orchestrates everything)
│   ├── transit_router.py       # GTFS route finding (outbound + return)
│   ├── hike_scorer.py          # Ranking/scoring (hiking_ratio, transfers, etc.)
│   └── time_budget.py          # Time calculations + Shabbat deadlines
└── output/                     # Result formatting
    ├── cli_formatter.py        # Rich CLI tables/colors
    └── json_formatter.py       # JSON for web API
```

### Query Engine Flow

The planner in `query/planner.py` follows this logic:

1. **Determine deadline** — Friday: `candle_lighting - safety_margin` (default 2h). Weekday: `latest_return` (default 18:00). Saturday: check for Shabbat service exceptions (Haifa, Nazareth, East Jerusalem, Tel Aviv).
2. **Filter trails** by area, distance, difficulty, season, and whether they have bus-accessible access points.
3. **For each trail, work backwards from deadline** — find latest viable return bus from each access point toward origin (via `transit_router.py`).
4. **Work forwards from morning** — find earliest outbound bus from origin to entry stop.
5. **Compute hiking window** — `hike_end = latest_return_departure - walk_time`. Verify window >= estimated hike time (Naismith's rule: `distance_km/4 + elevation_gain_m/600` hours).
6. **Score and rank** — primary metric is `hiking_ratio = hiking_time / total_trip_time`. Secondary: fewer transfers, shorter walks, preference match.

Through-hikes: for trails with 2+ access points, generate entry/exit pairs 3–20 km apart along the trail where both points have bus service.

## Key Data Structures

Defined as dataclasses in the source:

- `Trail` — id, name, source (osm/hike_israel/both), geometry, distance, elevation, difficulty, trail colors (ITC marking system), season, is_loop, and list of `TrailAccessPoint`s.
- `TrailAccessPoint` — stop_id, walk_distance_m, trail_entry_point (lat/lon), trail_km_from_start.
- `BusLeg` — line, operator, from/to stop, departure/arrival times.
- `HikePlan` — combines a Trail + outbound legs + hike segment + return legs + safety info + score.
- `HikeQuery` — origin, date, max_transfers (default 2), safety_margin_hours (default 2.0), max_walk_to_trail_m (default 1000), hike_type, distance range, difficulty, time constraints, area filter, sort_by.

## Shabbat Transit Rules

- **Friday**: buses end 2–3h before Shabbat. GTFS `calendar.txt` already encodes reduced Friday schedules. Apply safety margin (default 2h before candle lighting).
- **Saturday**: no service except — Haifa (limited), Nazareth (limited), East Jerusalem (Arab operators), Tel Aviv (free weekend buses Fri 18:00–02:00, Sat 09:00–17:00).
- **Sherut (shared taxis)**: run 24/7 on major routes but are NOT in GTFS data. Manual database for v1.0.
- Shabbat times from Hebcal API: `https://www.hebcal.com/shabbat`.

## Data Sources

| Source | Type | Update | Notes |
|--------|------|--------|-------|
| GTFS (MoT FTP) | Transit schedules | Weekly | Primary transit backbone. ~30K stops, 60-day window |
| OSM Overpass | Trail geometry | Monthly | All `route=hiking` relations. ITC color markings in `osmc:symbol` tag |
| hike-israel.com | Curated hikes | Monthly | ~30–50 hikes with English descriptions + free GPX. Rate limit scraping. Attribute and link back. Personal use only per ToS |
| Hebcal | Shabbat times | Cache 60 days | Candle lighting + havdalah by location |
| Israel Hiking Map API | Trail enrichment | As needed | Swagger at `israelhiking.osm.org.il/swagger/`. GraphHopper routing backend |
| SRTM GL3 | Elevation | Once | NASA 90m resolution. Read with `rasterio` |

## hike-israel.com Scraper

Regions to crawl: `hikes-in-the-central-negev`, `hikes-in-the-galilee`, `hikes-around-jerusalem`, `hiking-masada-deadsea`, `hikes-near-eilat`, `golan`, `wet-summer-hikes`, `hikes-near-tel-aviv-israel-by-foot`.

Per-hike page: extract metrics table (distance, climb, level, season), start point GPS (from Google Maps links), GPX download URL (pattern: `wp-content/uploads/.../Israel_by_Foot_{name}_GPS.gpx?download=1`), bus mentions in description text, trail color markings.

GPS coordinates come as Google Maps links (`goo.gl/maps/...` needing redirect follow, or direct lat/lon in text).

## ITC Trail Color Markings

Israel uses a distinctive 3-stripe system (two white stripes with a colored stripe between). Colors: Red, Blue, Green, Black, Orange, Purple. These are tagged in OSM via `osmc:symbol` and mentioned in hike-israel.com descriptions.

## Key Libraries

`gtfs-kit`/`partridge` (GTFS), `geopandas` (geospatial dataframes), `shapely` (geometry), `rtree` (spatial index), `osmnx` (OSM), `rasterio` (SRTM), `gpxpy` (GPX parsing), `beautifulsoup4` (scraping), `click`/`typer` (CLI), `rich` (CLI output), `fastapi` (web API, v0.3+), `leaflet.js` (map UI, v0.3+).

## Development Phases

- **v0.1**: CLI proof of concept — answer "What can I hike from Rehovot this Friday?"
- **v0.2**: Through-hike support, filters, multiple origins, elevation profiles, season awareness
- **v0.3**: FastAPI + Leaflet web GUI with map
- **v1.0**: Real-time SIRI data, user accounts, push notifications, Hebrew support, offline maps
