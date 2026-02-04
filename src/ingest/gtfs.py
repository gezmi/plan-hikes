"""GTFS data download and parsing for Israel public transit.

Downloads the Israeli GTFS feed from the Ministry of Transport server,
loads it with partridge, and provides helpers for filtering by date and
finding nearby stops.
"""

from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path

import pandas as pd
import partridge as ptg
import requests

from src.config import (
    GTFS_CACHE_DAYS,
    GTFS_DIR,
    GTFS_FTP_FILE,
    GTFS_FTP_HOST,
    GTFS_HTTPS_URL,
    GTFS_ZIP_PATH,
    STOP_SEARCH_RADIUS_M,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in **meters** between two points.

    Uses the Haversine formula with the math standard library only.
    """
    R = 6_371_000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# ---------------------------------------------------------------------------
# GTFS download
# ---------------------------------------------------------------------------

def download_gtfs() -> Path:
    """Download the Israeli GTFS zip via HTTPS (preferred) or FTP fallback.

    The file is saved to ``data/gtfs/israel-public-transportation.zip``.
    If a copy already exists and is less than ``GTFS_CACHE_DAYS`` (7) days
    old the download is skipped.

    Returns
    -------
    Path
        Absolute path to the downloaded zip file.
    """
    GTFS_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache freshness
    if GTFS_ZIP_PATH.exists():
        age_seconds = datetime.datetime.now().timestamp() - GTFS_ZIP_PATH.stat().st_mtime
        age_days = age_seconds / 86_400
        if age_days < GTFS_CACHE_DAYS:
            logger.info(
                "GTFS zip is %.1f days old (< %d); skipping download.",
                age_days,
                GTFS_CACHE_DAYS,
            )
            return GTFS_ZIP_PATH
        logger.info(
            "GTFS zip is %.1f days old (>= %d); re-downloading.",
            age_days,
            GTFS_CACHE_DAYS,
        )

    logger.info("Downloading GTFS from %s ...", GTFS_HTTPS_URL)
    print(f"Downloading GTFS from {GTFS_HTTPS_URL} ...")

    # The MoT server has a misconfigured SSL certificate — verify=False
    # is required for programmatic access.  The data itself is public.
    resp = requests.get(GTFS_HTTPS_URL, stream=True, timeout=300, verify=False)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    if total:
        print(f"File size: {total / 1_048_576:.1f} MB")

    tmp_path = GTFS_ZIP_PATH.with_suffix(".zip.part")
    downloaded = 0
    last_pct = -1

    try:
        with open(tmp_path, "wb") as f_out:
            for chunk in resp.iter_content(chunk_size=262_144):
                f_out.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct >= last_pct + 5:
                        print(
                            f"  {downloaded / 1_048_576:.1f} / "
                            f"{total / 1_048_576:.1f} MB ({pct}%)"
                        )
                        last_pct = pct
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    tmp_path.rename(GTFS_ZIP_PATH)
    print(f"Download complete: {GTFS_ZIP_PATH}  ({downloaded / 1_048_576:.1f} MB)")
    logger.info("GTFS download complete: %s", GTFS_ZIP_PATH)
    return GTFS_ZIP_PATH


# ---------------------------------------------------------------------------
# Service-ID helpers
# ---------------------------------------------------------------------------

def get_active_service_ids(feed, date: datetime.date) -> set[str]:
    """Return the set of GTFS service_ids active on *date*.

    Checks both ``calendar.txt`` (regular schedules) and
    ``calendar_dates.txt`` (exceptions).  Handles feeds that only supply
    one of the two files.

    Parameters
    ----------
    feed
        A partridge raw feed (has ``.calendar`` and ``.calendar_dates``
        DataFrames, either of which may be empty).
    date : datetime.date
        The query date.

    Returns
    -------
    set[str]
        Active service IDs.
    """
    date_str = date.strftime("%Y%m%d")
    date_int = int(date_str)  # partridge may parse dates as int

    # Day-of-week column name: monday=0 ... sunday=6
    day_name = date.strftime("%A").lower()  # e.g. "friday"

    active: set[str] = set()

    # --- calendar.txt (regular weekly schedule) ---
    if hasattr(feed, "calendar") and not feed.calendar.empty:
        cal = feed.calendar
        # Ensure start_date / end_date are comparable with date_int
        # partridge may store them as int or str depending on version
        start_dates = cal["start_date"].astype(str).astype(int)
        end_dates = cal["end_date"].astype(str).astype(int)

        in_range = (start_dates <= date_int) & (end_dates >= date_int)

        if day_name in cal.columns:
            day_active = cal[day_name].astype(int) == 1
        else:
            day_active = pd.Series([False] * len(cal), index=cal.index)

        matching = cal.loc[in_range & day_active, "service_id"]
        active.update(matching.astype(str))

    # --- calendar_dates.txt (exceptions) ---
    if hasattr(feed, "calendar_dates") and not feed.calendar_dates.empty:
        cd = feed.calendar_dates
        cd_dates = cd["date"].astype(str).astype(int)
        on_date = cd_dates == date_int

        # exception_type 1 = service added, 2 = service removed
        additions = cd.loc[on_date & (cd["exception_type"].astype(int) == 1), "service_id"]
        removals = cd.loc[on_date & (cd["exception_type"].astype(int) == 2), "service_id"]

        active.update(additions.astype(str))
        active -= set(removals.astype(str))

    logger.info(
        "Found %d active service_ids for %s (%s).",
        len(active),
        date.isoformat(),
        day_name,
    )
    return active


# ---------------------------------------------------------------------------
# Feed wrapper (partridge Feed properties are read-only)
# ---------------------------------------------------------------------------

class _FilteredFeed:
    """Lightweight wrapper holding filtered GTFS DataFrames."""

    def __init__(self, *, stops, stop_times, trips, routes, agency,
                 calendar, calendar_dates):
        self.stops = stops
        self.stop_times = stop_times
        self.trips = trips
        self.routes = routes
        self.agency = agency
        self.calendar = calendar
        self.calendar_dates = calendar_dates


# ---------------------------------------------------------------------------
# Feed loading
# ---------------------------------------------------------------------------

def load_feed_for_date(gtfs_path: Path, date: datetime.date):
    """Load the GTFS feed filtered to services active on *date*.

    Uses ``partridge.load_raw_feed`` to read all data from the zip and then
    filters ``trips`` and ``stop_times`` to only those rows whose
    ``service_id`` is active on the requested date.

    Parameters
    ----------
    gtfs_path : Path
        Path to the GTFS zip file.
    date : datetime.date
        The date to filter schedules for.

    Returns
    -------
    feed
        A partridge feed object with ``.stops``, ``.stop_times``,
        ``.trips``, ``.routes``, and ``.agency`` DataFrames (among others).
        The ``.trips`` and ``.stop_times`` tables are filtered to only
        contain entries for services running on *date*.
    """
    logger.info("Loading raw GTFS feed from %s ...", gtfs_path)
    raw = ptg.load_raw_feed(str(gtfs_path))

    active_ids = get_active_service_ids(raw, date)
    if not active_ids:
        logger.warning(
            "No active service IDs found for %s — the feed may not cover this date.",
            date.isoformat(),
        )

    # Filter trips to active services
    mask_trips = raw.trips["service_id"].astype(str).isin(active_ids)
    filtered_trips = raw.trips.loc[mask_trips].copy()
    logger.info("Trips after date filter: %d", len(filtered_trips))

    # Filter stop_times to remaining trip_ids
    active_trip_ids = set(filtered_trips["trip_id"].astype(str))
    mask_st = raw.stop_times["trip_id"].astype(str).isin(active_trip_ids)
    filtered_stop_times = raw.stop_times.loc[mask_st].copy()
    logger.info("Stop-times after date filter: %d", len(filtered_stop_times))

    # Partridge Feed properties are read-only, so wrap in a simple namespace
    feed = _FilteredFeed(
        stops=raw.stops,
        stop_times=filtered_stop_times,
        trips=filtered_trips,
        routes=raw.routes,
        agency=raw.agency,
        calendar=raw.calendar if hasattr(raw, "calendar") else pd.DataFrame(),
        calendar_dates=raw.calendar_dates if hasattr(raw, "calendar_dates") else pd.DataFrame(),
    )
    return feed


# ---------------------------------------------------------------------------
# Origin stop search
# ---------------------------------------------------------------------------

def find_origin_stops(
    feed,
    lat: float,
    lon: float,
    radius_m: float = STOP_SEARCH_RADIUS_M,
) -> list[str]:
    """Return stop_ids within *radius_m* meters of (*lat*, *lon*).

    Iterates over ``feed.stops`` and uses :func:`haversine` to compute the
    distance from each stop to the query point.

    Parameters
    ----------
    feed
        A partridge feed with a ``.stops`` DataFrame containing at least
        ``stop_id``, ``stop_lat``, and ``stop_lon`` columns.
    lat, lon : float
        Query point coordinates (WGS-84).
    radius_m : float
        Search radius in meters (default from config: 500 m).

    Returns
    -------
    list[str]
        Stop IDs within the radius, sorted by distance (nearest first).
    """
    stops = feed.stops
    results: list[tuple[float, str]] = []

    for _, row in stops.iterrows():
        try:
            stop_lat = float(row["stop_lat"])
            stop_lon = float(row["stop_lon"])
        except (ValueError, TypeError):
            continue

        dist = haversine(lat, lon, stop_lat, stop_lon)
        if dist <= radius_m:
            results.append((dist, str(row["stop_id"])))

    results.sort(key=lambda x: x[0])
    stop_ids = [sid for _, sid in results]

    logger.info(
        "Found %d stops within %.0f m of (%.4f, %.4f).",
        len(stop_ids),
        radius_m,
        lat,
        lon,
    )
    return stop_ids
