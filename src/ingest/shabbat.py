"""Hebcal Shabbat times client — fetch candle-lighting times and compute
transit deadlines for Friday / weekday hikes.

Caching: module-level dict keyed by ISO date string.  One CLI run queries
at most one date, so this is sufficient.
"""

import datetime
import logging

import requests

from src.config import (
    DEFAULT_LATEST_RETURN_HOUR,
    HEBCAL_URL,
    JERUSALEM_LAT,
    JERUSALEM_LON,
    SAFETY_MARGIN_HOURS,
)

logger = logging.getLogger(__name__)

# Simple in-memory cache: "YYYY-MM-DD" -> datetime.datetime (candle lighting)
_candle_cache: dict[str, datetime.datetime] = {}


# ── Conservative fallback estimates when Hebcal is unreachable ────────
# Winter months (Oct-Mar) sunset is earlier; summer (Apr-Sep) is later.
_WINTER_MONTHS = {1, 2, 3, 10, 11, 12}
_FALLBACK_WINTER = datetime.time(16, 30)  # ~16:30 candle lighting in winter
_FALLBACK_SUMMER = datetime.time(19, 0)   # ~19:00 candle lighting in summer


def get_deadline(
    date: datetime.date,
    safety_margin_hours: float = SAFETY_MARGIN_HOURS,
) -> datetime.datetime:
    """Return the latest datetime by which the hiker must be on a return bus.

    - **Saturday** (day 5): raises ``ValueError`` — not supported in v0.1.
    - **Friday** (day 4): candle-lighting minus *safety_margin_hours*.
    - **Weekday** (Mon-Thu, Sun): date at ``DEFAULT_LATEST_RETURN_HOUR`` (18:00).
    """
    if date.weekday() == 5:  # Saturday
        raise ValueError("Saturday hiking not supported in v0.1")

    if date.weekday() == 4:  # Friday
        candle_dt = fetch_candle_lighting(date)
        margin = datetime.timedelta(hours=safety_margin_hours)
        deadline = candle_dt - margin
        logger.info(
            "Friday deadline: candle lighting %s minus %.1fh margin -> %s",
            candle_dt.strftime("%H:%M"),
            safety_margin_hours,
            deadline.strftime("%H:%M"),
        )
        return deadline

    # Weekday (Sun=6, Mon=0, Tue=1, Wed=2, Thu=3)
    return datetime.datetime.combine(
        date, datetime.time(DEFAULT_LATEST_RETURN_HOUR, 0)
    )


def fetch_candle_lighting(
    date: datetime.date,
    lat: float | None = None,
    lon: float | None = None,
) -> datetime.datetime:
    """Fetch candle-lighting time from Hebcal for the Shabbat of the week
    containing *date*.

    Returns a naive ``datetime.datetime`` in Israel local time.

    Falls back to a conservative estimate when the API is unreachable.
    """
    if lat is None:
        lat = JERUSALEM_LAT
    if lon is None:
        lon = JERUSALEM_LON

    cache_key = date.isoformat()
    if cache_key in _candle_cache:
        logger.debug("Candle-lighting cache hit for %s", cache_key)
        return _candle_cache[cache_key]

    params = {
        "cfg": "json",
        "geo": "pos",
        "latitude": lat,
        "longitude": lon,
        "tzid": "Asia/Jerusalem",
        "M": "on",
        "b": 18,  # 18 minutes before sunset (standard)
        "gy": date.year,
        "gm": date.month,
        "gd": date.day,
    }

    try:
        response = requests.get(HEBCAL_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            if item.get("category") == "candles":
                iso_str = item["date"]  # e.g. "2026-02-06T16:53:00+02:00"
                candle_dt = _parse_iso_datetime(iso_str)
                _candle_cache[cache_key] = candle_dt
                logger.info(
                    "Hebcal candle lighting for %s: %s",
                    cache_key,
                    candle_dt.strftime("%Y-%m-%d %H:%M"),
                )
                return candle_dt

        # No "candles" entry found — unexpected response structure.
        logger.warning(
            "Hebcal response contained no 'candles' item for %s; "
            "falling back to conservative estimate.",
            cache_key,
        )

    except requests.RequestException as exc:
        logger.warning(
            "Hebcal API request failed (%s); using conservative fallback "
            "for candle-lighting time on %s.",
            exc,
            cache_key,
        )

    # ── Fallback: conservative candle-lighting estimate ──────────────
    fallback = _conservative_candle_estimate(date)
    _candle_cache[cache_key] = fallback
    return fallback


def _parse_iso_datetime(iso_str: str) -> datetime.datetime:
    """Parse an ISO 8601 datetime string and return a naive datetime in
    Israel local time (strip timezone info)."""
    dt = datetime.datetime.fromisoformat(iso_str)
    # Strip timezone — we treat everything as local Israel time.
    return dt.replace(tzinfo=None)


def _conservative_candle_estimate(date: datetime.date) -> datetime.datetime:
    """Return a conservative (early) candle-lighting estimate based on
    whether the month is in the winter or summer half of the year."""
    if date.month in _WINTER_MONTHS:
        fallback_time = _FALLBACK_WINTER
    else:
        fallback_time = _FALLBACK_SUMMER

    logger.info(
        "Using conservative candle-lighting fallback %s for %s",
        fallback_time.strftime("%H:%M"),
        date.isoformat(),
    )
    return datetime.datetime.combine(date, fallback_time)
