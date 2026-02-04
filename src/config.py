"""Constants and configuration for the Israel Hiking Transit Planner."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GTFS_DIR = DATA_DIR / "gtfs"
TRAILS_DIR = DATA_DIR / "trails"

# ── GTFS ───────────────────────────────────────────────────────────────
GTFS_FTP_HOST = "gtfs.mot.gov.il"
GTFS_FTP_FILE = "israel-public-transportation.zip"
GTFS_HTTPS_URL = "https://gtfs.mot.gov.il/gtfsfiles/israel-public-transportation.zip"
GTFS_ZIP_PATH = GTFS_DIR / GTFS_FTP_FILE
GTFS_CACHE_DAYS = 7

# ── Overpass (OSM trails) ──────────────────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """
[out:json][timeout:300];
area["ISO3166-1"="IL"]->.israel;
relation["route"="hiking"](area.israel);
out body;
>;
out skel qt;
"""
OVERPASS_CACHE_PATH = TRAILS_DIR / "overpass_response.json"
OVERPASS_CACHE_DAYS = 30

# ── Hebcal (Shabbat times) ────────────────────────────────────────────
HEBCAL_URL = "https://www.hebcal.com/shabbat"
# Jerusalem coordinates used as default (max ~5 min variance across Israel)
JERUSALEM_LAT = 31.7683
JERUSALEM_LON = 35.2137

# ── SRTM elevation data ──────────────────────────────────────────────
SRTM_DIR = DATA_DIR / "srtm"
SRTM_SAMPLE_INTERVAL_M = 50

# ── Routing constants ─────────────────────────────────────────────────
MAX_WALK_TO_TRAIL_M = 1000
SAFETY_MARGIN_HOURS = 2.0
NAISMITH_SPEED_KMH = 4.0
NAISMITH_CLIMB_FACTOR = 600  # meters of climb per hour
DEFAULT_LATEST_RETURN_HOUR = 18  # weekday deadline hour
MIN_HIKING_HOURS = 1.0
MAX_TRANSFERS = 1
WALK_SPEED_KMH = 4.5  # walking speed to/from bus stop
STOP_SEARCH_RADIUS_M = 500  # radius to find origin bus stops
DEDUP_TRAIL_DISTANCE_M = 200  # dedup nearby access points along trail
MAX_TRAIL_DISTANCE_KM = 30   # skip trails longer than this (mega-trails like Israel Trail)

# ── Through-hike constraints ────────────────────────────────────────
THROUGH_HIKE_MIN_DISTANCE_KM = 3.0
THROUGH_HIKE_MAX_DISTANCE_KM = 20.0

# ── City coordinates (lat, lon) — central bus/train station area ──────
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "rehovot":       (31.8928, 34.8113),
    "jerusalem":     (31.7892, 35.2033),
    "tel aviv":      (32.0564, 34.7796),
    "haifa":         (32.7940, 34.9896),
    "beer sheva":    (31.2430, 34.7932),
    "netanya":       (32.3215, 34.8532),
    "herzliya":      (32.1629, 34.8447),
    "petah tikva":   (32.0868, 34.8867),
    "rishon lezion": (31.9642, 34.8048),
    "ashdod":        (31.8014, 34.6435),
}
