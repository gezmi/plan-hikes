"""OSM hiking trail data via Overpass API.

Downloads all hiking route relations in Israel from the Overpass API,
parses the response into Trail objects with stitched LineString geometries,
and caches the raw JSON response to avoid repeated queries.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import requests
from shapely.geometry import LineString

from src.config import (
    OVERPASS_CACHE_DAYS,
    OVERPASS_CACHE_PATH,
    OVERPASS_QUERY,
    OVERPASS_URL,
    TRAILS_DIR,
)
from src.ingest.gtfs import haversine
from src.models import Trail

logger = logging.getLogger(__name__)

# ITC trail marking colors used in Israel
KNOWN_COLORS = {"red", "blue", "green", "black", "orange", "purple"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
    """Return True if the Overpass response cache exists and is < OVERPASS_CACHE_DAYS old."""
    if not OVERPASS_CACHE_PATH.exists():
        return False
    age_seconds = datetime.datetime.now().timestamp() - OVERPASS_CACHE_PATH.stat().st_mtime
    age_days = age_seconds / 86_400
    if age_days < OVERPASS_CACHE_DAYS:
        logger.info(
            "Overpass cache is %.1f days old (< %d); using cached data.",
            age_days,
            OVERPASS_CACHE_DAYS,
        )
        return True
    logger.info(
        "Overpass cache is %.1f days old (>= %d); will re-fetch.",
        age_days,
        OVERPASS_CACHE_DAYS,
    )
    return False


def _load_cache() -> dict:
    """Load and return the cached Overpass JSON response."""
    logger.info("Loading cached Overpass response from %s", OVERPASS_CACHE_PATH)
    with open(OVERPASS_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(data: dict) -> None:
    """Save the Overpass JSON response to the cache file."""
    OVERPASS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OVERPASS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("Saved Overpass response to %s", OVERPASS_CACHE_PATH)


# ---------------------------------------------------------------------------
# Overpass API query
# ---------------------------------------------------------------------------

def _fetch_overpass() -> dict:
    """POST the Overpass query and return the parsed JSON response.

    Raises
    ------
    RuntimeError
        On HTTP 429 (rate limit) or 504 (timeout) from the Overpass API.
    requests.HTTPError
        On other non-2xx responses.
    """
    logger.info("Sending Overpass query (this may take a few minutes)...")
    print("Querying Overpass API for Israeli hiking trails (may take a few minutes)...")

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": OVERPASS_QUERY},
            timeout=360,
        )
    except requests.Timeout:
        raise RuntimeError(
            "Overpass API request timed out after 360 seconds. "
            "Try again later or increase the timeout."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "Overpass API rate limit exceeded (HTTP 429). "
            "Please wait a few minutes and try again."
        )
    if response.status_code == 504:
        raise RuntimeError(
            "Overpass API query timed out on the server side (HTTP 504). "
            "The query may be too large; try again later."
        )

    response.raise_for_status()

    data = response.json()
    n_elements = len(data.get("elements", []))
    logger.info("Overpass returned %d elements.", n_elements)
    print(f"Overpass returned {n_elements} elements.")

    return data


# ---------------------------------------------------------------------------
# Color parsing
# ---------------------------------------------------------------------------

def _parse_colors(tags: dict) -> list[str]:
    """Extract trail marking colors from OSM tags.

    Checks the ``osmc:symbol`` tag (format ``color:background:foreground``,
    e.g. ``red:white:red_bar``), and the ``colour`` / ``color`` tags.

    Returns
    -------
    list[str]
        Unique color strings that match the known ITC trail colors.
    """
    colors: set[str] = set()

    # osmc:symbol — the color is the first segment before the first colon
    osmc = tags.get("osmc:symbol", "")
    if osmc:
        parts = osmc.split(":")
        if parts:
            candidate = parts[0].strip().lower()
            if candidate in KNOWN_COLORS:
                colors.add(candidate)

    # colour / color tags
    for key in ("colour", "color"):
        value = tags.get(key, "").strip().lower()
        if value in KNOWN_COLORS:
            colors.add(value)

    return sorted(colors)


# ---------------------------------------------------------------------------
# Season / desert detection
# ---------------------------------------------------------------------------

# Keywords in trail names that indicate desert or wadi trails
_DESERT_KEYWORDS = {
    "wadi", "nahal", "נחל", "ein", "עין", "negev", "נגב", "ramon", "רמון",
    "arava", "ערבה", "zin", "צין", "paran", "פארן", "mitzpe",
}

# Deep Negev latitude threshold
_DEEP_NEGEV_LAT = 31.0

FLASH_FLOOD_WARNING = (
    "Flash flood danger during rainy season (Nov-Mar). Check IMS forecast."
)


def _parse_season_info(
    name: str,
    tags: dict,
    coords: list[tuple[float, float]],
) -> tuple[list[str], list[str]]:
    """Detect desert/wadi trails and return (recommended_seasons, season_warnings).

    Parameters
    ----------
    name : str
        Trail name.
    tags : dict
        OSM relation tags.
    coords : list of (lat, lon)
        Trail coordinates used for geographic detection.

    Returns
    -------
    tuple[list[str], list[str]]
        (recommended_seasons, season_warnings)
    """
    is_desert = False

    # Check name keywords
    name_lower = name.lower()
    for keyword in _DESERT_KEYWORDS:
        if keyword in name_lower:
            is_desert = True
            break

    # Check geographic location (deep Negev)
    if not is_desert and coords:
        avg_lat = sum(lat for lat, _ in coords) / len(coords)
        if avg_lat < _DEEP_NEGEV_LAT:
            is_desert = True

    # Check OSM tags
    if not is_desert:
        for tag_key in ("seasonal", "description", "note"):
            tag_val = tags.get(tag_key, "").lower()
            if any(kw in tag_val for kw in ("flood", "wadi", "desert", "dry")):
                is_desert = True
                break

    if is_desert:
        return (
            ["spring", "autumn", "summer"],
            [FLASH_FLOOD_WARNING],
        )

    return ([], [])


# ---------------------------------------------------------------------------
# Way stitching
# ---------------------------------------------------------------------------

def _stitch_ways(
    way_refs: list[int],
    way_nodes: dict[int, list[int]],
    node_coords: dict[int, tuple[float, float]],
) -> list[tuple[float, float]]:
    """Stitch way segments into the longest contiguous coordinate sequence.

    Parameters
    ----------
    way_refs : list[int]
        Ordered list of way IDs from the relation members.
    way_nodes : dict[int, list[int]]
        Mapping from way ID to ordered list of node IDs.
    node_coords : dict[int, tuple[float, float]]
        Mapping from node ID to (lat, lon).

    Returns
    -------
    list[tuple[float, float]]
        Ordered (lat, lon) coordinates forming the best LineString.
    """
    # Build segments: each segment is a list of (lat, lon) tuples
    segments: list[list[tuple[float, float]]] = []
    for wref in way_refs:
        nids = way_nodes.get(wref, [])
        coords = []
        for nid in nids:
            if nid in node_coords:
                coords.append(node_coords[nid])
        if len(coords) >= 2:
            segments.append(coords)

    if not segments:
        return []

    # Chain segments together by matching start/end node IDs
    # We work with node IDs to do exact matching
    seg_node_ids: list[list[int]] = []
    for wref in way_refs:
        nids = way_nodes.get(wref, [])
        # Only include segments whose nodes all have coordinates
        valid_nids = [nid for nid in nids if nid in node_coords]
        if len(valid_nids) >= 2:
            seg_node_ids.append(valid_nids)

    if not seg_node_ids:
        return []

    # Greedy chaining: start with the first segment, then try to attach others
    used = [False] * len(seg_node_ids)
    chains: list[list[int]] = []

    for start_idx in range(len(seg_node_ids)):
        if used[start_idx]:
            continue

        # Start a new chain
        chain = list(seg_node_ids[start_idx])
        used[start_idx] = True
        changed = True

        while changed:
            changed = False
            for i in range(len(seg_node_ids)):
                if used[i]:
                    continue
                seg = seg_node_ids[i]
                seg_start = seg[0]
                seg_end = seg[-1]
                chain_start = chain[0]
                chain_end = chain[-1]

                if seg_start == chain_end:
                    # Append segment (skip first node to avoid duplicate)
                    chain.extend(seg[1:])
                    used[i] = True
                    changed = True
                elif seg_end == chain_start:
                    # Prepend segment (skip last node to avoid duplicate)
                    chain = seg[:-1] + chain
                    used[i] = True
                    changed = True
                elif seg_end == chain_end:
                    # Reverse segment and append
                    chain.extend(reversed(seg[:-1]))
                    used[i] = True
                    changed = True
                elif seg_start == chain_start:
                    # Reverse segment and prepend
                    chain = list(reversed(seg[1:])) + chain
                    used[i] = True
                    changed = True

        chains.append(chain)

    # Pick the longest chain
    best_chain = max(chains, key=len)

    # Convert node IDs to coordinates
    coords = [node_coords[nid] for nid in best_chain if nid in node_coords]
    return coords


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_hiking_trails() -> list[Trail]:
    """Fetch Israeli hiking trails from OSM via Overpass and return Trail objects.

    Uses a cached response if available and fresh (< 30 days old).
    Otherwise queries the Overpass API, caches the response, and parses it.

    Returns
    -------
    list[Trail]
        Parsed Trail objects with stitched LineString geometries.
    """
    # Load or fetch the Overpass data
    if _cache_is_fresh():
        data = _load_cache()
    else:
        data = _fetch_overpass()
        _save_cache(data)

    elements = data.get("elements", [])

    # Build lookup tables for nodes and ways
    node_coords: dict[int, tuple[float, float]] = {}
    way_nodes: dict[int, list[int]] = {}

    for el in elements:
        el_type = el.get("type")
        if el_type == "node":
            node_coords[el["id"]] = (el["lat"], el["lon"])
        elif el_type == "way":
            way_nodes[el["id"]] = el.get("nodes", [])

    logger.info(
        "Built lookup: %d nodes, %d ways.", len(node_coords), len(way_nodes)
    )

    # Parse relations into Trail objects
    trails: list[Trail] = []

    for el in elements:
        if el.get("type") != "relation":
            continue

        rel_id = el["id"]
        tags = el.get("tags", {})

        # Name
        name = tags.get("name", tags.get("name:en", f"Trail {rel_id}"))

        # Colors
        colors = _parse_colors(tags)

        # Collect way member refs
        members = el.get("members", [])
        way_refs = [m["ref"] for m in members if m.get("type") == "way"]

        if not way_refs:
            logger.debug("Relation %d (%s): no way members, skipping.", rel_id, name)
            continue

        # Stitch ways into a coordinate sequence
        coords = _stitch_ways(way_refs, way_nodes, node_coords)

        if len(coords) < 2:
            logger.debug(
                "Relation %d (%s): fewer than 2 coordinates after stitching, skipping.",
                rel_id,
                name,
            )
            continue

        # Build LineString geometry (Shapely uses (x, y) = (lon, lat))
        geometry = LineString([(lon, lat) for lat, lon in coords])

        # Compute distance in km by summing haversine between consecutive points
        distance_m = 0.0
        for i in range(len(coords) - 1):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[i + 1]
            distance_m += haversine(lat1, lon1, lat2, lon2)
        distance_km = distance_m / 1000.0

        # Detect loop: first coordinate within 100 m of last coordinate
        first_lat, first_lon = coords[0]
        last_lat, last_lon = coords[-1]
        loop_distance = haversine(first_lat, first_lon, last_lat, last_lon)
        is_loop = loop_distance < 100.0

        # Season detection
        recommended_seasons, season_warnings = _parse_season_info(name, tags, coords)

        trail = Trail(
            id=f"osm:{rel_id}",
            name=name,
            source="osm",
            geometry=geometry,
            distance_km=round(distance_km, 2),
            elevation_gain_m=0,
            difficulty="unknown",
            colors=colors,
            is_loop=is_loop,
            recommended_seasons=recommended_seasons,
            season_warnings=season_warnings,
        )
        trails.append(trail)

    logger.info("Parsed %d hiking trails from Overpass data.", len(trails))
    print(f"Parsed {len(trails)} hiking trails from OSM data.")

    return trails


def enrich_trails_with_elevation(trails: list[Trail]) -> None:
    """Enrich trails with SRTM elevation data (in-place).

    Separate from fetch_hiking_trails() so that trail fetching works
    even without rasterio or SRTM data on disk.
    """
    from src.ingest.elevation import ElevationSampler

    sampler = ElevationSampler()
    enriched = 0

    try:
        for trail in trails:
            stats = sampler.sample_trail(trail.geometry, trail.distance_km)
            if stats["elevation_gain_m"] > 0 or stats["elevation_loss_m"] > 0:
                trail.elevation_gain_m = stats["elevation_gain_m"]
                trail.elevation_loss_m = stats["elevation_loss_m"]
                trail.max_elevation_m = stats["max_elevation_m"]
                trail.min_elevation_m = stats["min_elevation_m"]
                trail.elevation_profile = stats["elevation_profile"]
                enriched += 1
    finally:
        sampler.close()

    logger.info("Enriched %d / %d trails with elevation data.", enriched, len(trails))
