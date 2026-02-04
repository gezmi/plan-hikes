#!/usr/bin/env python3
"""Refresh pre-processed trail data.

Downloads fresh GTFS and Overpass data, runs spatial join and elevation
enrichment, and saves the processed index to data/processed/.

Intended to be run weekly by a GitHub Action or cron job.

Usage:
    python scripts/refresh_data.py
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download GTFS
    logger.info("Step 1: Downloading GTFS data...")
    from src.ingest.gtfs import download_gtfs, load_feed_for_date
    gtfs_path = download_gtfs()
    logger.info("GTFS downloaded to %s", gtfs_path)

    # 2. Fetch trails from Overpass
    logger.info("Step 2: Fetching trails from Overpass...")
    from src.ingest.osm_trails import fetch_hiking_trails, enrich_trails_with_elevation
    trails = fetch_hiking_trails()
    logger.info("Fetched %d trails", len(trails))

    # 3. Enrich with elevation
    logger.info("Step 3: Enriching trails with elevation data...")
    try:
        enrich_trails_with_elevation(trails)
    except Exception as e:
        logger.warning("Elevation enrichment failed (continuing without): %s", e)

    # 4. Filter mega-trails
    from src.config import MAX_TRAIL_DISTANCE_KM
    trails = [t for t in trails if t.distance_km <= MAX_TRAIL_DISTANCE_KM]
    logger.info("%d trails after distance filter", len(trails))

    # 5. Spatial join with all GTFS stops
    logger.info("Step 4: Running spatial join with bus stops...")
    # Load feed for a sample date (we need stops, which don't change)
    sample_date = datetime.date.today() + datetime.timedelta(days=1)
    if sample_date.weekday() == 5:  # Saturday
        sample_date += datetime.timedelta(days=1)
    feed = load_feed_for_date(gtfs_path, sample_date)

    from src.index.spatial_join import build_trail_access_points
    trails = build_trail_access_points(trails, feed.stops, max_distance_m=1000)
    logger.info("%d trails with bus-accessible entry points", len(trails))

    # 6. Serialize to JSON
    logger.info("Step 5: Saving processed index...")
    index = []
    for trail in trails:
        access_points = []
        for ap in trail.access_points:
            access_points.append({
                "stop_id": ap.stop_id,
                "stop_name": ap.stop_name,
                "walk_distance_m": round(ap.walk_distance_m, 1),
                "trail_entry_lat": round(ap.trail_entry_lat, 6),
                "trail_entry_lon": round(ap.trail_entry_lon, 6),
                "trail_km_from_start": round(ap.trail_km_from_start, 3),
            })

        # Trail geometry as [[lat, lon], ...]
        coords = list(trail.geometry.coords)
        geometry = [[round(lat, 6), round(lon, 6)] for lon, lat in coords]

        index.append({
            "id": trail.id,
            "name": trail.name,
            "source": trail.source,
            "distance_km": trail.distance_km,
            "elevation_gain_m": trail.elevation_gain_m,
            "elevation_loss_m": trail.elevation_loss_m,
            "min_elevation_m": trail.min_elevation_m,
            "max_elevation_m": trail.max_elevation_m,
            "difficulty": trail.difficulty,
            "colors": trail.colors,
            "is_loop": trail.is_loop,
            "recommended_seasons": trail.recommended_seasons,
            "season_warnings": trail.season_warnings,
            "elevation_profile": [round(e, 1) for e in trail.elevation_profile],
            "geometry": geometry,
            "access_points": access_points,
        })

    output_path = PROCESSED_DIR / "trail_index.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(),
            "n_trails": len(index),
            "trails": index,
        }, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved %d trails to %s (%.1f MB)",
        len(index), output_path, size_mb,
    )
    print(f"Done. {len(index)} trails saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
