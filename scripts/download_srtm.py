#!/usr/bin/env python3
"""Download SRTM GL3 (90m) tiles covering Israel.

Israel spans roughly lat 29-33, lon 34-36, requiring 9 tiles.

Data source: NASA / USGS EarthData.
Requires a free EarthData account and a ~/.netrc file:

    machine urs.earthdata.nasa.gov
    login <your_username>
    password <your_password>

Register at: https://urs.earthdata.nasa.gov/users/new

Usage:
    python scripts/download_srtm.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

# Israel SRTM tiles (named by SW corner)
TILES = [
    "N29E034", "N29E035",
    "N30E034", "N30E035",
    "N31E034", "N31E035",
    "N32E034", "N32E035",
    "N33E035",
]

# USGS EarthData SRTM GL3 v003 base URL
BASE_URL = (
    "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11"
)

# Destination directory
SRTM_DIR = Path(__file__).resolve().parent.parent / "data" / "srtm"


def download_tile(tile_name: str) -> None:
    """Download a single SRTM tile as .hgt from EarthData."""
    hgt_path = SRTM_DIR / f"{tile_name}.hgt"
    if hgt_path.exists():
        print(f"  {tile_name}.hgt already exists, skipping.")
        return

    zip_name = f"{tile_name}.SRTMGL3.hgt.zip"
    url = f"{BASE_URL}/{zip_name}"

    print(f"  Downloading {zip_name}...")
    # Uses ~/.netrc for EarthData authentication
    response = requests.get(url, timeout=120)

    if response.status_code == 401:
        print(
            "ERROR: Authentication failed (HTTP 401).\n"
            "You need a free NASA EarthData account.\n"
            "  1. Register at https://urs.earthdata.nasa.gov/users/new\n"
            "  2. Create ~/.netrc with:\n"
            "     machine urs.earthdata.nasa.gov\n"
            "     login <your_username>\n"
            "     password <your_password>\n"
        )
        sys.exit(1)

    if response.status_code == 404:
        print(f"  WARNING: Tile {tile_name} not found (HTTP 404), skipping.")
        return

    response.raise_for_status()

    # The download is a zip containing the .hgt file
    zip_path = SRTM_DIR / zip_name
    zip_path.write_bytes(response.content)

    # Extract the .hgt file
    with zipfile.ZipFile(zip_path) as zf:
        hgt_members = [m for m in zf.namelist() if m.endswith(".hgt")]
        if hgt_members:
            zf.extract(hgt_members[0], SRTM_DIR)
            extracted = SRTM_DIR / hgt_members[0]
            if extracted != hgt_path:
                extracted.rename(hgt_path)

    zip_path.unlink()  # Clean up the zip
    size_mb = hgt_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {tile_name}.hgt ({size_mb:.1f} MB)")


def main() -> None:
    SRTM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(TILES)} SRTM tiles to {SRTM_DIR}")

    for tile in TILES:
        download_tile(tile)

    # Summary
    hgt_files = list(SRTM_DIR.glob("*.hgt"))
    tif_files = list(SRTM_DIR.glob("*.tif"))
    total = len(hgt_files) + len(tif_files)
    print(f"\nDone. {total} SRTM tiles available ({len(hgt_files)} .hgt, {len(tif_files)} .tif)")


if __name__ == "__main__":
    main()
