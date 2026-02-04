"""SRTM elevation sampling for trail geometries.

Reads SRTM GL3 (90m resolution) tiles (.tif or .hgt) and samples elevation
along trail LineStrings to compute gain, loss, and min/max elevation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from shapely.geometry import LineString

from src.config import SRTM_DIR, SRTM_SAMPLE_INTERVAL_M

logger = logging.getLogger(__name__)


class ElevationSampler:
    """Sample elevation from SRTM tiles (.tif or .hgt format)."""

    def __init__(self, srtm_dir: Path | None = None) -> None:
        self._srtm_dir = srtm_dir or SRTM_DIR
        self._datasets: dict[str, object] = {}  # tile_name -> rasterio dataset

    def _tile_name(self, lat: float, lon: float) -> str:
        """Return the SRTM tile base name for a given coordinate.

        SRTM tiles are named by their SW corner, e.g. N31E034.
        The extension (.tif or .hgt) is resolved in _get_dataset.
        """
        lat_prefix = "N" if lat >= 0 else "S"
        lon_prefix = "E" if lon >= 0 else "W"
        lat_int = int(math.floor(lat))
        lon_int = int(math.floor(lon))
        return f"{lat_prefix}{abs(lat_int):02d}{lon_prefix}{abs(lon_int):03d}"

    def _get_dataset(self, tile_name: str):
        """Open (or return cached) rasterio dataset for the given tile.

        Tries .tif first, then .hgt.
        """
        if tile_name in self._datasets:
            return self._datasets[tile_name]

        try:
            import rasterio
        except ImportError:
            logger.debug("rasterio not installed; elevation sampling unavailable.")
            self._datasets[tile_name] = None
            return None

        # Try both extensions
        tile_path = None
        for ext in (".tif", ".hgt"):
            candidate = self._srtm_dir / f"{tile_name}{ext}"
            if candidate.exists():
                tile_path = candidate
                break

        if tile_path is None:
            logger.debug("SRTM tile not found: %s (.tif or .hgt)", tile_name)
            self._datasets[tile_name] = None
            return None

        ds = rasterio.open(tile_path)
        self._datasets[tile_name] = ds
        return ds

    def sample_point(self, lat: float, lon: float) -> float | None:
        """Return elevation in meters for a point, or None if unavailable."""
        tile = self._tile_name(lat, lon)
        ds = self._get_dataset(tile)
        if ds is None:
            return None

        # rasterio expects (lon, lat) for geographic CRS
        row, col = ds.index(lon, lat)
        band = ds.read(1)
        val = band[row, col]

        # SRTM nodata is typically -32768
        if val <= -1000:
            return None
        return float(val)

    def sample_trail(
        self, geometry: LineString, distance_km: float
    ) -> dict[str, float]:
        """Sample elevation along a trail and compute statistics.

        Parameters
        ----------
        geometry : LineString
            Trail geometry in (lon, lat) coordinate order.
        distance_km : float
            Trail distance in km (used to determine sample count).

        Returns
        -------
        dict with keys: elevation_gain_m, elevation_loss_m,
                        max_elevation_m, min_elevation_m
        """
        distance_m = distance_km * 1000.0
        n_samples = max(int(distance_m / SRTM_SAMPLE_INTERVAL_M), 2)

        elevations: list[float] = []
        for i in range(n_samples + 1):
            fraction = i / n_samples
            point = geometry.interpolate(fraction, normalized=True)
            # geometry is (lon, lat), interpolated point is also (lon, lat)
            lon, lat = point.x, point.y
            elev = self.sample_point(lat, lon)
            if elev is not None:
                elevations.append(elev)

        if len(elevations) < 2:
            return {
                "elevation_gain_m": 0.0,
                "elevation_loss_m": 0.0,
                "max_elevation_m": 0.0,
                "min_elevation_m": 0.0,
                "elevation_profile": [],
            }

        gain = 0.0
        loss = 0.0
        for i in range(1, len(elevations)):
            delta = elevations[i] - elevations[i - 1]
            if delta > 0:
                gain += delta
            else:
                loss += abs(delta)

        return {
            "elevation_gain_m": round(gain, 1),
            "elevation_loss_m": round(loss, 1),
            "max_elevation_m": round(max(elevations), 1),
            "min_elevation_m": round(min(elevations), 1),
            "elevation_profile": elevations,
        }

    def close(self) -> None:
        """Close all open rasterio datasets."""
        for ds in self._datasets.values():
            if ds is not None and hasattr(ds, "close"):
                ds.close()
        self._datasets.clear()
