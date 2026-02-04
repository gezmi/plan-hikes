"""CLI entry point for the Israel Hiking Transit Planner."""

from __future__ import annotations

import datetime
import logging
import sys

import typer
from rich.console import Console

from src.config import CITY_COORDINATES, MAX_TRANSFERS, MAX_WALK_TO_TRAIL_M, MIN_HIKING_HOURS, SAFETY_MARGIN_HOURS
from src.models import HikeQuery
from src.output.cli_formatter import print_hike_plan, print_no_results, print_origin_header, print_query_header
from src.query.planner import plan_hikes, plan_hikes_for_origin, prepare_data

app = typer.Typer(help="Israel Hiking Transit Planner — find bus-accessible hikes.")
console = Console()


def _parse_date(value: str) -> datetime.date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(f"Invalid date format: '{value}'. Use YYYY-MM-DD.")


@app.command()
def main(
    origin: list[str] = typer.Option(..., "--origin", "-o", help="Origin city name (repeatable)"),
    date: str = typer.Option(..., "--date", "-d", help="Travel date (YYYY-MM-DD)"),
    max_results: int = typer.Option(20, "--max-results", "-n", help="Maximum number of results"),
    max_transfers: int = typer.Option(MAX_TRANSFERS, "--max-transfers", help="Maximum bus transfers (0 or 1)"),
    max_walk: int = typer.Option(MAX_WALK_TO_TRAIL_M, "--max-walk", help="Max walk to trail in meters"),
    min_hike: float = typer.Option(MIN_HIKING_HOURS, "--min-hike", help="Minimum hiking hours"),
    safety_margin: float = typer.Option(SAFETY_MARGIN_HOURS, "--safety-margin", help="Safety margin hours before Shabbat"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    # ── v0.2 filter flags ──
    color: list[str] = typer.Option([], "--color", "-c", help="Filter by trail color (red, blue, green, black, orange, purple)"),
    min_distance: float = typer.Option(None, "--min-distance", help="Minimum trail distance in km"),
    max_distance: float = typer.Option(None, "--max-distance", help="Maximum trail distance in km"),
    loop_only: bool = typer.Option(False, "--loop-only", help="Show only loop trails"),
    linear_only: bool = typer.Option(False, "--linear-only", help="Show only linear (non-loop) trails"),
    max_elevation_gain: float = typer.Option(None, "--max-elevation-gain", help="Maximum elevation gain in meters"),
    difficulty: str = typer.Option(None, "--difficulty", help="Filter by difficulty level"),
) -> None:
    """Find transit-accessible hiking trails from your city."""
    # ── Logging setup ─────────────────────────────────────────────────
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Parse inputs ──────────────────────────────────────────────────
    travel_date = _parse_date(date)

    if loop_only and linear_only:
        console.print("[red]Error:[/red] --loop-only and --linear-only are mutually exclusive.")
        raise typer.Exit(1)

    # ── Validate all origins upfront ──────────────────────────────────
    from src.query.planner import _resolve_origin
    for o in origin:
        try:
            _resolve_origin(o)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Build query from first origin (used for shared settings)
    base_query = HikeQuery(
        origin=origin[0],
        date=travel_date,
        max_transfers=max_transfers,
        safety_margin_hours=safety_margin,
        max_walk_to_trail_m=max_walk,
        min_hiking_hours=min_hike,
        max_results=max_results,
        colors=color if color else None,
        min_distance_km=min_distance,
        max_distance_km=max_distance,
        loop_only=loop_only,
        linear_only=linear_only,
        max_elevation_gain_m=max_elevation_gain,
        difficulty=difficulty,
    )

    # ── Prepare shared data (origin-independent) ─────────────────────
    try:
        with console.status("Loading data...", spinner="dots"):
            ctx = prepare_data(base_query)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    # ── Plan for each origin ──────────────────────────────────────────
    multi = len(origin) > 1
    any_results = False

    for o in origin:
        query = HikeQuery(
            origin=o,
            date=travel_date,
            max_transfers=max_transfers,
            safety_margin_hours=safety_margin,
            max_walk_to_trail_m=max_walk,
            min_hiking_hours=min_hike,
            max_results=max_results,
            colors=color if color else None,
            min_distance_km=min_distance,
            max_distance_km=max_distance,
            loop_only=loop_only,
            linear_only=linear_only,
            max_elevation_gain_m=max_elevation_gain,
            difficulty=difficulty,
        )

        try:
            with console.status(f"Planning hikes from {o}...", spinner="dots"):
                plans = plan_hikes_for_origin(query, ctx)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)

        if not plans:
            if multi:
                print_origin_header(o, 0)
            else:
                print_no_results(query)
            continue

        any_results = True

        if multi:
            print_origin_header(o, len(plans))
        else:
            deadline_str = plans[0].deadline.strftime("%H:%M")
            print_query_header(query, deadline_str, len(plans))

        origin_coords = CITY_COORDINATES.get(o.strip().lower())
        o_lat = origin_coords[0] if origin_coords else None
        o_lon = origin_coords[1] if origin_coords else None

        for i, plan in enumerate(plans, 1):
            print_hike_plan(i, plan, origin_lat=o_lat, origin_lon=o_lon)

    if not any_results and not multi:
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
