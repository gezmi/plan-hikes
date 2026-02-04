"""Rich CLI output for hiking plans."""

from __future__ import annotations

from urllib.parse import quote

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models import BusLeg, HikePlan, HikeQuery

console = Console()


# ── URL builders ──────────────────────────────────────────────────────

def _osm_url(trail_id: str) -> str | None:
    """Build an OpenStreetMap relation URL from trail.id like 'osm:12345'."""
    if trail_id.startswith("osm:"):
        rel_id = trail_id.split(":", 1)[1]
        return f"https://www.openstreetmap.org/relation/{rel_id}"
    return None


def _google_maps_url(lat: float, lon: float) -> str:
    """Google Maps pin at trail entry point."""
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def _israel_hiking_url(lat: float, lon: float) -> str:
    """Israel Hiking Map centered on the trail entry point (zoom 15)."""
    return f"https://israelhiking.osm.org.il/#/map/15/{lat:.5f}/{lon:.5f}"


def _transit_directions_url(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> str:
    """Google Maps transit directions from origin to trail entry."""
    return (
        f"https://www.google.com/maps/dir/?api=1"
        f"&origin={origin_lat:.6f},{origin_lon:.6f}"
        f"&destination={dest_lat:.6f},{dest_lon:.6f}"
        f"&travelmode=transit"
    )


# ── Sparkline ─────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    """Render a list of floats as a Unicode sparkline of the given width.

    Resamples the input to *width* bins and maps each to a block character.
    Returns empty string if fewer than 2 values.
    """
    if len(values) < 2:
        return ""

    n = len(values)
    # Resample to `width` points
    if n > width:
        # Downsample by averaging bins
        resampled = []
        for i in range(width):
            start = int(i * n / width)
            end = int((i + 1) * n / width)
            bin_vals = values[start:end] if start < end else [values[start]]
            resampled.append(sum(bin_vals) / len(bin_vals))
    elif n < width:
        # Upsample with linear interpolation
        resampled = []
        for i in range(width):
            frac = i * (n - 1) / (width - 1)
            lo = int(frac)
            hi = min(lo + 1, n - 1)
            t = frac - lo
            resampled.append(values[lo] * (1 - t) + values[hi] * t)
    else:
        resampled = list(values)

    lo = min(resampled)
    hi = max(resampled)
    span = hi - lo if hi != lo else 1.0

    chars = []
    for v in resampled:
        idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])

    return "".join(chars)


# ── Output functions ──────────────────────────────────────────────────

def print_query_header(query: HikeQuery, deadline_str: str, n_results: int) -> None:
    """Print a summary panel of the query parameters."""
    lines = [
        f"Origin:   {query.origin}",
        f"Date:     {query.date.strftime('%A, %B %d, %Y')}",
        f"Deadline: {deadline_str}",
        f"Results:  {n_results} hiking plans found",
    ]
    panel = Panel(
        "\n".join(lines),
        title="Israel Hiking Transit Planner",
        border_style="blue",
    )
    console.print(panel)


def print_hike_plan(
    idx: int,
    plan: HikePlan,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> None:
    """Print a single hiking plan with transit and trail details."""
    seg = plan.hike_segment
    trail = plan.trail
    ap = plan.access_point

    # ── Trail info line ───────────────────────────────────────────────
    if seg.is_through_hike:
        trail_type = "Through-hike"
    elif trail.is_loop:
        trail_type = "Loop"
    else:
        trail_type = "Out & Back"
    colors_str = ", ".join(seg.colors) if seg.colors else "unmarked"
    trail_info = (
        f"{trail.distance_km:.1f} km total | "
        f"{trail_type} | "
        f"Markings: {colors_str}"
    )
    if trail.elevation_gain_m > 0 or trail.elevation_loss_m > 0:
        trail_info += (
            f" | +{trail.elevation_gain_m:.0f}m / -{trail.elevation_loss_m:.0f}m"
            f" | {trail.min_elevation_m:.0f}-{trail.max_elevation_m:.0f}m"
        )

    # Elevation profile sparkline
    profile_line = ""
    if trail.elevation_profile:
        spark = _sparkline(trail.elevation_profile)
        if spark:
            profile_line = (
                f"  [dim]{trail.min_elevation_m:.0f}m[/dim] {spark}"
                f" [dim]{trail.max_elevation_m:.0f}m[/dim]"
            )

    # ── Hiking ratio bar ─────────────────────────────────────────────
    ratio_pct = plan.hiking_ratio * 100
    ratio_color = "green" if ratio_pct >= 50 else "yellow" if ratio_pct >= 30 else "red"

    # ── Build the content ─────────────────────────────────────────────
    parts: list[str] = []

    # Trail summary
    parts.append(f"[bold]{trail_info}[/bold]")
    if profile_line:
        parts.append(profile_line)
    parts.append(
        f"Hiking ratio: [{ratio_color}]{ratio_pct:.0f}%[/{ratio_color}]"
        f"  ({seg.hiking_hours:.1f}h hiking / {plan.total_hours:.1f}h total)"
    )
    parts.append("")

    # Outbound
    parts.append("[bold cyan]>>> Outbound[/bold cyan]")
    for leg in plan.outbound_legs:
        parts.append(_format_leg(leg))
    walk_min = seg.walk_to_trail_m / 1000.0 / 4.5 * 60
    parts.append(
        f"  Walk {seg.walk_to_trail_m:.0f}m to trail ({walk_min:.0f} min)"
    )
    parts.append("")

    # Hiking
    parts.append("[bold green]--- Hiking[/bold green]")
    hike_info = (
        f"  {seg.hike_start.strftime('%H:%M')} - {seg.hike_end.strftime('%H:%M')}"
        f"  |  ~{seg.estimated_distance_km:.1f} km"
        f"  |  {seg.hiking_hours:.1f}h"
    )
    if seg.is_through_hike and seg.exit_stop_name:
        hike_info += f"\n  {seg.entry_stop_name} -> {seg.exit_stop_name}"
    parts.append(hike_info)

    # Through-hike sub-profile
    if seg.is_through_hike and trail.elevation_profile and plan.exit_access_point:
        entry_km = plan.access_point.trail_km_from_start
        exit_km = plan.exit_access_point.trail_km_from_start
        n = len(trail.elevation_profile)
        if trail.distance_km > 0 and n >= 2:
            start_idx = int(min(entry_km, exit_km) / trail.distance_km * (n - 1))
            end_idx = int(max(entry_km, exit_km) / trail.distance_km * (n - 1))
            sub_profile = trail.elevation_profile[start_idx:end_idx + 1]
            if len(sub_profile) >= 2:
                spark = _sparkline(sub_profile)
                sub_min = min(sub_profile)
                sub_max = max(sub_profile)
                parts.append(
                    f"  Segment: [dim]{sub_min:.0f}m[/dim] {spark}"
                    f" [dim]{sub_max:.0f}m[/dim]"
                )

    parts.append("")

    # Return
    parts.append("[bold cyan]<<< Return[/bold cyan]")
    if seg.is_through_hike:
        walk_back_m = seg.walk_from_trail_m
        walk_back_min = walk_back_m / 1000.0 / 4.5 * 60
        parts.append(
            f"  Walk {walk_back_m:.0f}m from trail to stop ({walk_back_min:.0f} min)"
        )
    else:
        parts.append(
            f"  Walk {seg.walk_to_trail_m:.0f}m back to stop ({walk_min:.0f} min)"
        )
    for leg in plan.return_legs:
        parts.append(_format_leg(leg))
    parts.append("")

    # Deadline
    parts.append(
        f"[dim]Deadline: {plan.deadline.strftime('%H:%M')}"
        f"  |  Arrive home: {plan.arrival_at_origin.strftime('%H:%M')}[/dim]"
    )

    # Warnings (v0.2)
    if plan.warnings:
        for warning in plan.warnings:
            parts.append(f"[bold yellow]Warning: {warning}[/bold yellow]")

    parts.append("")

    # ── Links ─────────────────────────────────────────────────────────
    parts.append("[bold]Links[/bold]")

    osm = _osm_url(trail.id)
    if osm:
        parts.append(f"  OSM:            {osm}")

    entry_label = "Entry point" if seg.is_through_hike else "Google Maps"
    gmap = _google_maps_url(ap.trail_entry_lat, ap.trail_entry_lon)
    parts.append(f"  {entry_label}:    {gmap}")

    ihm = _israel_hiking_url(ap.trail_entry_lat, ap.trail_entry_lon)
    parts.append(f"  Israel Hiking:  {ihm}")

    if seg.is_through_hike and plan.exit_access_point:
        exit_ap = plan.exit_access_point
        exit_gmap = _google_maps_url(exit_ap.trail_entry_lat, exit_ap.trail_entry_lon)
        parts.append(f"  Exit point:     {exit_gmap}")

    if origin_lat is not None and origin_lon is not None:
        transit = _transit_directions_url(
            origin_lat, origin_lon, ap.trail_entry_lat, ap.trail_entry_lon,
        )
        parts.append(f"  Bus directions: {transit}")

    panel = Panel(
        "\n".join(parts),
        title=f"[bold]#{idx}  {trail.name}[/bold]",
        border_style="green" if ratio_pct >= 50 else "yellow" if ratio_pct >= 30 else "red",
    )
    console.print(panel)


def _format_leg(leg: BusLeg) -> str:
    """Format a single bus leg as a string."""
    dep = leg.departure.strftime("%H:%M")
    arr = leg.arrival.strftime("%H:%M")
    duration_min = (leg.arrival - leg.departure).total_seconds() / 60
    return (
        f"  Bus {leg.line} ({leg.operator})"
        f"  {dep} {leg.from_stop_name}"
        f" -> {arr} {leg.to_stop_name}"
        f"  ({duration_min:.0f} min)"
    )


def print_origin_header(origin: str, n_results: int) -> None:
    """Print a section header when showing results for multiple origins."""
    if n_results == 0:
        label = f"No hiking plans found from {origin}"
        style = "red"
    else:
        label = f"{origin} — {n_results} hiking plans"
        style = "blue"
    console.print()
    console.print(Panel(label, border_style=style, expand=True))


def print_no_results(query: HikeQuery) -> None:
    """Print a message when no hiking plans were found."""
    console.print(
        Panel(
            f"No hiking plans found from {query.origin} on {query.date}.\n"
            "Try a different date, wider search radius, or different origin.",
            title="No Results",
            border_style="red",
        )
    )
