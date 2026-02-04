"""Transit routing engine — finds bus routes between stops using GTFS data.

Builds an in-memory index from a partridge GTFS feed and provides
forward (outbound) and backward (return) route finding with 0 or 1 transfer.
"""

from __future__ import annotations

import bisect
import datetime
import logging
from collections import defaultdict

from src.models import BusLeg

logger = logging.getLogger(__name__)

# ── Limits for search space pruning ──────────────────────────────────────
_MAX_INTERMEDIATE_STOPS = 30   # max stops to consider for transfer along a trip
_MAX_CONNECTING_DEPARTURES = 10  # max connecting trips to check per intermediate stop
_MAX_RETURN_DEPARTURES = 10    # max latest departures to try from each trail stop
_MIN_TRANSFER_SECS = 60       # minimum transfer time in seconds (1 minute)


def _time_to_seconds(time_str: str) -> int:
    """Parse a GTFS time string "HH:MM:SS" to seconds since midnight.

    GTFS allows hours >= 24 for trips that extend past midnight.
    """
    parts = time_str.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s


class TransitRouter:
    """In-memory transit routing index built from a partridge GTFS feed."""

    def __init__(self, feed, date: datetime.date) -> None:
        """Build lookup structures from the feed DataFrames.

        Args:
            feed: A partridge feed object with .stops, .stop_times, .routes,
                  .agency, and .trips DataFrames.
            date: The date for which to build datetime objects.
        """
        self.date = date

        # ── stop_id -> stop_name ─────────────────────────────────────────
        self.stop_name: dict[str, str] = dict(
            zip(feed.stops["stop_id"].astype(str), feed.stops["stop_name"])
        )

        # ── route_info: route_id -> (route_short_name, agency_name) ──────
        routes = feed.routes.copy()
        agency = feed.agency.copy()
        merged = routes.merge(agency, on="agency_id", how="left")
        self.route_info: dict[str, tuple[str, str]] = {}
        for _, row in merged.iterrows():
            route_id = str(row["route_id"])
            short_name = str(row.get("route_short_name", ""))
            agency_name = str(row.get("agency_name", ""))
            self.route_info[route_id] = (short_name, agency_name)

        # ── trip_route: trip_id -> route_id ──────────────────────────────
        self.trip_route: dict[str, str] = dict(
            zip(
                feed.trips["trip_id"].astype(str),
                feed.trips["route_id"].astype(str),
            )
        )

        # ── Build stop_departures and trip_stop_sequence from stop_times ─
        self.stop_departures: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        self.trip_stop_sequence: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)

        st = feed.stop_times
        for _, row in st.iterrows():
            trip_id = str(row["trip_id"])
            stop_id = str(row["stop_id"])
            seq = int(row["stop_sequence"])
            arr_secs = _time_to_seconds(str(row["arrival_time"]))
            dep_secs = _time_to_seconds(str(row["departure_time"]))

            self.stop_departures[stop_id].append((dep_secs, trip_id, seq))
            self.trip_stop_sequence[trip_id].append((stop_id, arr_secs, dep_secs, seq))

        # Sort stop_departures by departure time
        for stop_id in self.stop_departures:
            self.stop_departures[stop_id].sort(key=lambda x: x[0])

        # Sort trip_stop_sequence by stop_sequence
        for trip_id in self.trip_stop_sequence:
            self.trip_stop_sequence[trip_id].sort(key=lambda x: x[3])

        n_stops = len(self.stop_departures)
        n_trips = len(self.trip_stop_sequence)
        logger.info(
            "TransitRouter built: %d stops with departures, %d trips indexed",
            n_stops,
            n_trips,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _seconds_to_datetime(self, secs: int) -> datetime.datetime:
        """Convert seconds since midnight to a datetime on self.date.

        If secs >= 86400, the datetime rolls over to the next day.
        """
        days_offset = secs // 86400
        remaining = secs % 86400
        base = datetime.datetime.combine(self.date, datetime.time())
        return base + datetime.timedelta(days=days_offset, seconds=remaining)

    def _make_bus_leg(
        self,
        trip_id: str,
        from_stop_id: str,
        from_dep_secs: int,
        to_stop_id: str,
        to_arr_secs: int,
    ) -> BusLeg:
        """Build a BusLeg from routing data."""
        route_id = self.trip_route.get(trip_id, "")
        short_name, agency_name = self.route_info.get(route_id, ("", ""))
        return BusLeg(
            line=short_name,
            operator=agency_name,
            from_stop_id=from_stop_id,
            from_stop_name=self.stop_name.get(from_stop_id, from_stop_id),
            to_stop_id=to_stop_id,
            to_stop_name=self.stop_name.get(to_stop_id, to_stop_id),
            departure=self._seconds_to_datetime(from_dep_secs),
            arrival=self._seconds_to_datetime(to_arr_secs),
        )

    # ── Outbound routing (earliest arrival) ──────────────────────────────

    def find_outbound(
        self,
        origin_stops: list[str],
        dest_stops: set[str],
        earliest_departure_secs: int,
    ) -> list[BusLeg] | None:
        """Find the earliest-arriving route from origin to destination.

        Searches for direct routes (0 transfers) and 1-transfer routes.

        Args:
            origin_stops: List of stop_ids near the origin.
            dest_stops: Set of stop_ids near the destination trail.
            earliest_departure_secs: Earliest departure in seconds since midnight.

        Returns:
            A list of 1 or 2 BusLeg objects, or None if no route found.
        """
        best_arrival = float("inf")
        best_legs: list[tuple[str, str, int, str, int]] | None = None  # raw data for legs

        # ── Phase 1: Direct routes ───────────────────────────────────────
        for origin_stop in origin_stops:
            deps = self.stop_departures.get(origin_stop)
            if not deps:
                continue

            # Find first departure >= earliest_departure_secs via bisect
            dep_times = [d[0] for d in deps]
            idx = bisect.bisect_left(dep_times, earliest_departure_secs)

            for i in range(idx, len(deps)):
                dep_secs, trip_id, origin_seq = deps[i]

                # Early termination: can't improve
                if dep_secs >= best_arrival:
                    break

                # Check if this trip visits any dest stop after origin
                trip_stops = self.trip_stop_sequence.get(trip_id)
                if not trip_stops:
                    continue

                for stop_id, arr_secs, _, seq in trip_stops:
                    if seq <= origin_seq:
                        continue
                    if stop_id in dest_stops and arr_secs < best_arrival:
                        best_arrival = arr_secs
                        best_legs = [
                            (trip_id, origin_stop, dep_secs, stop_id, arr_secs)
                        ]
                        break  # first dest hit on this trip is sufficient

        # ── Phase 2: 1-transfer routes ───────────────────────────────────
        for origin_stop in origin_stops:
            deps = self.stop_departures.get(origin_stop)
            if not deps:
                continue

            dep_times = [d[0] for d in deps]
            idx = bisect.bisect_left(dep_times, earliest_departure_secs)

            for i in range(idx, len(deps)):
                dep_secs, trip_id, origin_seq = deps[i]

                # Early termination
                if dep_secs >= best_arrival:
                    break

                trip_stops = self.trip_stop_sequence.get(trip_id)
                if not trip_stops:
                    continue

                # Iterate intermediate stops (after origin, up to limit)
                intermediates_checked = 0
                for stop_id, arr_secs, _, seq in trip_stops:
                    if seq <= origin_seq:
                        continue

                    # If this intermediate IS a dest stop, we already found
                    # it in Phase 1 (or would have). Skip to avoid duplicates.
                    if stop_id in dest_stops:
                        break

                    intermediates_checked += 1
                    if intermediates_checked > _MAX_INTERMEDIATE_STOPS:
                        break

                    # Arrival at intermediate already too late
                    if arr_secs >= best_arrival:
                        break

                    # Find connecting departures from intermediate stop
                    conn_deps = self.stop_departures.get(stop_id)
                    if not conn_deps:
                        continue

                    transfer_ready = arr_secs + _MIN_TRANSFER_SECS
                    conn_dep_times = [d[0] for d in conn_deps]
                    conn_idx = bisect.bisect_left(conn_dep_times, transfer_ready)

                    connections_checked = 0
                    for j in range(conn_idx, len(conn_deps)):
                        conn_dep_secs, conn_trip_id, conn_seq = conn_deps[j]

                        if conn_dep_secs >= best_arrival:
                            break

                        # Don't reboard the same trip
                        if conn_trip_id == trip_id:
                            continue

                        connections_checked += 1
                        if connections_checked > _MAX_CONNECTING_DEPARTURES:
                            break

                        # Check if connecting trip reaches dest
                        conn_trip_stops = self.trip_stop_sequence.get(conn_trip_id)
                        if not conn_trip_stops:
                            continue

                        for c_stop_id, c_arr_secs, _, c_seq in conn_trip_stops:
                            if c_seq <= conn_seq:
                                continue
                            if c_stop_id in dest_stops and c_arr_secs < best_arrival:
                                best_arrival = c_arr_secs
                                best_legs = [
                                    (trip_id, origin_stop, dep_secs, stop_id, arr_secs),
                                    (conn_trip_id, stop_id, conn_dep_secs, c_stop_id, c_arr_secs),
                                ]
                                break  # first dest on connecting trip

        if best_legs is None:
            return None

        return [
            self._make_bus_leg(trip_id, f_stop, f_dep, t_stop, t_arr)
            for trip_id, f_stop, f_dep, t_stop, t_arr in best_legs
        ]

    # ── Return routing (latest departure, arrive before deadline) ────────

    def find_return(
        self,
        trail_stops: list[str],
        origin_stops: set[str],
        deadline_secs: int,
    ) -> list[BusLeg] | None:
        """Find the latest-departing route from trail back to origin.

        The route must arrive at an origin stop before the deadline.

        Args:
            trail_stops: List of stop_ids near the trail exit.
            origin_stops: Set of stop_ids near the origin.
            deadline_secs: Latest allowed arrival in seconds since midnight.

        Returns:
            A list of 1 or 2 BusLeg objects, or None if no route found.
        """
        best_trail_dep = -1  # we want to maximize departure from trail
        best_legs: list[tuple[str, str, int, str, int]] | None = None

        # ── Phase 1: Direct routes ───────────────────────────────────────
        for trail_stop in trail_stops:
            deps = self.stop_departures.get(trail_stop)
            if not deps:
                continue

            # Iterate in reverse (latest first)
            checked = 0
            for i in range(len(deps) - 1, -1, -1):
                dep_secs, trip_id, trail_seq = deps[i]

                # Skip departures after deadline (can't possibly arrive before it)
                if dep_secs > deadline_secs:
                    continue

                # If we already have a better (later) departure, stop
                if dep_secs <= best_trail_dep:
                    break

                checked += 1
                if checked > _MAX_RETURN_DEPARTURES:
                    break

                trip_stops = self.trip_stop_sequence.get(trip_id)
                if not trip_stops:
                    continue

                for stop_id, arr_secs, _, seq in trip_stops:
                    if seq <= trail_seq:
                        continue
                    if stop_id in origin_stops and arr_secs <= deadline_secs:
                        if dep_secs > best_trail_dep:
                            best_trail_dep = dep_secs
                            best_legs = [
                                (trip_id, trail_stop, dep_secs, stop_id, arr_secs)
                            ]
                        break  # first origin hit on this trip

        # ── Phase 2: 1-transfer routes ───────────────────────────────────
        for trail_stop in trail_stops:
            deps = self.stop_departures.get(trail_stop)
            if not deps:
                continue

            checked = 0
            for i in range(len(deps) - 1, -1, -1):
                dep_secs, trip_id, trail_seq = deps[i]

                if dep_secs > deadline_secs:
                    continue

                if dep_secs <= best_trail_dep:
                    break

                checked += 1
                if checked > _MAX_RETURN_DEPARTURES:
                    break

                trip_stops = self.trip_stop_sequence.get(trip_id)
                if not trip_stops:
                    continue

                # Iterate intermediate stops after trail stop
                intermediates_checked = 0
                for stop_id, arr_secs, _, seq in trip_stops:
                    if seq <= trail_seq:
                        continue

                    # If intermediate IS an origin stop, Phase 1 handled it
                    if stop_id in origin_stops:
                        break

                    intermediates_checked += 1
                    if intermediates_checked > _MAX_INTERMEDIATE_STOPS:
                        break

                    if arr_secs > deadline_secs:
                        break

                    # Find connecting departures from intermediate
                    conn_deps = self.stop_departures.get(stop_id)
                    if not conn_deps:
                        continue

                    transfer_ready = arr_secs + _MIN_TRANSFER_SECS
                    conn_dep_times = [d[0] for d in conn_deps]
                    conn_idx = bisect.bisect_left(conn_dep_times, transfer_ready)

                    connections_checked = 0
                    for j in range(conn_idx, len(conn_deps)):
                        conn_dep_secs, conn_trip_id, conn_seq = conn_deps[j]

                        if conn_dep_secs > deadline_secs:
                            break

                        if conn_trip_id == trip_id:
                            continue

                        connections_checked += 1
                        if connections_checked > _MAX_CONNECTING_DEPARTURES:
                            break

                        conn_trip_stops = self.trip_stop_sequence.get(conn_trip_id)
                        if not conn_trip_stops:
                            continue

                        for c_stop_id, c_arr_secs, _, c_seq in conn_trip_stops:
                            if c_seq <= conn_seq:
                                continue
                            if c_stop_id in origin_stops and c_arr_secs <= deadline_secs:
                                if dep_secs > best_trail_dep:
                                    best_trail_dep = dep_secs
                                    best_legs = [
                                        (trip_id, trail_stop, dep_secs, stop_id, arr_secs),
                                        (conn_trip_id, stop_id, conn_dep_secs, c_stop_id, c_arr_secs),
                                    ]
                                break  # first origin on connecting trip

        if best_legs is None:
            return None

        return [
            self._make_bus_leg(trip_id, f_stop, f_dep, t_stop, t_arr)
            for trip_id, f_stop, f_dep, t_stop, t_arr in best_legs
        ]
