"""Overall water mass balance for the desalination chamber."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MassBalance:
    feed_kg: float
    evaporated_kg: float
    condensed_capacity_kg: float
    freshwater_collected_kg: float
    brine_kg: float
    recovery_ratio: float           # collected / feed
    closure_error_kg: float         # feed - collected - brine


def compute_mass_balance(
    feed_kg: float,
    evaporated_kg: float,
    condensed_capacity_kg: float,
    freshwater_collected_kg: float,
) -> MassBalance:
    fresh = min(freshwater_collected_kg, evaporated_kg, condensed_capacity_kg)
    fresh = max(fresh, 0.0)
    brine = max(feed_kg - fresh, 0.0)
    recovery = fresh / feed_kg if feed_kg > 0 else 0.0
    closure = feed_kg - fresh - brine
    return MassBalance(
        feed_kg=feed_kg,
        evaporated_kg=evaporated_kg,
        condensed_capacity_kg=condensed_capacity_kg,
        freshwater_collected_kg=fresh,
        brine_kg=brine,
        recovery_ratio=recovery,
        closure_error_kg=closure,
    )
