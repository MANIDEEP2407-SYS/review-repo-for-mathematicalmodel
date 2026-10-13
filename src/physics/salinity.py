"""Salt / TDS balance.

Salt mass in the feed side is conserved. Phase change does not carry
dissolved salts to the vapor. If the evaporator ever reaches saturation,
a warning is raised (scaling / crystallization risk).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SalinityResult:
    initial_tds_mg_per_L: float
    brine_tds_mg_per_L: float
    concentration_factor: float
    salt_mass_g: float
    saturated: bool


def compute_salinity(
    feed_volume_L: float,
    feed_tds_mg_per_L: float,
    m_evap_kg: float,
    water_density_kg_per_m3: float,
    saturation_limit_mg_per_L: float,
) -> SalinityResult:
    if feed_volume_L <= 0:
        raise ValueError("feed_volume_L must be positive.")

    salt_mass_g = feed_tds_mg_per_L * feed_volume_L * 1e-3   # mg->g
    V_evap_L = m_evap_kg / water_density_kg_per_m3 * 1000.0
    V_final_L = max(feed_volume_L - V_evap_L, 1e-6)
    brine_tds = (salt_mass_g * 1000.0) / V_final_L           # g->mg then /L
    cf = brine_tds / feed_tds_mg_per_L if feed_tds_mg_per_L > 0 else 0.0
    return SalinityResult(
        initial_tds_mg_per_L=feed_tds_mg_per_L,
        brine_tds_mg_per_L=brine_tds,
        concentration_factor=cf,
        salt_mass_g=salt_mass_g,
        saturated=brine_tds >= saturation_limit_mg_per_L,
    )
