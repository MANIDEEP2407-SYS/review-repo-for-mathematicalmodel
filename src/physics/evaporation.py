"""Evaporation models (four modes).

MODE A - energy-limited     m = Q_useful / h_fg
MODE B - empirical flux     m = J_emp * A_eff * t
MODE C - mass-transfer      J = k_m * (P_sat(T_surf) - P_vap)
MODE D - hybrid             m = min(energy-limited, mass-transfer)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def saturation_pressure_pa(T_celsius: float) -> float:
    """Magnus formula for saturation vapour pressure over liquid water [Pa]."""
    return 610.94 * math.exp(17.625 * T_celsius / (T_celsius + 243.04))


def latent_heat_vaporization(T_celsius: float) -> float:
    """Temperature-dependent h_fg [J/kg] (linear fit, Rogers & Yau)."""
    return (2500.9 - 2.365 * T_celsius) * 1000.0


def h_fg_or_default(T_celsius: Optional[float], fallback_J_per_kg: float) -> float:
    if T_celsius is None:
        return fallback_J_per_kg
    return latent_heat_vaporization(T_celsius)


@dataclass
class EvaporationResult:
    m_evap_kg: float             # kg per session (sunlight_hours)
    mode: str
    flux_kg_per_m2_h: float      # implied areal flux for reporting
    limited_by: str = ""         # e.g. "energy" or "mass-transfer" (MODE D)


def evaporate_energy_limited(
    Q_useful_J: float,
    h_fg_J_per_kg: float,
    area_effective: float,
    sunlight_hours: float,
) -> EvaporationResult:
    m = max(Q_useful_J, 0.0) / h_fg_J_per_kg
    flux = m / max(area_effective * sunlight_hours, 1e-12)
    return EvaporationResult(m, "energy_limited", flux)


def evaporate_empirical(
    J_emp_kg_per_m2_h: float,
    area_effective: float,
    sunlight_hours: float,
) -> EvaporationResult:
    m = max(J_emp_kg_per_m2_h, 0.0) * area_effective * sunlight_hours
    return EvaporationResult(m, "empirical", J_emp_kg_per_m2_h)


def evaporate_mass_transfer(
    T_surface_C: float,
    T_ambient_C: float,
    relative_humidity: float,
    k_m: float,
    area_effective: float,
    sunlight_hours: float,
) -> EvaporationResult:
    P_sat_surf = saturation_pressure_pa(T_surface_C)
    P_vap_amb = relative_humidity * saturation_pressure_pa(T_ambient_C)
    driving = max(P_sat_surf - P_vap_amb, 0.0)
    J = k_m * driving                                        # kg / (m^2 s)
    m = J * area_effective * sunlight_hours * 3600.0
    return EvaporationResult(m, "mass_transfer", J * 3600.0)


def evaporate_hybrid(
    Q_useful_J: float,
    h_fg_J_per_kg: float,
    T_surface_C: float,
    T_ambient_C: float,
    relative_humidity: float,
    k_m: float,
    area_effective: float,
    sunlight_hours: float,
) -> EvaporationResult:
    energy = evaporate_energy_limited(Q_useful_J, h_fg_J_per_kg, area_effective, sunlight_hours)
    mt = evaporate_mass_transfer(T_surface_C, T_ambient_C, relative_humidity,
                                 k_m, area_effective, sunlight_hours)
    if mt.m_evap_kg < energy.m_evap_kg:
        return EvaporationResult(mt.m_evap_kg, "hybrid", mt.flux_kg_per_m2_h, "mass-transfer")
    return EvaporationResult(energy.m_evap_kg, "hybrid", energy.flux_kg_per_m2_h, "energy")
