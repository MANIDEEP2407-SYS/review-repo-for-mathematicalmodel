"""Condensation on the sloped glass cover.

Two limits are computed:

    Q_glass_thru  = U_glass * A_g * (T_vapor - T_amb)     [W]
    m_cond_thermal_cap = Q_glass_thru * t / h_fg          [kg]

    Q_film_cap    = h_cond * A_g * (T_vapor - T_glass)    [W]
    m_cond_film_cap    = Q_film_cap * t / h_fg            [kg]

The condensation ceiling is the smaller of the two, then multiplied by
the collection efficiency to give the freshwater potentially harvested.
"""

from __future__ import annotations

from dataclasses import dataclass

from .heat_transfer import h_convection_wind, T_sky_celsius, SIGMA


@dataclass
class CondensationResult:
    m_cond_capacity_kg: float
    m_freshwater_collected_kg: float
    limited_by: str            # "film" or "glass_uvalue" or "evaporation"
    U_glass: float             # W/(m^2 K)
    Q_glass_reject_W: float


def compute_U_glass(
    h_cond_inner: float,
    glass_thickness: float,
    glass_k: float,
    T_glass_C: float,
    T_amb_C: float,
    wind_speed: float,
    emissivity_glass: float,
) -> float:
    """Overall glass U-value from inner condensation film to outdoor air."""
    h_conv_out = h_convection_wind(wind_speed)
    T_g_K = T_glass_C + 273.15
    T_sky_K = T_sky_celsius(T_amb_C) + 273.15
    # Linearised radiation coefficient about T_glass -> T_sky.
    if abs(T_g_K - T_sky_K) < 1e-6:
        h_rad = 4.0 * emissivity_glass * SIGMA * (T_g_K ** 3)
    else:
        h_rad = emissivity_glass * SIGMA * (T_g_K ** 2 + T_sky_K ** 2) * (T_g_K + T_sky_K)
    h_outer = h_conv_out + h_rad
    R_total = 1.0 / max(h_cond_inner, 1e-6) + glass_thickness / max(glass_k, 1e-6) + 1.0 / max(h_outer, 1e-6)
    return 1.0 / R_total


def compute_condensation(
    m_evap_kg: float,
    T_vapor_C: float,
    T_glass_C: float,
    T_ambient_C: float,
    glass_area: float,
    glass_thickness: float,
    glass_k: float,
    glass_emissivity: float,
    wind_speed: float,
    h_cond_inner: float,
    h_fg: float,
    sunlight_hours: float,
    collection_efficiency: float,
) -> CondensationResult:
    duration_s = sunlight_hours * 3600.0

    U_g = compute_U_glass(h_cond_inner, glass_thickness, glass_k,
                          T_glass_C, T_ambient_C, wind_speed, glass_emissivity)
    Q_reject = max(U_g * glass_area * (T_vapor_C - T_ambient_C), 0.0)
    m_cond_thermal = Q_reject * duration_s / h_fg

    Q_film = max(h_cond_inner * glass_area * (T_vapor_C - T_glass_C), 0.0)
    m_cond_film = Q_film * duration_s / h_fg

    capacity = min(m_cond_thermal, m_cond_film)
    if capacity >= m_evap_kg:
        limited = "evaporation"
        collected = m_evap_kg * collection_efficiency
    else:
        limited = "film" if m_cond_film < m_cond_thermal else "glass_uvalue"
        collected = capacity * collection_efficiency

    return CondensationResult(
        m_cond_capacity_kg=capacity,
        m_freshwater_collected_kg=collected,
        limited_by=limited,
        U_glass=U_g,
        Q_glass_reject_W=Q_reject,
    )
