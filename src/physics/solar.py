"""Solar input model.

Q_solar     = G * A * t
Q_absorbed  = G * A * tau_glass * optical_gain * t

optical_gain is the explicit reflector multiplier (1.0 = no reflector).
No arbitrary concentration is assumed; the user sets the value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolarInput:
    Q_incident: float   # J -- total solar energy incident during t_sun
    Q_absorbed: float   # J -- after optics
    G_effective: float  # W/m^2 -- irradiance seen by the evaporator surface


def compute_solar_input(
    irradiance: float,
    area_geometric: float,
    sunlight_hours: float,
    optical_gain: float = 1.0,
    glass_transmittance: float = 1.0,
    eta_optical: float = 1.0,
) -> SolarInput:
    """Bookkeeping for the solar energy chain.

    Parameters
    ----------
    irradiance : W/m^2      direct beam irradiance on the horizontal.
    area_geometric : m^2    raft / evaporator base area.
    sunlight_hours : h      effective sunlight duration.
    optical_gain : -        reflector multiplier (>= 1).
    glass_transmittance : - fraction of light passing the glass.
    eta_optical : -         additional optical losses (soiling, absorption).
    """
    duration_s = sunlight_hours * 3600.0
    G_eff = irradiance * optical_gain * glass_transmittance * eta_optical
    Q_incident = irradiance * area_geometric * duration_s
    Q_absorbed = G_eff * area_geometric * duration_s
    return SolarInput(Q_incident=Q_incident, Q_absorbed=Q_absorbed, G_effective=G_eff)


def reflector_gain(
    reflector_area: float,
    reflectivity: float,
    incidence_factor: float = 1.0,
    base_area: float = 1.0,
) -> float:
    """First-order energy-based reflector multiplier.

    Returns 1 + (A_r * rho * cos_factor) / A_base. This is an energy-balance
    approximation, not a ray-tracing result.
    """
    if base_area <= 0:
        raise ValueError("base_area must be positive.")
    return 1.0 + (reflector_area * reflectivity * incidence_factor) / base_area
