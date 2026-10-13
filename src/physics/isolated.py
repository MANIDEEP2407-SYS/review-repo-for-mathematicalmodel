"""Isolated open-surface evaporator under one sun: a CONSISTENCY check, not a validation.

A wetted solar absorber in open air (no cover, no chamber).  Per unit PROJECTED area:

    alpha G  +  m [ h_c (T_amb - T_s) - eps sigma (T_s^4 - T_sky^4) - mdot'' h_fg(T_s) ]  = 0

    mdot'' = h_m (rho_v,s - rho_v,inf),   h_m = h_c / (rho_air cp_air Le^(2/3))   (Lewis analogy)
    h_c    = 5.7 + 3.8 v   (the repo's McAdams-type fit),  Le = alpha_air / D_wa

m is the exchange-area enhancement of a 3-D structure (wetted area / projected area; 1 = flat plate).
The surface can sit BELOW ambient (evaporative cooling), in which case it harvests heat from the air:
that is why a 3-D evaporator can exceed the one-sun energy bound (~1.5 kg/m2/h for 1 kW/m2).

PROVENANCE: m, the boundary-layer (Lewis) closure, the air properties and the test conditions are
ASSUMED.  Used only to see whether a published lab flux is energetically and order-of-magnitude
consistent with this repo's heat/mass-transfer closures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from .coupled import D_wa, M_W, R_UNIV, hfg, psat
from .heat_transfer import SIGMA, T_sky_celsius, h_convection_wind

RHO_AIR = 1.17        # kg/m3 near 30 C
CP_AIR = 1007.0       # J/(kg K)
ALPHA_AIR = 2.2e-5    # m2/s thermal diffusivity of air near 30 C


@dataclass
class IsolatedResult:
    T_surface_C: float
    flux_kg_m2_h: float            # per projected area
    latent_W_m2: float
    solar_W_m2: float
    environment_gain_W_m2: float   # latent - solar  (> 0: energy taken from the air / sky)
    enhancement: float


def isolated_evaporator(G: float = 1000.0, T_amb_C: float = 25.0, RH: float = 0.5, wind: float = 0.0,
                        absorptance: float = 0.95, emissivity: float = 0.95, enhancement: float = 1.0,
                        P_Pa: float = 101325.0) -> IsolatedResult:
    T_sky = T_sky_celsius(T_amb_C)
    h_c = h_convection_wind(wind)
    Le = ALPHA_AIR / D_wa(273.15 + T_amb_C, P_Pa)
    h_m = h_c / (RHO_AIR * CP_AIR * Le ** (2.0 / 3.0))
    rho_inf = RH * psat(T_amb_C) * M_W / (R_UNIV * (273.15 + T_amb_C))

    def flux(Ts):                       # kg/(m2 surface s)
        rho_s = psat(Ts) * M_W / (R_UNIV * (273.15 + Ts))
        return max(h_m * (rho_s - rho_inf), 0.0)

    def resid(Ts):
        q_in = h_c * (T_amb_C - Ts) - emissivity * SIGMA * ((Ts + 273.15) ** 4 - (T_sky + 273.15) ** 4)
        return absorptance * G + enhancement * (q_in - flux(Ts) * hfg(Ts))

    Ts = brentq(resid, T_amb_C - 40.0, T_amb_C + 80.0, xtol=1e-10)
    m_dot = enhancement * flux(Ts)
    latent = m_dot * hfg(Ts)
    return IsolatedResult(Ts, m_dot * 3600.0, latent, absorptance * G, latent - absorptance * G, enhancement)


def enhancement_to_match(target_kg_m2_h: float, **kw) -> float:
    """Area enhancement m at which the open-surface model gives `target` per projected area (inf if unreachable)."""
    f = lambda m: isolated_evaporator(enhancement=m, **kw).flux_kg_m2_h - target_kg_m2_h
    if f(500.0) < 0:
        return float("inf")
    if f(1.0) >= 0:
        return 1.0
    return brentq(f, 1.0, 500.0, xtol=1e-8)
