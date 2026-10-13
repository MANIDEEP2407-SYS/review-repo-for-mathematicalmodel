"""Lumped-parameter thermal model.

Solves a steady-state energy balance for the evaporator temperature
T_evap and estimates T_glass from an inner-outer heat balance.
Reported as a "0D / lumped thermal estimate", NOT a CFD field.

Energy balance at the evaporator (per unit time):

    Q_absorbed = Q_evap + Q_conv_top + Q_rad_top + Q_cond_bottom

    Q_evap        = eta_thermal * Q_absorbed          [driven by user eta]
    Q_conv_top    = h_c(v_w) * A * (T_e - T_amb)
    Q_rad_top     = eps * sigma * A * (T_e^4 - T_sky^4)
    Q_cond_bottom = k_ins / d_ins * A * (T_e - T_amb)
"""

from __future__ import annotations

from dataclasses import dataclass

SIGMA = 5.670374419e-8   # Stefan-Boltzmann [W/(m^2 K^4)]


@dataclass
class ThermalState:
    T_evap_C: float
    T_glass_C: float
    T_sky_C: float
    h_conv_top: float           # W/(m^2 K)
    Q_convection_W: float
    Q_radiation_W: float
    Q_conduction_W: float
    Q_losses_total_W: float


def h_convection_wind(v_wind_m_s: float) -> float:
    """McAdams free/forced convection over a flat plate [W/(m^2 K)]."""
    return 5.7 + 3.8 * max(v_wind_m_s, 0.0)


def T_sky_celsius(T_amb_C: float) -> float:
    """Swinbank approximation for clear-sky effective temperature."""
    T_amb_K = T_amb_C + 273.15
    T_sky_K = 0.0552 * (T_amb_K ** 1.5)
    return T_sky_K - 273.15


def loss_at(T_evap_C: float,
            T_amb_C: float,
            T_sky_C: float,
            area: float,
            emissivity: float,
            h_conv: float,
            k_ins: float,
            d_ins: float) -> tuple:
    T_e_K = T_evap_C + 273.15
    T_sk_K = T_sky_C + 273.15
    Q_conv = h_conv * area * (T_evap_C - T_amb_C)
    Q_rad = emissivity * SIGMA * area * (T_e_K ** 4 - T_sk_K ** 4)
    Q_cond = (k_ins / d_ins) * area * (T_evap_C - T_amb_C)
    return Q_conv, Q_rad, Q_cond


def solve_evaporator_temperature(
    P_absorbed_W: float,
    P_evaporation_W: float,
    T_amb_C: float,
    area: float,
    emissivity: float,
    wind_speed: float,
    k_ins: float,
    d_ins: float,
    T_low: float = -10.0,
    T_high: float = 200.0,
    tol: float = 1e-3,
    max_iter: int = 200,
) -> ThermalState:
    """Bisection: find T_evap where absorbed = evap + convection + radiation + conduction."""
    T_sky = T_sky_celsius(T_amb_C)
    h_c = h_convection_wind(wind_speed)

    def residual(T_e):
        Q_conv, Q_rad, Q_cond = loss_at(T_e, T_amb_C, T_sky, area,
                                        emissivity, h_c, k_ins, d_ins)
        return P_absorbed_W - (P_evaporation_W + Q_conv + Q_rad + Q_cond)

    f_lo = residual(T_low)
    f_hi = residual(T_high)
    if f_lo * f_hi > 0:
        # No sign change: pick the endpoint with smaller residual magnitude.
        T = T_low if abs(f_lo) < abs(f_hi) else T_high
    else:
        lo, hi = T_low, T_high
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if abs(f_mid) < tol or (hi - lo) < tol:
                break
            if f_lo * f_mid < 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        T = mid

    Q_conv, Q_rad, Q_cond = loss_at(T, T_amb_C, T_sky, area,
                                    emissivity, h_c, k_ins, d_ins)

    # Simple T_glass estimate: sits between T_evap and T_amb weighted by
    # relative resistances (inner-vs-outer). Reasonable lumped approximation.
    T_glass = T_amb_C + 0.35 * (T - T_amb_C)

    return ThermalState(
        T_evap_C=T,
        T_glass_C=T_glass,
        T_sky_C=T_sky,
        h_conv_top=h_c,
        Q_convection_W=Q_conv,
        Q_radiation_W=Q_rad,
        Q_conduction_W=Q_cond,
        Q_losses_total_W=Q_conv + Q_rad + Q_cond,
    )
