"""Optional Venturi / vacuum module.

Ideal Bernoulli gives the theoretical throat pressure:

    v2  = Q / (Cd * A2)
    v1  = Q / A1
    dP  = 0.5 * rho * (v2^2 - v1^2)
    P2  = P1 - dP

Actual chamber vacuum is worse than P2 by a "loss fraction" that lumps
piping losses, seal leaks and the fact that Bernoulli is inviscid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VenturiResult:
    enabled: bool
    P_inlet_Pa: float
    P_throat_Pa: float               # theoretical, Bernoulli
    dP_theoretical_Pa: float
    P_chamber_effective_Pa: float    # after real-world loss fraction
    saturation_temperature_shift_C: float
    warning: str = ""


def _area(diameter_m: float) -> float:
    return math.pi * (diameter_m ** 2) / 4.0


def compute_venturi(
    enabled: bool,
    motive_flow_m3_s: float,
    inlet_diameter_m: float,
    throat_diameter_m: float,
    inlet_pressure_Pa: float,
    fluid_density: float,
    discharge_coefficient: float,
    loss_fraction: float,
) -> VenturiResult:
    if not enabled:
        return VenturiResult(
            enabled=False,
            P_inlet_Pa=inlet_pressure_Pa,
            P_throat_Pa=inlet_pressure_Pa,
            dP_theoretical_Pa=0.0,
            P_chamber_effective_Pa=inlet_pressure_Pa,
            saturation_temperature_shift_C=0.0,
            warning="",
        )

    A1 = _area(inlet_diameter_m)
    A2 = _area(throat_diameter_m)
    if A2 >= A1:
        raise ValueError("Throat diameter must be smaller than inlet diameter.")

    v1 = motive_flow_m3_s / A1
    v2 = motive_flow_m3_s / (max(discharge_coefficient, 1e-3) * A2)
    dP_theory = 0.5 * fluid_density * (v2 * v2 - v1 * v1)
    P2_theory = inlet_pressure_Pa - dP_theory

    dP_effective = dP_theory * (1.0 - loss_fraction)
    P_chamber = inlet_pressure_Pa - dP_effective
    P_chamber = max(P_chamber, 500.0)   # 500 Pa lower bound: nowhere near true vacuum

    # Approximate boiling-point shift from Clausius-Clapeyron near 60 C.
    # dT ~ (R * T^2 / h_fg) * ln(P_atm / P_chamber)
    R_v = 461.5
    T_ref_K = 333.15
    h_fg = 2358e3
    ratio = max(inlet_pressure_Pa / max(P_chamber, 1.0), 1.0)
    dT_boiling = (R_v * T_ref_K * T_ref_K / h_fg) * math.log(ratio)

    warning = ""
    if P_chamber > 0.9 * inlet_pressure_Pa:
        warning = "Theoretical vacuum is negligible; check flow rate / throat diameter."

    return VenturiResult(
        enabled=True,
        P_inlet_Pa=inlet_pressure_Pa,
        P_throat_Pa=P2_theory,
        dP_theoretical_Pa=dP_theory,
        P_chamber_effective_Pa=P_chamber,
        saturation_temperature_shift_C=dT_boiling,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Ejector model for the coupled solver (coupling="coupled").
#
# The legacy compute_venturi() above is kept unchanged for coupling="legacy".
# Here the same Bernoulli/loss-fraction pressure drop is used, but the chamber
# pressure is bounded below by physics instead of a 500 Pa constant:
#
#     P_chamber = max(P_atm - dP_eff, P_sat(T_motive), P_leak_floor)
#
#   * P_sat(T_motive): a water-jet ejector cannot pull below the vapour
#     pressure of its own motive water (cavitation), ~4.2 kPa at 30 C;
#   * P_leak_floor   : ASSUMED minimum pressure a leaky glass cover holds.
#
# The pump that drives the jet costs electricity: Q_m * dP_pump / eta_pump,
# with dP_pump = the nozzle pressure drop (Bernoulli).  The vacuum also loads
# the cover: (P_atm - P_chamber) * A_glass.
# ---------------------------------------------------------------------------
from .evaporation import saturation_pressure_pa as _psat_magnus  # noqa: E402

G_ACC = 9.80665


@dataclass
class EjectorResult:
    enabled: bool
    P_atm_Pa: float
    dP_theoretical_Pa: float
    dP_effective_Pa: float
    P_bernoulli_Pa: float          # P_atm - dP_eff before the floors
    P_sat_motive_Pa: float
    P_leak_floor_Pa: float
    P_chamber_set_Pa: float        # after the floors (what the ejector can hold with no vapour load)
    binding_floor: str             # "none" | "motive_vapour_pressure" | "leak_floor"
    pump_power_W: float
    warning: str = ""


def compute_ejector(
    enabled: bool,
    motive_flow_m3_s: float,
    inlet_diameter_m: float,
    throat_diameter_m: float,
    P_atm_Pa: float,
    fluid_density: float,
    discharge_coefficient: float,
    loss_fraction: float,
    motive_water_T_C: float,
    leak_floor_Pa: float,
    pump_efficiency: float,
) -> EjectorResult:
    p_sat_m = _psat_magnus(min(max(motive_water_T_C, -40.0), 150.0))
    if not enabled:
        return EjectorResult(False, P_atm_Pa, 0.0, 0.0, P_atm_Pa, p_sat_m, leak_floor_Pa,
                             P_atm_Pa, "none", 0.0)
    A1, A2 = _area(inlet_diameter_m), _area(throat_diameter_m)
    if A2 >= A1:
        raise ValueError("Throat diameter must be smaller than inlet diameter.")
    v1 = motive_flow_m3_s / A1
    v2 = motive_flow_m3_s / (max(discharge_coefficient, 1e-3) * A2)
    dP_theory = 0.5 * fluid_density * (v2 * v2 - v1 * v1)
    dP_eff = dP_theory * (1.0 - loss_fraction)
    P_bern = P_atm_Pa - dP_eff
    floors = {"motive_vapour_pressure": p_sat_m, "leak_floor": leak_floor_Pa}
    binding, P_set = "none", P_bern
    for name, floor in floors.items():
        if floor > P_set:
            binding, P_set = name, floor
    P_set = min(P_set, P_atm_Pa)
    power = motive_flow_m3_s * dP_theory / max(pump_efficiency, 1e-3)
    warn = ""
    if P_bern > 0.9 * P_atm_Pa:
        warn = "Ejector pressure drop is negligible; check flow rate / throat diameter."
    return EjectorResult(True, P_atm_Pa, dP_theory, dP_eff, P_bern, p_sat_m, leak_floor_Pa,
                         P_set, binding, power, warn)


@dataclass
class CoverVacuumLoad:
    load_kN: float
    equivalent_mass_t: float       # tonnes-force the cover must carry
    plate_stress_Pa: float
    implausible: bool
    message: str


def cover_vacuum_load(P_atm_Pa: float, P_chamber_Pa: float, glass_area_m2: float,
                      glass_thickness_m: float, allowable_stress_Pa: float) -> CoverVacuumLoad:
    """Net inward load on the cover and a plate-bending plausibility flag.

    Plate check: a simply supported SQUARE plate of the same area under uniform pressure,
    sigma_max = 0.2874 q a^2 / t^2 (Roark, small-deflection theory).  This is a screening
    estimate only (no edge fixity, support ribs, large-deflection membrane action, creep or
    glass-type factors).  It does NOT establish structural validity."""
    dP = max(P_atm_Pa - P_chamber_Pa, 0.0)
    load_N = dP * glass_area_m2
    tonnes = load_N / G_ACC / 1000.0
    a2 = glass_area_m2
    sigma = 0.2874 * dP * a2 / max(glass_thickness_m, 1e-6) ** 2
    implausible = sigma > allowable_stress_Pa
    msg = ""
    if dP > 0:
        msg = (f"Vacuum load on the cover: {load_N/1e3:.1f} kN (~{tonnes:.1f} tonnes) at "
               f"{dP/1e3:.1f} kPa below ambient over {glass_area_m2:.2f} m2.")
        if implausible:
            msg += (f" Plate-bending screen gives ~{sigma/1e6:.0f} MPa in a {glass_thickness_m*1e3:.0f} mm pane "
                    f"(allowable here {allowable_stress_Pa/1e6:.0f} MPa): a vacuum like this is only credible "
                    "for a SMALL, ribbed or curved, sealed chamber, not a flat glass cover of this size. "
                    "Screening estimate only, not structural validity.")
    return CoverVacuumLoad(load_N / 1e3, tonnes, sigma, implausible, msg)
