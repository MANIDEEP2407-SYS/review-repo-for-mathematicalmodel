"""Real-world derating: capillary supply limit, fouling, soiling, degradation,
diurnal sun profile.

The base model (system.simulate) assumes the wick can always deliver enough
water, the surfaces stay clean, and the sun is a flat block. This module
re-runs it hour by hour and applies, step by step:

  0  ideal            flat block, clean, unlimited water supply (simulate())
  1  diurnal          half-sine sun over `day_length_h`, SAME daily insolation
  2  + soiling        glass transmittance loses `soiling_rate` per day between cleanings
  3  + fouling/aging  evaporated mass x salt/biofilm factor x flower-ageing factor
  4  + capillary      evaporation capped by wick water supply (Jurin + Darcy)

Two solver paths (SystemConfig.coupling):
  coupled  fouling/ageing scale the evaporator's vapour-side conductance and flux caps
           (SystemConfig.evaporation_derating) and the capillary limit caps the vapour flow
           (SystemConfig.wick_supply_flux_limit).  Both act INSIDE the coupled energy balance, so
           a weaker evaporator runs hotter and loses more heat to the cover: nothing is masked by
           a glass ceiling.  (The ideal-evaporator mode A has no vapour-side resistance, so
           fouling cannot act on it; it is not the default.)
  legacy   fouling/ageing multiply the evaporated mass and the capillary cap is applied
           afterwards with min(), as before (README section 8 flaw 2: masked by the glass ceiling).

Capillary model (bundle of parallel pores, per m^2 of evaporator plan area):
    P_cap   = 2 sigma cos(theta) / r
    h_Jurin = P_cap / (rho g)                      max lift; wick_length above this = no supply
    K       = eps r^2 / (8 tau^2)                  permeability of the pore bundle
    u       = (K/mu) (P_cap - rho g L_w) / L_w     Darcy superficial velocity
    J_wick  = rho u phi_w                          kg/(m^2 s) that can be delivered

PROVENANCE: every parameter is ASSUMED - measure r, theta, phi_w with a
wicking-rise test on the real cotton flower before trusting step 4.
Not modelled: biofilm growth dynamics, salt-crust thermal resistance,
brine-viscosity effects, wind loading, rain/cloud days, water-quality changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from ..config.defaults import Parameter, ParameterSource
from .system import SystemConfig, simulate

_A = ParameterSource.ASSUMED
G_ACC = 9.80665

REALWORLD_REGISTRY: Dict[str, Parameter] = {}


def _reg(*a) -> None:
    p = Parameter(*a)
    REALWORLD_REGISTRY[p.name] = p


_reg("pore_radius", "r_p", "m", 2.0e-5, (1e-6, 2e-4), _A,
     "Effective capillary radius of the cotton wick. Measure with a wicking-rise test.")
_reg("contact_angle", "theta", "deg", 30.0, (0.0, 85.0), _A,
     "Water-cotton contact angle; rises with FeS coating hydrophobicity/ageing.")
_reg("wick_porosity", "eps_w", "-", 0.80, (0.3, 0.95), _A, "Void fraction of the wick.")
_reg("wick_tortuosity", "tau_w", "-", 1.5, (1.0, 4.0), _A, "Pore path tortuosity.")
_reg("wick_length", "L_w", "m", 0.05, (0.005, 1.0), _A,
     "Path length (= lift height) from the water surface to the evaporating tip.")
_reg("wick_area_fraction", "phi_w", "-", 0.0025, (1e-5, 0.5), _A,
     "Wick flow cross-section per m^2 of evaporator plan area (stems/strands, not the flower surface).")
_reg("day_length_h", "D", "h", 12.0, (6.0, 14.0), _A, "Sunrise-to-sunset length for the half-sine profile.")
_reg("cleaning_interval_days", "T_clean", "d", 14.0, (1.0, 180.0), _A, "Days between glass/wick cleaning.")
_reg("soiling_rate", "s", "1/d", 0.003, (0.0, 0.05), _A, "Fractional glass-transmittance loss per day (dust).")
_reg("fouling_timescale_days", "tau_f", "d", 60.0, (5.0, 1000.0), _A,
     "e-folding time of salt/biofilm yield loss; cleaning resets it.")
_reg("flower_degradation", "d_fl", "-", 0.15, (0.0, 0.8), _A,
     "Fractional yield loss by end of flower life (linear); the average over life is half.")


def resolve(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    v = {k: p.default for k, p in REALWORLD_REGISTRY.items()}
    for k, x in (overrides or {}).items():
        if k not in REALWORLD_REGISTRY:
            raise KeyError(f"Unknown real-world parameter '{k}'.")
        v[k] = REALWORLD_REGISTRY[k].validate(x)
    return v


def water_viscosity(T_K: float) -> float:
    """Vogel fit, Pa s."""
    return 2.414e-5 * 10 ** (247.8 / (T_K - 140.0))


def water_surface_tension(T_K: float) -> float:
    """Linear fit, N/m."""
    return 0.07275 * (1.0 - 0.002 * (T_K - 291.0))


@dataclass
class WickState:
    P_cap_Pa: float
    h_jurin_m: float
    K_m2: float
    J_wick_kg_m2_s: float          # 0 if the wick cannot lift water to its tip
    lift_ok: bool


def wick_capacity(T_C: float, p: Dict[str, float], rho: float = 985.0) -> WickState:
    T_K = T_C + 273.15
    sigma, mu = water_surface_tension(T_K), water_viscosity(T_K)
    cos_t = math.cos(math.radians(p["contact_angle"]))
    P_cap = 2.0 * sigma * cos_t / p["pore_radius"]
    h_j = P_cap / (rho * G_ACC)
    K = p["wick_porosity"] * p["pore_radius"] ** 2 / (8.0 * p["wick_tortuosity"] ** 2)
    net = P_cap - rho * G_ACC * p["wick_length"]
    if net <= 0:
        return WickState(P_cap, h_j, K, 0.0, False)
    u = (K / mu) * net / p["wick_length"]
    return WickState(P_cap, h_j, K, rho * u * p["wick_area_fraction"], True)


@dataclass
class RealWorldResult:
    steps: List[tuple]                    # (label, productivity L/m^2/day)
    ideal_L_m2_day: float
    final_L_m2_day: float
    derating: float                       # final / ideal
    wick: WickState                       # evaluated at the hottest hour
    min_wick_supply_ratio: float          # supply / demand at the tightest hour
    wick_limited_hours: int
    hourly: List[dict] = field(default_factory=list)
    factors: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _hourly_profile(cfg: SystemConfig, D: float) -> List[float]:
    """Half-sine over D hours with the same daily insolation as the flat block."""
    D = max(D, cfg.sunlight_hours)
    G_pk = cfg.solar_irradiance * cfg.sunlight_hours * math.pi / (2.0 * D)
    n = int(math.ceil(D))
    dt = D / n
    return [G_pk * math.sin(math.pi * (i + 0.5) * dt / D) for i in range(n)], dt


def _day(cfg: SystemConfig, p: Dict[str, float], *, soil: bool, foul: bool, wick: bool,
         rows: Optional[list] = None):
    Gs, dt = _hourly_profile(cfg, p["day_length_h"])
    T_clean = p["cleaning_interval_days"]
    tau_f = p["fouling_timescale_days"]
    f_soil = 1.0 - p["soiling_rate"] * T_clean / 2.0 if soil else 1.0
    f_foul = (tau_f / T_clean) * (1.0 - math.exp(-T_clean / tau_f)) if foul else 1.0
    f_age = 1.0 - p["flower_degradation"] / 2.0 if foul else 1.0
    fresh_kg, worst, limited, hot = 0.0, float("inf"), 0, None
    coupled = cfg.coupling != "legacy"
    for i, G in enumerate(Gs):
        if G < 100.0:      # below the registry's irradiance floor; yield ~ 0
            continue
        c = replace(cfg, solar_irradiance=G, sunlight_hours=dt,
                    glass_transmittance=cfg.glass_transmittance * f_soil)
        if coupled:
            # fouling/ageing weaken the evaporator inside the balance
            c = replace(c, evaporation_derating=cfg.evaporation_derating * f_foul * f_age)
            r = simulate(c)
            ws = wick_capacity(r.T_evaporator_C, p, cfg.water_density)
            A_eff = r.A_effective
            m_evap = r.m_evap_selected_kg                      # demand the wick has to meet
            m_wick = ws.J_wick_kg_m2_s * A_eff * dt * 3600.0 if wick else float("inf")
            if wick and ws.J_wick_kg_m2_s <= 0.0:
                fresh = 0.0                                    # wick cannot lift water: no supply at all
            else:
                if wick and m_wick < m_evap:
                    # supply-limited: re-solve with the wick flux as a cap (evaporator temperature rises)
                    r = simulate(replace(c, wick_supply_flux_limit=max(min(ws.J_wick_kg_m2_s, 1.0), 1e-9)))
                fresh = r.m_freshwater_kg
        else:
            r = simulate(c)
            # Fouling/ageing act on the evaporated mass directly. Routing them through
            # eta_thermal would heat the evaporator and RAISE the condensation ceiling
            # (a flaw of the legacy model: see README section 8), inflating yield.
            m_evap = r.m_evap_selected_kg * f_foul * f_age
            ws = wick_capacity(r.T_evaporator_C, p, cfg.water_density)
            m_wick = ws.J_wick_kg_m2_s * r.A_effective * dt * 3600.0 if wick else float("inf")
            fresh = min(m_evap, m_wick, r.m_condensation_capacity_kg) * cfg.collection_efficiency
        if wick and m_evap > 0:
            ratio = m_wick / m_evap
            worst = min(worst, ratio)
            if ratio < 1.0:
                limited += 1
        if hot is None or r.T_evaporator_C > hot[0]:
            hot = (r.T_evaporator_C, ws)
        fresh_kg += fresh
        if rows is not None:
            rows.append({"hour": round((i + 0.5) * dt, 2), "G_W_m2": G, "T_evap_C": r.T_evaporator_C,
                         "T_glass_C": r.T_glass_C, "T_condensate_C": r.T_condensate_C,
                         "evaporated_kg": r.m_evap_selected_kg if coupled else m_evap,
                         "wick_capacity_kg": m_wick,
                         "condensation_cap_kg": r.m_condensation_capacity_kg, "freshwater_kg": fresh})
    prod = fresh_kg / (cfg.water_density / 1000.0) / cfg.evaporator_area_geometric
    return prod, worst, limited, hot, dict(f_soil=f_soil, f_foul=f_foul, f_age=f_age)


def realworld_yield(cfg: Optional[SystemConfig] = None,
                    overrides: Optional[Dict[str, float]] = None) -> RealWorldResult:
    cfg = cfg or SystemConfig()
    p = resolve(overrides)
    ideal = simulate(cfg).productivity_L_per_m2_day
    s1, *_ = _day(cfg, p, soil=False, foul=False, wick=False)
    s2, *_ = _day(cfg, p, soil=True, foul=False, wick=False)
    s3, *_ = _day(cfg, p, soil=True, foul=True, wick=False)
    rows: list = []
    s4, worst, limited, hot, fac = _day(cfg, p, soil=True, foul=True, wick=True, rows=rows)
    ws = hot[1] if hot else wick_capacity(cfg.ambient_temperature, p, cfg.water_density)

    warnings: List[str] = []
    if not ws.lift_ok:
        warnings.append(
            f"Wick cannot lift water: Jurin height {ws.h_jurin_m*100:.1f} cm < wick length "
            f"{p['wick_length']*100:.1f} cm. Zero evaporation supply. Use finer pores, a lower "
            "contact angle or a shorter wick."
        )
    elif limited:
        warnings.append(
            f"Capillary supply is below evaporation demand in {limited} hour(s) "
            f"(worst supply/demand = {worst:.2f}): dry-out and local salt crust risk."
        )
        if cfg.coupling != "legacy" and hot is not None and hot[0] > 100.0:
            warnings.append(
                f"With the wick as the bottleneck the evaporator runs at up to {hot[0]:.0f} C (the absorbed heat "
                "has nowhere else to go): that is a dried-out, superheated surface, i.e. salt crust and damage risk, "
                "not a stable evaporating state."
            )
    base = simulate(cfg)
    if (cfg.coupling == "legacy" and base.condensation_limited_by != ""
            and base.m_condensation_capacity_kg < base.m_evap_selected_kg):
        warnings.append(
            "The glass/condensation ceiling binds in this configuration, so evaporator-side limits "
            "(fouling, ageing, capillary supply) are partly MASKED. That ceiling rests on heuristics "
            "(assumed T_glass and h_cond); treat the masking as model-dependent, not as robustness."
        )
    if fac["f_soil"] < 0.9:
        warnings.append("Soiling loss above 10 % between cleanings: shorten cleaning interval.")
    steps = [("Ideal (flat sun, clean, unlimited supply)", ideal),
             ("Diurnal sun profile", s1), ("+ dust soiling", s2),
             ("+ fouling & flower ageing", s3), ("+ capillary supply limit", s4)]
    return RealWorldResult(steps, ideal, s4, s4 / ideal if ideal > 0 else 0.0, ws,
                           worst if worst != float("inf") else float("inf"), limited, rows, fac, warnings)
