"""Cost, carbon and green-cover model layered on top of the module sizer.

Scope: cradle-to-gate embodied carbon of materials, replacement parts,
operating electricity (if a pump is used), and a levelised cost of water
(LCOW). End-of-life, transport and water-treatment chemicals are NOT
modelled.

PROVENANCE: almost every price and emission factor below is tagged ASSUMED.
They are placeholders to be replaced with supplier quotes / measured data /
a cited LCA database. Results are scale-up *projections*, not validated
figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config.defaults import Parameter, ParameterSource

_A = ParameterSource.ASSUMED
_L = ParameterSource.LITERATURE

SUSTAINABILITY_REGISTRY: Dict[str, Parameter] = {}


def _reg(*args) -> None:
    p = Parameter(*args)
    SUSTAINABILITY_REGISTRY[p.name] = p


# ---- materials: mass basis, unit cost [INR/kg], embodied carbon [kgCO2e/kg]
_reg("glass_density", "rho_g", "kg/m^3", 2500.0, (2000.0, 3000.0), _L, "Soda-lime glass, typical.")
_reg("glass_cost", "c_glass", "INR/kg", 120.0, (20.0, 1000.0), _A, "PLACEHOLDER - replace with supplier quote.")
_reg("glass_embodied_co2", "e_glass", "kgCO2e/kg", 1.2, (0.3, 5.0), _A, "PLACEHOLDER - replace with cited LCA database value.")
_reg("frame_mass_per_m2", "m_frame", "kg/m^2", 8.0, (1.0, 50.0), _A, "Basin/frame/raft mass per m^2 of module. Design-dependent.")
_reg("frame_cost", "c_frame", "INR/kg", 100.0, (20.0, 1000.0), _A, "PLACEHOLDER.")
_reg("frame_embodied_co2", "e_frame", "kgCO2e/kg", 2.0, (0.3, 10.0), _A, "PLACEHOLDER (material-dependent: steel/HDPE/Al differ widely).")
_reg("insulation_density", "rho_ins", "kg/m^3", 30.0, (10.0, 200.0), _A, "Foam/wool insulation, typical order.")
_reg("insulation_cost", "c_ins", "INR/kg", 200.0, (20.0, 2000.0), _A, "PLACEHOLDER.")
_reg("insulation_embodied_co2", "e_ins", "kgCO2e/kg", 3.0, (0.3, 15.0), _A, "PLACEHOLDER.")
_reg("cotton_mass_per_flower", "m_cot", "kg", 0.005, (0.0005, 0.1), _A, "Cotton per 3D flower; measure on prototype.")
_reg("cotton_cost", "c_cot", "INR/kg", 400.0, (50.0, 3000.0), _A, "PLACEHOLDER.")
_reg("cotton_embodied_co2", "e_cot", "kgCO2e/kg", 5.0, (0.5, 20.0), _A, "PLACEHOLDER.")
_reg("fes_mass_per_flower", "m_fes", "kg", 0.001, (0.0001, 0.05), _A, "FeS coating per flower; measure on prototype.")
_reg("fes_cost", "c_fes", "INR/kg", 1000.0, (50.0, 20000.0), _A, "PLACEHOLDER.")
_reg("fes_embodied_co2", "e_fes", "kgCO2e/kg", 3.0, (0.3, 50.0), _A, "PLACEHOLDER.")
_reg("flower_life_years", "L_fl", "yr", 2.0, (0.25, 15.0), _A, "Cotton/FeS replacement interval. Needs durability testing.")
# ---- labour, overheads, scaling
_reg("assembly_hours_per_m2", "h_asm", "h/m^2", 3.0, (0.1, 40.0), _A, "Local assembly labour per m^2.")
_reg("wage", "w", "INR/h", 60.0, (10.0, 1000.0), _A, "PLACEHOLDER - local wage.")
_reg("fixed_setup_cost", "C_fix", "INR", 50000.0, (0.0, 1e8), _A, "Tooling/jigs/training, one-off, independent of module count.")
_reg("overhead_fraction", "f_oh", "-", 0.10, (0.0, 1.0), _A, "Transport/contingency as fraction of variable capex.")
_reg("learning_rate", "LR", "-", 0.90, (0.6, 1.0), _A,
     "Wright-law: unit variable cost x LR per doubling of module count. 1.0 = no economies of scale. Unverified for this device.")
# ---- operation
_reg("operating_days", "d_op", "d/yr", 300.0, (100.0, 365.0), _A, "Productive days per year.")
_reg("cleaning_hours_per_m2_year", "h_om", "h/(m^2*yr)", 2.0, (0.0, 50.0), _A, "Glass cleaning + brine handling labour.")
_reg("system_life_years", "T", "yr", 15.0, (1.0, 40.0), _A, "Design life of frame/glass.")
_reg("discount_rate", "r", "-", 0.08, (0.0, 0.30), _A, "For LCOW discounting.")
_reg("pump_power_W", "P_pump", "W", 0.0, (0.0, 5000.0), _A, "Per module. 0 = fully passive.")
_reg("pump_hours_per_day", "t_pump", "h/d", 8.0, (0.0, 24.0), _A, "")
_reg("electricity_price", "c_el", "INR/kWh", 8.0, (0.0, 50.0), _A, "PLACEHOLDER.")
_reg("grid_emission_factor", "EF_grid", "kgCO2e/kWh", 0.75, (0.0, 1.2), _L,
     "Approximate Indian grid average; confirm exact CEA figure/year before publishing.")
# ---- green cover
_reg("tree_sequestration", "s_tree", "kgCO2/tree/yr", 21.0, (5.0, 60.0), _L,
     "Commonly cited mature-tree average; species/climate dependent - confirm citation.")
_reg("tree_survival", "f_surv", "-", 0.80, (0.3, 1.0), _A, "Fraction of planted trees surviving.")
_reg("tree_maturity_years", "t_mat", "yr", 10.0, (1.0, 40.0), _A, "Years for linear ramp to full sequestration.")
_reg("trees_per_hectare", "n_ha", "1/ha", 1000.0, (100.0, 3000.0), _A, "Planting density.")


def resolve_params(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    vals = {k: p.default for k, p in SUSTAINABILITY_REGISTRY.items()}
    for k, v in (overrides or {}).items():
        if k not in SUSTAINABILITY_REGISTRY:
            raise KeyError(f"Unknown sustainability parameter '{k}'.")
        vals[k] = SUSTAINABILITY_REGISTRY[k].validate(v)
    return vals


@dataclass
class LifecycleResult:
    n_modules: int
    bom: List[dict]                 # per-module bill of materials
    capex_total_INR: float
    capex_per_module_INR: float
    scale_cost_multiplier: float
    embodied_co2_total_kg: float
    annual_opex_INR: float
    annual_operational_co2_kg: float
    annual_water_L: float
    lcow_INR_per_m3: float
    co2_kg_per_m3: float
    trees_steady_state: int
    trees_lifetime_balance: int
    green_area_ha: float
    net_zero_year: Optional[int]
    yearly: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def bill_of_materials(area_m2: float, flowers: int, glass_area_ratio: float,
                      glass_thickness_m: float, p: Dict[str, float]) -> List[dict]:
    """Per-module BOM. Quantities in kg; cost/CO2 are un-scaled."""
    q = {
        "glass cover": (area_m2 * glass_area_ratio * glass_thickness_m * p["glass_density"],
                        p["glass_cost"], p["glass_embodied_co2"]),
        "frame / basin": (area_m2 * p["frame_mass_per_m2"], p["frame_cost"], p["frame_embodied_co2"]),
        "insulation": (area_m2 * 0.05 * p["insulation_density"], p["insulation_cost"], p["insulation_embodied_co2"]),
        "cotton flowers": (flowers * p["cotton_mass_per_flower"], p["cotton_cost"], p["cotton_embodied_co2"]),
        "FeS coating": (flowers * p["fes_mass_per_flower"], p["fes_cost"], p["fes_embodied_co2"]),
    }
    return [{"item": k, "mass_kg": m, "cost_INR": m * c, "embodied_co2_kg": m * e,
             "replaceable": k in ("cotton flowers", "FeS coating")}
            for k, (m, c, e) in q.items()]


def analyse(ms, cfg, overrides: Optional[Dict[str, float]] = None,
            horizon_years: Optional[int] = None) -> LifecycleResult:
    """ms: ModuleSizing (from size_modules); cfg: SystemConfig."""
    p = resolve_params(overrides)
    warnings: List[str] = []
    if not math.isfinite(ms.A_module_m2) or ms.A_module_m2 <= 0:
        raise ValueError("Module sizing is not finite; fix simulator inputs first.")
    N = ms.n_modules
    T = int(horizon_years or round(p["system_life_years"]))
    ratio = cfg.glass_area / cfg.evaporator_area_geometric
    bom = bill_of_materials(ms.A_module_m2, ms.flowers_per_module, ratio, cfg.glass_thickness, p)

    mat_cost = sum(i["cost_INR"] for i in bom)
    labour = ms.A_module_m2 * p["assembly_hours_per_m2"] * p["wage"]
    unit_var = (mat_cost + labour) * (1 + p["overhead_fraction"])
    mult = N ** math.log2(p["learning_rate"])
    capex = unit_var * mult * N + p["fixed_setup_cost"]

    embodied = N * sum(i["embodied_co2_kg"] for i in bom)
    repl_cost = N * sum(i["cost_INR"] for i in bom if i["replaceable"]) * mult
    repl_co2 = N * sum(i["embodied_co2_kg"] for i in bom if i["replaceable"])

    pump_kwh = N * p["pump_power_W"] / 1000 * p["pump_hours_per_day"] * p["operating_days"]
    op_co2 = pump_kwh * p["grid_emission_factor"]
    opex = (N * ms.A_module_m2 * p["cleaning_hours_per_m2_year"] * p["wage"]
            + pump_kwh * p["electricity_price"])
    water = ms.target_L_per_day * p["operating_days"]

    r = p["discount_rate"]
    pv_cost = capex
    pv_water = 0.0
    yearly: List[dict] = []
    cum_em = embodied
    for t in range(1, T + 1):
        replace = t % p["flower_life_years"] < 1e-9 and t < T if p["flower_life_years"] >= 1 else False
        cost_t = opex + (repl_cost if replace else 0.0)
        em_t = op_co2 + (repl_co2 if replace else 0.0)
        cum_em += em_t
        d = (1 + r) ** t
        pv_cost += cost_t / d
        pv_water += water / d
        yearly.append({"year": t, "water_L": water, "cost_INR": cost_t,
                       "emissions_kg": em_t, "cum_emissions_kg": cum_em})
    if p["flower_life_years"] < 1:
        warnings.append("Flower life < 1 yr: replacement cost/carbon not modelled at sub-annual resolution.")
    yearly[0]["emissions_kg"] += embodied
    total_em = cum_em

    lcow = pv_cost / pv_water * 1000 if pv_water else float("inf")
    co2_m3 = total_em / (water * T / 1000) if water else float("inf")

    # ---- green cover
    per_tree = p["tree_sequestration"] * p["tree_survival"]
    trees_ss = math.ceil((total_em / T) / per_tree) if per_tree else 0

    def seq_cum(trees: float, years: int) -> float:
        return sum(trees * per_tree * min(1.0, t / p["tree_maturity_years"]) for t in range(1, years + 1))

    unit_seq = seq_cum(1, T)
    trees_life = math.ceil(total_em / unit_seq) if unit_seq else 0
    net_zero_year = None
    cum_seq = 0.0
    cum_em2 = 0.0
    for row in yearly:
        cum_seq += trees_life * per_tree * min(1.0, row["year"] / p["tree_maturity_years"])
        cum_em2 += row["emissions_kg"]
        row["cum_sequestered_kg"] = cum_seq
        row["net_kg"] = cum_em2 - cum_seq
        if net_zero_year is None and row["net_kg"] <= 0:
            net_zero_year = row["year"]
    if trees_life > 0 and net_zero_year is None:
        warnings.append("Net zero not reached within the horizon.")
    if T < p["tree_maturity_years"]:
        warnings.append("Horizon shorter than tree maturity; sequestration is heavily back-loaded.")

    return LifecycleResult(
        n_modules=N, bom=bom, capex_total_INR=capex, capex_per_module_INR=capex / N,
        scale_cost_multiplier=mult, embodied_co2_total_kg=embodied, annual_opex_INR=opex,
        annual_operational_co2_kg=op_co2, annual_water_L=water, lcow_INR_per_m3=lcow,
        co2_kg_per_m3=co2_m3, trees_steady_state=trees_ss, trees_lifetime_balance=trees_life,
        green_area_ha=trees_life / p["trees_per_hectare"], net_zero_year=net_zero_year,
        yearly=yearly, warnings=warnings,
    )
