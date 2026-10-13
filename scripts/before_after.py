"""Legacy-vs-coupled comparison table (the numbers for the paper update).

    python scripts/before_after.py

Writes results/before_after.csv and docs/before_after.md.  Every number is computed here from the two
solvers (coupling="legacy" / "coupled"); nothing is typed in.  Nothing in the table is validation:
both columns are model output, and the coupled column rests on assumed delta, h_gap, h_cond and leakage.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_model import enhanced_config, moderate_ejector_config     # noqa: E402
from src.physics.system import SystemConfig, simulate               # noqa: E402
from src.scaling.monte_carlo import ARTIFACT_PARAMETERS, monte_carlo  # noqa: E402
from src.scaling.sensitivity import default_tornado_ranges, tornado   # noqa: E402
from src.scaling.sizing import HouseholdDemand, size_for_village    # noqa: E402

L, C = "legacy", "coupled"
rows = []


def add(section, metric, leg, cpl, note=""):
    rows.append({"section": section, "metric": metric, "legacy": leg, "coupled": cpl, "note": note})


def base(coupling, **kw):
    return SystemConfig(coupling=coupling, **kw)


# ---------------------------------------------------------------- 1. baseline
b = {k: simulate(base(k)) for k in (L, C)}
add("baseline", "yield [L/day]", b[L].daily_freshwater_L, b[C].daily_freshwater_L,
    "legacy default mode = energy_limited; coupled default mode = mass_transfer (diffusion)")
add("baseline", "T_evaporator [C]", b[L].T_evaporator_C, b[C].T_evaporator_C)
add("baseline", "T_glass [C]", b[L].T_glass_C, b[C].T_glass_C,
    "legacy: T_amb + 0.35 (T_e - T_amb) heuristic; coupled: solved inner surface")
add("baseline", "T_condensate film [C]", b[L].T_condensate_C, b[C].T_condensate_C, "legacy has no film node (= T_glass)")
add("baseline", "emergent thermal efficiency, latent / absorbed [-]", b[L].Q_useful_J / b[L].Q_absorbed_J,
    b[C].thermal_efficiency_actual, "legacy: the INPUT eta_thermal = 0.65; coupled: result")
add("baseline", "energy efficiency, product latent / incident [-]", b[L].energy_efficiency, b[C].energy_efficiency)

# ---------------------------------------------------------------- 2. limits
add("limits", "evaporation prediction [kg/session]", b[L].m_evap_selected_kg, b[C].m_evap_selected_kg,
    "legacy: Q_useful/h_fg (input eta); coupled: diffusion solution")
add("limits", "condensation / cover capacity [kg/session]", b[L].m_condensation_capacity_kg, b[C].m_condensation_capacity_kg,
    "legacy: U A (T_e - T_amb) cap; coupled: ideal-evaporator (mode A) limit on the same cover")
add("limits", "binding resistance (diagnostic)", b[L].condensation_limited_by, b[C].limited_by)

# ---------------------------------------------------------------- 3. enhanced
e = {k: simulate(enhanced_config(k)) for k in (L, C)}
add("enhanced configuration", "yield [L/day]", e[L].daily_freshwater_L, e[C].daily_freshwater_L,
    "gain 2.0, eta_th 0.72 (legacy only), ejector 2 L/s through 8 mm throat, h_cond 12000, glass 1.4 m2")
add("enhanced configuration", "uplift over own baseline [%]", 100 * (e[L].daily_freshwater_L / b[L].daily_freshwater_L - 1),
    100 * (e[C].daily_freshwater_L / b[C].daily_freshwater_L - 1))
add("enhanced configuration", "chamber pressure [Pa]", e[L].P_chamber_Pa, e[C].P_chamber_Pa,
    f"coupled: ejector asks for {e[C].P_chamber_set_Pa:.0f} Pa (leak floor); the cover holds {e[C].P_chamber_Pa:.0f}")
add("enhanced configuration", "boiling-limited", "n/a", str(e[C].boiling_limited))
add("enhanced configuration", "ejector pump power [W]", e[L].ejector_power_W, e[C].ejector_power_W, "parasitic, not in the yield")
add("enhanced configuration", "vacuum load on cover [kN]", e[L].vacuum_load_kN, e[C].vacuum_load_kN)

# ---------------------------------------------------------------- 4. one-at-a-time changes
def oat(label, **kw):
    out = []
    for k in (L, C):
        y0 = simulate(base(k)).daily_freshwater_L
        y1 = simulate(base(k, **kw)).daily_freshwater_L
        out.append((y0, y1, y1 - y0, 100 * (y1 / y0 - 1)))
    add("one-at-a-time", f"{label}: yield [L/day]", f"{out[0][0]:.3f} -> {out[0][1]:.3f} ({out[0][3]:+.1f} %)",
        f"{out[1][0]:.3f} -> {out[1][1]:.3f} ({out[1][3]:+.1f} %)")


oat("G_opt 1.5 -> 2.0", optical_gain=2.0)
oat("glass area 1.1 -> 1.5 m2", glass_area=1.5)
oat("glass area 1.1 -> 2.5 m2", glass_area=2.5)
oat("eta_thermal 0.65 -> 0.72", eta_thermal=0.72)
oat("eta_thermal 0.65 -> 0.45", eta_thermal=0.45)
oat("area utilisation 0.85 -> 1.0", area_utilization_factor=1.0)
oat("wind 2 -> 5 m/s", wind_speed=5.0)
oat("Venturi on, default ejector (1 L/s, 10 mm throat)", venturi_enabled=True)
m_leg, m_cpl = moderate_ejector_config(L), moderate_ejector_config(C)
add("one-at-a-time", "Venturi on, moderate ejector (0.5 L/s, 8 mm): yield [L/day]",
    f"{b[L].daily_freshwater_L:.3f} -> {simulate(m_leg).daily_freshwater_L:.3f}",
    f"{b[C].daily_freshwater_L:.3f} -> {simulate(m_cpl).daily_freshwater_L:.3f}",
    f"coupled: P_chamber {simulate(m_cpl).P_chamber_Pa:.0f} Pa, pump {simulate(m_cpl).ejector_power_W:.0f} W")
add("one-at-a-time", "Venturi toggle changes yield?", str(simulate(base(L, venturi_enabled=True)).daily_freshwater_L
                                                     != b[L].daily_freshwater_L),
    str(simulate(base(C, venturi_enabled=True)).daily_freshwater_L != b[C].daily_freshwater_L))

# ---------------------------------------------------------------- 5. tornado
tor = {k: tornado(default_tornado_ranges(k), base(k)) for k in (L, C)}
for i in range(8):
    tl, tc = tor[L][i], tor[C][i]
    add("tornado (top 8 by range)", f"rank {i+1}", f"{tl.parameter} ({tl.range_output:.2f} L/day)",
        f"{tc.parameter} ({tc.range_output:.2f} L/day)")
rank_leg = [t.parameter for t in tor[L]].index("eta_thermal") + 1
add("tornado (top 8 by range)", "rank of eta_thermal", f"{rank_leg} of {len(tor[L])}", "not on the coupled path (zero effect)")

# ---------------------------------------------------------------- 6. Monte Carlo
for k in (L, C):
    pass
mc = {(k, ex): monte_carlo(2000, base(k), seed=42, exclude=ARTIFACT_PARAMETERS if ex else None)
      for k in (L, C) for ex in (False, True)}
for ex, label in ((False, "all parameters"), (True, "artifact parameters (eta_thermal, f_util) excluded")):
    for stat in ("mean", "std", "p05", "p95"):
        add("Monte Carlo (N=2000, seed 42)", f"{stat} [L/day], {label}", getattr(mc[(L, ex)], stat),
            getattr(mc[(C, ex)], stat))
rho = {k: mc[(k, False)].spearman() for k in (L, C)}
add("Monte Carlo (N=2000, seed 42)", "Spearman rho: eta_thermal vs yield", rho[L]["eta_thermal"], rho[C]["eta_thermal"])
add("Monte Carlo (N=2000, seed 42)", "Spearman rho: area_utilization_factor vs yield",
    rho[L]["area_utilization_factor"], rho[C]["area_utilization_factor"])
add("Monte Carlo (N=2000, seed 42)", "widths", "legacy list (13 parameters)",
    "legacy list + glass_transmittance, glass_area, insulation_k, gap_convection (30 %), delta (40 %), alpha_g (30 %)")

# ---------------------------------------------------------------- 7. glass-to-evaporator ratio
def legacy_ratio():
    f = lambda r: (lambda x: x.m_condensation_capacity_kg - x.m_evap_selected_kg)(simulate(base(L, glass_area=r)))
    return brentq(f, 0.3, 20.0, xtol=1e-6)


add("glass-to-evaporator ratio", "legacy: condensation capacity >= evaporation (A_g/A_geo)", legacy_ratio(), "")
ratios = np.geomspace(0.8, 400.0, 400)
yA = np.array([simulate(base(C, evaporation_mode="energy_limited", glass_area=r)).daily_freshwater_L for r in ratios])
r99 = ratios[np.argmax(yA >= 0.99 * yA[-1])]
add("glass-to-evaporator ratio", "coupled, mode A: yield within 1 % of its asymptote", "", r99)
add("glass-to-evaporator ratio", "coupled, mode A: asymptote [L/day]", "", yA[-1])
yC = np.array([simulate(base(C, glass_area=r)).daily_freshwater_L for r in ratios[:200]])
lim = [simulate(base(C, glass_area=r)).limited_by for r in ratios[:200]]
flip = next((r for r, l in zip(ratios, lim) if l == "evaporation"), float("nan"))
add("glass-to-evaporator ratio", "coupled, mode C: vapour gap becomes the dominant resistance (limited_by flips to 'evaporation')",
    "", flip)
def valid_min_ratio():
    """Smallest cover/evaporator ratio at which the chamber does not need more than ambient pressure
    (below it the cover would have to run hotter than the boiling point: steam venting, model invalid)."""
    f = lambda r: simulate(base(C, evaporation_mode="energy_limited", glass_area=r)).P_chamber_Pa - 101325.0 * (1 + 1e-9)
    return brentq(f, 0.2, 3.0, xtol=1e-6)


r_valid = valid_min_ratio()
add("glass-to-evaporator ratio", "coupled: smallest ratio without steam venting (cover T_c <= boiling point)", "", r_valid,
    "below this the model is outside its validity (flagged in the result warnings)")
pk = [simulate(base(C, glass_area=r)).daily_freshwater_L for r in (r_valid * 1.001, 1.1, 1.5, 2.5, 5.0, 18.0)]
add("glass-to-evaporator ratio", "coupled, mode C: yield at ratio r_valid, 1.1, 1.5, 2.5, 5, 18 [L/day]", "",
    " / ".join(f"{v:.2f}" for v in pk),
    "diffusion mode is highest at the smallest valid cover and FALLS with more glass (colder cover, larger sensible leak)")
capcfg = base(C, evaporation_mode="empirical")
cap_L = capcfg.empirical_flux * capcfg.area_utilization_factor * capcfg.sunlight_hours * capcfg.collection_efficiency
rr = np.geomspace(r_valid * 1.001, 50.0, 200)
yB = np.array([simulate(replace(capcfg, glass_area=r)).daily_freshwater_L for r in rr])
add("glass-to-evaporator ratio", "coupled, mode B: J_emp evaporation cap binds from ratio", "",
    rr[int(np.argmax(yB >= cap_L - 1e-9))],
    f"cap = {cap_L:.3f} L/day is below the ideal-evaporator limit ({yA[-1]:.2f}) at every valid ratio, so the cap binds throughout")

# ---------------------------------------------------------------- 8. Kalpar sizing
demand = HouseholdDemand(people=4, drinking_per_person_L=4.0, cooking_per_person_L=2.0, safety_factor=1.20)
for label, f in (("default module", lambda k: base(k)), ("enhanced module", enhanced_config)):
    sl, sc = size_for_village(61, demand, f(L)), size_for_village(61, demand, f(C))
    add("Kalpar sizing (61 households x 4 persons x 7.2 L/person/day = 1756.8 L/day)", f"{label}: productivity [L/m2/day]",
        sl.productivity_L_per_m2_day, sc.productivity_L_per_m2_day)
    add("Kalpar sizing (61 households x 4 persons x 7.2 L/person/day = 1756.8 L/day)", f"{label}: required area [m2]",
        sl.A_required_m2, sc.A_required_m2)
    add("Kalpar sizing (61 households x 4 persons x 7.2 L/person/day = 1756.8 L/day)", f"{label}: 1 m2 modules",
        sl.modules_required, sc.modules_required)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "before_after.csv", index=False)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


md = ["# Legacy vs coupled: before / after", "",
      "Generated by `scripts/before_after.py` from the two solvers (`coupling=\"legacy\"` / `\"coupled\"`). "
      "Not validation: both columns are model output; the coupled column depends on ASSUMED delta, h_gap, "
      "h_cond, glass absorptance and ejector leakage (see `docs/physics.md`, decisions list).", ""]
for sec in dict.fromkeys(df["section"]):
    md += [f"## {sec}", "", "| metric | legacy | coupled | note |", "|---|---|---|---|"]
    for _, r in df[df["section"] == sec].iterrows():
        md.append(f"| {r['metric']} | {fmt(r['legacy'])} | {fmt(r['coupled'])} | {r['note']} |")
    md.append("")
(ROOT / "docs" / "before_after.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
