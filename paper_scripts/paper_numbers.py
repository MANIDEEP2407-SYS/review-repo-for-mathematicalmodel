"""Re-derives every number used in the coupled-model version of the ITSEMC-2026 paper.
Nothing is typed in: all values come from running solar_desalination_simulator (read-only import).
Writes paper_numbers.json next to this file.   python -B paper_numbers.py [--mc]"""
import json, os, sys, math
from dataclasses import replace
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..")
sys.path.insert(0, SIM)
from run_model import enhanced_config, moderate_ejector_config           # noqa: E402
from src.physics.system import SystemConfig, simulate                     # noqa: E402
from src.physics.realworld import realworld_yield                         # noqa: E402
from src.physics.isolated import isolated_evaporator, enhancement_to_match  # noqa: E402
from src.scaling.sizing import HouseholdDemand, size_for_village          # noqa: E402
from src.scaling.sensitivity import default_tornado_ranges, tornado       # noqa: E402
from src.scaling.monte_carlo import monte_carlo, ARTIFACT_PARAMETERS      # noqa: E402
from src.config.defaults import PARAMETER_REGISTRY                        # noqa: E402

L, C = "legacy", "coupled"
out = {}


def cfg(k, **kw):
    return SystemConfig(coupling=k, **kw)


def brief(r):
    return dict(L=r.daily_freshwater_L, Te=r.T_evaporator_C, Tg=r.T_glass_C, Tc=r.T_condensate_C,
                P=r.P_chamber_Pa, Pset=r.P_chamber_set_Pa, boil=r.boiling_limited, pump=r.ejector_power_W,
                kWh_m3=r.ejector_kWh_per_m3, vac_kN=r.vacuum_load_kN, vac_t=r.vacuum_load_tonnes,
                stress=r.cover_plate_stress_MPa, implaus=r.cover_vacuum_implausible, conv=r.solver_converged,
                res=r.solver_residual_max, limited=r.limited_by, warns=list(r.warnings), floor=r.ejector_binding_floor)


# ---- baseline, both solvers, all modes
for k in (L, C):
    r = simulate(cfg(k))
    d = brief(r)
    d.update(mevap_A=r.m_evap_energy_limited_kg, mevap_B=r.m_evap_empirical_kg, mevap_C=r.m_evap_mass_transfer_kg,
             mcap=r.m_condensation_capacity_kg, Uglass=r.U_glass, rec=r.recovery_ratio, brine=r.brine_tds_mg_per_L,
             cf=r.concentration_factor, eff_energy=r.energy_efficiency, eff_th=r.thermal_efficiency_actual,
             Qinc=r.Q_solar_incident_J / 1e6, Qabs=r.Q_absorbed_J / 1e6, Quse=r.Q_useful_J / 1e6,
             Qconv=r.Q_convection_W, Qrad=r.Q_radiation_W, Qcond=r.Q_conduction_W, Qgap=r.Q_gap_W,
             Pabs=r.P_absorbed_W, mdot=r.mdot_kg_s, fresh_kg=r.m_freshwater_kg, feed=r.feed_kg, brine_kg=r.brine_kg,
             closure=r.mass_closure_error_kg)
    modes = {}
    for m in ("energy_limited", "empirical", "mass_transfer", "hybrid"):
        rm = simulate(cfg(k, evaporation_mode=m))
        modes[m] = dict(L=rm.daily_freshwater_L, mevap=rm.m_evap_selected_kg, Te=rm.T_evaporator_C)
    d["modes"] = modes
    out["baseline_" + k] = d
c = out["baseline_coupled"]
# coupled energy balance (average power over the session)
r = simulate(cfg(C))
e_evap = c["mdot"] * 0  # filled below
from src.physics.coupled import hfg
ev = c["mdot"] * (hfg(r.T_evaporator_C) + 4186.0 * (r.T_evaporator_C - 30.0))
out["coupled_balance"] = dict(Pabs=c["Pabs"], evap=ev, gap=c["Qgap"], gap_conv=c["Qconv"], gap_rad=c["Qrad"],
                              ins=c["Qcond"], resid=c["Pabs"] - ev - c["Qgap"] - c["Qcond"],
                              Pg=0.05 * 800 * 1.5 * 1.1, Qout=r.Q_ideal_cover_W)

# ---- enhanced
for k in (L, C):
    out["enhanced_" + k] = brief(simulate(enhanced_config(k)))
out["moderate_ejector_coupled"] = brief(simulate(moderate_ejector_config(C)))
out["venturi_default_coupled"] = brief(simulate(cfg(C, venturi_enabled=True)))
out["venturi_default_legacy"] = brief(simulate(cfg(L, venturi_enabled=True)))

# ---- one-at-a-time
oat = {}
for name, kw in {"gain2": dict(optical_gain=2.0), "glass1.5": dict(glass_area=1.5), "glass2.5": dict(glass_area=2.5),
                 "eta0.72": dict(eta_thermal=0.72), "eta0.45": dict(eta_thermal=0.45),
                 "futil1": dict(area_utilization_factor=1.0), "wind5": dict(wind_speed=5.0),
                 "venturi": dict(venturi_enabled=True), "gain1": dict(optical_gain=1.0),
                 "G700": dict(solar_irradiance=700.0)}.items():
    oat[name] = {k: (simulate(cfg(k)).daily_freshwater_L, simulate(cfg(k, **kw)).daily_freshwater_L) for k in (L, C)}
out["oat"] = oat

# ---- wind by mode (coupled)
out["wind"] = {m: {w: simulate(cfg(C, evaporation_mode=m, wind_speed=w)).daily_freshwater_L for w in (0.5, 2.0, 5.0)}
               for m in ("energy_limited", "mass_transfer")}

# ---- delta / h_gap / other assumed-parameter sensitivities (coupled baseline)
sens = {}
for pname, vals in {"diffusion_path_length": (0.005, 0.01, 0.02, 0.05, 0.30),
                    "gap_convection_coefficient": (0.5, 2.0, 5.0),
                    "glass_absorptance": (0.02, 0.05, 0.10),
                    "condensation_coefficient": (2000.0, 8000.0, 15000.0),
                    "chamber_leak_pressure_floor": (5000.0, 10000.0, 60000.0),
                    "motive_water_temperature": (15.0, 30.0, 45.0),
                    "pump_efficiency": (0.3, 0.5, 0.7),
                    "glass_allowable_stress": (1e7, 2e7, 5e7)}.items():
    sens[pname] = {str(v): brief(simulate(cfg(C, **{pname: v}))) | {} for v in vals}
    for v in vals:   # ejector-related ones only matter with the ejector on
        if pname in ("chamber_leak_pressure_floor", "motive_water_temperature", "pump_efficiency", "glass_allowable_stress"):
            sens[pname][str(v)] = brief(simulate(cfg(C, venturi_enabled=True, **{pname: v})))
out["assumed_sens"] = {p: {v: dict(L=d["L"], Te=d["Te"], pump=d["pump"], implaus=d["implaus"], P=d["P"]) for v, d in dd.items()}
                       for p, dd in sens.items()}
out["registry"] = {n: dict(default=p.default, range=list(p.allowed_range), tier=p.source.value, unit=p.unit,
                           notes=p.notes) for n, p in PARAMETER_REGISTRY.items()
                   if n in ("diffusion_path_length", "gap_convection_coefficient", "glass_absorptance",
                            "chamber_leak_pressure_floor", "motive_water_temperature", "pump_efficiency",
                            "glass_allowable_stress", "condensation_coefficient", "empirical_flux", "eta_thermal",
                            "mass_transfer_coefficient", "h_fg", "relative_humidity", "water_specific_heat")}

# ---- glass-to-evaporator ratio
def valid_min_ratio(**kw):
    A = kw.get("evaporator_area_geometric", 1.0)
    f = lambda r: simulate(cfg(C, evaporation_mode="energy_limited", glass_area=r * A, **kw)).P_chamber_Pa - 101325.0 * (1 + 1e-9)
    return brentq(f, 0.2, 3.0, xtol=1e-6)


rv = valid_min_ratio()
ratios = np.geomspace(0.8, 400.0, 400)
yA = np.array([simulate(cfg(C, evaporation_mode="energy_limited", glass_area=r)).daily_freshwater_L for r in ratios])
out["ratio"] = dict(r_valid=rv, Te_at_0p8=simulate(cfg(C, glass_area=0.8)).T_evaporator_C, r99=float(ratios[np.argmax(yA >= 0.99 * yA[-1])]), asymA=float(yA[-1]),
                    C_at={str(r): simulate(cfg(C, glass_area=r)).daily_freshwater_L for r in (round(rv * 1.001, 4), 1.1, 1.5, 2.5, 5.0, 18.0)},
                    B_cap=simulate(cfg(C, evaporation_mode="empirical")).daily_freshwater_L,
                    legacy_cross=brentq(lambda r: (lambda x: x.m_condensation_capacity_kg - x.m_evap_selected_kg)(
                        simulate(cfg(L, glass_area=r))), 0.3, 20.0, xtol=1e-6))
# mode-C ratio curves for several delta (for the figure)
rr = np.geomspace(rv * 1.001, 20.0, 60)
curves = {"A": [simulate(cfg(C, evaporation_mode="energy_limited", glass_area=float(r))).daily_freshwater_L for r in rr],
          "B": [simulate(cfg(C, evaporation_mode="empirical", glass_area=float(r))).daily_freshwater_L for r in rr]}
curves["legacy"] = [simulate(cfg(L, glass_area=float(r))).daily_freshwater_L for r in rr]
for dl in (0.005, 0.02, 0.05):
    curves[f"C_{dl}"] = [simulate(cfg(C, diffusion_path_length=dl, glass_area=float(r))).daily_freshwater_L for r in rr]
out["ratio_curves"] = dict(r=[float(x) for x in rr], **{k: [float(x) for x in v] for k, v in curves.items()})
out["ratio_delta"] = {str(dl): {str(r): simulate(cfg(C, diffusion_path_length=dl, glass_area=r)).daily_freshwater_L
                                for r in (1.1, 2.5, 18.0)} for dl in (0.005, 0.02, 0.05)}

# ---- tornado + Monte Carlo (both solvers)
for k in (L, C):
    out["tornado_" + k] = [dict(p=t.parameter, lo=t.low_value, hi=t.high_value, lo_out=t.low_output, hi_out=t.high_output,
                                rng=t.range_output, pct=t.percent_change) for t in tornado(default_tornado_ranges(k), cfg(k))]
out["tornado_ejector"] = [dict(p=t.parameter, rng=t.range_output, lo=t.low_value, hi=t.high_value)
                          for t in tornado(default_tornado_ranges(C, venturi=True), cfg(C, venturi_enabled=True))]
N_MC = 2000
for k in (L, C):
    m = monte_carlo(N_MC, cfg(k), seed=42)
    sp = m.spearman()
    out["mc_" + k] = dict(mean=m.mean, std=m.std, median=m.median, p05=m.p05, p95=m.p95, n=N_MC,
                          spearman={a: float(b) for a, b in sp.items()},
                          samples=[float(x) for x in m.daily_L])
    m2 = monte_carlo(N_MC, cfg(k), seed=42, exclude=ARTIFACT_PARAMETERS)
    out["mc_excl_" + k] = dict(mean=m2.mean, std=m2.std, p05=m2.p05, p95=m2.p95)

# ---- derating (hourly) on both solvers
for name, f in (("base", lambda k: cfg(k)), ("enh", enhanced_config)):
    for k in (L, C):
        rw = realworld_yield(f(k))
        out[f"derate_{name}_{k}"] = dict(steps=[(s[0], s[1]) for s in rw.steps], final=rw.final_productivity_L_per_m2_day
                                         if hasattr(rw, "final_productivity_L_per_m2_day") else rw.steps[-1][1],
                                         warns=list(rw.warnings))

# ---- Kalpar sizing (ideal and derated), village table
demand = HouseholdDemand(people=4, drinking_per_person_L=4.0, cooking_per_person_L=2.0, safety_factor=1.20)
siz = {}
for k in (L, C):
    for tag, f in (("default", lambda kk: cfg(kk)), ("enhanced", enhanced_config)):
        s = size_for_village(61, demand, f(k))
        siz[f"{tag}_{k}"] = dict(prod=s.productivity_L_per_m2_day, A=s.A_required_m2, modules=s.modules_required,
                                 target=s.target_L_per_day)
out["sizing"] = siz
out["sizing_derated"] = {k: out[f"derate_base_{k}"]["steps"][-1][1] for k in (L, C)}
out["sizing_derated_A"] = {k: 1756.8 / out["sizing_derated"][k] for k in (L, C)}
out["household_28.8"] = {k: 28.8 / out["baseline_" + k]["L"] for k in (L, C)}

# ---- isolated evaporator (consistency check, F)
iso = isolated_evaporator()
out["isolated"] = dict(flux=iso.flux_kg_m2_h, Ts=iso.T_surface_C, enh_to_match=enhancement_to_match(2.77),
                       one_sun_bound=0.95 * 1000.0 / hfg(25.0) * 3600,
                       env_gain_needed=2.77 / 3600.0 * hfg(25.0) - 0.95 * 1000.0)

# ---- bench prototype: 30 x 20 cm = 0.06 m2, G_opt = 1, 1 L feed
def bench(k, ratio, **kw):
    return cfg(k, evaporator_area_geometric=0.06, glass_area=0.06 * ratio, optical_gain=1.0, feed_volume=1.0, **kw)


proto = {"legacy": {}, "coupled": {}}
for ratio in (1, 2, 3):
    r = simulate(bench(L, ratio))
    proto["legacy"][ratio] = dict(mL=r.daily_freshwater_L * 1000, rec=r.recovery_ratio)
    proto["coupled"][ratio] = {}
    for dl in (0.005, 0.02, 0.05):
        for hg in (0.5, 2.0, 5.0):
            r = simulate(bench(C, ratio, diffusion_path_length=dl, gap_convection_coefficient=hg))
            proto["coupled"][ratio][f"d{dl}_h{hg}"] = dict(mL=r.daily_freshwater_L * 1000, Te=r.T_evaporator_C,
                                                           Tg=r.T_glass_C, boil=r.boiling_limited, conv=r.solver_converged,
                                                           warns=list(r.warnings))
    ra = simulate(bench(C, ratio, evaporation_mode="energy_limited"))
    proto["coupled"][ratio]["modeA"] = dict(mL=ra.daily_freshwater_L * 1000, Te=ra.T_evaporator_C, boil=ra.boiling_limited)
out["proto"] = proto
out["proto_rvalid_gain1"] = valid_min_ratio(evaporator_area_geometric=0.06, optical_gain=1.0)
# bench ejector (prototype B): small motive flow, two sizes
ej = {}
for q, thr in ((5e-5, 0.003), (1e-4, 0.004), (2e-4, 0.005)):
    row = {}
    for ratio in (2,):
        off = simulate(bench(C, ratio))
        on = simulate(bench(C, ratio, venturi_enabled=True, motive_flow_rate=q, venturi_throat_diameter=thr,
                            venturi_inlet_diameter=0.012))
        row = dict(off=off.daily_freshwater_L * 1000, on=on.daily_freshwater_L * 1000, P=on.P_chamber_Pa, pump=on.ejector_power_W,
                   vac_kN=on.vacuum_load_kN, boil=on.boiling_limited, stress=on.cover_plate_stress_MPa,
                   implaus=on.cover_vacuum_implausible, kWh_m3=on.ejector_kWh_per_m3, floor=on.ejector_binding_floor)
    ej[f"{q}_{thr}"] = row
out["proto_ejector"] = ej
out["proto_delta_only"] = {str(dl): {str(ratio): simulate(bench(C, ratio, diffusion_path_length=dl)).daily_freshwater_L * 1000
                                     for ratio in (1, 2, 3)} for dl in (0.005, 0.02, 0.05)}

# ---- Table I-style: parameter list for coupled baseline is read from SystemConfig defaults
json.dump(out, open(os.path.join(HERE, "paper_numbers.json"), "w"), indent=1, default=str)
print("wrote paper_numbers.json")
