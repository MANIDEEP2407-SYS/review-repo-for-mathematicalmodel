"""End-to-end demonstration run of the Solar Desalination Simulator.

Exercises every phase (physics, thermal, condensation, salinity, Venturi, scaling, sensitivity,
Monte Carlo, plausibility checks) and writes CSVs + PNG plots.

    python run_model.py                    # both: coupled -> results/, legacy -> results/legacy/
    python run_model.py --coupling coupled
    python run_model.py --coupling legacy

The legacy outputs are produced by the untouched decoupled solver (coupling="legacy") so the old
results stay reproducible.  A legacy-vs-coupled comparison table is made by scripts/before_after.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.physics.system import SystemConfig, simulate, EVAPORATION_MODES
from src.scaling.sizing import (
    HouseholdDemand, size_for_demand, size_for_household,
    size_for_village, sweep_scale,
)
from src.scaling.sensitivity import (
    VENTURI_TORNADO_RANGES, default_tornado_ranges, sweep_one, tornado,
)
from src.scaling.monte_carlo import ARTIFACT_PARAMETERS, monte_carlo
from src.validation.validation_cases import (
    EMPIRICAL_YIELD_KG_PER_M2_DAY, run_condensation_headroom, run_empirical_linear_scaling,
    run_energy_vs_empirical_consistency, run_isolated_evaporator_consistency,
    run_still_plausibility_check,
)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def enhanced_config(coupling: str) -> SystemConfig:
    """Reflector + ejector scenario of the concept diagram (same inputs for both solvers).
    eta_thermal is only used by the legacy solver."""
    cfg = SystemConfig(coupling=coupling)
    cfg.optical_gain = 2.0                # external mirrors + internal lining
    cfg.eta_thermal = 0.72                # legacy only: better thermal capture
    cfg.venturi_enabled = True
    cfg.motive_flow_rate = 2.0e-3
    cfg.venturi_inlet_diameter = 0.025
    cfg.venturi_throat_diameter = 0.008
    cfg.venturi_loss_fraction = 0.40
    cfg.condensation_coefficient = 12000.0
    cfg.glass_area = 1.4
    return cfg


def moderate_ejector_config(coupling: str) -> SystemConfig:
    """A modest ejector whose set pressure stays above the cover's vapour pressure (no boiling limit)."""
    cfg = SystemConfig(coupling=coupling)
    cfg.venturi_enabled = True
    cfg.motive_flow_rate = 5.0e-4
    cfg.venturi_throat_diameter = 0.008
    return cfg


def run(coupling: str, RESULTS: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    legacy = coupling == "legacy"
    print(f"\n######## coupling = {coupling}  ->  {RESULTS} ########")

    def dump_json(name: str, obj) -> Path:
        p = RESULTS / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    # ======================================================================
    # 1. Baseline simulation (Kalpar-like defaults)
    # ======================================================================
    hr("BASELINE SIMULATION")
    baseline_cfg = SystemConfig(coupling=coupling)
    r0 = simulate(baseline_cfg)
    print(f"Config: G={baseline_cfg.solar_irradiance} W/m^2, t_sun={baseline_cfg.sunlight_hours} h, "
          f"A_geo={baseline_cfg.evaporator_area_geometric} m^2, opt_gain={baseline_cfg.optical_gain}, "
          f"mode={baseline_cfg.evaporation_mode}" + (f", eta_th={baseline_cfg.eta_thermal}" if legacy else
                                                    ", (eta_thermal not used)"))
    print()
    print(f"  Q_incident      : {r0.Q_solar_incident_J/1e6:8.3f} MJ")
    print(f"  Q_absorbed      : {r0.Q_absorbed_J/1e6:8.3f} MJ")
    print(f"  Q_useful (latent): {r0.Q_useful_J/1e6:8.3f} MJ" + ("" if legacy else "   (emergent, not an input)"))
    print(f"  T_evap          : {r0.T_evaporator_C:8.2f} C")
    print(f"  T_glass (inner) : {r0.T_glass_C:8.2f} C")
    print(f"  T_condensate    : {r0.T_condensate_C:8.2f} C")
    print(f"  T_sky           : {r0.T_sky_C:8.2f} C")
    print(f"  Q_convection    : {r0.Q_convection_W:8.2f} W  (gap)" if not legacy else
          f"  Q_convection    : {r0.Q_convection_W:8.2f} W")
    print(f"  Q_radiation     : {r0.Q_radiation_W:8.2f} W")
    print(f"  Q_conduction    : {r0.Q_conduction_W:8.2f} W")
    print(f"  U_glass         : {r0.U_glass:8.3f} W/m^2K")
    print(f"  m_evap (mode={baseline_cfg.evaporation_mode:16s}): {r0.m_evap_selected_kg:6.3f} kg")
    for k, v in r0.mode_predictions_kg.items():
        print(f"     mode {k:14s}: {v:6.3f} kg")
    print(f"  m_cond capacity : {r0.m_condensation_capacity_kg:6.3f} kg  (binding: {r0.condensation_limited_by})")
    print(f"  P_chamber       : {r0.P_chamber_Pa:8.0f} Pa   boiling-limited: {r0.boiling_limited}")
    print(f"  Freshwater      : {r0.daily_freshwater_L:6.3f} L/day")
    print(f"  Productivity    : {r0.productivity_L_per_m2_day:6.3f} L/m^2/day")
    print(f"  Recovery        : {r0.recovery_ratio*100:6.2f} %")
    print(f"  Energy eff.     : {r0.energy_efficiency*100:6.2f} %")
    print(f"  Brine TDS       : {r0.brine_tds_mg_per_L:8.0f} mg/L (CF={r0.concentration_factor:.2f})")
    if not legacy:
        print(f"  Solver converged: {r0.solver_converged}, worst closure residual {r0.solver_residual_max:.2e}")
    if r0.warnings:
        print("\n  Warnings:")
        for w in r0.warnings:
            print(f"    - {w}")
    dump_json("baseline_result.json", {
        k: v for k, v in r0.__dict__.items() if not isinstance(v, (list, dict)) and not hasattr(v, "shape")
    })

    # ======================================================================
    # 2. Reflector-enhanced + ejector scenario (matches the concept diagram)
    # ======================================================================
    hr("REFLECTOR + VENTURI/EJECTOR SCENARIO")
    cfg_enh = enhanced_config(coupling)
    r_enh = simulate(cfg_enh)
    print(f"Freshwater      : {r_enh.daily_freshwater_L:6.3f} L/day  (baseline {r0.daily_freshwater_L:.3f} L/day)")
    print(f"Uplift          : {(r_enh.daily_freshwater_L/max(r0.daily_freshwater_L,1e-6)-1)*100:+.1f} %")
    print(f"P_chamber       : {r_enh.P_chamber_Pa:8.0f} Pa  (set {r_enh.P_chamber_set_Pa:.0f}; atmospheric 101325 Pa)")
    print(f"dT_saturation   : {r_enh.saturation_T_shift_C:6.2f} C boiling-point drop")
    print(f"T_evap / T_glass: {r_enh.T_evaporator_C:6.2f} / {r_enh.T_glass_C:6.2f} C")
    if not legacy:
        print(f"boiling-limited : {r_enh.boiling_limited}   ejector pump power {r_enh.ejector_power_W:.0f} W "
              f"({r_enh.ejector_kWh_per_m3:.0f} kWh/m3 of product)")
        print(f"vacuum load     : {r_enh.vacuum_load_kN:.1f} kN (~{r_enh.vacuum_load_tonnes:.1f} t) on the cover")
    rows = []
    for name, c in (("baseline", baseline_cfg), ("enhanced", cfg_enh), ("moderate_ejector", moderate_ejector_config(coupling))):
        r = simulate(c)
        rows.append({
            "case": name, "L_per_day": r.daily_freshwater_L, "T_evap_C": r.T_evaporator_C,
            "T_glass_inner_C": r.T_glass_C, "T_condensate_C": r.T_condensate_C,
            "P_chamber_Pa": r.P_chamber_Pa, "P_chamber_set_Pa": r.P_chamber_set_Pa,
            "boiling_limited": r.boiling_limited, "vacuum_load_kN": r.vacuum_load_kN,
            "ejector_power_W": r.ejector_power_W, "ejector_kWh_per_m3": r.ejector_kWh_per_m3,
            "limited_by": r.limited_by, "solver_converged": r.solver_converged,
        })
    pd.DataFrame(rows).to_csv(RESULTS / "scenario_diagnostics.csv", index=False)

    # ======================================================================
    # 3. Mode comparison table
    # ======================================================================
    hr("EVAPORATION MODE COMPARISON @ A=1 m^2")
    modes_rows = []
    for mode in EVAPORATION_MODES:
        c = SystemConfig(coupling=coupling); c.evaporation_mode = mode
        r = simulate(c)
        modes_rows.append({
            "mode": mode, "m_evap_kg": r.m_evap_selected_kg, "daily_L": r.daily_freshwater_L,
            "T_evap_C": r.T_evaporator_C, "T_glass_C": r.T_glass_C, "cond_limited_by": r.condensation_limited_by,
        })
    modes_df = pd.DataFrame(modes_rows)
    print(modes_df.to_string(index=False))
    modes_df.to_csv(RESULTS / "mode_comparison.csv", index=False)

    # ======================================================================
    # 4. Household + village scaling
    # ======================================================================
    hr("HOUSEHOLD + VILLAGE SCALING")
    demand = HouseholdDemand(people=4, drinking_per_person_L=4.0, cooking_per_person_L=2.0, safety_factor=1.20)
    print(f"Household demand: {demand.total_L_per_day:.1f} L/day (4 people * 6 L * 1.2 = 7.2 L/person/day)")
    kalpar = []
    for label, ref in (("default module", baseline_cfg), ("enhanced module", cfg_enh)):
        prod = simulate(ref).productivity_L_per_m2_day
        s = size_for_village(61, demand, ref, A_per_module=1.0)
        kalpar.append({"reference": label, "productivity_L_m2_day": prod, "households": 61,
                       "target_L_day": s.target_L_per_day, "A_required_m2": s.A_required_m2,
                       "modules_1m2": s.modules_required})
        print(f"  Kalpar 61 households, {label}: productivity {prod:.3f} L/m2/day -> "
              f"{s.target_L_per_day:.1f} L/day -> A = {s.A_required_m2:.1f} m2 ({s.modules_required} x 1 m2)")
        for w in s.warnings:
            print(f"      note: {w}")
    pd.DataFrame(kalpar).to_csv(RESULTS / "kalpar_sizing.csv", index=False)

    ref = cfg_enh      # as in the original run: the enhanced module is the sizing reference
    prod = simulate(ref).productivity_L_per_m2_day
    print(f"Reference productivity (enhanced): {prod:.3f} L/m^2/day\n")
    for pop in (1, 3, 10, 50, 100, 500):
        s = size_for_village(pop, demand, ref, A_per_module=1.0)
        print(f"  Households={pop:4d} -> target={s.target_L_per_day:8.1f} L/day, "
              f"A_req={s.A_required_m2:6.2f} m^2, modules={s.modules_required:3d}, "
              f"footprint={s.total_footprint_m2:6.2f} m^2")

    pop_rows = sweep_scale([1, 5, 10, 25, 50, 100, 250, 500, 1000], demand.per_person_L * demand.safety_factor,
                           1.0, ref, A_per_module=1.0)
    pop_df = pd.DataFrame(pop_rows)
    pop_df.to_csv(RESULTS / "village_scaling.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pop_df["population"], pop_df["A_required_m2"], "-o", label="Required area (m²)")
    ax.set_xlabel("Population")
    ax.set_ylabel("Required evaporator area [m²]")
    ax.set_title(f"Village scaling ({coupling}): population vs required evaporator area")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(pop_df["population"], pop_df["target_L_per_day"], "-s", color="tab:red", label="Target [L/day]")
    ax2.set_ylabel("Target freshwater [L/day]")
    fig.tight_layout()
    fig.savefig(RESULTS / "village_scaling.png", dpi=140)
    plt.close(fig)

    # ======================================================================
    # 5. Sensitivity tornado
    # ======================================================================
    hr("SENSITIVITY TORNADO")
    entries = tornado(default_tornado_ranges(coupling), baseline_cfg)
    tornado_df = pd.DataFrame([e.__dict__ for e in entries])
    print(tornado_df[["parameter", "low_output", "high_output", "range_output", "percent_change"]].to_string(index=False))
    tornado_df.to_csv(RESULTS / "sensitivity_tornado.csv", index=False)

    def tornado_plot(df, title, fname):
        fig, ax = plt.subplots(figsize=(9, 6))
        d = df.sort_values("range_output")
        ax.barh(d["parameter"], d["range_output"], color="steelblue")
        ax.set_xlabel("Range of daily freshwater [L/day]")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=140)
        plt.close(fig)

    tornado_plot(tornado_df, f"Sensitivity tornado ({coupling}, default baseline)", "sensitivity_tornado.png")

    if not legacy:
        hr("SENSITIVITY TORNADO: ejector parameters (moderate-ejector baseline)")
        v_entries = tornado(VENTURI_TORNADO_RANGES, moderate_ejector_config(coupling))
        v_df = pd.DataFrame([e.__dict__ for e in v_entries])
        print(v_df[["parameter", "low_output", "high_output", "range_output", "percent_change"]].to_string(index=False))
        v_df.to_csv(RESULTS / "sensitivity_tornado_ejector.csv", index=False)
        tornado_plot(v_df, "Sensitivity tornado: ejector parameters (coupled)", "sensitivity_tornado_ejector.png")

    # ======================================================================
    # 6. Parameter sweeps
    # ======================================================================
    hr("PARAMETER SWEEPS")
    sweep_targets = [
        ("evaporator_area_geometric", np.linspace(0.5, 10.0, 20)),
        ("solar_irradiance",          np.linspace(300, 1000, 20)),
        ("optical_gain",              np.linspace(1.0, 3.0, 20)),
    ]
    if legacy:
        sweep_targets += [("empirical_flux", np.linspace(0.3, 3.0, 20)), ("eta_thermal", np.linspace(0.30, 0.90, 20))]
    else:
        sweep_targets += [("glass_area", np.geomspace(0.9, 20.0, 20)),
                          ("diffusion_path_length", np.geomspace(0.004, 0.2, 20))]
    ncol = 3
    fig, axes = plt.subplots(2, ncol, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (p, values) in zip(axes, sweep_targets):
        sr = sweep_one(p, values.tolist(), baseline_cfg)
        ax.plot(sr.values, sr.outputs, "-o", markersize=4)
        ax.set_xlabel(p)
        ax.set_ylabel("Daily freshwater [L/day]")
        ax.set_title(f"L/day vs {p}")
        ax.grid(alpha=0.3)
        pd.DataFrame({p: sr.values, "daily_L": sr.outputs}).to_csv(RESULTS / f"sweep_{p}.csv", index=False)
    for ax in axes[len(sweep_targets):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "parameter_sweeps.png", dpi=140)
    plt.close(fig)
    print(f"  Sweeps saved to {RESULTS.name}/sweep_*.csv and parameter_sweeps.png")

    if not legacy:
        # glass-area and chamber-pressure curves: where does the cover / the vacuum stop helping?
        hr("COVER AREA AND VACUUM CURVES (coupled)")
        rows = []
        for ratio in np.geomspace(0.8, 50.0, 28):
            row = {"glass_to_evaporator_ratio": ratio}
            for mode in ("energy_limited", "mass_transfer", "empirical"):
                r = simulate(SystemConfig(evaporation_mode=mode, glass_area=ratio))
                row[f"L_day_{mode}"] = r.daily_freshwater_L
                if mode == "mass_transfer":
                    row["T_glass_C"] = r.T_glass_C
                    row["T_evap_C"] = r.T_evaporator_C
                    row["limited_by"] = r.limited_by
            rows.append(row)
        gdf = pd.DataFrame(rows)
        gdf.to_csv(RESULTS / "glass_area_curve.csv", index=False)
        i_pk = int(gdf["L_day_mass_transfer"].idxmax())
        print(f"  diffusion-mode yield peaks at glass/evaporator = {gdf.loc[i_pk, 'glass_to_evaporator_ratio']:.2f} "
              f"({gdf.loc[i_pk, 'L_day_mass_transfer']:.2f} L/day) and falls to "
              f"{gdf['L_day_mass_transfer'].iloc[-1]:.2f} at ratio 50; ideal-evaporator limit saturates at "
              f"{gdf['L_day_energy_limited'].iloc[-1]:.2f}.")
        vrows = []
        for P in np.linspace(101325, 8000, 24):
            # set the chamber pressure directly through the leak floor with an over-strong ejector
            c = SystemConfig(venturi_enabled=True, motive_flow_rate=4e-3, chamber_leak_pressure_floor=P)
            r = simulate(c)
            vrows.append({"P_set_Pa": r.P_chamber_set_Pa, "P_chamber_Pa": r.P_chamber_Pa,
                          "L_per_day": r.daily_freshwater_L, "T_evap_C": r.T_evaporator_C,
                          "T_condensate_C": r.T_condensate_C, "boiling_limited": r.boiling_limited,
                          "vacuum_load_kN": r.vacuum_load_kN})
        vdf = pd.DataFrame(vrows)
        vdf.to_csv(RESULTS / "vacuum_curve.csv", index=False)
        fig, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
        axes2[0].plot(gdf["glass_to_evaporator_ratio"], gdf["L_day_energy_limited"], label="mode A (ideal evaporator)")
        axes2[0].plot(gdf["glass_to_evaporator_ratio"], gdf["L_day_mass_transfer"], label="mode C (diffusion)")
        axes2[0].plot(gdf["glass_to_evaporator_ratio"], gdf["L_day_empirical"], label="mode B (J_emp cap)")
        axes2[0].set_xscale("log"); axes2[0].set_xlabel("glass area / evaporator area"); axes2[0].set_ylabel("L/day")
        axes2[0].legend(); axes2[0].grid(alpha=0.3); axes2[0].set_title("Cover area")
        axes2[1].plot(vdf["P_set_Pa"] / 1e3, vdf["L_per_day"], "-o", ms=3, label="yield")
        axes2[1].set_xlabel("pressure the ejector would set [kPa]"); axes2[1].set_ylabel("L/day")
        axes2[1].invert_xaxis(); axes2[1].grid(alpha=0.3); axes2[1].set_title("Vacuum: gain stops at the cover's vapour pressure")
        fig.tight_layout()
        fig.savefig(RESULTS / "cover_and_vacuum_curves.png", dpi=140)
        plt.close(fig)

    # ======================================================================
    # 7. Monte Carlo uncertainty
    # ======================================================================
    hr("MONTE CARLO UNCERTAINTY")
    mc = monte_carlo(n_samples=2000, baseline=baseline_cfg, seed=42)
    mc_x = monte_carlo(n_samples=2000, baseline=baseline_cfg, seed=42, exclude=ARTIFACT_PARAMETERS)
    for label, m in (("all parameters", mc), (f"excluding {', '.join(ARTIFACT_PARAMETERS)}", mc_x)):
        print(f"  [{label}]  mean {m.mean:6.3f}  median {m.median:6.3f}  std {m.std:6.3f}  "
              f"P05 {m.p05:6.3f}  P95 {m.p95:6.3f}  L/day")
    rho = pd.Series(mc.spearman(), name="spearman_rho_with_yield").sort_values(key=abs, ascending=False)
    print("  Spearman rho with yield (largest |rho|):")
    print(rho.head(8).round(3).to_string())
    rho.to_csv(RESULTS / "monte_carlo_spearman.csv")
    pd.DataFrame([
        {"set": "all parameters", "mean": mc.mean, "median": mc.median, "std": mc.std, "p05": mc.p05, "p95": mc.p95},
        {"set": "artifact parameters excluded", "mean": mc_x.mean, "median": mc_x.median, "std": mc_x.std,
         "p05": mc_x.p05, "p95": mc_x.p95},
    ]).to_csv(RESULTS / "monte_carlo_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mc.daily_L, bins=50, color="teal", alpha=0.8, edgecolor="black")
    ax.axvline(mc.mean, color="red", linestyle="--", label=f"mean {mc.mean:.2f}")
    ax.axvline(mc.p05, color="orange", linestyle=":", label=f"P05 {mc.p05:.2f}")
    ax.axvline(mc.p95, color="orange", linestyle=":", label=f"P95 {mc.p95:.2f}")
    ax.set_xlabel("Daily freshwater [L/day]")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo uncertainty ({coupling}, N={mc.n_samples})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "monte_carlo.png", dpi=140)
    plt.close(fig)
    pd.DataFrame({"daily_L": mc.daily_L}).to_csv(RESULTS / "monte_carlo_samples.csv", index=False)

    # ======================================================================
    # 8. Validation / plausibility (NOT validation of the physics)
    # ======================================================================
    hr("ARITHMETIC, PLAUSIBILITY AND CONSISTENCY CHECKS (none of these validates the model)")
    print(f"Empirical reference: {EMPIRICAL_YIELD_KG_PER_M2_DAY} kg/m^2/day (concept diagram, cite 396). "
          "Linear-scaling arithmetic, NOT validation.\n")
    for c in run_empirical_linear_scaling([1.0, 3.6, 10.0, 25.0], coupling=coupling):
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.name}: predicted={c.predicted_L_per_day:.3f} kg, "
              f"empirical={c.empirical_L_per_day:.3f} kg, err={c.relative_error_percent:.3f}%")
    hd = run_condensation_headroom(1.0, coupling=coupling)
    print(f"\n  [{'PASS' if hd.passed else 'DESIGN-FLAG'}] {hd.name}: cond_cap={hd.predicted_L_per_day:.3f} kg, "
          f"m_evap={hd.empirical_L_per_day:.3f} kg")
    cons = run_energy_vs_empirical_consistency(coupling=coupling)
    print(f"\n  [{'PASS' if cons.passed else 'FLAG'}] {cons.name}: energy={cons.predicted_L_per_day:.3f} kg, "
          f"empirical={cons.empirical_L_per_day:.3f} kg (ratio-1 = {cons.relative_error_percent:.1f} %)")
    chk = []
    pl = run_still_plausibility_check(baseline_cfg)
    print(f"\n  [{'IN BAND' if pl.passed else 'OUT OF BAND'}] {pl.name}\n      {pl.notes}")
    chk.append(pl)
    if not legacy:
        pl1 = run_still_plausibility_check(SystemConfig(optical_gain=1.0))
        print(f"\n  [{'IN BAND' if pl1.passed else 'OUT OF BAND'}] (no reflector, gain 1.0): "
              f"{pl1.predicted_L_per_day:.2f} L/m2/day")
        iso = run_isolated_evaporator_consistency()
        print(f"\n  [{'CONSISTENT' if iso.passed else 'NOT CONSISTENT'}] {iso.name}\n      {iso.notes}")
        chk.append(iso)
    pd.DataFrame([{"check": c.name, "predicted": c.predicted_L_per_day, "reference": c.empirical_L_per_day,
                   "passed": c.passed, "notes": c.notes} for c in chk]).to_csv(
        RESULTS / "plausibility_checks.csv", index=False)

    hr("OPENFOAM PARAMETER MAP - documentation only")
    for k, v in [
        ("solar_irradiance", "heat-flux BC on evaporator wall (constantHeatFlux)"),
        ("ambient_temperature", "far-field thermal BC (fixedValue on outer patches)"),
        ("glass_temperature", "wall thermal condition on glass patch (coupled: T_gi)"),
        ("evaporator_area", "geometry / patch definition in blockMeshDict"),
        ("wind_speed", "external convection h_conv on top wall"),
        ("venturi_pressure", "pressure BC at throat"),
    ]:
        print(f"  {k:22s} -> {v}")
    print("\nNO OpenFOAM case files generated by this script.")

    hr("DONE")
    print(f"Artifacts written to: {RESULTS}")
    for p in sorted(RESULTS.glob("*")):
        if p.is_file() and p.name != ".gitkeep":
            print(f"  {p.stat().st_size/1024:6.1f} kB  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupling", choices=["coupled", "legacy", "both"], default="both")
    args = ap.parse_args()
    if args.coupling in ("legacy", "both"):
        run("legacy", ROOT / "results" / "legacy")
    if args.coupling in ("coupled", "both"):
        run("coupled", ROOT / "results")
