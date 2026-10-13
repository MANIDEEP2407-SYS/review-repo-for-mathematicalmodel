"""Streamlit dashboard for the solar desalination simulator.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dataclasses import replace

from src.physics.system import SystemConfig, simulate, EVAPORATION_MODES
from src.scaling.sizing import (
    HouseholdDemand,
    size_for_demand,
    size_modules,
    sweep_scale,
)
import time
from src.openfoam import runner as foam
from src.physics.realworld import REALWORLD_REGISTRY, realworld_yield
from src.sustainability.lifecycle import SUSTAINABILITY_REGISTRY, analyse
from src.scaling.sensitivity import (
    VENTURI_TORNADO_RANGES,
    default_tornado_ranges,
    sweep_one,
    tornado,
)
from src.scaling.monte_carlo import monte_carlo, ARTIFACT_PARAMETERS
from src.validation.validation_cases import (
    run_empirical_linear_scaling,
    run_energy_vs_empirical_consistency,
    run_isolated_evaporator_consistency,
    run_still_plausibility_check,
    EMPIRICAL_YIELD_KG_PER_M2_DAY,
)


st.set_page_config(page_title="Solar Desalination Simulator", page_icon="💧", layout="wide")
st.title("💧 Solar Desalination Simulator")
st.caption(
    "FeS-functionalised cotton 3D-flower evaporator, passive solar still + optional Venturi. "
    "All values are configurable; internal SI units, L/day on display only."
)


# ---------- Sidebar: system configuration ----------
st.sidebar.header("System configuration")

with st.sidebar.expander("Solver (Coupling)", expanded=True):
    coupling = st.radio(
        "Coupling", ["coupled", "legacy"], index=0, horizontal=True,
        help="coupled: evaporator, vapour gap and glass condenser solved as ONE balance; the ejector pressure "
             "feeds evaporation. legacy: the old decoupled model (glass heuristic, Venturi never reaches yield), "
             "kept so old results reproduce.")
    COUPLED = coupling == "coupled"
    if COUPLED:
        st.caption("η_thermal, constant h_fg, k_m, RH and Venturi inlet pressure are NOT used by the coupled solver.")
    else:
        st.caption("Legacy solver: condensation capped separately from evaporation; known flaws in README §8.")

with st.sidebar.expander("Solar", expanded=True):
    solar_irradiance = st.slider("Solar irradiance G [W/m²]", 100, 1361, 800)
    sunlight_hours = st.slider("Sunlight hours [h]", 1.0, 14.0, 8.0, 0.5)
    optical_gain = st.slider("Optical / reflector gain [-]", 1.0, 3.0, 1.5, 0.05)
    glass_transmittance = st.slider("Glass transmittance τ [-]", 0.5, 0.95, 0.90, 0.01)
    eta_optical = st.slider("Optical efficiency η_opt [-]", 0.3, 1.0, 0.90, 0.01)

with st.sidebar.expander("Evaporator", expanded=True):
    evaporator_area_geometric = st.number_input(
        "Geometric evaporator area A_geo [m²]", 0.05, 100.0, 1.0, 0.1
    )
    area_utilization_factor = st.slider(
        "Area utilization f_util [-]", 0.1, 1.0, 0.85, 0.01
    )
    eta_thermal = st.slider("Thermal efficiency η_th [-] (legacy only)", 0.1, 0.95, 0.65, 0.01,
                            disabled=COUPLED,
                            help="Not used by the coupled solver: evaporation is whatever the energy balance gives.")
    evaporator_emissivity = st.slider("Evaporator emissivity ε_e [-]", 0.7, 1.0, 0.95, 0.01)

with st.sidebar.expander("Environment"):
    ambient_temperature = st.slider("Ambient T [°C]", -10.0, 55.0, 30.0, 0.5)
    wind_speed = st.slider("Wind speed [m/s]", 0.0, 20.0, 2.0, 0.1)
    relative_humidity = st.slider("Relative humidity [-] (legacy only)", 0.0, 1.0, 0.50, 0.01, disabled=COUPLED,
                                  help="The coupled chamber is sealed: ambient humidity does not enter.")
    insulation_thickness = st.slider("Insulation thickness [m]", 0.005, 0.30, 0.05, 0.005)
    insulation_k = st.slider("Insulation k [W/m·K]", 0.02, 0.20, 0.04, 0.005)

with st.sidebar.expander("Glass cover"):
    glass_area = st.number_input("Glass area [m²]", 0.05, 120.0, 1.1, 0.1)
    glass_thickness = st.number_input("Glass thickness [m]", 0.003, 0.020, 0.010, 0.001, format="%.3f")
    glass_k = st.slider("Glass k [W/m·K]", 0.7, 1.4, 1.0, 0.05)
    glass_emissivity = st.slider("Glass emissivity ε_g [-]", 0.7, 0.95, 0.88, 0.01)

with st.sidebar.expander("Evaporation model"):
    evaporation_mode = st.selectbox(
        "Mode", EVAPORATION_MODES, index=(2 if COUPLED else 0),
        help="energy_limited = ideal evaporator (upper bound for the cover); empirical = J_emp cap; "
             "mass_transfer = vapour diffusion through air; hybrid = min(empirical cap, diffusion).")
    empirical_flux = st.number_input(
        "Empirical flux J_emp [kg/(m²·h)]", 0.1, 10.0, 1.05, 0.05
    )
    mass_transfer_coefficient = st.number_input(
        "Mass-transfer coeff k_m [kg/(m²·s·Pa)] (legacy only)", 1e-9, 1e-6, 3e-8, 1e-9, format="%.2e", disabled=COUPLED
    )

with st.sidebar.expander("Condensation"):
    condensation_coefficient = st.slider(
        "Inner h_cond [W/m²·K]", 500, 20000, 8000, 100
    )
    collection_efficiency = st.slider("Collection efficiency [-]", 0.5, 1.0, 0.90, 0.01)

with st.sidebar.expander("Coupled-model parameters (ASSUMED, calibrate on prototype)", expanded=False):
    diffusion_path_length = st.number_input(
        "Effective diffusion path δ [m]", 0.002, 0.30, 0.02, 0.002, format="%.3f", disabled=not COUPLED,
        help="Effective air path vapour must diffuse through, natural convection included. Biggest unknown.")
    gap_convection_coefficient = st.number_input(
        "Gap convection h_c,gap [W/m²K]", 0.1, 15.0, 2.0, 0.1, disabled=not COUPLED,
        help="Sensible coupling evaporator→glass; the radiative part is added from the emissivities.")
    glass_absorptance = st.slider("Glass absorptance α_g [-]", 0.0, 0.30, 0.05, 0.01, disabled=not COUPLED)
    sensible_heat = st.checkbox("Sensible-heat bookkeeping (energy-conserving)", value=True, disabled=not COUPLED,
                                help="Off = the literal brief equations; yield then DEcreases with cover area at high T.")

with st.sidebar.expander("Salinity / feed"):
    feed_tds = st.number_input("Feed TDS [mg/L]", 100.0, 60000.0, 5000.0, 100.0)
    feed_volume = st.number_input("Feed volume [L]", 0.5, 100000.0, 20.0, 0.5)

with st.sidebar.expander("Venturi (optional)"):
    venturi_enabled = st.checkbox("Enable Venturi", value=False)
    motive_flow_rate = st.number_input("Motive flow Q_m [m³/s]", 1e-6, 1e-1, 1e-3, 1e-4, format="%.4f")
    venturi_inlet_diameter = st.number_input("Inlet D₁ [m]", 0.005, 0.100, 0.025, 0.001, format="%.3f")
    venturi_throat_diameter = st.number_input("Throat D₂ [m]", 0.002, 0.050, 0.010, 0.001, format="%.3f")
    venturi_loss_fraction = st.slider("Venturi loss fraction [-]", 0.0, 0.9, 0.30, 0.05)
    motive_water_temperature = st.number_input("Motive water T [°C]", 5.0, 60.0, 30.0, 1.0, disabled=not COUPLED,
                                               help="Its vapour pressure is the ejector's hard floor.")
    chamber_leak_pressure_floor = st.number_input("Leak pressure floor [Pa]", 1000.0, 101325.0, 10000.0, 1000.0,
                                                  disabled=not COUPLED, help="ASSUMED; measure with a vacuum-decay test.")
    pump_efficiency = st.slider("Motive pump efficiency [-]", 0.1, 0.9, 0.5, 0.05, disabled=not COUPLED)


with st.sidebar.expander("Real-world limits (capillary, fouling…)"):
    apply_limits = st.checkbox("Apply to module sizing", value=True,
                               help="Sizes modules with the derated productivity instead of the ideal one.")
    rw_over = {}
    for _n, _p in REALWORLD_REGISTRY.items():
        _lo, _hi = _p.allowed_range
        rw_over[_n] = st.number_input(f"{_n} [{_p.unit}] · {_p.source.value}", float(_lo), float(_hi),
                                      float(_p.default), float((_hi - _lo) / 100), format="%g",
                                      help=_p.notes, key=f"rw_{_n}")


cfg = SystemConfig(
    coupling=coupling,
    diffusion_path_length=diffusion_path_length,
    gap_convection_coefficient=gap_convection_coefficient,
    glass_absorptance=glass_absorptance,
    sensible_heat=sensible_heat,
    motive_water_temperature=motive_water_temperature,
    chamber_leak_pressure_floor=chamber_leak_pressure_floor,
    pump_efficiency=pump_efficiency,
    solar_irradiance=solar_irradiance,
    sunlight_hours=sunlight_hours,
    optical_gain=optical_gain,
    glass_transmittance=glass_transmittance,
    eta_optical=eta_optical,
    eta_thermal=eta_thermal,
    evaporator_area_geometric=evaporator_area_geometric,
    area_utilization_factor=area_utilization_factor,
    ambient_temperature=ambient_temperature,
    wind_speed=wind_speed,
    relative_humidity=relative_humidity,
    insulation_thickness=insulation_thickness,
    insulation_k=insulation_k,
    evaporator_emissivity=evaporator_emissivity,
    glass_area=glass_area,
    glass_thickness=glass_thickness,
    glass_k=glass_k,
    glass_emissivity=glass_emissivity,
    condensation_coefficient=condensation_coefficient,
    collection_efficiency=collection_efficiency,
    evaporation_mode=evaporation_mode,
    empirical_flux=empirical_flux,
    mass_transfer_coefficient=mass_transfer_coefficient,
    feed_tds=feed_tds,
    feed_volume=feed_volume,
    venturi_enabled=venturi_enabled,
    motive_flow_rate=motive_flow_rate,
    venturi_inlet_diameter=venturi_inlet_diameter,
    venturi_throat_diameter=venturi_throat_diameter,
    venturi_loss_fraction=venturi_loss_fraction,
)

result = simulate(cfg)
rw = realworld_yield(cfg, rw_over)
prod_override = rw.final_L_m2_day if apply_limits else None

# ---------- Main dashboard ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Freshwater", f"{result.daily_freshwater_L:.2f} L/day")
c2.metric("Productivity", f"{result.productivity_L_per_m2_day:.2f} L/m²·day")
c3.metric("Recovery", f"{result.recovery_ratio*100:.1f} %")
c4.metric("Energy efficiency", f"{result.energy_efficiency*100:.1f} %")
c5.metric("T_evap (lumped)" if not COUPLED else "T_e evaporator", f"{result.T_evaporator_C:.1f} °C")
if COUPLED:
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("T_gi glass inner", f"{result.T_glass_C:.1f} °C")
    d2.metric("T_c condensate", f"{result.T_condensate_C:.1f} °C")
    d3.metric("P_chamber", f"{result.P_chamber_Pa/1e3:.1f} kPa",
              help=f"ejector/atmosphere set {result.P_chamber_set_Pa/1e3:.1f} kPa")
    d4.metric("Vacuum load on cover", f"{result.vacuum_load_kN:.1f} kN",
              f"~{result.vacuum_load_tonnes:.1f} t" if result.vacuum_load_kN > 0 else None, delta_color="off")
    d5.metric("Boiling-limited", "YES" if result.boiling_limited else "no")
    d6.metric("Ejector power", f"{result.ejector_power_W:.0f} W")
    if not result.solver_converged:
        st.error("Coupled solver did not close for this configuration (see warnings): the numbers are not meaningful.")
    st.caption(f"Closure: worst relative residual of the node energy balances / mass balance = "
               f"{result.solver_residual_max:.1e}. Limiting resistance (diagnostic): **{result.limited_by}**.")

if result.warnings:
    with st.expander(f"⚠️ Warnings ({len(result.warnings)})", expanded=False):
        for w in result.warnings:
            st.warning(w)

tabs = st.tabs([
    "Module sizer", "Cost & sustainability", "OpenFOAM (CFD)", "Real-world limits", "Results", "Modes", "Scaling",
    "Sensitivity", "Uncertainty", "Validation", "Parameters"
])

with tabs[0]:
    st.subheader("Module sizer: modules + required water → module size")
    in1, in2, in3 = st.columns(3)
    n_modules = in1.number_input("Number of modules", 1, 10000, 1, 1)
    req_L_per_hour = in2.number_input("Water required, total [L/hour]", 0.1, 100000.0, 10.0, 1.0)
    flowers_per_m2 = in3.number_input(
        "Flowers per m² of module [assumed]", 1.0, 500.0, 25.0, 1.0,
        help="Assumed packing density (registry default N_f = 25). Not measured.",
    )
    ms = size_modules(int(n_modules), req_L_per_hour, cfg, flowers_per_m2,
                      productivity_override=prod_override)
    st.caption(
        f"Rate is averaged over {ms.sunlight_hours:g} sunlight h/day → "
        f"{ms.target_L_per_day:.1f} L/day total. Productivity from the current "
        f"model settings: {ms.productivity_L_per_m2_day:.2f} L/m²·day "
        + (f"(real-world derated; ideal would be {rw.ideal_L_m2_day:.2f})." if apply_limits
           else "(ideal, no real-world limits).")
    )
    for w in ms.warnings:
        st.warning(w)
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Area per module", f"{ms.A_module_m2:.3f} m²")
    o2.metric("Square module side", f"{ms.module_side_m:.2f} m")
    o3.metric("…or circular Ø", f"{ms.module_diameter_m:.2f} m")
    o4.metric("Total area", f"{ms.A_total_m2:.2f} m²")
    o5, o6, o7, o8 = st.columns(4)
    o5.metric("Flowers per module", f"{ms.flowers_per_module}")
    o6.metric("Flowers total", f"{ms.flowers_total}")
    o7.metric("Output per module", f"{ms.L_per_hour_per_module:.2f} L/h")
    o8.metric("Flower coverage", f"{ms.flower_coverage_fraction*100:.0f} %")
    st.caption(
        "Sizes scale from the model's productivity; the flowers/m² density is an "
        "assumption and the model is not CFD- or experimentally validated."
    )

with tabs[1]:
    st.subheader("Cost, carbon and green-cover projection")
    st.warning(
        "Prices and most emission factors are ASSUMED placeholders (see provenance table). "
        "Replace with supplier quotes / cited LCA data before quoting any number. "
        "Scope: materials, replacements, pump electricity, LCOW. Excludes transport and end-of-life."
    )
    overrides = {}
    with st.expander("Assumptions (edit to run what-if scenarios)", expanded=False):
        cols = st.columns(3)
        for i, (name, prm) in enumerate(SUSTAINABILITY_REGISTRY.items()):
            lo, hi = prm.allowed_range
            overrides[name] = cols[i % 3].number_input(
                f"{name} [{prm.unit}] · {prm.source.value}", float(lo), float(hi),
                float(prm.default), float((hi - lo) / 100) or 0.01, format="%g",
                help=prm.notes, key=f"sus_{name}",
            )
    horizon = st.slider("Forecast horizon [years]", 1, 40, int(overrides["system_life_years"]))
    lc = analyse(ms, cfg, overrides, horizon)
    for w in lc.warnings:
        st.warning(w)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capex total", f"₹{lc.capex_total_INR:,.0f}")
    m2.metric("Capex per module", f"₹{lc.capex_per_module_INR:,.0f}")
    m3.metric("Cost of water (LCOW)", f"₹{lc.lcow_INR_per_m3:,.0f} /m³")
    m4.metric("Annual opex", f"₹{lc.annual_opex_INR:,.0f}")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Embodied CO₂", f"{lc.embodied_co2_total_kg:,.0f} kg")
    m6.metric("Operational CO₂", f"{lc.annual_operational_co2_kg:,.0f} kg/yr")
    m7.metric("CO₂ per m³ water", f"{lc.co2_kg_per_m3:.1f} kg")
    m8.metric("Water per year", f"{lc.annual_water_L/1000:,.1f} m³")

    st.markdown("#### Green cover for net zero")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Trees (steady-state balance)", f"{lc.trees_steady_state:,}")
    g2.metric("Trees (lifetime balance)", f"{lc.trees_lifetime_balance:,}")
    g3.metric("Green area", f"{lc.green_area_ha:.3f} ha")
    g4.metric("Net-zero year", lc.net_zero_year if lc.net_zero_year else "not reached")
    st.caption(
        "Lifetime balance: trees planted at year 0 ramp linearly to full sequestration at maturity; "
        "count chosen so cumulative sequestration ≥ cumulative emissions over the horizon. "
        "Steady-state: mature-tree annual uptake vs average annual emissions."
    )
    dfy = pd.DataFrame(lc.yearly)
    figy = go.Figure()
    figy.add_trace(go.Scatter(x=dfy["year"], y=dfy["cum_emissions_kg"], name="Cumulative emissions"))
    figy.add_trace(go.Scatter(x=dfy["year"], y=dfy["cum_sequestered_kg"], name="Cumulative sequestered"))
    figy.update_layout(title="Emissions vs green-cover sequestration [kg CO₂e]",
                       xaxis_title="Year", yaxis_title="kg CO₂e")
    st.plotly_chart(figy, width="stretch")
    st.dataframe(dfy, width="stretch")

    st.markdown("#### Per-module bill of materials (un-scaled prices)")
    st.dataframe(pd.DataFrame(lc.bom), width="stretch")

    st.markdown("#### Scale-up forecast (same per-module output, more modules)")
    rows = []
    for n in [1, 10, 100, 1000, 10000]:
        m_n = size_modules(n, ms.L_per_hour_per_module * n, cfg, flowers_per_m2,
                           productivity_override=prod_override)
        r_n = analyse(m_n, cfg, overrides, horizon)
        rows.append({
            "modules": n, "water m³/yr": r_n.annual_water_L / 1000,
            "capex ₹": r_n.capex_total_INR, "capex/module ₹": r_n.capex_per_module_INR,
            "LCOW ₹/m³": r_n.lcow_INR_per_m3, "embodied CO₂ kg": r_n.embodied_co2_total_kg,
            "trees": r_n.trees_lifetime_balance, "green ha": r_n.green_area_ha,
        })
    dfs = pd.DataFrame(rows)
    st.dataframe(dfs, width="stretch")
    st.plotly_chart(px.line(dfs, x="modules", y="LCOW ₹/m³", log_x=True, markers=True,
                            title="LCOW vs number of modules (driven by learning_rate assumption)"),
                    width="stretch")

with tabs[2]:
    st.subheader("Run the OpenFOAM case from the simulator")
    st.warning(
        "The current case solves buoyant heat transfer + flow ONLY. It has no evaporation/condensation "
        "mass-transfer source, so it gives NO freshwater yield and is NOT CFD validation of the Python "
        "model. The mesh is fixed (one 3.6 m² module): only fluxes, temperatures and run controls are mapped."
    )
    fc1, fc2 = st.columns(2)
    case_dir = fc1.text_input("Case directory", str(foam.DEFAULT_CASE))
    distro = fc2.text_input("WSL distro", foam.DEFAULT_DISTRO)
    if "foam_ok" not in st.session_state or st.button("Re-check OpenFOAM"):
        st.session_state.foam_ok = foam.openfoam_available(distro)
    if st.session_state.foam_ok:
        st.success("OpenFOAM v2406 found in WSL.")
    else:
        st.error("OpenFOAM not found in that WSL distro.")

    mapped = foam.map_simulator_to_case(cfg, result)
    st.markdown("**Parameters sent to OpenFOAM** (prefilled from the current simulator settings; editable)")
    pc = st.columns(5)
    fparams = {
        "q_net_W_m2": pc[0].number_input("Evaporator net flux q [W/m²]", 1.0, 5000.0, round(mapped["q_net_W_m2"], 1), 10.0,
                                         help="Simulator Q_useful / (A_eff · sunlight time)."),
        "T_ambient_K": pc[1].number_input("Ambient T [K]", 250.0, 340.0, round(mapped["T_ambient_K"], 2), 0.5),
        "h_glass_W_m2K": pc[2].number_input("Glass outer h [W/m²K]", 1.0, 100.0, mapped["h_glass_W_m2K"], 1.0,
                                            help="Case default; not derived from the Python model."),
        "end_time_s": pc[3].number_input("Simulated time [s]", 10.0, 28800.0, mapped["end_time_s"], 60.0,
                                         help="Full solar day = 28800 s; runs in real time far slower."),
        "write_interval_s": pc[4].number_input("Write interval [s]", 1.0, 3600.0, mapped["write_interval_s"], 10.0),
    }

    b1, b2 = st.columns(2)
    if b1.button("▶ Run OpenFOAM", type="primary", disabled=not st.session_state.foam_ok):
        try:
            rd = foam.prepare_case(fparams, Path(case_dir))
            foam.start_run(rd, distro)
            st.session_state.foam_run = str(rd)
        except Exception as exc:  # surface prep errors to the user
            st.error(f"Could not start run: {exc}")
    if b2.button("■ Stop run"):
        if st.session_state.get("foam_run"):
            foam.stop_run(Path(st.session_state.foam_run), distro)

    prev = sorted(foam.RUNS_DIR.glob("run_*")) if foam.RUNS_DIR.exists() else []
    if prev and not st.session_state.get("foam_run"):
        st.session_state.foam_run = str(prev[-1])
    if st.session_state.get("foam_run"):
        rd = Path(st.session_state.foam_run)
        stt = foam.run_status(rd)
        st.markdown(f"**Run:** `{rd.name}` — **{stt.state}**")
        if stt.end_time_s > 0:
            st.progress(min(1.0, stt.sim_time_s / stt.end_time_s),
                        text=f"simulated {stt.sim_time_s:.0f} / {stt.end_time_s:.0f} s"
                             + (f" · solver CPU {stt.exec_time_s:.0f} s" if stt.exec_time_s else ""))
        if stt.T_evap_series:
            tt = pd.DataFrame(stt.T_evap_series, columns=["time_s", "T_K"])
            tt["T_C"] = tt["T_K"] - 273.15
            figf = px.line(tt, x="time_s", y="T_C", title="CFD area-average evaporator temperature [°C]")
            figf.add_hline(y=result.T_evaporator_C, line_dash="dash",
                           annotation_text="Python lumped T_evap (steady-state)")
            st.plotly_chart(figf, width="stretch")
            st.caption(
                "The dashed line is the lumped model's steady value; a short CFD run is still heating up and "
                "the two use different physics. Agreement/disagreement here is NOT validation."
            )
        with st.expander("Solver log (tail)"):
            st.code(stt.log_tail or "(no output yet)")
        st.caption(f"Results in `{rd}` — open `case.foam` in ParaView for fields.")
        if stt.state == "running":
            time.sleep(3)
            st.rerun()

with tabs[3]:
    st.subheader("Real-world limits: what the ideal model leaves out")
    st.caption(
        "Hour-by-hour re-run with a half-sine sun (same daily insolation), dust soiling, salt/biofilm "
        "fouling, flower ageing and the capillary water-supply limit. All limit parameters are ASSUMED "
        "(sidebar) until measured on the real cotton flower."
    )
    for w in rw.warnings:
        st.warning(w)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ideal", f"{rw.ideal_L_m2_day:.2f} L/m²·day")
    k2.metric("Real-world", f"{rw.final_L_m2_day:.2f} L/m²·day", f"{(rw.derating-1)*100:.0f} %")
    k3.metric("Jurin max lift", f"{rw.wick.h_jurin_m*100:.1f} cm",
              help="Highest the wick's capillarity can raise water; the wick must be shorter.")
    ratio = rw.min_wick_supply_ratio
    k4.metric("Wick supply / demand", "no limit" if ratio == float("inf") else f"{ratio:.2f}×",
              help="Tightest hour. Below 1× the evaporator dries out.")
    wf = pd.DataFrame(rw.steps, columns=["stage", "L/m²/day"])
    st.plotly_chart(px.bar(wf, x="L/m²/day", y="stage", orientation="h",
                           title="Yield after each real-world effect").update_yaxes(autorange="reversed"),
                    width="stretch")
    st.markdown("**Capillary supply**")
    st.write({
        "capillary pressure [Pa]": round(rw.wick.P_cap_Pa),
        "Jurin height [cm]": round(rw.wick.h_jurin_m * 100, 1),
        "wick permeability K [m²]": f"{rw.wick.K_m2:.2e}",
        "deliverable flux at hottest hour [kg/m²/h]": round(rw.wick.J_wick_kg_m2_s * 3600, 2),
        "wick can lift water": rw.wick.lift_ok,
        "hours where supply < demand": rw.wick_limited_hours,
    })
    st.markdown("**Hour by hour**")
    st.dataframe(pd.DataFrame(rw.hourly), width="stretch")
    st.caption(
        "Not modelled: biofilm growth dynamics, salt-crust thermal resistance, brine viscosity, wind/rain "
        "damage, cloudy/monsoon days, water-quality change. See README §8."
    )

with tabs[4]:
    st.subheader("Legacy vs coupled (same inputs, same mode name)")
    cmp_rows = []
    for k in ("legacy", "coupled"):
        ck = replace(cfg, coupling=k)
        rk = simulate(ck)
        cmp_rows.append({
            "coupling": k, "yield L/day": rk.daily_freshwater_L, "T_evap °C": rk.T_evaporator_C,
            "T_glass °C": rk.T_glass_C, "T_condensate °C": rk.T_condensate_C,
            "evaporated kg": rk.m_evap_selected_kg, "cover capacity kg": rk.m_condensation_capacity_kg,
            "P_chamber kPa": rk.P_chamber_Pa / 1e3, "boiling-limited": rk.boiling_limited,
            "vacuum load kN": rk.vacuum_load_kN, "ejector W": rk.ejector_power_W,
            "binding": rk.condensation_limited_by if k == "legacy" else rk.limited_by,
        })
    st.dataframe(pd.DataFrame(cmp_rows), width="stretch")
    st.caption("Neither column is validated. Legacy: evaporation uses the INPUT η_thermal and condensation is capped "
               "separately by a glass heuristic; Venturi never reaches yield. Coupled: one energy/mass balance, but "
               "δ, h_c,gap, h_cond, α_g and the leak floor are ASSUMED. Full table: docs/before_after.md.")
    st.subheader("Full result vector")
    kv = {
        "Effective area A_eff [m²]": result.A_effective,
        "Effective irradiance G_eff [W/m²]": result.G_effective,
        "Q_incident [MJ]": result.Q_solar_incident_J / 1e6,
        "Q_absorbed [MJ]": result.Q_absorbed_J / 1e6,
        "Q_useful [MJ]": result.Q_useful_J / 1e6,
        "T_evaporator [°C]": result.T_evaporator_C,
        "T_glass (inner) [°C]": result.T_glass_C,
        "T_condensate [°C]": result.T_condensate_C,
        "T_sky [°C]": result.T_sky_C,
        "Q_convection [W]": result.Q_convection_W,
        "Q_radiation [W]": result.Q_radiation_W,
        "Q_conduction [W]": result.Q_conduction_W,
        "m_evap (selected) [kg]": result.m_evap_selected_kg,
        "m_cond capacity [kg]": result.m_condensation_capacity_kg,
        "Freshwater [kg]": result.daily_freshwater_kg,
        "Feed [kg]": result.feed_kg,
        "Brine [kg]": result.brine_kg,
        "Brine TDS [mg/L]": result.brine_tds_mg_per_L,
        "Concentration factor": result.concentration_factor,
        "U_glass [W/m²·K]": result.U_glass,
        "P_chamber (effective) [Pa]": result.P_chamber_Pa,
        "P_chamber (ejector set) [Pa]": result.P_chamber_set_Pa,
        "Boiling-limited": result.boiling_limited,
        "Vacuum load on cover [kN]": result.vacuum_load_kN,
        "Ejector pump power [W]": result.ejector_power_W,
        "ΔT_sat (Venturi) [°C]": result.saturation_T_shift_C,
        "Solver converged": result.solver_converged,
        "Worst closure residual": result.solver_residual_max,
    }
    st.dataframe(pd.DataFrame([(k, str(round(v, 6)) if isinstance(v, float) else str(v)) for k, v in kv.items()],
                              columns=["Quantity", "Value"]), width="stretch")

with tabs[5]:
    st.subheader("Evaporation mode comparison")
    modes_df = pd.DataFrame(
        [(k, v) for k, v in result.mode_predictions_kg.items()],
        columns=["Mode", "m_evap [kg/session]"],
    )
    st.dataframe(modes_df, width="stretch")
    fig = px.bar(modes_df, x="Mode", y="m_evap [kg/session]",
                 title="Evaporated mass by mode (before condensation limit)")
    st.plotly_chart(fig, width="stretch")
    mrows = []
    for k in ("legacy", "coupled"):
        for mode in EVAPORATION_MODES:
            rk = simulate(replace(cfg, coupling=k, evaporation_mode=mode))
            mrows.append({"coupling": k, "mode": mode, "evaporated kg": rk.m_evap_selected_kg,
                          "yield L/day": rk.daily_freshwater_L, "T_evap °C": rk.T_evaporator_C})
    mdf = pd.DataFrame(mrows)
    st.dataframe(mdf, width="stretch")
    st.plotly_chart(px.bar(mdf, x="mode", y="yield L/day", color="coupling", barmode="group",
                           title="Yield by evaporation mode: legacy vs coupled"), width="stretch")
    st.info(
        f"Selected mode: **{result.evaporation_mode_used}** (limited by: {result.limited_by or 'n/a'}). "
        + ("Coupled: freshwater = ṁ·t·η_coll, where ṁ is the solution of the coupled balance; mode A is the "
           "ideal-evaporator limit of the cover, B caps ṁ by J_emp·A_eff, C is vapour diffusion, D = min(B, C)."
           if COUPLED else "Legacy: freshwater = min(evap, condensation) × collection_efficiency.")
    )

with tabs[6]:
    st.subheader("Household / village scaling")
    colL, colR = st.columns(2)
    with colL:
        people = st.number_input("People per household", 1, 20, 4)
        drink = st.number_input("Drinking L/person/day", 1.0, 10.0, 4.0, 0.5)
        cook = st.number_input("Cooking L/person/day", 0.5, 5.0, 2.0, 0.5)
        sf = st.slider("Safety factor", 1.0, 2.0, 1.20, 0.05)
        households = st.number_input("Number of households", 1, 5000, 1)
        A_per_module = st.number_input("Area per module [m²]", 0.1, 20.0, 1.0, 0.1)
    demand = HouseholdDemand(int(people), drink, cook, sf)
    target = households * demand.total_L_per_day
    sizing = size_for_demand(target, cfg, A_per_module)
    with colR:
        st.metric("Target", f"{target:.1f} L/day")
        st.metric("Required area", f"{sizing.A_required_m2:.2f} m²")
        st.metric("Modules", sizing.modules_required)
        st.metric("Footprint", f"{sizing.total_footprint_m2:.2f} m²")
        st.caption(f"Productivity used: {sizing.productivity_L_per_m2_day:.2f} L/m²·day")

    st.markdown("### Population sweep")
    pops = [1, 10, 50, 100, 250, 500, 1000]
    rows = sweep_scale(pops, demand.per_person_L * sf, 1.0, cfg, A_per_module)
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")
    if sizing.productivity_L_per_m2_day > 0:
        fig = px.line(df, x="population", y=["target_L_per_day", "A_required_m2"],
                      title="Village-scale sizing")
        st.plotly_chart(fig, width="stretch")

with tabs[7]:
    st.subheader("Sensitivity (one-at-a-time tornado)")
    TORN = default_tornado_ranges(cfg.coupling)
    entries = tornado(TORN, cfg)
    df = pd.DataFrame([e.__dict__ for e in entries])
    st.dataframe(df, width="stretch")
    if not df.empty:
        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                y=[row["parameter"]],
                x=[row["range_output"]],
                orientation="h",
                name=row["parameter"],
                showlegend=False,
                hovertext=(f"Low={row['low_output']:.2f}, "
                           f"High={row['high_output']:.2f} L/day"),
            ))
        fig.update_layout(title="Sensitivity range on daily freshwater [L]",
                          xaxis_title="Δ freshwater [L/day]",
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Parameter sweep")
    param = st.selectbox("Sweep parameter", list(TORN.keys()))
    lo, hi = TORN[param]
    values = np.linspace(lo, hi, 25).tolist()
    sr = sweep_one(param, values, cfg)
    dfp = pd.DataFrame({param: sr.values, "daily_L": sr.outputs})
    fig = px.line(dfp, x=param, y="daily_L", markers=True,
                  title=f"Freshwater vs {param}")
    st.plotly_chart(fig, width="stretch")
    if COUPLED:
        st.markdown("### Ejector parameters (tornado on a venturi-enabled copy of the current settings)")
        vcfg = replace(cfg, venturi_enabled=True)
        vdf = pd.DataFrame([e.__dict__ for e in tornado(VENTURI_TORNADO_RANGES, vcfg)])
        st.dataframe(vdf, width="stretch")
        st.caption("Parameters whose floor is not binding (leak floor, motive temperature) show zero range, and a "
                   "boiling-limited baseline is insensitive to flow: the vacuum cannot help past the cover's vapour pressure.")

with tabs[8]:
    st.subheader("Monte Carlo uncertainty")
    n = st.slider("Samples", 100, 5000, 1000, 100)
    seed = st.number_input("Seed", 0, 10_000, 42)
    if st.button("Run Monte Carlo"):
        mc = monte_carlo(n_samples=int(n), baseline=cfg, seed=int(seed))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{mc.mean:.2f} L/day")
        c2.metric("Median", f"{mc.median:.2f} L/day")
        c3.metric("5th pct", f"{mc.p05:.2f} L/day")
        c4.metric("95th pct", f"{mc.p95:.2f} L/day")
        fig = px.histogram(mc.daily_L, nbins=40,
                           title="Daily freshwater distribution [L/day]")
        st.plotly_chart(fig, width="stretch")
        mcx = monte_carlo(n_samples=int(n), baseline=cfg, seed=int(seed), exclude=ARTIFACT_PARAMETERS)
        st.caption(f"Excluding the legacy artifact parameters ({', '.join(ARTIFACT_PARAMETERS)}): "
                   f"mean {mcx.mean:.2f}, σ {mcx.std:.2f}, P05 {mcx.p05:.2f}, P95 {mcx.p95:.2f} L/day.")
        rho = mc.spearman()
        st.dataframe(
            pd.DataFrame([(k, v, rho.get(k, float("nan"))) for k, v in mc.parameter_relative_uncertainty.items()],
                         columns=["parameter", "relative σ", "Spearman ρ with yield"]),
            width="stretch",
        )

with tabs[9]:
    st.subheader("Reference validation")
    st.caption(
        f"Empirical yield reference: **{EMPIRICAL_YIELD_KG_PER_M2_DAY} kg/m²/day** "
        "(concept diagram, cite 396). Linear-scaling test, NOT CFD validation."
    )
    cases = run_empirical_linear_scaling([1.0, 3.6, 10.0], coupling=cfg.coupling)
    df = pd.DataFrame([c.__dict__ for c in cases])
    st.dataframe(df, width="stretch")
    consistency = run_energy_vs_empirical_consistency(coupling=cfg.coupling)
    st.write("**Energy vs empirical mode consistency at A=1 m²:**", consistency.__dict__)
    st.markdown("#### Plausibility and consistency (NOT validation)")
    pl = run_still_plausibility_check(cfg)
    (st.success if pl.passed else st.warning)(f"{pl.name}: {pl.predicted_L_per_day:.2f} L/m²/day vs 2–5. {pl.notes}")
    iso = run_isolated_evaporator_consistency()
    (st.success if iso.passed else st.warning)(f"{iso.name}: {iso.notes}")

with tabs[10]:
    st.subheader("Parameter registry (provenance)")
    from src.config.defaults import PARAMETER_REGISTRY
    rows = []
    for name, p in PARAMETER_REGISTRY.items():
        rows.append({
            "name": name,
            "symbol": p.symbol,
            "unit": p.unit,
            "default": p.default,
            "range": f"[{p.allowed_range[0]}, {p.allowed_range[1]}]",
            "source": p.source.value,
            "notes": p.notes,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")
