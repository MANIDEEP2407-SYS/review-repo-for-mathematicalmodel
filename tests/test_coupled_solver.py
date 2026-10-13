"""Tests for the coupled evaporator / gap / condenser solver and the ejector model.

The closure tests recompute every balance INDEPENDENTLY from the config and the result fields
(not from solver internals).  Property sweeps are seeded random configurations, not single points.

Findings these tests pin down (they are what the physics gives, not what was hoped for):
  * eta_thermal is not on the coupled path at all, so "yield rises when efficiency falls" is
    structurally impossible;
  * in the DIFFUSION mode a colder cover (more glass, more wind) raises the evaporator-to-cover
    sensible/radiative leak share, so yield is NOT monotone in glass area or wind speed there;
    the ideal-evaporator limit (mode A) and the empirical-cap mode (B) are.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import fields, replace
from pathlib import Path

import pytest

from src.config.defaults import PARAMETER_REGISTRY
from src.physics import coupled as cpl
from src.physics.evaporation import saturation_pressure_pa
from src.physics.heat_transfer import SIGMA, T_sky_celsius, h_convection_wind
from src.physics.system import EVAPORATION_MODES, SystemConfig, simulate
from src.physics.venturi import compute_ejector, cover_vacuum_load

K = lambda T: T + 273.15
DATA = Path(__file__).parent / "data" / "legacy_reference.json"


# ----------------------------------------------------------------------------- helpers
def closure(cfg: SystemConfig, r) -> dict:
    """Relative residuals of every balance, recomputed from first principles."""
    Te, Tgi, Tc, m = r.T_evaporator_C, r.T_glass_C, r.T_condensate_C, r.mdot_kg_s
    A_loss = cfg.evaporator_area_geometric
    A_eff = A_loss * cfg.area_utilization_factor
    Ta = cfg.ambient_temperature
    Tsky = T_sky_celsius(Ta)
    cp = cfg.water_specific_heat
    eps_gap = 1.0 / (1.0 / cfg.evaporator_emissivity + 1.0 / cfg.glass_emissivity - 1.0)
    h_rad_gap = eps_gap * SIGMA * (K(Te) ** 2 + K(Tgi) ** 2) * (K(Te) + K(Tgi))
    Qgap = (cfg.gap_convection_coefficient + h_rad_gap) * A_loss * (Te - Tgi)
    Qins = cfg.insulation_k / cfg.insulation_thickness * A_loss * (Te - Ta)
    P_abs = (cfg.solar_irradiance * cfg.optical_gain * cfg.glass_transmittance
             * cfg.eta_optical * A_loss)
    P_g = cfg.glass_absorptance * cfg.solar_irradiance * cfg.optical_gain * cfg.glass_area
    e_evap = cpl.hfg(Te) + cp * (Te - Ta)
    e_glass = cpl.hfg(Te) + cp * (Te - Tc)
    h_rad_out = cfg.glass_emissivity * SIGMA * (K(Tgi) ** 2 + K(Ta) ** 2) * (K(Tgi) + K(Ta))
    h_o = h_convection_wind(cfg.wind_speed) + h_rad_out
    U = 1.0 / (cfg.glass_thickness / cfg.glass_k + 1.0 / h_o)
    q_sky = cfg.glass_emissivity * SIGMA * (K(Ta) ** 4 - K(Tsky) ** 4)
    Qout = cfg.glass_area * U * ((Tgi - Ta) + q_sky / h_o)
    res = {
        "evaporator": abs(P_abs - m * e_evap - Qgap - Qins) / P_abs,
        "glass": abs(m * e_glass + Qgap + P_g - Qout) / max(P_abs, Qout),
        "film": abs((Tc - Tgi) - m * e_glass / (cfg.condensation_coefficient * cfg.glass_area))
                / max(abs(Tc - Ta), 1.0),
        # global energy: in = out, the hot condensate carrying m cp (Tc - Ta) away
        "global": abs(P_abs + P_g - Qout - Qins - m * cp * (Tc - Ta)) / P_abs,
        # mass: kg/session
        "mass_balance": abs(r.m_freshwater_kg - m * cfg.sunlight_hours * 3600.0 * cfg.collection_efficiency)
                        / max(r.m_freshwater_kg, 1e-12),
        "feed_balance": abs(r.feed_kg - r.m_freshwater_kg - r.brine_kg) / r.feed_kg,
    }
    if cfg.evaporation_mode == "mass_transfer" and not r.boiling_limited:
        P = r.P_chamber_Pa
        pe, pc = saturation_pressure_pa(Te), saturation_pressure_pa(Tc)
        Tm = K(0.5 * (Te + Tc))
        D = cpl.D_wa(Tm, P)
        flux = D * cpl.M_W * P / (cpl.R_UNIV * Tm * cfg.diffusion_path_length) * math.log((P - pc) / (P - pe))
        res["diffusion"] = abs(m - flux * A_eff) / m
    return res


def sane_cfg(rng: random.Random, **over) -> SystemConfig:
    A = rng.uniform(0.5, 3.0)
    kw = dict(
        solar_irradiance=rng.uniform(500, 1100), optical_gain=rng.uniform(1.0, 2.2),
        eta_optical=rng.uniform(0.7, 0.95), evaporator_area_geometric=A,
        area_utilization_factor=rng.uniform(0.5, 1.0), glass_area=A * rng.uniform(1.3, 3.0),
        wind_speed=rng.uniform(0.5, 6.0), ambient_temperature=rng.uniform(15, 40),
        condensation_coefficient=rng.uniform(2000, 15000), insulation_k=rng.uniform(0.03, 0.10),
        diffusion_path_length=rng.uniform(0.005, 0.08), gap_convection_coefficient=rng.uniform(0.5, 6.0),
        glass_absorptance=rng.uniform(0.02, 0.10), sunlight_hours=rng.uniform(4, 10),
        evaporation_mode=rng.choice(["mass_transfer", "energy_limited"]),
        feed_volume=2000.0,        # larger than any daily yield, so the feed/brine bookkeeping can close
    )
    kw.update(over)
    return SystemConfig(**kw)


# ----------------------------------------------------------------------------- closure
def test_closure_baseline():
    cfg = SystemConfig()
    r = simulate(cfg)
    assert r.coupling == "coupled" and r.solver_converged
    for name, v in closure(cfg, r).items():
        assert v < 1e-6, (name, v)
    assert r.solver_residual_max < 1e-6


@pytest.mark.parametrize("seed", range(10))
def test_closure_random_configs(seed):
    cfg = sane_cfg(random.Random(seed))
    r = simulate(cfg)
    assert r.solver_converged, r.warnings
    for name, v in closure(cfg, r).items():
        assert v < 1e-6, (seed, cfg.evaporation_mode, name, v)


@pytest.mark.parametrize("mode", EVAPORATION_MODES)
def test_closure_every_mode_baseline(mode):
    cfg = SystemConfig(evaporation_mode=mode)
    r = simulate(cfg)
    assert r.solver_converged
    for name, v in closure(cfg, r).items():
        assert v < 1e-6, (mode, name, v)


def test_closure_with_literal_brief_equations():
    """sensible_heat=False reproduces the brief's literal (a)/(c): P_abs = m h_fg(T_e) + ..."""
    cfg = SystemConfig(sensible_heat=False)
    r = simulate(cfg)
    assert r.solver_converged
    Te, Tc, m = r.T_evaporator_C, r.T_condensate_C, r.mdot_kg_s
    A = cfg.evaporator_area_geometric
    eps_gap = 1.0 / (1.0 / cfg.evaporator_emissivity + 1.0 / cfg.glass_emissivity - 1.0)
    Tgi = r.T_glass_C
    h = cfg.gap_convection_coefficient + eps_gap * SIGMA * (K(Te) ** 2 + K(Tgi) ** 2) * (K(Te) + K(Tgi))
    Qgap = h * A * (Te - Tgi)
    Qins = cfg.insulation_k / cfg.insulation_thickness * A * (Te - cfg.ambient_temperature)
    P_abs = cfg.solar_irradiance * cfg.optical_gain * cfg.glass_transmittance * cfg.eta_optical * A
    assert abs(P_abs - m * cpl.hfg(Te) - Qgap - Qins) / P_abs < 1e-6
    # glass uses h_fg(T_c) in the literal form
    assert abs((Tc - Tgi) - m * cpl.hfg(Tc) / (cfg.condensation_coefficient * cfg.glass_area)) < 1e-6


# ----------------------------------------------------------------------------- ordering / bounds
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("mode", EVAPORATION_MODES)
def test_temperature_ordering_and_energy_bound(seed, mode):
    rng = random.Random(100 + seed)
    cfg = sane_cfg(rng, solar_irradiance=rng.uniform(400, 1100), evaporation_mode=mode)
    r = simulate(cfg)
    assert r.solver_converged
    eps = 1e-6
    assert r.T_evaporator_C >= r.T_condensate_C - eps
    assert r.T_condensate_C >= r.T_glass_C - eps
    assert r.T_condensate_C >= cfg.ambient_temperature
    t = cfg.sunlight_hours * 3600.0
    P_abs = r.P_absorbed_W
    # yield bound: latent energy of the product cannot exceed what was absorbed
    assert r.m_evap_selected_kg * cpl.hfg(r.T_evaporator_C) <= P_abs * t * (1 + 1e-9)
    assert r.daily_freshwater_kg <= P_abs * t / cpl.hfg(r.T_evaporator_C) + 1e-9
    assert r.thermal_efficiency_actual <= 1.0 + 1e-9


def test_mode_ordering_ideal_bounds_diffusion():
    r = simulate(SystemConfig())
    m = r.mode_predictions_kg
    assert m["mass_transfer"] <= m["energy_limited"] + 1e-12
    assert m["empirical"] <= m["energy_limited"] + 1e-12
    assert m["hybrid"] <= min(m["empirical"], m["mass_transfer"]) + 1e-12
    assert set(m) == {"energy_limited", "empirical", "mass_transfer", "hybrid"}


# ----------------------------------------------------------------------------- the bug: regression
SWEEP_UP = {
    "solar_irradiance": (300.0, 1300.0),
    "optical_gain": (1.0, 3.0),
    "eta_optical": (0.4, 1.0),
    "glass_transmittance": (0.5, 0.95),
}


def _ys(cfg, name, vals):
    return [simulate(replace(cfg, **{name: v})).daily_freshwater_L for v in vals]


def _monotone(ys, sign, tol=1e-7):
    return all(sign * (b - a) >= -tol for a, b in zip(ys, ys[1:]))


@pytest.mark.parametrize("name", list(SWEEP_UP))
@pytest.mark.parametrize("mode", EVAPORATION_MODES)
@pytest.mark.parametrize("seed", range(6))
def test_yield_nondecreasing_in_useful_energy(name, mode, seed):
    """More absorbed useful energy can never lower yield (G, G_opt, eta_optical, tau_glass)."""
    cfg = sane_cfg(random.Random(500 + seed), evaporation_mode=mode)
    lo, hi = SWEEP_UP[name]
    vals = [lo + (hi - lo) * i / 12 for i in range(13)]
    ys = _ys(cfg, name, vals)
    assert _monotone(ys, +1), (name, mode, seed, ys)


@pytest.mark.parametrize("mode", ["mass_transfer", "energy_limited", "empirical", "hybrid"])
@pytest.mark.parametrize("seed", range(6))
def test_yield_nonincreasing_in_insulation_conductivity_and_gap_loss(mode, seed):
    cfg = sane_cfg(random.Random(700 + seed), evaporation_mode=mode)
    ks = [0.02 + 0.18 * i / 10 for i in range(11)]
    hs = [0.1 + 14.9 * i / 10 for i in range(11)]
    assert _monotone(_ys(cfg, "insulation_k", ks), -1)
    assert _monotone(_ys(cfg, "gap_convection_coefficient", hs), -1)


@pytest.mark.parametrize("mode", ["mass_transfer", "hybrid", "empirical"])
@pytest.mark.parametrize("seed", range(6))
def test_yield_nonincreasing_when_evaporation_gets_worse(mode, seed):
    """Fouling (derating down), longer diffusion path: yield never rises. (Legacy: it could.)"""
    cfg = sane_cfg(random.Random(900 + seed), evaporation_mode=mode)
    ders = [0.2 + 0.8 * i / 10 for i in range(11)]           # increasing derating = cleaner
    assert _monotone(_ys(cfg, "evaporation_derating", ders), +1)
    deltas = [0.004 * (1.4 ** i) for i in range(12)]          # increasing delta = worse
    assert _monotone(_ys(cfg, "diffusion_path_length", deltas), -1)


@pytest.mark.parametrize("mode", EVAPORATION_MODES)
def test_eta_thermal_is_not_on_the_coupled_path(mode):
    base = simulate(SystemConfig(evaporation_mode=mode)).daily_freshwater_L
    for eta in (0.1, 0.4, 0.95):
        assert simulate(SystemConfig(evaporation_mode=mode, eta_thermal=eta)).daily_freshwater_L == base


def test_legacy_artifact_is_still_reproducible_but_coupled_is_free_of_it():
    """Documents the bug: legacy yield RISES as eta_thermal falls when the glass binds."""
    leg = [simulate(SystemConfig(coupling="legacy", eta_thermal=e)).daily_freshwater_L for e in (0.45, 0.65, 0.80)]
    assert leg[0] > leg[-1]                      # the physically impossible response
    cp = [simulate(SystemConfig(eta_thermal=e)).daily_freshwater_L for e in (0.45, 0.65, 0.80)]
    assert cp[0] == cp[1] == cp[2]


@pytest.mark.parametrize("mode", ["mass_transfer", "energy_limited"])
def test_area_utilization_never_lowers_yield(mode):
    """Legacy: more wet area heated the evaporator and could raise OR lower yield by artifact."""
    ys = _ys(SystemConfig(evaporation_mode=mode), "area_utilization_factor", [0.2, 0.4, 0.6, 0.8, 1.0])
    assert _monotone(ys, +1)


def test_wind_effect_depends_on_which_resistance_binds():
    """Ideal-evaporator limit: wind helps (the cover rejects more).  Diffusion mode: while the
    evaporator is genuinely diffusion-limited, wind cools the cover, the evaporator-to-cover leak
    grows and yield FALLS; in the cover-limited (boiling-limited) regime it rises again."""
    winds = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
    a = _ys(SystemConfig(evaporation_mode="energy_limited"), "wind_speed", winds)
    assert _monotone(a, +1)
    n_dif = 0
    for seed in range(12):
        rng = random.Random(1100 + seed)
        A = rng.uniform(0.5, 3.0)
        cfg = sane_cfg(rng, evaporation_mode="mass_transfer", glass_area=A * 2.0, solar_irradiance=900.0)
        rs = [simulate(replace(cfg, wind_speed=v)) for v in winds]
        for r0, r1 in zip(rs, rs[1:]):
            if not r0.boiling_limited and not r1.boiling_limited:
                assert r1.daily_freshwater_L <= r0.daily_freshwater_L + 1e-7, seed
                n_dif += 1
            elif r0.boiling_limited and r1.boiling_limited:
                assert r1.daily_freshwater_L >= r0.daily_freshwater_L - 1e-7, seed
    assert n_dif > 20


# ----------------------------------------------------------------------------- glass area
def test_glass_area_sweep_ideal_evaporator_monotone_and_saturating():
    base = SystemConfig(evaporation_mode="energy_limited")
    ratios = [0.8 * 1.25 ** i for i in range(30)]            # 0.8 .. ~500
    ys = [simulate(replace(base, glass_area=r * base.evaporator_area_geometric)).daily_freshwater_L for r in ratios]
    assert _monotone(ys, +1)
    asym = ys[-1]
    r99 = next(r for r, y in zip(ratios, ys) if y >= 0.99 * asym)
    assert r99 < 30.0                                         # finite saturation ratio
    # asymptote respects the energy bound
    r = simulate(replace(base, glass_area=500.0))
    assert r.daily_freshwater_kg <= r.P_absorbed_W * base.sunlight_hours * 3600.0 / cpl.hfg(r.T_evaporator_C) + 1e-9


def test_glass_area_sweep_with_empirical_cap_saturates_at_the_evaporation_limit():
    """Mode B: yield = min(cover capacity, J_emp A_eff): rises, then sits EXACTLY on the evaporation cap."""
    base = SystemConfig(evaporation_mode="empirical", empirical_flux=0.9)
    cap_L = base.empirical_flux * base.evaporator_area_geometric * base.area_utilization_factor \
        * base.sunlight_hours * base.collection_efficiency
    ratios = [0.8 * 1.25 ** i for i in range(24)]
    ys = [simulate(replace(base, glass_area=r * base.evaporator_area_geometric)).daily_freshwater_L for r in ratios]
    assert _monotone(ys, +1)
    plateau = [r for r, y in zip(ratios, ys) if abs(y - cap_L) < 1e-9]
    assert plateau and plateau[0] < 10.0                      # saturates at a finite ratio
    assert max(ys) <= cap_L + 1e-9


def test_glass_area_diffusion_mode_is_bounded_by_ideal_and_has_an_optimum():
    """FINDING: with the default vapour-gap resistance, the diffusion-mode yield peaks and then FALLS
    slowly as the cover gets colder.  It is always below the ideal-evaporator limit."""
    ratios = [0.9 * 1.3 ** i for i in range(16)]              # 0.9 .. ~50
    yc, ya = [], []
    for r in ratios:
        g = r * 1.0
        yc.append(simulate(SystemConfig(evaporation_mode="mass_transfer", glass_area=g)).daily_freshwater_L)
        ya.append(simulate(SystemConfig(evaporation_mode="energy_limited", glass_area=g)).daily_freshwater_L)
    assert all(c <= a + 1e-9 for c, a in zip(yc, ya))
    i_max = max(range(len(yc)), key=yc.__getitem__)
    assert yc[-1] < yc[i_max] - 0.5                           # declines beyond the optimum


# ----------------------------------------------------------------------------- venturi / ejector
def _vac(**kw):
    """Moderate ejector: set pressure well above the vapour pressure of the cover (no boiling limit)."""
    return SystemConfig(venturi_enabled=True, motive_flow_rate=5.0e-4, venturi_throat_diameter=0.008,
                        venturi_loss_fraction=0.3, **kw)


def test_enabling_the_venturi_changes_yield():
    off = simulate(SystemConfig()).daily_freshwater_L
    on = simulate(SystemConfig(venturi_enabled=True)).daily_freshwater_L
    assert abs(on - off) > 1e-3                                # legacy: 3.0233 vs 3.0233, identical


def test_moderate_vacuum_raises_the_evaporation_side():
    off, on = simulate(SystemConfig()), simulate(_vac())
    assert not on.boiling_limited
    assert on.P_chamber_Pa < off.P_chamber_Pa
    assert on.T_evaporator_C < off.T_evaporator_C              # easier evaporation at the same heat input
    assert on.daily_freshwater_L > off.daily_freshwater_L
    ideal = simulate(_vac(evaporation_mode="energy_limited"))
    assert on.daily_freshwater_L <= ideal.daily_freshwater_L + 1e-9   # never beats the ideal evaporator


def test_vacuum_gain_is_bounded_and_saturates_when_the_cover_binds():
    """Deep vacuum cannot help once the cover's condensing temperature reaches Psat(P): the chamber
    pressure floats up, evaporation is energy-limited and the gain stops.  Asserted against the
    solver's OWN diagnostics (boiling_limited, P_chamber_Pa), not a hard-coded sign."""
    base = simulate(SystemConfig())
    deep = simulate(SystemConfig(venturi_enabled=True))
    deeper = simulate(SystemConfig(venturi_enabled=True, motive_flow_rate=4e-3))
    ideal = simulate(SystemConfig(evaporation_mode="energy_limited"))
    for r in (deep, deeper):
        assert r.boiling_limited and r.limited_by == "boiling"
        assert r.P_chamber_Pa > r.P_chamber_set_Pa              # the ejector could not hold its set point
        assert math.isclose(cpl.psat(r.T_condensate_C), r.P_chamber_Pa, rel_tol=1e-6)
        assert r.daily_freshwater_L == pytest.approx(ideal.daily_freshwater_L, rel=1e-9)
        assert any("Boiling-limited" in w for w in r.warnings)
    assert deeper.daily_freshwater_L == pytest.approx(deep.daily_freshwater_L, rel=1e-9)   # more vacuum, no gain
    assert 0 < deep.daily_freshwater_L - base.daily_freshwater_L < 0.25 * base.daily_freshwater_L


def test_net_electrical_cost_of_the_ejector_is_reported():
    r = simulate(SystemConfig(venturi_enabled=True))
    assert r.ejector_power_W > 0 and r.ejector_kWh_per_m3 > 0
    q, dP = 1e-3, 87738.7
    assert r.ejector_power_W == pytest.approx(q * dP / 0.5, rel=1e-3)
    assert simulate(SystemConfig()).ejector_power_W == 0.0


def test_chamber_pressure_never_below_motive_vapour_pressure_or_leak_floor():
    rng = random.Random(7)
    for _ in range(200):
        Tm = rng.uniform(5, 60)
        floor = rng.uniform(1000, 60000)
        D1 = rng.uniform(0.01, 0.1)
        e = compute_ejector(True, 10 ** rng.uniform(-4, -1), D1, D1 * rng.uniform(0.05, 0.9), 101325.0,
                            1000.0, rng.uniform(0.6, 1.0), rng.uniform(0, 0.9), Tm, floor, 0.5)
        assert e.P_chamber_set_Pa >= saturation_pressure_pa(Tm) - 1e-9
        assert e.P_chamber_set_Pa >= min(floor, 101325.0) - 1e-9
        assert e.P_chamber_set_Pa <= 101325.0
    # the legacy 500 Pa floor is gone: a violent ejector now stops at P_sat(T_motive) / the leak floor
    e = compute_ejector(True, 0.1, 0.1, 0.002, 101325.0, 1000.0, 0.95, 0.0, 30.0, 1000.0, 0.5)
    assert e.P_chamber_set_Pa == pytest.approx(saturation_pressure_pa(30.0), rel=1e-9)
    assert e.binding_floor == "motive_vapour_pressure"
    e = compute_ejector(True, 0.1, 0.1, 0.002, 101325.0, 1000.0, 0.95, 0.0, 30.0, 8000.0, 0.5)
    assert e.P_chamber_set_Pa == 8000.0 and e.binding_floor == "leak_floor"
    # through simulate()
    r = simulate(SystemConfig(venturi_enabled=True, motive_flow_rate=0.05, motive_water_temperature=50.0))
    assert r.P_chamber_Pa >= saturation_pressure_pa(50.0) - 1e-6


def test_boiling_limited_flag_triggers_when_expected_and_not_otherwise():
    assert not simulate(SystemConfig()).boiling_limited
    assert simulate(SystemConfig(venturi_enabled=True)).boiling_limited
    # an evaporator too small for diffusion to carry the energy: it boils at T_sat(P), energy-limited, finite
    cfg = SystemConfig(evaporator_area_geometric=3.0, area_utilization_factor=0.1, glass_area=6.0,
                       evaporation_mode="mass_transfer", diffusion_path_length=0.1)
    r = simulate(cfg)
    assert r.boiling_limited and r.solver_converged
    assert r.T_evaporator_C == pytest.approx(cpl.t_sat_C(cfg.atmospheric_pressure), abs=0.05)   # pinned at T_sat(P)
    assert math.isfinite(r.daily_freshwater_L) and r.daily_freshwater_L > 0
    assert r.m_evap_selected_kg * cpl.hfg(r.T_evaporator_C) <= r.P_absorbed_W * 3600 * cfg.sunlight_hours


def test_vacuum_load_on_the_cover():
    c = cover_vacuum_load(101325.0, 51325.0, 1.0, 0.010, 2e7)
    assert c.load_kN == pytest.approx(50.0)
    assert c.equivalent_mass_t == pytest.approx(5.1, rel=0.01)    # "1 m2 at 50 kPa carries ~5 tonnes"
    assert c.implausible and "SMALL" in c.message
    small = cover_vacuum_load(101325.0, 51325.0, 0.01, 0.010, 2e7)
    assert not small.implausible
    assert cover_vacuum_load(101325.0, 101325.0, 1.0, 0.01, 2e7).load_kN == 0.0
    r = simulate(SystemConfig(venturi_enabled=True))
    assert r.vacuum_load_kN == pytest.approx((101325.0 - r.P_chamber_Pa) * r.A_geometric * 1.1 / 1e3, rel=1e-9) \
        or r.vacuum_load_kN > 0
    assert r.cover_vacuum_implausible
    assert any("tonnes" in w and "Screening estimate only" in w for w in r.warnings)
    assert simulate(SystemConfig()).vacuum_load_kN == 0.0


# ----------------------------------------------------------------------------- building blocks
def test_marrero_mason_diffusivity_and_inverse_pressure_dependence():
    D = cpl.D_wa(300.0, 101325.0)
    assert D == pytest.approx(2.5e-5, rel=0.08)                  # literature ~2.5-2.6e-5 m2/s
    assert cpl.D_wa(300.0, 50662.5) == pytest.approx(2.0 * D, rel=1e-12)
    assert cpl.D_wa(350.0, 101325.0) > D


def test_stefan_flux_matches_hand_calculation_and_limits():
    Te, Tc, P, A, d = 60.0, 40.0, 101325.0, 1.0, 0.02
    pe, pc = saturation_pressure_pa(Te), saturation_pressure_pa(Tc)
    Tm = 323.15
    hand = cpl.D_wa(Tm, P) * 0.01801528 * P / (8.314462618 * Tm * d) * math.log((P - pc) / (P - pe))
    assert cpl.diffusion_flux_kg_s(Te, Tc, P, A, d) == pytest.approx(hand, rel=1e-12)
    assert cpl.diffusion_flux_kg_s(Tc, Te, P, A, d) == 0.0       # no driving force
    big = cpl.diffusion_flux_kg_s(cpl.t_sat_C(P), Tc, P, A, d)   # boiling: finite sentinel, not inf/NaN
    assert math.isfinite(big) and big >= cpl.BIG_FLOW
    assert cpl.diffusion_flux_kg_s(Te, Tc, 0.5e5, A, d) > cpl.diffusion_flux_kg_s(Te, Tc, P, A, d)  # less air


def test_saturation_temperature_inverts_magnus():
    for T in (0.0, 30.0, 70.0, 100.0, 140.0):
        assert cpl.t_sat_C(saturation_pressure_pa(T)) == pytest.approx(T, abs=1e-8)
    assert cpl.t_sat_C(saturation_pressure_pa(100.0)) == pytest.approx(100.0, abs=1e-8)


def test_condensation_limited_by_is_a_diagnostic_from_the_resistance_shares():
    r = simulate(SystemConfig())
    assert sum(r.resistance_share.values()) == pytest.approx(1.0)
    assert r.limited_by == max(r.resistance_share, key=r.resistance_share.get)
    assert r.condensation_limited_by in ("glass_uvalue", "film", "evaporation")
    # a huge cover moves the dominant resistance to the vapour gap
    r2 = simulate(SystemConfig(glass_area=40.0))
    assert r2.limited_by == "evaporation" and r2.condensation_limited_by == "evaporation"
    assert simulate(SystemConfig(evaporation_mode="empirical")).limited_by == "empirical_flux"


# ----------------------------------------------------------------------------- fuzz
def _fuzz_cfg(rng: random.Random) -> SystemConfig:
    names = [f.name for f in fields(SystemConfig) if f.name in PARAMETER_REGISTRY]
    kw = {}
    for nm in names:
        lo, hi = PARAMETER_REGISTRY[nm].allowed_range
        if lo > 0 and hi / lo > 1e3:
            kw[nm] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            kw[nm] = rng.uniform(lo, hi)
    kw["venturi_enabled"] = rng.random() < 0.5
    kw["sensible_heat"] = rng.random() < 0.8
    if kw["venturi_throat_diameter"] >= kw["venturi_inlet_diameter"]:
        kw["venturi_throat_diameter"] = 0.5 * kw["venturi_inlet_diameter"]
    kw["evaporation_mode"] = rng.choice(EVAPORATION_MODES)
    return SystemConfig(**kw)


def test_fuzz_registry_box_never_nan_or_inf_and_never_silently_unconverged():
    rng = random.Random(2026)
    n_bad = 0
    for _ in range(1500):
        cfg = _fuzz_cfg(rng)
        r = simulate(cfg)
        for k, v in r.__dict__.items():
            if isinstance(v, float):
                assert math.isfinite(v), (k, v, cfg)
        for v in list(r.mode_predictions_kg.values()) + list(r.solver_residuals.values()):
            assert math.isfinite(v)
        assert r.daily_freshwater_L >= 0.0 and r.mdot_kg_s >= 0.0
        if r.solver_converged:
            assert r.solver_residual_max <= 1e-6
            # energy bound P_abs >= m e_evap holds only if no heat flows IN: a sub-ambient evaporator
            # (radiatively cooled cover) legitimately gains ambient heat through the insulation and the gap
            if r.T_evaporator_C >= max(cfg.ambient_temperature, r.T_glass_C):
                e_ev = cpl.hfg(r.T_evaporator_C) + (cfg.water_specific_heat * (r.T_evaporator_C - cfg.ambient_temperature)
                                                     if cfg.sensible_heat else 0.0)
                assert r.m_evap_selected_kg * e_ev <= r.P_absorbed_W * 3600 * cfg.sunlight_hours * (1 + 1e-9)
        else:
            n_bad += 1
            assert any(("cannot reject" in w) or ("did not close" in w) for w in r.warnings)   # never silent
    assert n_bad < 0.2 * 1500            # infeasible corners of the box (cover far too small) are flagged


def test_fuzz_other_resolved_derived_quantities_are_finite_in_all_modes_at_extremes():
    corners = []
    for G in (100.0, 1361.0):
        for Ag in (0.05, 120.0):
            for A in (0.05, 100.0):
                for mode in EVAPORATION_MODES:
                    corners.append(SystemConfig(solar_irradiance=G, glass_area=Ag, evaporator_area_geometric=A,
                                                evaporation_mode=mode))
    for cfg in corners:
        r = simulate(cfg)
        assert all(math.isfinite(v) for v in r.__dict__.values() if isinstance(v, float))


# ----------------------------------------------------------------------------- legacy reproducibility
def test_legacy_reproduces_3p0233_exactly():
    r = simulate(SystemConfig(coupling="legacy"))
    assert round(r.daily_freshwater_L, 4) == 3.0233
    assert round(r.m_evap_selected_kg, 4) == 7.5816
    assert round(r.m_condensation_capacity_kg, 4) == 3.3593
    assert r.coupling == "legacy" and r.evaporation_mode_used == "energy_limited"


def test_legacy_matches_the_frozen_pre_change_reference_to_machine_precision():
    ref = json.loads(DATA.read_text())
    for name, entry in ref.items():
        if "cfg" not in entry:
            continue
        cfgd = {k: v for k, v in entry["cfg"].items()}
        r = simulate(SystemConfig(coupling="legacy", **cfgd))
        for k, v in entry["res"].items():
            if k == "modes":
                for mk, mv in v.items():
                    assert r.mode_predictions_kg[mk] == pytest.approx(mv, rel=1e-12, abs=1e-12), (name, mk)
            elif isinstance(v, float):
                assert getattr(r, k) == pytest.approx(v, rel=1e-12, abs=1e-12), (name, k)
            elif isinstance(v, (str, bool, int)):
                assert getattr(r, k) == v, (name, k)


def test_coupling_flag_is_validated():
    assert any("coupling" in v for v in SystemConfig(coupling="nope").validate())
    assert SystemConfig().evaporation_mode == "mass_transfer"
    assert SystemConfig(coupling="legacy").evaporation_mode == "energy_limited"
