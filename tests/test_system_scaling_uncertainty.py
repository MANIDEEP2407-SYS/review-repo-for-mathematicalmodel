"""Phase 6-11 tests: system aggregator, scaling, sensitivity, Monte Carlo, validation."""

import math

import pytest

from src.physics.system import SystemConfig, simulate
from src.scaling.sizing import (
    HouseholdDemand, size_for_demand, size_for_household,
    size_for_village, sweep_scale,
)
from src.scaling.sensitivity import sweep_one, tornado, DEFAULT_TORNADO_RANGES
from src.scaling.monte_carlo import monte_carlo
from src.validation.validation_cases import (
    run_empirical_linear_scaling, run_energy_vs_empirical_consistency,
)


# ---------- System aggregator ----------

def test_simulate_returns_positive_freshwater_with_defaults():
    r = simulate()
    assert r.daily_freshwater_L > 0
    assert r.recovery_ratio >= 0
    assert r.energy_efficiency <= 1.0


def test_simulate_freshwater_le_evaporated():
    r = simulate()
    assert r.daily_freshwater_kg <= r.m_evap_selected_kg + 1e-9


def test_simulate_freshwater_le_condensation_capacity():
    r = simulate()
    assert r.daily_freshwater_kg <= r.m_condensation_capacity_kg + 1e-9


def test_simulate_doubles_with_area_in_empirical_mode():
    cfg = SystemConfig()
    cfg.evaporation_mode = "empirical"
    cfg.condensation_coefficient = 20000.0
    cfg.glass_area = 5.0
    cfg.collection_efficiency = 1.0
    r1 = simulate(cfg)
    cfg.evaporator_area_geometric = 2.0
    cfg.glass_area = 10.0
    r2 = simulate(cfg)
    assert math.isclose(r2.daily_freshwater_L / r1.daily_freshwater_L, 2.0, rel_tol=1e-3)


def test_simulate_temperature_bounded():
    r = simulate()
    assert -50 < r.T_evaporator_C < 200
    assert -50 < r.T_glass_C < 200


def test_simulate_produces_mode_predictions_for_all_modes():
    r = simulate()
    assert set(r.mode_predictions_kg.keys()) == {
        "energy_limited", "empirical", "mass_transfer", "hybrid",
    }


def test_simulate_validates_config():
    cfg = SystemConfig()
    cfg.solar_irradiance = 5000  # out of range
    r = simulate(cfg)
    # Should be reported in warnings
    assert any("solar_irradiance" in w for w in r.warnings)


# ---------- Scaling ----------

def test_size_for_demand_gives_positive_area():
    cfg = SystemConfig()
    cfg.evaporation_mode = "empirical"
    cfg.glass_area = 10.0
    cfg.condensation_coefficient = 20000.0
    cfg.collection_efficiency = 1.0
    s = size_for_demand(target_L_per_day=30.0, reference=cfg)
    assert s.A_required_m2 > 0
    assert s.modules_required >= 1
    assert s.total_footprint_m2 >= s.A_required_m2


def test_household_demand_math():
    d = HouseholdDemand(people=4, drinking_per_person_L=4.0,
                        cooking_per_person_L=2.0, safety_factor=1.2)
    assert math.isclose(d.total_L_per_day, 4 * 6 * 1.2)


def test_village_scales_linearly_in_area():
    cfg = SystemConfig()
    cfg.evaporation_mode = "empirical"
    cfg.glass_area = 50.0
    cfg.condensation_coefficient = 20000.0
    cfg.collection_efficiency = 1.0
    d = HouseholdDemand(4, 4.0, 2.0, 1.0)
    s1 = size_for_village(1, d, cfg)
    s10 = size_for_village(10, d, cfg)
    assert math.isclose(s10.A_required_m2, 10 * s1.A_required_m2, rel_tol=1e-6)


def test_population_sweep_monotonic_area():
    cfg = SystemConfig()
    cfg.evaporation_mode = "empirical"
    cfg.glass_area = 5.0
    cfg.condensation_coefficient = 20000.0
    cfg.collection_efficiency = 1.0
    rows = sweep_scale([1, 10, 100, 1000], 6.0, 1.0, cfg)
    areas = [r["A_required_m2"] for r in rows]
    assert areas == sorted(areas)


# ---------- Sensitivity ----------

def test_sweep_one_matches_baseline_at_baseline_value():
    cfg = SystemConfig()
    sr = sweep_one("solar_irradiance", [cfg.solar_irradiance], cfg)
    assert math.isclose(sr.outputs[0], sr.baseline_output, rel_tol=1e-9)


def test_tornado_returns_sorted_descending():
    entries = tornado(DEFAULT_TORNADO_RANGES)
    ranges = [e.range_output for e in entries]
    assert ranges == sorted(ranges, reverse=True)


# ---------- Monte Carlo ----------

def test_monte_carlo_reproducible_with_seed():
    a = monte_carlo(n_samples=200, seed=7)
    b = monte_carlo(n_samples=200, seed=7)
    assert math.isclose(a.mean, b.mean)
    assert math.isclose(a.p95, b.p95)


def test_monte_carlo_percentiles_ordered():
    r = monte_carlo(n_samples=500, seed=42)
    assert r.p05 <= r.p50 <= r.p95


# ---------- Validation ----------

def test_empirical_linear_scaling_within_tolerance():
    cases = run_empirical_linear_scaling([1.0, 3.6, 10.0], tolerance_percent=1.0)
    assert all(c.passed for c in cases)


def test_energy_vs_empirical_within_order_of_magnitude():
    c = run_energy_vs_empirical_consistency()
    assert c.passed  # tolerance ratio 3x
