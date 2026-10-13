"""coupling="legacy": every behaviour the pre-coupling test-suite checked must still hold, unchanged.

These are the original system / scaling / uncertainty / validation tests re-run with the legacy solver
(the originals in the other files now run on the default coupled solver as well).
"""

import math

import pytest

from src.physics.system import SystemConfig, simulate
from src.scaling.monte_carlo import monte_carlo
from src.scaling.sensitivity import DEFAULT_TORNADO_RANGES, sweep_one, tornado
from src.scaling.sizing import HouseholdDemand, size_for_demand, size_for_village, sweep_scale
from src.validation.validation_cases import (
    run_condensation_headroom, run_empirical_linear_scaling, run_energy_vs_empirical_consistency,
)

LEG = dict(coupling="legacy")


def L(**kw):
    return SystemConfig(**LEG, **kw)


def test_simulate_returns_positive_freshwater_with_defaults():
    r = simulate(L())
    assert r.daily_freshwater_L > 0 and r.recovery_ratio >= 0 and r.energy_efficiency <= 1.0
    assert r.daily_freshwater_kg <= r.m_evap_selected_kg + 1e-9
    assert r.daily_freshwater_kg <= r.m_condensation_capacity_kg + 1e-9
    assert -50 < r.T_evaporator_C < 200 and -50 < r.T_glass_C < 200
    assert set(r.mode_predictions_kg) == {"energy_limited", "empirical", "mass_transfer", "hybrid"}


def test_legacy_baseline_is_condensation_limited_as_documented():
    r = simulate(L())
    assert r.condensation_limited_by == "glass_uvalue"
    assert r.m_condensation_capacity_kg < r.m_evap_selected_kg
    assert any("Condensation is limiting production" in w for w in r.warnings)


def test_legacy_doubles_with_area_in_empirical_mode():
    cfg = L(evaporation_mode="empirical", condensation_coefficient=20000.0, glass_area=5.0, collection_efficiency=1.0)
    r1 = simulate(cfg)
    cfg.evaporator_area_geometric, cfg.glass_area = 2.0, 10.0
    assert math.isclose(simulate(cfg).daily_freshwater_L / r1.daily_freshwater_L, 2.0, rel_tol=1e-3)


def test_legacy_validates_config():
    r = simulate(L(solar_irradiance=5000))
    assert any("solar_irradiance" in w for w in r.warnings)


def test_legacy_venturi_never_reaches_yield():
    """The defect the coupled model fixes, pinned so nobody 'fixes' legacy by accident."""
    a = simulate(L(venturi_enabled=False)).daily_freshwater_L
    b = simulate(L(venturi_enabled=True)).daily_freshwater_L
    assert a == b
    assert simulate(L(venturi_enabled=True)).P_chamber_Pa < 101325.0


def test_legacy_sizing_and_scaling():
    cfg = L(evaporation_mode="empirical", glass_area=10.0, condensation_coefficient=20000.0, collection_efficiency=1.0)
    s = size_for_demand(30.0, cfg)
    assert s.A_required_m2 > 0 and s.modules_required >= 1 and s.total_footprint_m2 >= s.A_required_m2
    d = HouseholdDemand(4, 4.0, 2.0, 1.0)
    cfg2 = L(evaporation_mode="empirical", glass_area=50.0, condensation_coefficient=20000.0, collection_efficiency=1.0)
    assert math.isclose(size_for_village(10, d, cfg2).A_required_m2, 10 * size_for_village(1, d, cfg2).A_required_m2, rel_tol=1e-6)
    rows = sweep_scale([1, 10, 100, 1000], 6.0, 1.0, cfg2)
    areas = [r["A_required_m2"] for r in rows]
    assert areas == sorted(areas)


def test_legacy_kalpar_household_sizing_is_reproduced():
    prod = simulate(L()).productivity_L_per_m2_day
    d = HouseholdDemand(4, 4.0, 2.0, 1.2)
    s = size_for_village(61, d, L(), A_per_module=1.0)
    assert s.target_L_per_day == pytest.approx(61 * 4 * 7.2)
    assert s.A_required_m2 == pytest.approx(61 * 4 * 7.2 / prod)


def test_legacy_sensitivity_and_montecarlo():
    cfg = L()
    sr = sweep_one("solar_irradiance", [cfg.solar_irradiance], cfg)
    assert math.isclose(sr.outputs[0], sr.baseline_output, rel_tol=1e-9)
    ranges = [e.range_output for e in tornado(DEFAULT_TORNADO_RANGES, cfg)]
    assert ranges == sorted(ranges, reverse=True)
    a, b = monte_carlo(200, cfg, seed=7), monte_carlo(200, cfg, seed=7)
    assert math.isclose(a.mean, b.mean) and math.isclose(a.p95, b.p95)
    r = monte_carlo(500, cfg, seed=42)
    assert r.p05 <= r.p50 <= r.p95


def test_legacy_montecarlo_reproduces_the_published_spread():
    mc = monte_carlo(2000, L(), seed=42)
    assert mc.mean == pytest.approx(3.0386873, rel=1e-6)
    assert mc.std == pytest.approx(1.0460123, rel=1e-6)
    rho = mc.spearman()
    assert rho["eta_thermal"] == pytest.approx(-0.64, abs=0.02)          # the artifact
    assert rho["area_utilization_factor"] == pytest.approx(-0.33, abs=0.02)


def test_legacy_tornado_still_shows_the_eta_thermal_artifact():
    t = {e.parameter: e for e in tornado(DEFAULT_TORNADO_RANGES, L())}
    assert t["eta_thermal"].low_output > t["eta_thermal"].high_output   # lower efficiency, MORE water


def test_legacy_validation_cases():
    assert all(c.passed for c in run_empirical_linear_scaling([1.0, 3.6, 10.0], 1.0, coupling="legacy"))
    assert run_energy_vs_empirical_consistency(coupling="legacy").passed
    # legacy flags the 8.4 kg/m2/day claim as condensation-limited (a DESIGN FLAG, not a pass)
    assert not run_condensation_headroom(coupling="legacy").passed
