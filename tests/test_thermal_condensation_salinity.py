"""Phase 2/4/5 tests: thermal, condensation, mass balance, salinity, Venturi."""

import math

import pytest

from src.physics.heat_transfer import (
    h_convection_wind,
    T_sky_celsius,
    solve_evaporator_temperature,
)
from src.physics.condensation import compute_condensation, compute_U_glass
from src.physics.mass_balance import compute_mass_balance
from src.physics.salinity import compute_salinity
from src.physics.venturi import compute_venturi


def test_h_convection_increases_with_wind():
    assert h_convection_wind(0) < h_convection_wind(5)


def test_T_sky_below_ambient():
    assert T_sky_celsius(30) < 30


def test_thermal_solve_conserves_energy():
    state = solve_evaporator_temperature(
        P_absorbed_W=800, P_evaporation_W=400, T_amb_C=30, area=1.0,
        emissivity=0.95, wind_speed=2.0, k_ins=0.04, d_ins=0.05,
    )
    losses = state.Q_convection_W + state.Q_radiation_W + state.Q_conduction_W
    assert math.isclose(losses, 800 - 400, rel_tol=5e-2)


def test_thermal_T_evap_above_ambient_when_absorbing():
    state = solve_evaporator_temperature(
        P_absorbed_W=600, P_evaporation_W=200, T_amb_C=25, area=1.0,
        emissivity=0.95, wind_speed=1.0, k_ins=0.04, d_ins=0.05,
    )
    assert state.T_evap_C > 25


def test_condensation_reduces_to_evap_when_capacity_high():
    r = compute_condensation(
        m_evap_kg=5.0, T_vapor_C=60, T_glass_C=35, T_ambient_C=30,
        glass_area=100.0, glass_thickness=0.010, glass_k=1.0,
        glass_emissivity=0.88, wind_speed=2.0, h_cond_inner=20000,
        h_fg=2.4e6, sunlight_hours=8.0, collection_efficiency=1.0,
    )
    assert r.limited_by == "evaporation"
    assert math.isclose(r.m_freshwater_collected_kg, 5.0, rel_tol=1e-6)


def test_condensation_caps_at_glass_capacity_when_low():
    r = compute_condensation(
        m_evap_kg=50.0, T_vapor_C=45, T_glass_C=40, T_ambient_C=30,
        glass_area=0.5, glass_thickness=0.010, glass_k=1.0,
        glass_emissivity=0.88, wind_speed=2.0, h_cond_inner=8000,
        h_fg=2.4e6, sunlight_hours=8.0, collection_efficiency=0.9,
    )
    assert r.limited_by in ("glass_uvalue", "film")
    assert r.m_freshwater_collected_kg < 50.0


def test_mass_balance_freshwater_bounded_by_evap_and_cond():
    mb = compute_mass_balance(feed_kg=10.0, evaporated_kg=3.0,
                              condensed_capacity_kg=4.0,
                              freshwater_collected_kg=10.0)
    assert mb.freshwater_collected_kg <= 3.0
    assert mb.brine_kg >= 0
    assert abs(mb.closure_error_kg) < 1e-9


def test_salinity_conservation_of_salt_mass():
    s = compute_salinity(feed_volume_L=10.0, feed_tds_mg_per_L=5000.0,
                         m_evap_kg=1.0, water_density_kg_per_m3=1000.0,
                         saturation_limit_mg_per_L=250000.0)
    assert s.brine_tds_mg_per_L > s.initial_tds_mg_per_L
    assert math.isclose(s.brine_tds_mg_per_L * (10.0 - 1.0), 5000.0 * 10.0,
                        rel_tol=1e-3)


def test_salinity_flag_saturated():
    # Almost all water removed -> extreme concentration.
    s = compute_salinity(feed_volume_L=10.0, feed_tds_mg_per_L=50000.0,
                         m_evap_kg=9.8, water_density_kg_per_m3=1000.0,
                         saturation_limit_mg_per_L=250000.0)
    assert s.saturated is True


def test_venturi_disabled_returns_atmospheric():
    v = compute_venturi(False, 1e-3, 0.025, 0.010, 101325.0, 1000.0, 0.95, 0.3)
    assert v.enabled is False
    assert v.P_chamber_effective_Pa == 101325.0


def test_venturi_pressure_drop_positive_when_enabled():
    v = compute_venturi(True, 1e-3, 0.025, 0.010, 101325.0, 1000.0, 0.95, 0.3)
    assert v.dP_theoretical_Pa > 0
    assert v.P_chamber_effective_Pa < v.P_inlet_Pa
    assert v.saturation_temperature_shift_C > 0


def test_venturi_rejects_bigger_throat_than_inlet():
    with pytest.raises(ValueError):
        compute_venturi(True, 1e-3, 0.010, 0.020, 101325.0, 1000.0, 0.95, 0.3)


def test_U_glass_reasonable_order_of_magnitude():
    U = compute_U_glass(h_cond_inner=8000, glass_thickness=0.010,
                        glass_k=1.0, T_glass_C=40, T_amb_C=30,
                        wind_speed=2.0, emissivity_glass=0.88)
    assert 1.0 < U < 30.0
