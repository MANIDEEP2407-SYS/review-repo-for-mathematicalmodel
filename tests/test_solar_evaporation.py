"""Phase 1 tests: solar input and evaporation modes."""

import math

import pytest

from src.physics.solar import compute_solar_input, reflector_gain
from src.physics.evaporation import (
    saturation_pressure_pa,
    latent_heat_vaporization,
    evaporate_energy_limited,
    evaporate_empirical,
    evaporate_mass_transfer,
    evaporate_hybrid,
)


def test_solar_incident_scales_linearly_with_area():
    a = compute_solar_input(800, 1.0, 8.0, 1.0, 1.0, 1.0).Q_incident
    b = compute_solar_input(800, 2.0, 8.0, 1.0, 1.0, 1.0).Q_incident
    assert math.isclose(b, 2.0 * a)


def test_solar_effective_irradiance_uses_gain_and_transmittance():
    s = compute_solar_input(1000, 1.0, 1.0, optical_gain=1.5,
                            glass_transmittance=0.9, eta_optical=0.9)
    assert math.isclose(s.G_effective, 1000 * 1.5 * 0.9 * 0.9)


def test_reflector_gain_greater_than_one():
    g = reflector_gain(reflector_area=1.0, reflectivity=0.85, base_area=1.0)
    assert g > 1.0
    assert math.isclose(g, 1.85, rel_tol=1e-6)


def test_saturation_pressure_monotonic():
    assert saturation_pressure_pa(30) > saturation_pressure_pa(20)
    assert saturation_pressure_pa(60) > saturation_pressure_pa(30)


def test_saturation_pressure_at_100C_close_to_atm():
    assert 90_000 < saturation_pressure_pa(100) < 115_000


def test_latent_heat_decreases_with_temperature():
    assert latent_heat_vaporization(20) > latent_heat_vaporization(80)


def test_energy_limited_mode_matches_manual():
    Q = 1_000_000  # J
    h_fg = 2.4e6
    r = evaporate_energy_limited(Q, h_fg, area_effective=1.0, sunlight_hours=1.0)
    assert math.isclose(r.m_evap_kg, Q / h_fg)


def test_empirical_mode_scales_linearly_with_area():
    r1 = evaporate_empirical(1.05, 1.0, 8.0)
    r2 = evaporate_empirical(1.05, 3.6, 8.0)
    assert math.isclose(r2.m_evap_kg / r1.m_evap_kg, 3.6, rel_tol=1e-9)


def test_empirical_mode_reproduces_8p4_per_day():
    """8.4 kg/m^2/day = 1.05 kg/m^2/h * 8 h -> 8.4 kg on 1 m^2 with f_util=1."""
    r = evaporate_empirical(8.4 / 8.0, 1.0, 8.0)
    assert math.isclose(r.m_evap_kg, 8.4, rel_tol=1e-9)


def test_mass_transfer_zero_when_no_driving_force():
    r = evaporate_mass_transfer(T_surface_C=30, T_ambient_C=30,
                                relative_humidity=1.0, k_m=3e-8,
                                area_effective=1.0, sunlight_hours=8.0)
    assert r.m_evap_kg == 0.0


def test_mass_transfer_increases_with_temperature():
    r_low = evaporate_mass_transfer(35, 30, 0.5, 3e-8, 1.0, 8.0)
    r_hi = evaporate_mass_transfer(60, 30, 0.5, 3e-8, 1.0, 8.0)
    assert r_hi.m_evap_kg > r_low.m_evap_kg


def test_hybrid_picks_the_smaller():
    energy = evaporate_energy_limited(1e6, 2.4e6, 1.0, 8.0).m_evap_kg
    mt = evaporate_mass_transfer(50, 30, 0.4, 3e-8, 1.0, 8.0).m_evap_kg
    h = evaporate_hybrid(1e6, 2.4e6, 50, 30, 0.4, 3e-8, 1.0, 8.0).m_evap_kg
    assert math.isclose(h, min(energy, mt))
