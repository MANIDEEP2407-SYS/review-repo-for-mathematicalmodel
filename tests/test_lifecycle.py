import math
import pytest
from src.physics.system import SystemConfig
from src.scaling.sizing import size_modules
from src.sustainability.lifecycle import analyse, resolve_params


def _run(n=1, rate=10.0, **ov):
    cfg = SystemConfig()
    return analyse(size_modules(n, rate, cfg), cfg, ov)


def test_passive_system_has_no_operational_co2():
    assert _run().annual_operational_co2_kg == 0.0


def test_pump_adds_operational_co2():
    assert _run(pump_power_W=50.0).annual_operational_co2_kg > 0


def test_learning_rate_one_gives_linear_variable_cost():
    a = _run(1, 10, learning_rate=1.0, fixed_setup_cost=0.0)
    b = _run(10, 100, learning_rate=1.0, fixed_setup_cost=0.0)
    assert b.capex_total_INR == pytest.approx(10 * a.capex_total_INR, rel=1e-9)


def test_learning_curve_reduces_per_module_cost():
    a = _run(1, 10, fixed_setup_cost=0.0)
    b = _run(100, 1000, fixed_setup_cost=0.0)
    assert b.capex_per_module_INR < a.capex_per_module_INR


def test_net_zero_reached_with_lifetime_trees():
    r = _run()
    assert r.trees_lifetime_balance > 0 and r.net_zero_year is not None
    assert r.yearly[-1]["net_kg"] <= 1e-6


def test_out_of_range_override_rejected():
    with pytest.raises(ValueError):
        resolve_params({"discount_rate": 5.0})
    with pytest.raises(KeyError):
        resolve_params({"nope": 1})
