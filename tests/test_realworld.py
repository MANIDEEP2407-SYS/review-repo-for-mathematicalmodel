import pytest
from src.physics.realworld import realworld_yield, wick_capacity, resolve
from src.physics.system import SystemConfig


@pytest.mark.parametrize("coupling", ["coupled", "legacy"])
def test_each_step_never_increases_yield(coupling):
    v = [s[1] for s in realworld_yield(SystemConfig(coupling=coupling)).steps]
    assert all(b <= a + 1e-9 for a, b in zip(v, v[1:]))


@pytest.mark.parametrize("mode", ["energy_limited", "empirical", "mass_transfer", "hybrid"])
def test_derating_steps_after_the_diurnal_profile_never_increase_yield_in_any_mode(mode):
    """The diurnal step re-times the same insolation over a longer day: with a per-hour flux cap (modes B/D)
    that legitimately ADDS capacity, so only the soiling / fouling / capillary steps are derating steps."""
    v = [s[1] for s in realworld_yield(SystemConfig(evaporation_mode=mode)).steps]
    assert all(b <= a + 1e-9 for a, b in zip(v[1:], v[2:]))


def test_coupled_fouling_and_capillary_act_inside_the_balance_not_behind_a_glass_ceiling():
    """Legacy: fouling/ageing/capillary were hidden behind the glass ceiling.  Coupled: they lower yield and
    RAISE the evaporator temperature, because the heat that cannot leave as vapour stays in the evaporator."""
    cfg = SystemConfig()
    clean = realworld_yield(cfg, {"fouling_timescale_days": 1000.0, "flower_degradation": 0.0})
    fouled = realworld_yield(cfg, {"fouling_timescale_days": 5.0, "cleaning_interval_days": 90.0})
    assert fouled.final_L_m2_day < clean.final_L_m2_day
    Te = lambda rw: max(h["T_evap_C"] for h in rw.hourly)
    assert Te(fouled) > Te(clean)
    thin = realworld_yield(cfg, {"wick_length": 0.5, "wick_area_fraction": 2e-4})
    assert Te(thin) > Te(realworld_yield(cfg))
    assert thin.final_L_m2_day < realworld_yield(cfg).final_L_m2_day


def test_jurin_height_matches_textbook():
    p = resolve({"pore_radius": 1e-5, "contact_angle": 0.0})
    ws = wick_capacity(20.0, p)
    assert ws.h_jurin_m == pytest.approx(2 * 0.0728 / (985 * 9.80665 * 1e-5), rel=0.02)


def test_wick_longer_than_jurin_height_gives_zero_supply():
    r = realworld_yield(SystemConfig(), {"pore_radius": 1e-4, "wick_length": 0.3})
    assert r.final_L_m2_day == 0.0 and not r.wick.lift_ok and r.warnings


def test_thin_wick_limits_yield_below_demand():
    ok = realworld_yield(SystemConfig())
    thin = realworld_yield(SystemConfig(), {"wick_length": 0.5, "wick_area_fraction": 2e-4})
    assert thin.final_L_m2_day < ok.final_L_m2_day and thin.wick_limited_hours > 0


def test_finer_pores_lift_higher_but_flow_slower():
    coarse = wick_capacity(30.0, resolve({"pore_radius": 1e-4}))
    fine = wick_capacity(30.0, resolve({"pore_radius": 1e-5}))
    assert fine.h_jurin_m > coarse.h_jurin_m and fine.K_m2 < coarse.K_m2


def test_more_cleaning_reduces_fouling_loss():
    a = realworld_yield(SystemConfig(), {"cleaning_interval_days": 90.0, "fouling_timescale_days": 30.0})
    b = realworld_yield(SystemConfig(), {"cleaning_interval_days": 7.0, "fouling_timescale_days": 30.0})
    assert a.factors["f_foul"] < b.factors["f_foul"]
