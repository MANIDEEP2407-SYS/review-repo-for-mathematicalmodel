import pytest
from src.openfoam.runner import (DEFAULT_CASE, map_simulator_to_case, prepare_case,
                                 to_wsl_path)
from src.physics.system import SystemConfig, simulate

needs_case = pytest.mark.skipif(not (DEFAULT_CASE / "system" / "controlDict").exists(),
                                reason="OpenFOAM case not present")


def test_wsl_path():
    from pathlib import Path
    assert to_wsl_path(Path("C:/Users/x/a b")) == "/mnt/c/Users/x/a b"


def test_mapping_is_positive_and_kelvin():
    cfg = SystemConfig()
    p = map_simulator_to_case(cfg, simulate(cfg))
    assert p["q_net_W_m2"] > 0
    assert p["T_ambient_K"] == pytest.approx(cfg.ambient_temperature + 273.15)


@needs_case
def test_prepare_case_edits_only_requested_values(tmp_path):
    params = {"q_net_W_m2": 500.0, "T_ambient_K": 300.0, "h_glass_W_m2K": 12.0,
              "end_time_s": 120.0, "write_interval_s": 20.0}
    rd = prepare_case(params, runs_dir=tmp_path)
    T = (rd / "0" / "T").read_text(encoding="utf-8", errors="replace")
    assert "q           uniform 500;" in T
    assert "uniform 300;" in T and "uniform 12;" in T
    ctrl = (rd / "system" / "controlDict").read_text(errors="replace")
    assert "endTime         120;" in ctrl
    assert (rd / "constant" / "polyMesh" / "points").exists()
    # source case untouched
    assert "uniform 980;" in (DEFAULT_CASE / "0" / "T").read_text(encoding="utf-8", errors="replace")
