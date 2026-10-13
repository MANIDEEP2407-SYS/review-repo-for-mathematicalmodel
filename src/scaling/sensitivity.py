"""Sensitivity analysis: one-at-a-time sweeps and tornado data.

DEFAULT_TORNADO_RANGES is the LEGACY list (it contains eta_thermal and mass_transfer_coefficient,
which are not on the coupled path).  Use default_tornado_ranges(coupling) to get the right set; for the
coupled model it adds the vapour-gap, cover, insulation and (separately) ejector parameters.
Ejector parameters only matter when venturi_enabled=True, so they live in VENTURI_TORNADO_RANGES and
must be run on a venturi-enabled baseline.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..physics.system import SystemConfig, SystemResult, simulate


def _metric_daily_L(r: SystemResult) -> float:
    return r.daily_freshwater_L


@dataclass
class SweepResult:
    parameter: str
    values: List[float]
    outputs: List[float]
    baseline_output: float


@dataclass
class TornadoEntry:
    parameter: str
    low_value: float
    high_value: float
    low_output: float
    high_output: float
    range_output: float
    percent_change: float


def sweep_one(parameter: str,
              values: List[float],
              baseline: Optional[SystemConfig] = None,
              metric: Callable[[SystemResult], float] = _metric_daily_L) -> SweepResult:
    base = baseline or SystemConfig()
    if not hasattr(base, parameter):
        raise KeyError(f"SystemConfig has no attribute '{parameter}'")
    baseline_out = metric(simulate(base))
    outs = []
    for v in values:
        cfg = deepcopy(base)
        setattr(cfg, parameter, v)
        outs.append(metric(simulate(cfg)))
    return SweepResult(parameter, list(values), outs, baseline_out)


def tornado(parameters: Dict[str, Tuple[float, float]],
            baseline: Optional[SystemConfig] = None,
            metric: Callable[[SystemResult], float] = _metric_daily_L) -> List[TornadoEntry]:
    base = baseline or SystemConfig()
    baseline_out = metric(simulate(base))
    entries: List[TornadoEntry] = []
    for name, (lo, hi) in parameters.items():
        cfg_lo = deepcopy(base); setattr(cfg_lo, name, lo)
        cfg_hi = deepcopy(base); setattr(cfg_hi, name, hi)
        y_lo = metric(simulate(cfg_lo))
        y_hi = metric(simulate(cfg_hi))
        rng = abs(y_hi - y_lo)
        pct = rng / baseline_out * 100.0 if baseline_out else float("inf")
        entries.append(TornadoEntry(name, lo, hi, y_lo, y_hi, rng, pct))
    entries.sort(key=lambda e: e.range_output, reverse=True)
    return entries


DEFAULT_TORNADO_RANGES: Dict[str, Tuple[float, float]] = {
    "solar_irradiance": (500.0, 1000.0),
    "sunlight_hours": (5.0, 10.0),
    "optical_gain": (1.0, 2.2),
    "eta_thermal": (0.45, 0.80),
    "eta_optical": (0.75, 0.95),
    "evaporator_area_geometric": (0.5, 2.0),
    "area_utilization_factor": (0.6, 1.0),
    "ambient_temperature": (20.0, 40.0),
    "relative_humidity": (0.3, 0.75),
    "wind_speed": (0.5, 5.0),
    "glass_area": (0.8, 2.5),
    "condensation_coefficient": (2000.0, 15000.0),
    "collection_efficiency": (0.7, 0.98),
    "mass_transfer_coefficient": (1.0e-8, 8.0e-8),
}


# --- coupled model ---------------------------------------------------------
# eta_thermal, mass_transfer_coefficient, relative_humidity, empirical_flux (default mode) and
# venturi_inlet_pressure are not on the coupled path; the ranges below are assumed plausible bands.
COUPLED_TORNADO_RANGES: Dict[str, Tuple[float, float]] = {
    "solar_irradiance": (500.0, 1000.0),
    "sunlight_hours": (5.0, 10.0),
    "optical_gain": (1.0, 2.2),
    "eta_optical": (0.75, 0.95),
    "glass_transmittance": (0.80, 0.93),
    "evaporator_area_geometric": (0.5, 2.0),
    "area_utilization_factor": (0.6, 1.0),
    "ambient_temperature": (20.0, 40.0),
    "wind_speed": (0.5, 5.0),
    "glass_area": (0.8, 2.5),
    "condensation_coefficient": (2000.0, 15000.0),
    "collection_efficiency": (0.7, 0.98),
    "insulation_k": (0.03, 0.08),
    "insulation_thickness": (0.03, 0.10),
    "gap_convection_coefficient": (0.5, 5.0),
    "diffusion_path_length": (0.005, 0.06),
    "glass_absorptance": (0.02, 0.10),
}

VENTURI_TORNADO_RANGES: Dict[str, Tuple[float, float]] = {
    "motive_flow_rate": (3.0e-4, 1.0e-3),
    "venturi_throat_diameter": (0.007, 0.012),
    "venturi_loss_fraction": (0.1, 0.5),
    "chamber_leak_pressure_floor": (5000.0, 60000.0),
    "motive_water_temperature": (15.0, 45.0),
    "atmospheric_pressure": (95000.0, 103000.0),
}


def default_tornado_ranges(coupling: str = "coupled", venturi: bool = False) -> Dict[str, Tuple[float, float]]:
    if coupling == "legacy":
        return dict(DEFAULT_TORNADO_RANGES)
    return dict(VENTURI_TORNADO_RANGES if venturi else COUPLED_TORNADO_RANGES)
