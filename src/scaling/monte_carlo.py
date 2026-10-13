"""Monte Carlo uncertainty analysis.

Each parameter is perturbed by an independent Gaussian (or uniform)
draw. simulate() is evaluated N times and mean / percentiles are
reported. Random seed is exposed so results are reproducible.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..config.defaults import PARAMETER_REGISTRY
from ..physics.system import SystemConfig, simulate


@dataclass
class MonteCarloResult:
    n_samples: int
    daily_L: np.ndarray
    productivity_L_per_m2_day: np.ndarray
    mean: float
    median: float
    std: float
    p05: float
    p95: float
    p50: float
    parameter_relative_uncertainty: Dict[str, float]
    parameter_samples: Dict[str, np.ndarray] = field(default_factory=dict)

    def spearman(self) -> Dict[str, float]:
        """Spearman rank correlation of each sampled parameter with daily yield."""
        from scipy.stats import spearmanr
        out = {}
        for k, v in self.parameter_samples.items():
            if np.ptp(v) == 0 or np.ptp(self.daily_L) == 0:
                out[k] = 0.0
            else:
                out[k] = float(spearmanr(v, self.daily_L).statistic)
        return out


DEFAULT_UNCERTAINTIES: Dict[str, float] = {
    "solar_irradiance": 0.10,
    "sunlight_hours": 0.10,
    "eta_thermal": 0.10,
    "eta_optical": 0.10,
    "optical_gain": 0.10,
    "empirical_flux": 0.20,
    "mass_transfer_coefficient": 0.30,
    "condensation_coefficient": 0.20,
    "collection_efficiency": 0.05,
    "area_utilization_factor": 0.10,
    "ambient_temperature": 0.05,
    "relative_humidity": 0.15,
    "wind_speed": 0.25,
}


# Coupled model.  eta_thermal and empirical_flux are kept (with their legacy widths) although they are
# inert on the default coupled path, so a legacy-vs-coupled comparison uses the same parameter list;
# the new widths are ASSUMED (delta and the gap coefficient are the least known numbers in the model).
COUPLED_UNCERTAINTIES: Dict[str, float] = {
    **DEFAULT_UNCERTAINTIES,
    "glass_transmittance": 0.03,
    "glass_area": 0.05,
    "insulation_k": 0.20,
    "gap_convection_coefficient": 0.30,
    "diffusion_path_length": 0.40,
    "glass_absorptance": 0.30,
}

# Parameters whose legacy role was an artifact of the decoupled model (Spearman rho with yield
# -0.64 and -0.33 at the legacy baseline).  Excluding them gives the "artifact-free" spread.
ARTIFACT_PARAMETERS = ("eta_thermal", "area_utilization_factor")


def default_uncertainties(coupling: str = "coupled") -> Dict[str, float]:
    return dict(DEFAULT_UNCERTAINTIES if coupling == "legacy" else COUPLED_UNCERTAINTIES)


def monte_carlo(
    n_samples: int = 1000,
    baseline: Optional[SystemConfig] = None,
    uncertainties: Optional[Dict[str, float]] = None,
    seed: Optional[int] = 42,
    exclude: Optional[Iterable[str]] = None,
) -> MonteCarloResult:
    base = baseline or SystemConfig()
    unc = dict(uncertainties or default_uncertainties(base.coupling))
    for name in (exclude or ()):
        unc.pop(name, None)
    coupled = base.coupling != "legacy"
    rng = np.random.default_rng(seed)
    samples = {k: np.empty(n_samples) for k in unc if hasattr(base, k)}

    daily_L = np.empty(n_samples)
    prod = np.empty(n_samples)

    for i in range(n_samples):
        cfg = deepcopy(base)
        for name, sigma_rel in unc.items():
            if not hasattr(cfg, name):
                continue
            nominal = getattr(cfg, name)
            perturbed = nominal * float(rng.normal(1.0, sigma_rel))
            # Clamp to strictly positive for physical inputs.
            if isinstance(nominal, (int, float)) and nominal > 0:
                perturbed = max(perturbed, nominal * 1e-3)
            if coupled and name in PARAMETER_REGISTRY:       # stay inside the registered physical range
                lo, hi = PARAMETER_REGISTRY[name].allowed_range
                perturbed = min(max(perturbed, lo), hi)
            setattr(cfg, name, perturbed)
            samples[name][i] = perturbed
        r = simulate(cfg)
        daily_L[i] = r.daily_freshwater_L
        prod[i] = r.productivity_L_per_m2_day

    return MonteCarloResult(
        n_samples=n_samples,
        daily_L=daily_L,
        productivity_L_per_m2_day=prod,
        mean=float(daily_L.mean()),
        median=float(np.median(daily_L)),
        std=float(daily_L.std()),
        p05=float(np.percentile(daily_L, 5)),
        p95=float(np.percentile(daily_L, 95)),
        p50=float(np.percentile(daily_L, 50)),
        parameter_relative_uncertainty=dict(unc),
        parameter_samples=samples,
    )
