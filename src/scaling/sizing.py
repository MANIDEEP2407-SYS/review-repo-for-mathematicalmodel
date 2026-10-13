"""Sizing calculator: household and village scaling.

A_required = target_output / productivity

Productivity is measured by running simulate() on a unit-area reference
system, NOT hard-coded.  The reference's solver diagnostics (non-convergence,
boiling limit, vacuum load on the cover) are carried into the sizing warnings, so
a size is never quoted from a physically invalid reference silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from ..physics.system import SystemConfig, simulate


@dataclass
class HouseholdDemand:
    people: int
    drinking_per_person_L: float
    cooking_per_person_L: float
    safety_factor: float

    @property
    def per_person_L(self) -> float:
        return self.drinking_per_person_L + self.cooking_per_person_L

    @property
    def total_L_per_day(self) -> float:
        return self.people * self.per_person_L * self.safety_factor


@dataclass
class SizingResult:
    target_L_per_day: float
    productivity_L_per_m2_day: float
    A_required_m2: float
    modules_required: int
    A_per_module: float
    total_footprint_m2: float
    n_flowers_estimated: int
    warnings: List[str] = field(default_factory=list)


def measure_productivity(reference: Optional[SystemConfig] = None) -> float:
    ref = reference or SystemConfig()
    return simulate(ref).productivity_L_per_m2_day


def reference_diagnostics(reference: Optional[SystemConfig] = None) -> List[str]:
    """Warnings about the reference system that a sizing result should not hide."""
    ref = reference or SystemConfig()
    r = simulate(ref)
    notes: List[str] = []
    if r.coupling == "coupled":
        if not r.solver_converged:
            notes.append("Reference system: coupled solver did not close (cover too small or evaporator capped); "
                         "productivity is not meaningful.")
        if r.boiling_limited:
            notes.append("Reference system is boiling-limited: the evaporator/cover pair sits at the vapour-pressure "
                         "limit, so extra vacuum or heat will not raise yield.")
        if r.cover_vacuum_implausible:
            notes.append(f"Reference system needs {r.vacuum_load_kN:.0f} kN (~{r.vacuum_load_tonnes:.1f} t) of vacuum load on the "
                         "cover: credible only for a small sealed chamber; scaling this area up with a flat glass "
                         "cover is not.")
        if r.ejector_power_W > 0:
            notes.append(f"Reference ejector draws {r.ejector_power_W:.0f} W per module (parasitic, not in the yield).")
    return notes


def size_for_demand(
    target_L_per_day: float,
    reference: Optional[SystemConfig] = None,
    A_per_module: float = 1.0,
    flowers_per_m2: float = 25.0,
) -> SizingResult:
    if target_L_per_day <= 0:
        raise ValueError("target_L_per_day must be positive.")
    if A_per_module <= 0:
        raise ValueError("A_per_module must be positive.")

    productivity = measure_productivity(reference)
    warnings: List[str] = reference_diagnostics(reference)
    if productivity <= 0:
        warnings.append(
            "Reference system has zero productivity; cannot size. "
            "Check solar / thermal / condensation parameters."
        )
        return SizingResult(
            target_L_per_day=target_L_per_day,
            productivity_L_per_m2_day=productivity,
            A_required_m2=float("inf"),
            modules_required=0,
            A_per_module=A_per_module,
            total_footprint_m2=float("inf"),
            n_flowers_estimated=0,
            warnings=warnings,
        )

    A_req = target_L_per_day / productivity
    modules = math.ceil(A_req / A_per_module)
    footprint = modules * A_per_module
    n_flowers = int(round(A_req * flowers_per_m2))

    return SizingResult(
        target_L_per_day=target_L_per_day,
        productivity_L_per_m2_day=productivity,
        A_required_m2=A_req,
        modules_required=modules,
        A_per_module=A_per_module,
        total_footprint_m2=footprint,
        n_flowers_estimated=n_flowers,
        warnings=warnings,
    )


@dataclass
class ModuleSizing:
    n_modules: int
    target_L_per_hour: float
    sunlight_hours: float
    target_L_per_day: float
    productivity_L_per_m2_day: float
    L_per_hour_per_module: float
    A_module_m2: float
    A_total_m2: float
    module_side_m: float          # side of a square module
    module_diameter_m: float      # diameter of a circular module
    flowers_per_module: int
    flowers_total: int
    flower_coverage_fraction: float  # flower footprint / module area
    warnings: List[str] = field(default_factory=list)


def size_modules(
    n_modules: int,
    target_L_per_hour: float,
    reference: Optional[SystemConfig] = None,
    flowers_per_m2: float = 25.0,
    flower_diameter_m: float = 0.07,
    productivity_override: Optional[float] = None,
) -> ModuleSizing:
    """Size identical modules for a total output rate.

    target_L_per_hour is the TOTAL rate over all modules, averaged over the
    reference system's sunlight hours (daily yield = rate x sunlight_hours).
    productivity_override [L/m^2/day] replaces the ideal simulate() value, e.g. with the
    real-world derated productivity from physics.realworld.
    """
    if n_modules < 1:
        raise ValueError("n_modules must be >= 1.")
    if target_L_per_hour <= 0:
        raise ValueError("target_L_per_hour must be positive.")
    ref = reference or SystemConfig()
    hours = ref.sunlight_hours
    productivity = productivity_override if productivity_override is not None else measure_productivity(ref)
    target_day = target_L_per_hour * hours
    warnings: List[str] = reference_diagnostics(ref) if productivity_override is None else []
    if productivity <= 0:
        warnings.append("Reference system has zero productivity; cannot size.")
        inf = float("inf")
        return ModuleSizing(n_modules, target_L_per_hour, hours, target_day, productivity,
                            target_L_per_hour / n_modules, inf, inf, inf, inf, 0, 0, 0.0, warnings)

    A_total = target_day / productivity
    A_mod = A_total / n_modules
    flowers_mod = max(1, int(math.ceil(A_mod * flowers_per_m2)))
    coverage = flowers_mod * math.pi * (flower_diameter_m / 2) ** 2 / A_mod
    if coverage > 0.9069:  # hexagonal close packing of circles
        warnings.append(
            f"Flower footprint is {coverage*100:.0f} % of module area, above the "
            "~91 % maximum for circles; reduce flowers/m² or flower diameter."
        )
    return ModuleSizing(
        n_modules=n_modules,
        target_L_per_hour=target_L_per_hour,
        sunlight_hours=hours,
        target_L_per_day=target_day,
        productivity_L_per_m2_day=productivity,
        L_per_hour_per_module=target_L_per_hour / n_modules,
        A_module_m2=A_mod,
        A_total_m2=A_total,
        module_side_m=math.sqrt(A_mod),
        module_diameter_m=2 * math.sqrt(A_mod / math.pi),
        flowers_per_module=flowers_mod,
        flowers_total=flowers_mod * n_modules,
        flower_coverage_fraction=coverage,
        warnings=warnings,
    )


def size_for_household(demand: HouseholdDemand,
                       reference: Optional[SystemConfig] = None,
                       A_per_module: float = 1.0) -> SizingResult:
    return size_for_demand(demand.total_L_per_day, reference, A_per_module)


def size_for_village(households: int,
                     demand_per_household: HouseholdDemand,
                     reference: Optional[SystemConfig] = None,
                     A_per_module: float = 1.0) -> SizingResult:
    total = households * demand_per_household.total_L_per_day
    return size_for_demand(total, reference, A_per_module)


def sweep_scale(populations: List[int],
                demand_per_person_L: float,
                safety_factor: float = 1.2,
                reference: Optional[SystemConfig] = None,
                A_per_module: float = 1.0):
    productivity = measure_productivity(reference)
    rows = []
    for pop in populations:
        target = pop * demand_per_person_L * safety_factor
        A_req = target / productivity if productivity > 0 else float("inf")
        modules = math.ceil(A_req / A_per_module) if productivity > 0 else 0
        rows.append({
            "population": pop,
            "target_L_per_day": target,
            "productivity_L_per_m2_day": productivity,
            "A_required_m2": A_req,
            "modules_required": modules,
            "total_footprint_m2": modules * A_per_module,
        })
    return rows
