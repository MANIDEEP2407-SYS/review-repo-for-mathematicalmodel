"""Reference validation cases.

NOTHING in this module is a validation of the physical model.  The cases are arithmetic checks
(linear scaling of an empirical input), internal-consistency checks, and two plausibility /
consistency comparisons against published numbers that are labelled "consistency, not validation".

Empirical productivity value (8.4 kg/m^2/day) from the conceptual
diagram is treated as an INPUT the simulator must reproduce in MODE B.
Reproducing it demonstrates linear scaling of an empirical claim,
NOT experimental confirmation of the physical model.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List

from ..physics.system import SystemConfig, simulate


EMPIRICAL_YIELD_KG_PER_M2_DAY = 8.4    # concept diagram, cite 396


@dataclass
class ValidationCase:
    name: str
    predicted_L_per_day: float
    empirical_L_per_day: float
    absolute_error_L: float
    relative_error_percent: float
    passed: bool
    notes: str


def _empirical_reference_config(area_m2: float, coupling: str = "coupled") -> SystemConfig:
    """MODE B reference: strip out physics limits so we test the empirical scaling."""
    cfg = SystemConfig(coupling=coupling)
    cfg.evaporator_area_geometric = area_m2
    cfg.area_utilization_factor = 1.0
    cfg.evaporation_mode = "empirical"
    # 8.4 kg/m2/day over 8 h -> 1.05 kg/m2/h
    cfg.empirical_flux = 8.4 / 8.0
    # Prevent condensation from limiting the empirical linear test.
    cfg.glass_area = area_m2 * 3.0
    cfg.condensation_coefficient = 20000.0
    cfg.collection_efficiency = 1.0
    return cfg


def run_empirical_linear_scaling(areas: List[float] = None,
                                 tolerance_percent: float = 1.0,
                                 coupling: str = "coupled") -> List[ValidationCase]:
    """Test the linear scaling of MODE B (evaporation law) against 8.4 kg/m^2/day.

    We compare the *evaporated* mass predicted by MODE B, not the collected
    freshwater. The condensation model can cap collection independently of
    whether MODE B's scaling law is correct; validating them together would
    conflate two independent claims. Collection-vs-condensation adequacy is
    tested separately in run_condensation_headroom below.
    """
    areas = areas or [1.0, 3.6, 10.0]
    results: List[ValidationCase] = []
    for A in areas:
        cfg = _empirical_reference_config(A, coupling)
        r = simulate(cfg)
        predicted_kg = r.m_evap_selected_kg   # MODE B's own prediction
        empirical_kg = EMPIRICAL_YIELD_KG_PER_M2_DAY * A
        abs_err = abs(predicted_kg - empirical_kg)
        rel_err = abs_err / empirical_kg * 100.0 if empirical_kg else 0.0
        results.append(ValidationCase(
            name=f"Empirical linear scaling @ A={A} m^2",
            predicted_L_per_day=predicted_kg,       # kg == L for water at rho=1
            empirical_L_per_day=empirical_kg,
            absolute_error_L=abs_err,
            relative_error_percent=rel_err,
            passed=rel_err <= tolerance_percent,
            notes="Empirical linear evaporation-law scaling -- NOT CFD validation.",
        ))
    return results


def run_condensation_headroom(area_m2: float = 1.0, coupling: str = "coupled") -> ValidationCase:
    """Independent check: at the empirical evaporation rate, is condensation
    capacity enough to collect what MODE B evaporates? A binding condensation
    limit is a design flag, not a physics-law error."""
    cfg = _empirical_reference_config(area_m2, coupling)
    r = simulate(cfg)
    ratio = r.m_condensation_capacity_kg / max(r.m_evap_selected_kg, 1e-12)
    return ValidationCase(
        name=f"Condensation headroom @ A={area_m2} m^2",
        predicted_L_per_day=r.m_condensation_capacity_kg,
        empirical_L_per_day=r.m_evap_selected_kg,
        absolute_error_L=r.m_evap_selected_kg - r.m_condensation_capacity_kg,
        relative_error_percent=(1.0 - min(ratio, 1.0)) * 100.0,
        passed=ratio >= 1.0,
        notes=("Design check: if <1, condensation caps collection below the "
               "empirical yield claim -- consider larger glass area, lower "
               "T_glass or higher inner condensation coefficient."),
    )


def run_energy_vs_empirical_consistency(area_m2: float = 1.0,
                                        tolerance_ratio: float = 3.0,
                                        coupling: str = "coupled") -> ValidationCase:
    """Sanity check: energy-limited and empirical modes should agree within an
    order of magnitude at reasonable defaults; otherwise model discrepancy is flagged."""
    cfg_e = SystemConfig(coupling=coupling)
    cfg_e.evaporator_area_geometric = area_m2
    cfg_e.evaporation_mode = "energy_limited"
    cfg_b = deepcopy(cfg_e)
    cfg_b.evaporation_mode = "empirical"
    r_e = simulate(cfg_e)
    r_b = simulate(cfg_b)

    e_kg = r_e.m_evap_selected_kg
    b_kg = r_b.m_evap_selected_kg
    ratio = max(e_kg, b_kg) / max(min(e_kg, b_kg), 1e-9)
    passed = ratio <= tolerance_ratio
    return ValidationCase(
        name=f"Energy vs empirical evaporation consistency @ A={area_m2} m^2",
        predicted_L_per_day=e_kg,
        empirical_L_per_day=b_kg,
        absolute_error_L=abs(e_kg - b_kg),
        relative_error_percent=(ratio - 1.0) * 100.0,
        passed=passed,
        notes=("Energy-limited vs empirical predictions of evaporated mass "
               "(NOT of collected freshwater). Large disagreement flags "
               "either an unrealistic eta_thermal or an unrealistic J_emp."),
    )


# ---------------------------------------------------------------------------
# Plausibility / consistency checks against published numbers.
# These are NOT validation.  They say whether the model is in the right ballpark and
# why it might not be; they cannot say the model is right.
# ---------------------------------------------------------------------------
PUBLISHED_STILL_L_PER_M2_DAY = (2.0, 5.0)     # typical single-basin passive solar still productivity
LAL_ONE_SUN_FLUX_KG_M2_H = 2.77               # Lal et al., Chem. Eng. J. 522, 167097 (2025), as given in the
                                              # project brief; the number was NOT re-checked against the paper.


def run_still_plausibility_check(cfg: SystemConfig = None) -> ValidationCase:
    """CONSISTENCY, NOT VALIDATION: is the model's productivity inside the 2-5 L/m2/day band published for
    passive solar stills?  A value above the band is a flag (missing loss paths or optimistic inputs), a
    value inside it proves nothing."""
    cfg = cfg or SystemConfig()
    r = simulate(cfg)
    lo, hi = PUBLISHED_STILL_L_PER_M2_DAY
    prod = r.productivity_L_per_m2_day
    inside = lo <= prod <= hi
    return ValidationCase(
        name=f"Still plausibility ({cfg.coupling}, mode={cfg.evaporation_mode}) -- consistency, not validation",
        predicted_L_per_day=prod,
        empirical_L_per_day=0.5 * (lo + hi),
        absolute_error_L=max(prod - hi, lo - prod, 0.0),
        relative_error_percent=100.0 * max(prod - hi, lo - prod, 0.0) / hi,
        passed=inside,
        notes=(f"Predicted {prod:.2f} L/m2/day vs a typical published band of {lo:g}-{hi:g}. "
               "Includes the user-set reflector gain "
               f"(x{cfg.optical_gain:g}), a perfectly sealed chamber and no side/edge or basin losses beyond the "
               "insulation term, so an over-the-band value is expected from the model's missing loss paths, not "
               "evidence of a better device. Inside the band would prove nothing either."),
    )


def run_isolated_evaporator_consistency(target_kg_m2_h: float = LAL_ONE_SUN_FLUX_KG_M2_H,
                                        G: float = 1000.0) -> ValidationCase:
    """CONSISTENCY, NOT VALIDATION: isolated open-surface evaporator under one sun versus the 2.77 kg/m2/h
    lab flux reported by Lal et al. (Chem. Eng. J. 522, 167097, 2025).

    Two facts are reported: (1) the one-sun energy bound (alpha G / h_fg) is far below 2.77, so the lab flux
    needs a large environmental energy gain (surface below ambient); (2) the exchange-area enhancement the
    repo's flat-plate heat/mass-transfer closure needs to reproduce it."""
    from ..physics.isolated import enhancement_to_match, isolated_evaporator
    from ..physics.coupled import hfg
    flat = isolated_evaporator(G=G, enhancement=1.0)
    m_req = enhancement_to_match(target_kg_m2_h, G=G)
    bound = 0.95 * G / hfg(25.0) * 3600.0
    need = target_kg_m2_h / 3600.0 * hfg(25.0) - 0.95 * G
    plausible = m_req <= 10.0
    return ValidationCase(
        name="Isolated evaporator at one sun vs Lal et al. 2.77 kg/m2/h -- consistency, not validation",
        predicted_L_per_day=flat.flux_kg_m2_h,
        empirical_L_per_day=target_kg_m2_h,
        absolute_error_L=abs(target_kg_m2_h - flat.flux_kg_m2_h),
        relative_error_percent=100.0 * (target_kg_m2_h - flat.flux_kg_m2_h) / target_kg_m2_h,
        passed=plausible,
        notes=(f"Units here are kg/m2/h, not L/day. Flat open surface: {flat.flux_kg_m2_h:.2f} kg/m2/h. "
               f"One-sun energy bound alpha*G/h_fg = {bound:.2f} kg/m2/h, so 2.77 needs ~{need:.0f} W/m2 taken from the "
               f"air/sky. The repo's closure reaches 2.77 only if the exchange area is ~{m_req:.0f}x the projected "
               "area (3-D flower enhancement is unmeasured); a plausible 3-D enhancement is assumed <= 10. "
               "Inputs (25 C, 50 % RH, still air, Lewis-analogy mass transfer) are assumptions."),
    )
