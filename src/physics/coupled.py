"""Coupled evaporator / vapour-gap / glass-condenser steady-state solver.

Replaces the decoupled model (evaporation and condensation capped separately, glass
temperature a heuristic) with ONE closed balance.  SI throughout; temperatures in degC
only where noted.

Unknowns: T_e (evaporator), T_gi (glass inner surface), T_c (condensate film surface), mdot.

 (a) evaporator   P_abs = mdot h_fg(T_e) + Q_gap + Q_ins
        Q_gap = h_gap(T_e,T_gi) A_loss (T_e - T_gi)       h_gap = h_conv,gap + h_rad,gap
        Q_ins = (k_ins/d_ins) A_loss (T_e - T_amb)
        A_loss = A_geo (whole absorber), A_eff = f_util A_geo (wet part) is used for evaporation only.
        The brief writes A_eff in the loss terms; with that, RAISING f_util lowers yield (more wet area
        = more leak), the same kind of artifact as the legacy one.  Identical at f_util = 1.
 (b) vapour diffusion through air (Stefan flow, stagnant air film of length delta)
        mdot'' = D_wa M_w P / (R T delta) * ln((P - p_c)/(P - p_e)),   p_e = Psat(T_e), p_c = Psat(T_c)
        D_wa   = 1.87e-10 T^2.072 / P[atm]                (Marrero & Mason 1972, 282-450 K)
 (c) glass        mdot h_fg(T_c) + Q_gap + P_glass_abs = Q_out(T_gi)
        Q_out  = A_g U_out [(T_gi - T_amb) + q_sky / h_o],  U_out = [d_g/k_g + 1/h_o]^-1,
        h_o = h_conv(wind) + h_rad,out(T_gi vs ambient);  q_sky = eps_g sigma (T_amb^4 - T_sky^4)
        film  T_c = T_gi + mdot h_fg(T_c) / (h_cond A_g)
 (d) h_fg is temperature dependent everywhere (Rogers & Yau linear fit).
 (e) SENSIBLE-HEAT BOOKKEEPING (deliberate, switchable deviation from the literal (a)/(c) above):
     with `sensible=True` (default) the evaporator also heats the make-up water from T_amb to T_e and the
     glass receives the vapour's enthalpy above the condensate:
        e_evap  = h_fg(T_e) + cp (T_e - T_amb)          instead of h_fg(T_e)
        e_glass = h_fg(T_e) + cp (T_e - T_c)            instead of h_fg(T_c)
     so that total energy is conserved (P_abs + P_glass_abs = Q_out + Q_ins + mdot cp (T_c - T_amb); the hot
     condensate leaves, no heat recovery).  Without it h_fg(T) falling with temperature makes a HOTTER
     still look cheaper per kg, and yield would spuriously DEcrease as the cover grows.
     `sensible=False` gives the literal brief equations.

Evaporation modes (names unchanged):
  A energy_limited  ideal evaporator: no vapour-side resistance (T_e = T_c).  The upper bound
                    for the given cover; independent of P and delta.
  B empirical       prescribed flux J_emp*A_eff*f_der, capped by A.
  C mass_transfer   diffusion law (b).
  D hybrid          min(B, C) (and therefore also <= A).
A supply (wick) limit, when set, caps every mode.

Boiling limit: as p_e -> P the log in (b) diverges, so the flux becomes energy-limited and the
root sits just below T_sat(P).  If the cover needs T_c >= T_sat(P_set) no steady state exists at
the set pressure: the chamber is then a pure-vapour space whose pressure floats up to Psat(T_c)
(the ejector cannot hold the vacuum against the condensing load; vapour pumped out by the ejector
is NOT modelled).  The state is then the ideal-evaporator state and `boiling_limited` is True.

Everything is solved with nested bracketed brentq (no unbounded Newton).  Closure residuals
of (a), (b), (c) and the film are reported; non-convergence raises a warning, never NaN.

PROVENANCE: delta, h_c,gap, alpha_g, h_cond and the leak floor are ASSUMED (see the registry).
Not modelled: basin/feed sensible heating, side-wall loss other than through k_ins/d_ins, vapour
extraction by the ejector, thermal inertia (quasi-steady), non-uniform temperatures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy.optimize import brentq

from .evaporation import latent_heat_vaporization, saturation_pressure_pa
from .heat_transfer import SIGMA

M_W = 0.01801528          # kg/mol
R_UNIV = 8.314462618      # J/(mol K)
P_STD = 101325.0          # Pa
T_MIN_C = -60.0           # validity clamp for the property fits
T_MAX_C = 250.0
BIG_FLOW = 1.0e3          # kg/s, stands for "no diffusion limit" (boiling)
CLOSURE_TOL = 1.0e-6
_HFG_A = 2500.9e3         # h_fg(T) = A - B*T[degC]  (same fit as evaporation.latent_heat_vaporization)
_HFG_B = 2365.0


# ----------------------------------------------------------------------------- properties
def _clamp_T(T_C: float) -> float:
    return min(max(T_C, T_MIN_C), T_MAX_C)


def psat(T_C: float) -> float:
    """Magnus saturation pressure [Pa], clamped to the fit's validity range (no overflow)."""
    return saturation_pressure_pa(_clamp_T(T_C))


def t_sat_C(P_Pa: float) -> float:
    """Inverse of the Magnus formula: boiling temperature at total pressure P [degC]."""
    x = math.log(max(P_Pa, 1.0) / 610.94)
    if x >= 17.625 - 1e-6:
        return T_MAX_C
    return _clamp_T(243.04 * x / (17.625 - x))


def hfg(T_C: float) -> float:
    return latent_heat_vaporization(_clamp_T(T_C))


def D_wa(T_K: float, P_Pa: float) -> float:
    """Water-vapour-in-air binary diffusivity [m^2/s], Marrero & Mason (1972): 1.87e-10 T^2.072 / P[atm]."""
    return 1.87e-10 * max(T_K, 200.0) ** 2.072 / (max(P_Pa, 1.0) / P_STD)


def diffusion_flux_kg_s(T_e_C: float, T_c_C: float, P_Pa: float, area: float, delta: float,
                        derating: float = 1.0) -> float:
    """Stefan-flow vapour transport evaporator -> condenser [kg/s].  0 if p_e <= p_c; BIG at boiling."""
    pe, pc = psat(T_e_C), psat(T_c_C)
    if pe <= pc or pc >= P_Pa:
        return 0.0
    if pe >= P_Pa * (1.0 - 1e-12):
        return BIG_FLOW
    Tm = 273.15 + 0.5 * (_clamp_T(T_e_C) + _clamp_T(T_c_C))
    flux = D_wa(Tm, P_Pa) * M_W * P_Pa / (R_UNIV * Tm * delta) * math.log((P_Pa - pc) / (P_Pa - pe))
    return derating * area * flux


def diffusion_flux_s(T_e_C: float, T_c_C: float, P_Pa: float, s_gap: float, area: float, delta: float,
                     derating: float = 1.0) -> float:
    """Same law as diffusion_flux_kg_s but with the boiling margin s = (P - p_e)/P given exactly
    (avoids the cancellation P - p_e when the evaporator is within ~1e-12 of boiling)."""
    pc = psat(T_c_C)
    if pc >= P_Pa * (1.0 - s_gap):
        return 0.0
    Tm = 273.15 + 0.5 * (_clamp_T(T_e_C) + _clamp_T(T_c_C))
    flux = D_wa(Tm, P_Pa) * M_W * P_Pa / (R_UNIV * Tm * delta) * (
        math.log(P_Pa - pc) - math.log(P_Pa) - math.log(max(s_gap, 1e-300)))
    return derating * area * flux


# ----------------------------------------------------------------------------- inputs / outputs
@dataclass(frozen=True)
class CoupledParams:
    P_abs_W: float            # light absorbed by the evaporator
    P_glass_abs_W: float      # light absorbed in the cover
    A_eff: float              # evaporating (wet) area [m2]: used for vapour transport and the flux caps
    A_loss: float             # absorber plan area [m2]: used for the gap and insulation heat leaks
    A_glass: float
    T_amb_C: float
    T_sky_C: float
    h_conv_out: float         # wind-driven outer convection [W/m2K]
    eps_e: float
    eps_g: float
    k_ins: float
    d_ins: float
    h_gap_conv: float
    delta: float
    d_g: float
    k_g: float
    h_cond: float
    P_chamber_Pa: float       # set pressure (atmospheric, or the ejector result)
    J_emp_kg_m2_s: float
    f_der: float = 1.0
    supply_kg_m2_s: float = 1.0
    cp_w: float = 4186.0
    sensible: bool = True


@dataclass
class CoupledState:
    T_e: float
    T_gi: float
    T_c: float
    mdot: float                   # kg/s
    Q_gap: float
    Q_gap_conv: float
    Q_gap_rad: float
    Q_ins: float
    Q_out: float
    U_out: float
    h_gap: float
    regime: str                   # ideal | diffusion | prescribed
    P_eff: float
    converged: bool = True
    overheated: bool = False
    s_gap: float = -1.0           # exact (P - p_e)/P for diffusion states; -1 otherwise
    residuals: Dict[str, float] = field(default_factory=dict)


@dataclass
class CoupledSolution:
    state: CoupledState               # selected mode
    mode: str
    mdot_modes: Dict[str, float]      # kg/s for the four modes
    state_ideal: CoupledState         # mode-A state (the cover's capacity)
    state_diffusion: CoupledState
    P_set_Pa: float
    P_eff_Pa: float
    boiling_limited: bool
    near_boiling: bool
    limited_by: str                   # which resistance / cap dominates
    resistance_share: Dict[str, float]
    warnings: List[str]


# ----------------------------------------------------------------------------- the solver
def _root(f, lo: float, hi: float) -> Tuple[float, bool]:
    flo, fhi = f(lo), f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        return (lo if abs(flo) < abs(fhi) else hi), False
    if flo == 0.0:
        return lo, True
    if fhi == 0.0:
        return hi, True
    if flo * fhi > 0.0:
        return (lo if abs(flo) < abs(fhi) else hi), False
    try:
        x, info = brentq(f, lo, hi, xtol=1e-12, rtol=1e-13, maxiter=300, full_output=True, disp=False)
        return x, bool(info.converged)
    except (RuntimeError, ValueError):
        return 0.5 * (lo + hi), False


class _Solver:
    def __init__(self, p: CoupledParams):
        self.p = p
        self.T_lo_gi = min(p.T_amb_C, p.T_sky_C) - 40.0
        self.T_lo_e = min(p.T_amb_C, p.T_sky_C) - 10.0
        self.eps_gap = 1.0 / (1.0 / p.eps_e + 1.0 / p.eps_g - 1.0)

    # --- heat-transfer pieces -------------------------------------------------
    def h_rad_gap(self, T_e: float, T_gi: float) -> float:
        a, b = T_e + 273.15, T_gi + 273.15
        return self.eps_gap * SIGMA * (a * a + b * b) * (a + b)

    def gap(self, T_e: float, T_gi: float) -> Tuple[float, float, float, float]:
        hr = self.h_rad_gap(T_e, T_gi)
        dT = T_e - T_gi
        h = self.p.h_gap_conv + hr
        return h * self.p.A_loss * dT, self.p.h_gap_conv * self.p.A_loss * dT, hr * self.p.A_loss * dT, h

    def ins(self, T_e: float) -> float:
        return self.p.k_ins / self.p.d_ins * self.p.A_loss * (T_e - self.p.T_amb_C)

    def out(self, T_gi: float) -> Tuple[float, float]:
        p = self.p
        Tg, Ta, Ts = T_gi + 273.15, p.T_amb_C + 273.15, p.T_sky_C + 273.15
        h_rad = p.eps_g * SIGMA * (Tg * Tg + Ta * Ta) * (Tg + Ta)
        h_o = p.h_conv_out + h_rad
        U = 1.0 / (p.d_g / p.k_g + 1.0 / h_o)
        q_sky = p.eps_g * SIGMA * (Ta ** 4 - Ts ** 4)
        return p.A_glass * U * ((T_gi - p.T_amb_C) + q_sky / h_o), U

    def e_evap(self, T_e: float) -> float:
        """Heat the evaporator spends per kg of vapour [J/kg]."""
        p = self.p
        return hfg(T_e) + (p.cp_w * (T_e - p.T_amb_C) if p.sensible else 0.0)

    def e_glass(self, T_e: float, T_c: float) -> float:
        """Heat the vapour delivers to the cover per kg condensed [J/kg]."""
        p = self.p
        return hfg(T_e) + p.cp_w * (T_e - T_c) if p.sensible else hfg(T_c)

    def mdot_from_evaporator(self, T_e: float, T_gi: float) -> float:
        Qg = self.gap(T_e, T_gi)[0]
        return max(self.p.P_abs_W - Qg - self.ins(T_e), 0.0) / self.e_evap(T_e)

    def film_T(self, T_gi: float, m: float, T_e: float) -> float:
        """Closed form of T_c = T_gi + m e_glass/(h_cond A_g) (e_glass is linear in T_c)."""
        k = m / (self.p.h_cond * self.p.A_glass)
        if self.p.sensible:
            return (T_gi + k * (hfg(T_e) + self.p.cp_w * T_e)) / (1.0 + k * self.p.cp_w)
        return (T_gi + k * _HFG_A) / (1.0 + k * _HFG_B)

    # --- inner solve: glass node for a given T_e -----------------------------
    def glass_node(self, T_e: float):
        p = self.p

        def phi(T_gi: float) -> float:
            m = self.mdot_from_evaporator(T_e, T_gi)
            Tc = self.film_T(T_gi, m, T_e)
            return m * self.e_glass(T_e, Tc) + self.gap(T_e, T_gi)[0] + p.P_glass_abs_W - self.out(T_gi)[0]

        lo = self.T_lo_gi
        hi = min(max(T_e, p.T_amb_C) + 5.0, T_MAX_C)
        for hi in (hi, min(hi + 50.0, T_MAX_C), min(hi + 150.0, T_MAX_C), T_MAX_C):
            if phi(hi) < 0.0:
                break
        over = phi(hi) >= 0.0
        if over:
            T_gi, ok = hi, False
        else:
            T_gi, ok = _root(phi, lo, hi)
        m = self.mdot_from_evaporator(T_e, T_gi)
        return T_gi, m, self.film_T(T_gi, m, T_e), ok and not over, over

    def assemble(self, T_e: float, regime: str, ok_outer: bool, P_eff: float,
                 s_gap: float = -1.0) -> CoupledState:
        T_gi, m, T_c, ok_in, over = self.glass_node(T_e)
        over = over or (T_e >= T_MAX_C - 1e-9 and not ok_outer)   # evaporator cannot shed the power
        Qg, Qgc, Qgr, h = self.gap(T_e, T_gi)
        Qo, U = self.out(T_gi)
        s = CoupledState(T_e, T_gi, T_c, m, Qg, Qgc, Qgr, self.ins(T_e), Qo, U, h, regime, P_eff,
                         converged=ok_outer and ok_in and not over, overheated=over, s_gap=s_gap)
        s.residuals = self.residuals(s)
        if max(s.residuals.values()) > CLOSURE_TOL:
            s.converged = False
        return s

    # --- closure diagnostics (recomputed independently of the solve path) ----
    def residuals(self, s: CoupledState) -> Dict[str, float]:
        p = self.p
        Qg = self.gap(s.T_e, s.T_gi)[0]
        Qo = self.out(s.T_gi)[0]
        scale = max(p.P_abs_W, 1e-9)
        res = {
            "energy_evaporator": abs(p.P_abs_W - s.mdot * self.e_evap(s.T_e) - Qg - self.ins(s.T_e)) / scale,
            "energy_glass": abs(s.mdot * self.e_glass(s.T_e, s.T_c) + Qg + p.P_glass_abs_W - Qo) / max(scale, abs(Qo)),
            "film": abs((s.T_c - s.T_gi) - s.mdot * self.e_glass(s.T_e, s.T_c) / (p.h_cond * p.A_glass))
                    / max(abs(s.T_c - p.T_amb_C), 1.0),
        }
        if s.regime == "diffusion":
            cap = diffusion_flux_s(s.T_e, s.T_c, s.P_eff, s.s_gap, p.A_eff, p.delta, p.f_der)
            res["mass_transfer"] = abs(s.mdot - cap) / max(s.mdot, 1e-12)
        elif s.regime == "ideal":
            res["ideal_T_e_eq_T_c"] = abs(s.T_e - s.T_c) / max(abs(s.T_c - p.T_amb_C), 1.0)
        elif s.regime == "prescribed":
            pass  # mdot is the prescribed cap; (a) and (c) closure already cover it
        return res

    # --- outer solves ---------------------------------------------------------
    def solve_ideal(self) -> CoupledState:
        f = lambda Te: (lambda r: r[2] - Te)(self.glass_node(Te))
        Te, ok = _root(f, self.T_lo_e, T_MAX_C)
        return self.assemble(Te, "ideal", ok, self.p.P_chamber_Pa)

    def solve_diffusion(self, P_eff: float) -> Optional[Tuple[CoupledState, bool]]:
        """Diffusion-limited state at total pressure P_eff.

        Solved in x = ln s, s = (P - p_e)/P, because near boiling T_e(s) is flat to machine
        precision while the flux ~ -ln s keeps varying: in x the residual is smooth and monotone.
        Returns (state, boiled): boiled=True when even s -> 0 cannot carry the energy flux (the
        evaporator boils at T_sat(P) and evaporation is energy-limited).  None if the cover is too
        hot for this pressure (T_c >= T_sat)."""
        p = self.p
        X_LO = math.log(1e-300)

        def Te_of(x: float) -> float:
            return t_sat_C(P_eff * (1.0 - math.exp(x)))

        def r(x: float) -> float:
            Te = Te_of(x)
            _, m, Tc, _, _ = self.glass_node(Te)
            return m - diffusion_flux_s(Te, Tc, P_eff, math.exp(x), p.A_eff, p.delta, p.f_der)

        T_boil = min(t_sat_C(P_eff), T_MAX_C)
        _, m_b, Tc_b, _, _ = self.glass_node(T_boil)
        if psat(Tc_b) >= P_eff * (1.0 - 1e-9):
            return None
        s_hi = max(1.0 - psat(self.T_lo_e) / P_eff, 1e-12)
        x_hi = math.log(min(s_hi, 1.0 - 1e-12))
        if r(X_LO) >= 0.0:
            return self.assemble(T_boil, "boiling", True, P_eff), True
        x, ok = _root(r, X_LO, x_hi)
        return self.assemble(Te_of(x), "diffusion", ok, P_eff, s_gap=math.exp(x)), False

    def solve_prescribed(self, m_target: float) -> CoupledState:
        def r(Te: float) -> float:
            return self.glass_node(Te)[1] - m_target

        Te, ok = _root(r, self.T_lo_e, T_MAX_C)
        return self.assemble(Te, "prescribed", ok, self.p.P_chamber_Pa)


def _shares(s: CoupledState, T_amb: float) -> Dict[str, float]:
    """Fraction of the evaporator-to-ambient temperature drop spent in each stage."""
    parts = {"evaporation": max(s.T_e - s.T_c, 0.0),
             "film": max(s.T_c - s.T_gi, 0.0),
             "glass_uvalue": max(s.T_gi - T_amb, 0.0)}
    tot = sum(parts.values())
    return {k: (v / tot if tot > 0 else 0.0) for k, v in parts.items()}


def solve(p: CoupledParams, mode: str) -> CoupledSolution:
    """Solve the coupled balance for the requested evaporation mode and for all four (masses)."""
    S = _Solver(p)
    warnings: List[str] = []
    sA = S.solve_ideal()

    P_set = p.P_chamber_Pa
    P_eff = P_set
    boiling = False
    pc_A = psat(sA.T_c)
    sC: Optional[CoupledState] = None
    boiled = False
    if pc_A >= P_set * (1.0 - 1e-9):
        boiling = True
    else:
        out = S.solve_diffusion(P_set)
        if out is None:
            boiling = True
        else:
            sC, boiled = out
    if boiling:
        P_eff = max(pc_A, P_set)
        sA.P_eff = P_eff
        sC = sA
        warnings.append(
            f"Boiling-limited: the cover needs T_c = {sA.T_c:.1f} C (Psat = {pc_A/1e3:.1f} kPa) but the "
            f"set chamber pressure is {P_set/1e3:.1f} kPa (T_sat = {t_sat_C(P_set):.1f} C). No steady state at the "
            f"set pressure; the chamber floats to {P_eff/1e3:.1f} kPa, the vacuum does not help and evaporation is "
            "energy-limited (vapour extraction by the ejector is not modelled).")

    A_eff = p.A_eff
    m_B = p.J_emp_kg_m2_s * p.f_der * A_eff
    m_W = p.supply_kg_m2_s * A_eff
    mA, mC = sA.mdot, sC.mdot
    modes = {
        "energy_limited": min(mA, m_W),
        "empirical": min(m_B, mA, m_W),
        "mass_transfer": min(mC, m_W),
        "hybrid": min(m_B, mC, m_W),
    }
    m_sel = modes[mode]
    nat_state = sA if mode in ("energy_limited", "empirical") else sC
    cap_label = ""
    if m_sel < nat_state.mdot * (1.0 - 1e-12):
        state = S.solve_prescribed(m_sel)
        cap_label = "wick_supply" if abs(m_sel - m_W) <= 1e-12 * m_W else "empirical_flux"
    else:
        state = nat_state

    s_state = 1.0 - psat(state.T_e) / state.P_eff       # boiling margin; <= 0: at/above the boiling point
    near = s_state < 0.05 or state.regime == "boiling"
    boiling = boiling or boiled or s_state < 0.01
    boiled = boiled and state is sC
    shares = _shares(state, p.T_amb_C)
    if cap_label:
        limited = cap_label
    elif boiling:
        limited = "boiling"
    else:
        limited = max(shares, key=shares.get)
    if cap_label and s_state < 0.0:
        warnings.append(
            f"The {cap_label} cap forces the evaporator to T_e = {state.T_e:.0f} C, above the boiling point "
            f"{t_sat_C(state.P_eff):.0f} C at {state.P_eff/1e3:.0f} kPa: dry-out / superheating, not a stable evaporating surface.")
    elif near and not boiling:
        warnings.append(
            f"Evaporator within 5 % of boiling (p_e = {psat(state.T_e)/1e3:.1f} kPa vs P = {state.P_eff/1e3:.1f} kPa): "
            "the stagnant-air diffusion model is a poor description of a nearly pure-vapour gap.")
    for name, st in (("selected", state), ("ideal", sA)):
        if st.overheated:
            warnings.append(f"The cover cannot reject the absorbed power below {T_MAX_C:.0f} C ({name} state): "
                            "model invalid for this configuration (temperatures capped).")
        elif not st.converged:
            worst = max(st.residuals.values()) if st.residuals else float("nan")
            warnings.append(f"Coupled solver did not close to {CLOSURE_TOL:g} ({name} state, worst relative "
                            f"residual {worst:.2e}).")
    if boiled:
        warnings.append(
            f"Evaporator boils at T_sat(P) = {state.T_e:.1f} C (P = {state.P_eff/1e3:.1f} kPa): diffusion cannot carry the "
            "energy flux, so evaporation is purely energy-limited.")
    return CoupledSolution(state, mode, modes, sA, sC, P_set, P_eff, boiling, near or boiling,
                           limited, shares, warnings)
