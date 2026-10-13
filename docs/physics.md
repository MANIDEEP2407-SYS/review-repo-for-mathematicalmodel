# Physics

Equations, symbols, assumptions and known failures of the two solvers selected by
`SystemConfig.coupling`:

* **`"coupled"` (default)** - evaporator, vapour gap, glass condenser and ejector solved as one balance
  (`src/physics/coupled.py`, `venturi.compute_ejector`, `system._simulate_coupled`).
* **`"legacy"`** - the original decoupled model, kept bit-for-bit so old results reproduce
  (`results/legacy/`, `tests/data/legacy_reference.json`).

Nothing here is validated against measurements of our prototype. Every parameter tagged ASSUMED in the
registry is a guess with a range. The 8.4 kg/m2/day and 180 % figures come from a concept diagram and are
inputs to test, never evidence.

---------------------------------------------------------------------------------------------------

## 1. Coupled solver

Steady state during the sunlight session, lumped (0-D), SI units internally. Unknowns:
`T_e` (evaporator), `T_gi` (glass inner surface), `T_c` (condensate-film surface), `mdot` [kg/s].
Yield = `mdot * t_sun * eta_coll`.

```
P_abs   = G * G_opt * tau_g * eta_opt * A_geo                      light absorbed by the evaporator
P_g     = alpha_g * G * G_opt * A_g                                light absorbed in the cover (ASSUMED alpha_g)
```

### 1.1 Evaporator node (a)

```
P_abs = mdot * e_evap + Q_gap + Q_ins
Q_gap = h_gap(T_e, T_gi) * A_loss * (T_e - T_gi)      h_gap = h_c,gap + h_rad,gap          [ASSUMED h_c,gap]
h_rad,gap = sigma (T_e^2 + T_gi^2)(T_e + T_gi) / (1/eps_e + 1/eps_g - 1)                   [K]
Q_ins = (k_ins / d_ins) * A_loss * (T_e - T_amb)
e_evap = h_fg(T_e) + c_p (T_e - T_amb)                                                     (see 1.5)
```

`A_loss = A_geo` (whole absorber) and `A_eff = f_util * A_geo` (wet part) is used for evaporation only.
The brief writes `A_eff` in both loss terms; with that, *raising* `f_util` lowers yield (more wet area means more
leak), the same kind of artifact as the legacy one, so the loss terms use the full plan area. Identical at
`f_util = 1`.

`eta_thermal` is **not on this path**: evaporation is whatever the balance gives. A photothermal efficiency would
only be physical as "fraction of absorbed light that does not become heat at the surface", which is already
inside `eta_opt` (registry note: transmittance x absorptance x reflector losses), so it was dropped, not kept.
`h_fg` (constant) and `relative_humidity` (the chamber is sealed) are not used either.

### 1.2 Vapour transport (b)

Stefan flow through a stagnant air film of effective length `delta`:

```
mdot'' = D_wa(T,P) M_w P / (R T delta) * ln[(P - p_c) / (P - p_e)]        p_e = Psat(T_e),  p_c = Psat(T_c)   (Magnus)
D_wa   = 1.87e-10 T^2.072 / P[atm]      Marrero & Mason, J. Phys. Chem. Ref. Data 1 (1972), 282-450 K; T = (T_e+T_c)/2
mdot   = f_der * A_eff * mdot''
```

`D_wa ~ 1/P`, so `D_wa * P` is pressure-independent; **the vacuum enters through the log**: less air means a larger
`ln` for the same vapour-pressure difference. `delta` is the *effective* path (natural convection included): the
geometric gap is 10-30 cm, but a Dunkle-type convective enhancement shortens the effective path to ~1-2 cm.
**It is the largest unknown in the model - calibrate on a prototype.**

Limits handled without NaN/inf: `p_e <= p_c` gives zero flux; `p_e -> P` gives a very large flux (boiling).
Near boiling the problem is solved in `x = ln s`, `s = (P - p_e)/P`, because `T_e(s)` is flat to machine precision
while the flux keeps changing (a root-find in `T_e` loses precision there).

### 1.3 Glass / condenser node (c)

```
mdot * e_glass + Q_gap + P_g = Q_out(T_gi)                                  e_glass = h_fg(T_e) + c_p (T_e - T_c)
T_c = T_gi + mdot * e_glass / (h_cond * A_g)                                condensate film (closed form)
Q_out = A_g * U_out * [ (T_gi - T_amb) + q_sky / h_o ]
U_out = [ d_g/k_g + 1/h_o ]^-1,   h_o = h_conv(wind) + h_rad,out,   h_conv = 5.7 + 3.8 v
h_rad,out = eps_g sigma (T_gi^2 + T_amb^2)(T_gi + T_amb)                    radiation linearised about ambient
q_sky = eps_g sigma (T_amb^4 - T_sky^4)                                     constant sky-depression term (Swinbank T_sky)
```

The 0.35 heuristic is gone; `T_gi` is solved. `h_rad,out` is evaluated at `T_gi` (the 7 K drop across a 10 mm
pane is ignored inside the radiation coefficient: ~2 % effect on `U_out`). The sky term is the exact remainder when
radiation is referred to ambient instead of the sky.

### 1.4 Evaporation modes (names unchanged)

| mode | coupled meaning |
|---|---|
| A `energy_limited` | **ideal evaporator**: no vapour-side resistance (`T_e = T_c`). The upper bound this cover allows; independent of `P` and `delta`. |
| B `empirical` | prescribed `mdot = J_emp * f_der * A_eff`, capped by A |
| C `mass_transfer` | vapour diffusion (1.2) |
| D `hybrid` | `min(B, C)` (and so also `<= A`) |

An optional wick supply limit caps every mode. Default mode is **`mass_transfer`** for the coupled solver
(`energy_limited` for legacy): a diffusion mode is needed for chamber pressure to matter, and the empirical cap
(a concept-diagram number) must not silently set the baseline (at baseline it would: 6.43 vs 8.59 L/day).
`m_condensation_capacity_kg` is the mode-A limit on the same cover; `limited_by` /
`condensation_limited_by` are diagnostics from the share of the evaporator-to-ambient temperature drop spent in
the vapour gap, film and cover - not a separate `min()`.

### 1.5 Temperature-dependent latent heat and sensible-heat bookkeeping

`h_fg(T) = 2500.9 - 2.365 T[C]` kJ/kg (Rogers & Yau) is used everywhere. The brief's literal (a)/(c) use
`h_fg(T_e)` and `h_fg(T_c)` with no sensible terms. With those, `h_fg` falling with temperature makes a *hotter*
still look cheaper per kg and yield spuriously **decreases** as the cover grows (monotonicity fails even for the
ideal evaporator), and energy is not conserved. The default `sensible_heat=True` adds the make-up water heating
(`c_p (T_e - T_amb)`) and gives the glass the vapour's enthalpy above the condensate (`c_p (T_e - T_c)`), so

```
P_abs + P_g = Q_out + Q_ins + mdot c_p (T_c - T_amb)        (the hot condensate leaves; no heat recovery)
```

`sensible_heat=False` gives the literal equations (tested to close as written).

### 1.6 Numerical method

Nested bracketed `brentq` (no unbounded Newton): outer root in `T_e` (or `ln s` for diffusion), inner root in
`T_gi` (monotone: heat out grows ~17 W/K, heat in barely changes). The reported tolerance is the worst relative
residual of the evaporator balance, glass balance, film relation and mass-transfer law, recomputed after the
solve (`solver_residual_max`, `solver_residuals`); `SystemResult.solver_converged` is false and a warning is
raised if it exceeds 1e-6 or if the cover cannot reject the power below 250 C. Baseline closes to ~1e-16; tests
check < 1e-6 with an independent recomputation from the config and result fields.

### 1.7 Boiling limit and the vacuum that cannot be held

* Evaporator at the boiling point (`p_e -> P`): the log diverges, `T_e` pins at `T_sat(P)` and evaporation is
  energy-limited (`boiling_limited = True`).
* **The cover needs `T_c >= T_sat(P_set)`**: total heat to the cover is `~P_abs - Q_ins + P_g`, nearly independent
  of how it splits between latent and sensible heat, so `T_gi` is set by the cover's `U*A`. If that `T_c` has a
  vapour pressure above the set pressure there is **no steady state** at the set pressure. The model then lets the
  chamber float to `Psat(T_c)` (`P_chamber_Pa` > `P_chamber_set_Pa`): a pure-vapour space, ideal-evaporator
  state, vacuum gives nothing more. Vapour extracted by the ejector (direct-contact condensation in the motive
  stream) is **not** modelled.

So in this model a vacuum can only help through the air-removal (log) effect, up to the mode-A bound, and the
coupled solve - not an assumption - decides where that stops. It does not predict a net *loss* in yield from
vacuum; it predicts a saturating gain plus a parasitic pump cost (see 2).

---------------------------------------------------------------------------------------------------

## 2. Ejector / Venturi and cover loading

```
dP_theory = 1/2 rho (v2^2 - v1^2),  v2 = Q/(C_d A2), v1 = Q/A1
dP_eff    = dP_theory (1 - eta_v)
P_chamber,set = max( P_atm - dP_eff , Psat(T_motive) , P_leak )        500 Pa floor removed
P_pump = Q_m dP_theory / eta_pump                                       parasitic electrical power
Vacuum load on cover = (P_atm - P_chamber) * A_g                        kN; reported with tonnes-force
```

`Psat(T_motive)` (4.2 kPa at 30 C) is a hard cavitation floor; `P_leak` (default 10 kPa, ASSUMED) is the lowest
pressure a leaky glass cover holds. A plate-bending screen (`0.2874 q a^2 / t^2`, square simply-supported plate of
equal area, Roark) is compared with an assumed allowable stress; when exceeded the result carries a warning that a
vacuum like this is credible only for a small, ribbed or curved sealed chamber. **This is a screening estimate, not
structural validity** (no edge fixity, ribs, membrane action, creep). A 1 m2 cover at 50 kPa vacuum carries
~5.1 tonnes-force (tested).

Default ejector (1 L/s, 10 mm throat): `dP_theory = 87.7 kPa`, pump 175 W (hydraulic 88 W) = 146 kWh/m3 of
product at 9.6 L/day; the concept-diagram ejector (2 L/s, 8 mm) draws 3.5 kW for 12.9 L/day. Pump power is
reported next to the yield so the gain is never shown without its electrical cost.

Non-condensable air (option (i) in the brief) was **not** added as a separate correlation: the stagnant-air film
in (b) already puts the air resistance of the whole path (evaporation and condensation side) into `delta`, and
the log term already rises as air is removed. A separate Othmer/Uchida-type condensation correction would count
it twice.

---------------------------------------------------------------------------------------------------

## 3. Real-world derating on the coupled solver (`physics/realworld.py`)

Hourly half-sine sun, soiling, fouling/ageing, capillary supply. In coupled mode fouling and ageing scale the
evaporator's vapour-side conductance and flux caps (`evaporation_derating`) and the capillary limit caps the vapour
flow (`wick_supply_flux_limit`); both act **inside** the balance, so a weaker evaporator runs hotter and leaks more
heat to the cover instead of being hidden behind a glass ceiling. Legacy keeps the old post-hoc multipliers.
The ideal-evaporator mode A has no vapour-side resistance, so fouling cannot act on it (not the default).

The diurnal step re-times the same daily insolation over a longer day; with a per-hour flux cap (modes B/D) this
adds capacity, so only the soiling/fouling/capillary steps are derating steps in every mode. In the default mode
all steps are non-increasing (tested for both couplings).

---------------------------------------------------------------------------------------------------

## 4. What changed, and the evidence

| legacy defect | coupled | test(s) proving it |
|---|---|---|
| condensation cap independent of vapour made; `T_glass` a heuristic | one balance, `T_gi`, `T_c` solved | `test_closure_*` (independent recomputation < 1e-6, baseline + 10 random + all modes) |
| `eta_thermal` raises evaporator `T` and can *raise* yield | not on the path | `test_eta_thermal_is_not_on_the_coupled_path`, `test_legacy_artifact_is_still_reproducible_but_coupled_is_free_of_it` |
| `f_util` artifact (rho = -0.33) | losses on the whole absorber | `test_area_utilization_never_lowers_yield`, MC rho = 0.03 |
| yield not monotone in "more useful energy" | monotone in G, G_opt, eta_opt, tau_g (4 modes x 6 random configs) | `test_yield_nondecreasing_in_useful_energy` |
| evaporator-side limits masked by glass | fouling/capillary lower yield and *raise* `T_e` | `test_coupled_fouling_and_capillary_act_inside_the_balance...` |
| Venturi toggle changes nothing (3.023 vs 3.023) | yield changes; gain saturates at the cover's vapour pressure | `test_enabling_the_venturi_changes_yield`, `test_vacuum_gain_is_bounded_and_saturates...` |
| 500 Pa floor | `Psat(T_motive)` and leak floor | `test_chamber_pressure_never_below_...` |
| NaN risk at boiling | `ln s` solve, fuzz of the full registry box | `test_fuzz_registry_box_never_nan...` |
| legacy numbers must survive | `coupling="legacy"` | `test_legacy_reproduces_3p0233_exactly` + frozen JSON of 8 configs to 1e-12; `results/legacy/*.csv` byte-identical to the pre-change outputs |

### Findings where physics disagrees with the brief's expected signs

* **Wind speed.** Not monotone. Ideal-evaporator limit (A) and the cover-limited / boiling-limited regime:
  yield rises with wind (cover rejects more). While the evaporator is genuinely diffusion-limited: yield **falls**
  with wind (colder cover, larger evaporator-to-cover sensible+radiative leak). The brief's "non-increasing in
  wind" holds only in that second regime and is tested there.
* **Glass area.** Ideal evaporator: monotone, saturates (within 1 % at ratio ~9.4). Empirical cap (B): rises then sits
  exactly on the cap. **Diffusion mode: peaks at the smallest valid cover and falls slowly** (9.3 -> 8.6 -> 7.6 -> 6.6
  L/day at ratios 0.7, 1.1, 2.5, 18) because the cooler cover raises the leak share. Below ratio ~0.68 the cover
  would need to exceed the boiling point (steam venting, flagged invalid). The brief's "monotone non-decreasing"
  therefore holds for A and B, not for C; this depends on `delta` and `h_c,gap` and is only as good as they are.
* **Glass absorptance** lowers mode A and raises mode C (hotter cover, better latent share): same mechanism.

---------------------------------------------------------------------------------------------------

## 5. Remaining failures of the coupled model

1. **`delta`, `h_c,gap`, `h_cond`, `alpha_g`, `P_leak` are guesses.** `delta` alone moves the baseline from
   9.3 L/day (0.5 cm, T_e 80 C) through 8.6 (2 cm, 85 C) to 6.9 (30 cm, 99 C, near boiling); `h_c,gap` over 0.5-5
   W/m2K moves it 8.7 -> 8.4. Nothing here has been measured.
2. **Optimistic by construction.** No side/edge losses beyond the insulation term, perfectly sealed chamber, quasi-steady
   (no basin/wick thermal inertia, morning and evening are over-predicted), no heat recovery from the hot condensate,
   uniform temperatures, no salt-crust thermal resistance. The baseline gives 8.6 L/m2/day (5.1 without the 1.5x
   reflector gain) against a published 2-5 L/m2/day band for passive stills: out of band, flagged by
   `run_still_plausibility_check`. Inside the band would prove nothing either.
3. **Isolated evaporator.** The repo's open-surface closures give 0.97 kg/m2/h at one sun against 2.77 reported by
   Lal et al.; reaching it needs ~53x exchange area or ~930 W/m2 of environmental energy (the one-sun bound is
   1.40). The check says the repo's flat-plate closures cannot reproduce the lab flux without an unmeasured 3-D
   enhancement (consistency, not validation; the 2.77 figure was taken from the brief, not re-checked).
4. **Vacuum.** Ejector extraction of vapour, ejector-as-condenser operation, air in-leak dynamics and the cover's
   mechanical response are not modelled; the plate screen is a flag only.
5. **Diffusion model near boiling** (stagnant air film) is a poor description of a nearly pure-vapour gap; the
   result is flagged within 5 % of boiling.
6. **Linear area scaling, uniform sun, no clouds/monsoon, water quality** - unchanged from README section 8.
7. **Energy bookkeeping** neglects the c_p-vs-c_pv difference in vapour sensible heat and the temperature
   dependence of c_p (<1 %).

---------------------------------------------------------------------------------------------------

## 6. Legacy solver (coupling="legacy")

```
Q_useful = eta_thermal * Q_absorbed                       INPUT
P_abs = P_evap + h_c A (T_e - T_amb) + eps sigma A (T_e^4 - T_sky^4) + (k_ins/d_ins) A (T_e - T_amb)
T_glass = T_amb + 0.35 (T_e - T_amb)                      heuristic
capacity = min( U_glass A_g (T_vapour - T_amb), h_cond A_g (T_vapour - T_glass) ) * t / h_fg   independent of mdot
freshwater = min(m_evap, capacity) * eta_coll
```

Kept unchanged for reproducibility. Known flaws, still true in this mode and pinned by tests: raising
`eta_thermal` can lower yield, lowering it can raise yield; `f_util` has the wrong sign; the Venturi never reaches
yield; evaporator-side limits are masked while the glass binds; the 500 Pa floor. Do not quote legacy sensitivity
results for `eta_thermal` or `area_utilization_factor`.
