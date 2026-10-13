# Solar Desalination Simulator

A mathematical simulator, sizing tool and sustainability model for a **passive solar desalination
system** built on an FeS-functionalised cotton 3D-flower evaporator (Kalpar village context).

It answers three questions:

1. Given dimensions and weather, **how much freshwater per day** does one module make?
2. Given a target (e.g. *1 module, 10 L/hour*), **how big must the module be and how many flowers**?
3. If we scale up, **what does it cost, what does it emit, and how much green cover offsets it**?

> **Status: a projection tool, not a validated design tool.** The model has not been calibrated against
> measurements of our own prototype. Read [§8 Where the physics will fail](#8-where-the-physics-will-fail)
> before quoting any number from it.
>
> **Two solvers.** `SystemConfig.coupling = "coupled"` (default) solves the evaporator, vapour gap, glass
> condenser and ejector as one balance; `"legacy"` is the original decoupled model, kept bit-for-bit so old
> results reproduce (`results/legacy/`). The coupled model removes the defects that were admitted in §8
> (condensation decoupled from vapour production, Venturi with no effect on yield) but its new parameters
> (diffusion path, gap coefficient, leakage...) are **assumed**: see [docs/physics.md](docs/physics.md).

---

## Contents
1. [Quick start](#1-quick-start)
2. [How the model is put together](#2-how-the-model-is-put-together)
3. [The physics, stage by stage](#3-the-physics-stage-by-stage)
4. [Worked example (baseline)](#4-worked-example-baseline)
5. [How we scale](#5-how-we-scale)
6. [Cost, carbon and green cover](#6-cost-carbon-and-green-cover)
7. [OpenFOAM connection](#7-openfoam-connection)
8. [Where the physics will fail](#8-where-the-physics-will-fail)
9. [Sensitivity and uncertainty](#9-sensitivity-and-uncertainty)
10. [Parameter provenance and validation status](#10-parameter-provenance-and-validation-status)
11. [Repository layout](#11-repository-layout)

---

## 1. Quick start

```bash
python -m pip install -r requirements.txt
streamlit run app/app.py
pytest tests/
python run_model.py                  # coupled -> results/, legacy -> results/legacy/
python scripts/before_after.py       # legacy-vs-coupled table -> docs/before_after.md, results/before_after.csv
```

The dashboard tabs: **Module sizer**, **Cost & sustainability**, **OpenFOAM (CFD)**, Real-world limits, Results
(legacy-vs-coupled comparison), Modes (all four modes, both solvers), Scaling, Sensitivity, Uncertainty,
Validation, Parameters. Every sidebar setting feeds every tab; the **Coupling** toggle (sidebar, top) switches
solver and the header shows T_e, T_gi, T_c, P_chamber, vacuum load on the cover and the boiling flag.

The OpenFOAM tab needs OpenFOAM v2406 in WSL (see [§7](#7-openfoam-connection)); everything else is pure Python.

---

## 2. How the model is put together

```
sidebar settings (SystemConfig)
        │      coupling="coupled": physics/coupled.py (evaporator ⇄ gap ⇄ glass, ejector pressure) | "legacy": decoupled
        ▼
 physics/system.simulate()  ──►  productivity  [L / m² / day]   ◄── the ONE number scaling relies on
        │                                │
        │                                ▼
        │                     scaling/sizing.size_modules()   target L/h, N modules → area, size, flowers
        │                                │
        │                                ▼
        │                     sustainability/lifecycle.analyse()   capex, LCOW, CO₂, trees
        ▼
 openfoam/runner   (optional)   same temperatures/fluxes → CFD case in WSL
```

Design rules (enforced in code review):

* Internal units are **SI**; L/day is converted only for display.
* Freshwater output is **never hard-coded**; it falls out of the model.
* Physics, scaling, sustainability and UI live in separate modules and don't import each other upward.
* Every parameter has name, symbol, unit, default, allowed range and a source tier
  (measured / literature / assumed / calculated) — see `src/config/defaults.py` and
  `src/sustainability/lifecycle.py`.

---

## 3. The physics, stage by stage

The model is **lumped (0-D) and steady-state**: one representative temperature per node, constant irradiance for
`sunlight_hours`. Everything is computed per sunlight session. Full equations, symbols, numerical method and the
list of deliberate deviations from the original specification: **[docs/physics.md](docs/physics.md)**.

### 3.1 Coupled solver (default) - `physics/coupled.py`

Four unknowns, `T_e` (evaporator), `T_gi` (glass inner surface), `T_c` (condensate film) and `ṁ` [kg/s], three
closed balances, nested bracketed root-finding (no unbounded Newton), residuals reported:

```
G_eff   = G · G_opt · τ_glass · η_opt                       P_abs = G_eff · A_geo        (η_thermal is NOT used)
(a) evaporator  P_abs = ṁ·e_evap + h_gap·A_geo·(T_e − T_gi) + (k_ins/d_ins)·A_geo·(T_e − T_amb)
                h_gap = h_c,gap + h_rad,gap(ε_e, ε_g, T_e, T_gi)                       [h_c,gap ASSUMED]
(b) vapour      ṁ = f_der · A_eff · D_wa·M_w·P/(R·T·δ) · ln[(P − p_c)/(P − p_e)]       [δ ASSUMED]
                p_e = Psat(T_e), p_c = Psat(T_c) (Magnus); D_wa = 1.87e-10·T^2.072/P[atm] (Marrero & Mason 1972)
(c) glass       ṁ·e_glass + Q_gap + α_g·G·G_opt·A_g = U_out·A_g·[(T_gi − T_amb) + q_sky/h_o]
                T_c = T_gi + ṁ·e_glass/(h_cond·A_g)             U_out = [d_g/k_g + 1/(h_conv(wind) + h_rad,out)]⁻¹
yield = ṁ · t_sun · η_coll
```

* `h_fg` is temperature dependent everywhere (Rogers & Yau). `e_evap = h_fg(T_e) + c_p(T_e − T_amb)` and
  `e_glass = h_fg(T_e) + c_p(T_e − T_c)` keep energy conserved (make-up water is heated, the hot condensate leaves);
  `sensible_heat=False` gives the literal `ṁ·h_fg` forms.
* Heat leaks use the whole absorber area `A_geo`; only evaporation uses the wet area `A_eff = f_util·A_geo`.
* **Evaporation modes** (names unchanged): **A** `energy_limited` = ideal evaporator (no vapour resistance), the upper
  bound for this cover; **B** `empirical` = `J_emp·A_eff` flux cap, itself capped by A; **C** `mass_transfer` = (b);
  **D** `hybrid` = `min(B, C)`. Default for the coupled solver is **C**, so the empirical (concept-diagram) number
  does not set the baseline. The Modes tab shows all four.
* `condensation_limited_by` / `limited_by` are *diagnostics* from which stage takes most of the temperature drop
  (vapour gap / film / cover), no longer a separate `min()`.
* **Boiling limit:** when `p_e → P` the log diverges; evaporation becomes energy-limited with `T_e` pinned at
  `T_sat(P)`; no NaN. If the cover needs `T_c ≥ T_sat(P_set)` there is no steady state at the set pressure and the
  chamber floats to `Psat(T_c)`: vacuum cannot help past that point (`boiling_limited = True`).

### 3.2 Ejector / Venturi and cover load

```
P_chamber,set = max( P_atm − ΔP_eff , Psat(T_motive) , P_leak )              (500 Pa floor removed)
pump power    = Q_m · ΔP / η_pump          vacuum load on cover = (P_atm − P_chamber) · A_g   [kN]
```

`P_chamber` feeds (b) as `P`; because `T_c` and `T_gi` are solved, the loss of rejection driving force emerges
instead of being assumed. Outputs: `P_chamber`, boiling flag, ejector pump power (and kWh per m³), vacuum load
with a plate-bending warning (screen only, **not structural validity**: a 1 m² cover at 50 kPa carries ~5 t, so a
vacuum is only credible for a small, ribbed or curved sealed chamber).

### 3.3 Freshwater, mass balance, salinity - `mass_balance.py`, `salinity.py`

```
freshwater = ṁ · t · η_coll         recovery = freshwater / feed
brine TDS  = (feed TDS · V_feed) / (V_feed − V_evaporated)      (salt is conserved; vapour carries none)
```

Warnings are raised, not hidden: solver not closed, boiling limit, vacuum load, latent > absorbed energy, brine above
saturation, mass-closure error, `τ_glass + α_g > 1`.

### 3.4 Legacy solver (`coupling="legacy"`) - unchanged

`Q_useful = η_thermal · Q_absorbed` (an input); `T_e` from a balance with convection + radiation to ambient;
`T_glass = T_amb + 0.35·(T_e − T_amb)` (heuristic); condensation capacity
`min(U_glass·A_g·(T_vap − T_amb), h_cond·A_g·(T_vap − T_glass))·t/h_fg`, independent of the vapour produced;
`freshwater = min(m_evap, capacity)·η_coll`; constant `h_fg`; Venturi computes ΔP but never reaches evaporation or
condensation. Its flaws are listed in [docs/physics.md §6](docs/physics.md); it is kept so old results reproduce.

---

## 4. Worked example (baseline)

Defaults: G = 800 W/m², 8 h, gain 1.5, τ = 0.90, η_opt = 0.90, A_geo = 1 m², f_util = 0.85, glass 1.1 m²,
ambient 30 °C, wind 2 m/s, δ = 2 cm, h_c,gap = 2 W/m²K. (Numbers below are computed by the code, not typed in;
`python scripts/before_after.py` regenerates the full comparison.)

| Step | Legacy | Coupled |
|---|---|---|
| Incident / absorbed energy | 23.0 / 28.0 MJ | 23.0 / 28.0 MJ |
| Latent energy of the vapour | 18.2 MJ (**input** η_th = 0.65) | 21.9 MJ (**result**, 78 % of absorbed) |
| Evaporator T / glass T | 46.2 / 35.7 °C (glass: heuristic) | 85.5 / 77.0 °C (solved), condensate film 77.1 °C |
| Evaporated mass A / B / C / D [kg] | 7.58 / 7.14 / 5.93 / 5.93 | 10.69 / 7.14 / 9.54 / 7.14 |
| Cover / condensation capacity | **3.36 kg ← the bottleneck** | 10.69 kg (ideal-evaporator limit); cover takes 85 % of the temperature budget |
| Freshwater (× 0.90) | **3.02 L/day** | **8.59 L/day** |
| Energy efficiency (latent of product / incident) | 31.5 % | 85.7 % |

The legacy reading ("the evaporator could make 7.6 kg but the glass limits to 3.4 kg") was an artifact of the
decoupled cap. In the coupled model the cover simply runs hot (77 °C) until it rejects the absorbed power, and
the yield is set by how that heat splits between vapour and sensible leak. **The coupled number is optimistic**:
no side or edge losses beyond the insulation term, a perfectly sealed chamber, no thermal inertia, no heat
recovery, and the user-set reflector gain 1.5 (5.1 L/m²/day without it). It lies above the 2-5 L/m²/day band
published for passive stills and the tool flags that (`run_still_plausibility_check`: *consistency, not
validation*). Treat 8.6 as an upper estimate conditional on the assumed δ, h_c,gap, h_cond.

## 5. How we scale

### 5.1 Sizing rule

`scaling/sizing.py` measures productivity by **running `simulate()`** on the reference configuration
(it is never typed in), then:

```
A_required = target [L/day] / productivity [L/m²/day]
```

### 5.2 Module sizer (modules + water → size, flowers)

Inputs: `N` modules, total `L/hour`, flowers per m².

```
target_L_per_day = L_per_hour · sunlight_hours        (rate is averaged over sunlight hours)
A_total          = target_L_per_day / productivity
A_module         = A_total / N
square side      = √A_module          circular Ø = 2·√(A_module/π)
flowers/module   = ⌈A_module · flowers_per_m²⌉
coverage         = flowers · π(d_flower/2)² / A_module        (warn if > 0.9069, the max packing of circles)
```

Example: 1 module at 10 L/h → 80 L/day ÷ 3.02 = **26.5 m²** (≈5.1 m square), ≈662 flowers.
4 modules at the same total → 6.6 m² each, 166 flowers each. Total area is unchanged; splitting only
changes module size.

If you meant 10 L/h round the clock, daily demand is 240 L/day and every area triples — the
"sunlight-hours" reading is an explicit assumption.

### 5.3 What scaling assumes (and where it is weakest)

* **Productivity per m² is constant with size.** Area is scaled linearly. Real arrays have edge effects,
  flower-to-flower shading, longer vapour paths and wind exposure that change with size.
* **Glass area scales with evaporator area** (the reference ratio 1.1 is held). In the legacy model the (artifactual)
  condensation ceiling made this assumption drive the answer; in the coupled model yield is *not* monotone in cover
  area (it peaks at the smallest valid cover and falls slowly in diffusion mode, saturates in the ideal-evaporator
  limit; docs/physics.md §4), so the ratio is a design variable, not a free assumption.
* **Same weather for every module**, no shading between modules.
* **Flowers per m² is assumed** (default 25, the registry value), not measured; it only sets the flower
  *count* and packing check, not the water yield.

### 5.4 Real-world derating (`physics/realworld.py`)

Sizing can use a **derated** productivity instead of the ideal one (sidebar → *Real-world limits* →
"Apply to module sizing", on by default). The model is re-run **hour by hour** and each effect is applied
in turn; the *Real-world limits* tab shows the waterfall.

| Step | What it does |
|---|---|
| Diurnal sun | Half-sine over `day_length_h` with the **same daily insolation** as the flat block, so only the non-linear losses at low sun change the answer |
| Dust soiling | Glass transmittance × `(1 − soiling_rate · T_clean / 2)` (average over a cleaning cycle) |
| Fouling & ageing | Evaporated mass × salt/biofilm factor `(τ/T)(1 − e^(−T/τ))` × flower-ageing factor `(1 − d/2)` |
| **Capillary supply** | Evaporation is capped by what the cotton wick can deliver (coupled: caps the vapour flow *inside* the balance, so the evaporator heats up) |

**Capillary model** (bundle of parallel pores, per m² of evaporator):

```
P_cap   = 2 σ cosθ / r                     capillary pressure
h_Jurin = P_cap / (ρ g)                    highest the wick can lift water
K       = ε r² / (8 τ²)                    permeability
u       = (K/μ) (P_cap − ρ g L_w) / L_w    Darcy flow through a wick of length L_w
J_wick  = ρ u φ_w                          kg/m²/s deliverable  →  m_wick = J_wick · A_eff · t
freshwater(hour) = min(evaporated, m_wick, condensation capacity) · collection_efficiency      [legacy]
coupled: fouling/ageing scale the vapour-side conductance and flux caps, the wick flux caps ṁ; both enter the energy balance
```

`σ(T)` and `μ(T)` are evaluated at the hourly evaporator temperature. If `L_w > h_Jurin` the wick cannot
lift water at all and supply is **zero** (the tab says so). If supply < demand in any hour the tab warns
of dry-out and local salt crust. Trade-off it captures: **finer pores lift higher but flow slower**.

All these parameters are **assumed** (defaults: r = 20 µm, θ = 30°, L_w = 5 cm, φ_w = 0.25 %). At those
defaults the wick has ~30× headroom, so **capillarity is not binding — that is a statement about the
assumed numbers, not a finding.** Measure r, θ and φ_w with a wicking-rise test on the real flower.

Baseline result, legacy: ideal 3.02 → **2.69 L/m²·day** (diurnal −8.5 %, soiling −2.6 %; fouling, ageing and
capillary hidden behind the condensation ceiling). Coupled: ideal 8.59 → **7.62 L/m²·day** (diurnal −6.5 %,
soiling −2.6 %, fouling+ageing −2.6 %; capillary not binding at the assumed numbers); a thin wick pushes the
evaporator above 100 °C, which the tab flags as dry-out.

### 5.5 Household / village demand

`HouseholdDemand`: `people · (drinking + cooking L/person/day) · safety_factor`, times households, then the
same `A_required` rule; modules = `⌈A_required / A_per_module⌉`.

---

## 6. Cost, carbon and green cover

`sustainability/lifecycle.py`. **Almost every price and emission factor is an ASSUMED placeholder** and is
labelled so in the UI; replace with quotes / cited LCA data. Scope: materials, replacements, pump
electricity, LCOW. Excluded: transport, end-of-life, water-treatment chemicals.

**Bill of materials per module** (mass × unit cost × embodied CO₂ factor): glass cover
(`A · glass_ratio · thickness · density`), frame/basin, insulation, cotton flowers, FeS coating. Cotton and
FeS are *replaceable* every `flower_life_years`.

**Cost scaling (Wright's law):**

```
unit_variable = (materials + assembly_hours·wage) · (1 + overhead)
capex(N)      = unit_variable · N^(log₂ learning_rate) · N  +  fixed_setup_cost
```

`learning_rate = 0.90` means each doubling of module count cuts unit cost by 10 %. **This is an assumption
about how manufacturing improves, unverified for this device**; set it to 1.0 for no economies of scale.

**Levelised cost of water** discounts costs and water over the design life:

```
LCOW = [ capex + Σ (opex_t + replacements_t)/(1+r)^t ] / [ Σ water_t/(1+r)^t ]
```

**Carbon:** embodied CO₂ = Σ mass × factor (again at each replacement); operational CO₂ =
`pump kWh × grid factor` (zero for the passive design). Reported per year, cumulative, and per m³ of water.

**Green cover for net zero:**

```
steady-state trees = ⌈ average annual CO₂ / (uptake per tree · survival) ⌉
lifetime trees     = ⌈ total CO₂ / Σₜ(uptake·survival·min(1, t / maturity)) ⌉      (saplings ramp up)
green area (ha)    = trees / trees_per_hectare
net-zero year      = first year cumulative emissions − cumulative uptake ≤ 0
```

Caveats: tree offsetting is not emission *reduction*, needs land and permanent survival, and the uptake
per tree is species/climate dependent. The tool warns if net zero isn't reached inside the horizon.

---

## 7. OpenFOAM connection

`src/openfoam/runner.py` + the **OpenFOAM (CFD)** tab (maps `Q_useful` = the latent heat the solver says leaves as vapour, and ambient; does not use the coupled T_gi). It copies the case in `../openfoam_desalination`
to `results/openfoam_runs/run_<time>/`, patches a few values, runs `buoyantPimpleFoam` in WSL
(OpenFOAM v2406, `source /usr/lib/openfoam/openfoam2406/etc/bashrc`), and reads back progress and the
area-average evaporator temperature.

| Simulator quantity | OpenFOAM entry patched |
|---|---|
| `Q_useful / (A_eff · t_sun)` | `0/T` → `evaporatorSurface` `q` (net flux) |
| `ambient_temperature + 273.15` | `0/T` → `internalField`, `glassCover` `Ta` |
| glass outer `h` (case default 18) | `0/T` → `glassCover` `h` |
| run length / write interval | `system/controlDict` `endTime`, `writeInterval` |

**What this is and is not:**

* The current case solves **buoyant heat transfer and flow only** — no evaporation/condensation
  mass-transfer source. It **produces no freshwater number**, and agreement or disagreement with the
  Python temperature is **not validation**. Its `controlDict` says so.
* The mesh is fixed (one 3.6 m² module). Only intensive quantities (fluxes, temperatures) map; changing
  area would need re-meshing.
* A short run is still warming up, so it will not have reached the Python model's steady value.

---

## 8. Where the physics will fail

Being explicit, in rough order of how much it could change the answer. Items 1-2 of the original list (η_thermal /
decoupled condensation, Venturi without effect) are **fixed in `coupling="coupled"`** - the regression tests in
`tests/test_coupled_solver.py` prove it (yield cannot rise when evaporation gets worse; η_thermal is off the path;
the Venturi changes yield; fouling/capillary limits are no longer masked) - and **remain true in
`coupling="legacy"`**, pinned by `tests/test_legacy_mode.py`; do not quote legacy `η_thermal` or `f_util`
sensitivities.

1. **The new coupled parameters are guesses, and the answer moves with them.** δ (effective diffusion path, default
   2 cm), h_c,gap (2 W/m²K), h_cond (8000), glass absorptance (5 %), leak floor (10 kPa) are ASSUMED. δ alone moves
   the baseline from 9.3 L/day (0.5 cm) to 6.9 (30 cm); the evaporator temperature from 80 to 99 °C. **This is the
   biggest gap: measure δ, h_c,gap and the leak floor on a prototype.**
2. **The coupled model is optimistic by construction.** No side/edge loss beyond k_ins/d_ins, perfectly sealed
   chamber, quasi-steady (no thermal mass: morning and evening over-predicted), no recovery of the hot condensate's
   heat, no salt-crust thermal resistance, uniform temperatures. Baseline 8.6 L/m²/day (5.1 without the reflector)
   against a published 2-5: out of band. The isolated-evaporator check gives 0.97 kg/m²/h for a flat open surface vs
   2.77 reported by Lal et al. (Chem. Eng. J. 522, 167097, 2025; figure taken from the project brief, not re-checked):
   the repo's closures reach it only with ~53× exchange area, so the 3-D flower enhancement is unmeasured.
   Both are *consistency, not validation*.
3. **Cover area and wind are not monotone** in the diffusion regime (colder cover → larger sensible leak): yield
   peaks at the smallest valid cover (ratio ≈ 0.7) and falls to 6.6 L/day at ratio 18, while the ideal-evaporator limit
   saturates. Wind helps in the cover-limited regime and hurts in the diffusion-limited one. Both depend on δ and h_c,gap.
4. **Vacuum is bounded by the cover.** The chamber floats to Psat(T_c) when the cover needs a hotter condensing
   temperature than the ejector's set pressure allows, so the modelled gain saturates (+12 % at the default ejector)
   and costs electricity (175 W for ~10 L/day = 146 kWh/m³; the concept-diagram ejector draws 3.5 kW for 12.9 L/day).
   Vapour extraction by the ejector and ejector-as-condenser operation are **not modelled**. A flat cover under
   vacuum is not credible beyond a small chamber (the plate screen flags it; it is *not* structural validity).
5. **Capillary/wicking** is a bundle-of-pores Darcy model with assumed inputs (README §5.4): it ignores non-uniform pores,
   drying/rewetting, salt blocking, brine viscosity, 3-D flow paths and contact-angle hysteresis. Real wicks often
   behave worse once salted. Not modelled: biofilm dynamics, salt-crust resistance, wind/rain damage, cloudy and monsoon days
   (`operating_days` in the cost model is a blunt stand-in), night-time salt re-dissolution, water-quality change.
6. **Linear area scaling** (§5.3). Fails when modules are large, close together, shaded, or vapour paths lengthen.
7. **Steady state, uniform irradiance.** The hourly derating uses a half-sine but no thermal inertia, clouds, monsoon,
   dust events or night. The L/h→L/day conversion averages over `sunlight_hours`.
8. **Lumped temperatures.** No in-plane gradients, no flower-scale physics, no flow in the basin; the stagnant-air
   diffusion model is a poor description when the gap is nearly pure vapour (flagged within 5 % of boiling).
9. **Salinity and fouling.** Salt conservation is exact but crystallisation only raises a warning; scale, crust and
   biofouling over months are not modelled (only a replacement interval in the cost model). The 20 L feed batch is conceptual.
10. **Water quality.** Output is assumed potable distillate.
11. **The `8.4 kg/m²/day` and `180 %` figures come from a concept diagram.** They are used as an empirical input
    (mode B cap). The "linear scaling" case checks that `8.4 × A` equals `8.4 × A`: arithmetic, **not** physics.
    Agreement between modes A and B is not independent evidence.
12. **Cost/carbon numbers are placeholders**, the learning curve is unverified, grid and tree uptake factors need citations.
13. **Uncertainty analysis assumes** independent Gaussian errors with assumed widths (§9); real inputs (irradiance,
    temperature, humidity) are correlated; δ's 40 % width is itself a guess.

**Built-in tripwires** (warnings in the UI/result): coupled solver not closed or cover unable to reject the power below 250 °C,
boiling limit, vacuum load / plate screen, latent > absorbed energy, `τ_glass + α_g > 1`, brine above saturation,
mass-closure error, flower coverage above the packing limit, net zero not reached in the horizon, wick dry-out.

**What would tell us the model is wrong:** build and instrument a ~1 m² prototype and measure daily yield under
known irradiance, T_e, T_gi (cover inner surface), chamber pressure, and fit δ and h_c,gap. If daily yield falls
short of the model by more than the Monte-Carlo 5-95 % band, an assumption above has broken.

---

## 9. Sensitivity and uncertainty

* **Tornado / one-at-a-time sweeps** (`scaling/sensitivity.py`): each parameter is moved across a
  plausible range (e.g. irradiance 500–1000 W/m², wind 0.5–5 m/s) with the others held, and the change in L/day
  is ranked. `default_tornado_ranges("coupled")` adds glass area, h_cond, δ, h_c,gap, insulation and glass
  absorptance and drops η_th and k_m (not on the coupled path); ejector parameters are a separate tornado on a
  venturi-enabled baseline.
* **Monte Carlo** (`scaling/monte_carlo.py`): each parameter is multiplied by an independent
  `Normal(1, σ_rel)` draw (σ from 5 % for collection efficiency to 30 % for the mass-transfer
  coefficient; assumed, not measured), `simulate()` is run N times with a fixed seed, and mean/median/
  5th/95th percentiles are reported. Values are clamped positive (and, for the coupled solver, to the registry
  range). The coupled list adds glass transmittance, glass area, insulation k, h_c,gap (30 %), δ (40 %) and α_g
  (30 %). `exclude=ARTIFACT_PARAMETERS` drops `eta_thermal` and `area_utilization_factor`, the two parameters whose
  legacy role was an artifact (Spearman ρ with yield −0.64 and −0.33 in legacy; −0.01 and 0.03 coupled).

Use both to decide **which parameter to measure first**, not as a confidence interval on the truth.

---

## 10. Parameter provenance and validation status

* Source tiers: **measured** · **literature** · **assumed** · **calculated**. The *Parameters* tab lists
  each with unit, range and source; the cost tab shows the tier in each input's label.
* Numbers from concept diagrams are **empirical inputs to test scaling**, never validation.
* Nothing in `src/validation` validates the physics: it holds arithmetic checks, internal-consistency checks and two
  plausibility comparisons (2-5 L/m²/day published still productivity; Lal et al. isolated-evaporator flux) that are
  labelled "consistency, not validation".
* The new coupled parameters (δ, h_c,gap, α_g, P_leak, pump efficiency, allowable glass stress) are all **assumed**.
* **CFD validation has not been done.** OpenFOAM results will only be described as validation after a run
  that includes the phase-change physics is compared with experimental data.

---

## 11. Repository layout

```
solar_desalination_simulator/
├── app/app.py                Streamlit dashboard
├── src/
│   ├── config/defaults.py    parameter registry with provenance
│   ├── physics/              solar, heat_transfer, evaporation, condensation,
│   │                         mass_balance, salinity, venturi (legacy Venturi + ejector, cover load),
│   │                         coupled (evaporator ⇄ gap ⇄ glass solver), isolated (open-surface check),
│   │                         system (simulate; coupling="coupled"|"legacy"),
│   │                         realworld (capillary limit, fouling, soiling, diurnal)
│   ├── scaling/              sizing (module sizer), sensitivity, monte_carlo
│   ├── sustainability/       lifecycle (cost, carbon, green cover)
│   ├── openfoam/             runner (WSL OpenFOAM connection)
│   └── validation/           reference cases (linear-scaling arithmetic checks)
├── scripts/before_after.py   legacy-vs-coupled comparison table
├── tests/                    pytest suite (test_coupled_solver, test_legacy_mode, data/legacy_reference.json ...)
├── docs/                     physics, assumptions, parameters, scaling, validation, before_after
└── results/                  coupled CSV/plots; results/legacy/ = legacy solver (OpenFOAM run folders are git-ignored)
openfoam_desalination/        the OpenFOAM case the CFD tab drives
```
