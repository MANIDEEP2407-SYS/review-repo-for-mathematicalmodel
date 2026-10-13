# Assumptions and scientific rules

Applied globally across the simulator.

## Provenance tiers

Every registered parameter is tagged with one of:

- **measured** — measured in-lab or in-field by this project.
- **literature** — value cited from a peer-reviewed source (with cite).
- **assumed** — engineering assumption with no direct source.
- **calculated** — derived from other registered parameters.

## Rules

1. No fabricated experimental results.
2. Numbers appearing on conceptual system diagrams (e.g. `8.4 kg/m2/day`,
   `180 % yield increase`) are **empirical inputs used to test the model**;
   they are not experimental validation of the current design.
3. Every parameter carries name / symbol / unit / default / range / source.
4. Internal computation is in **SI units**. Conversion to `L/day` happens at
   the presentation layer only.
5. The physical model is separated from the scaling engine which is separated
   from the UI.
6. Freshwater production is never hard-coded. It falls out of the model.
7. CFD claims are only made after an actual OpenFOAM run is completed and
   compared against experimental data.

## Simplifications (Phase 0 baseline)

- Steady-state analysis will be the first target; transient dynamics come later.
- Lumped-parameter thermal nodes (0D) — not spatially resolved.
- Uniform solar irradiance during `sunlight_hours` — no diurnal profile in v0.
- Uniform evaporator temperature — no in-plane gradients in v0.

## Coupled solver (added with `coupling="coupled"`, see physics.md)

Assumed, not measured; every one has a registry entry with a range and tier ASSUMED (never tagged literature):

- `diffusion_path_length` (delta = 2 cm): effective vapour diffusion path in the gap. Largest unknown; calibrate on prototype.
- `gap_convection_coefficient` (2 W/m2K), `glass_absorptance` (5 %), `chamber_leak_pressure_floor` (10 kPa),
  `motive_water_temperature` (30 C), `pump_efficiency` (0.5), `glass_allowable_stress` (20 MPa, warning only).
- Literature: `atmospheric_pressure`; Marrero-Mason diffusivity, Magnus vapour pressure, Rogers-Yau h_fg, Swinbank sky
  temperature, McAdams-type outer convection (as already used by the repo).

Modelling choices that differ from the first specification of the coupled equations (deliberate, documented, switchable
where noted): sensible-heat bookkeeping (`sensible_heat`), heat-leak terms on the absorber plan area instead of the wet
area, a constant sky-depression term in the glass balance, default evaporation mode `mass_transfer`, and no separate
non-condensable-air correlation (air resistance is already inside delta). Not modelled: vapour extraction by the ejector,
thermal inertia, edge/side losses beyond the insulation term, heat recovery from hot condensate.

Rule 8 (validation honesty): the 2-5 L/m2/day still productivity band and the Lal et al. 2.77 kg/m2/h isolated-evaporator
flux are used only for plausibility / consistency checks and are labelled "consistency, not validation".
