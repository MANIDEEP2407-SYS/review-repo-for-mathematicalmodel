# Validation

Reference cases the simulator must reproduce, plus honest labelling of
what each reproduction actually demonstrates.

## Reference points (Phase 11)

- **Empirical evaporator productivity** — `8.4 kg/m2/day`
  (source: conceptual system diagram, cite marker 396).
  Reproducing this with the empirical-flux mode (MODE B) demonstrates
  *linear scaling of an empirical input*, **not** physics validation.

- **180 % yield increase over a traditional solar still**
  (source: conceptual system diagram, cite marker 393).
  Same caveat: this is an input claim to be tested, not evidence.

## What Phase 11 will NOT claim

- CFD validation. That requires the OpenFOAM module (Phase 12+).
- Field validation. That requires prototype measurements.
- Universal accuracy. Model discrepancy flags will be surfaced whenever
  MODE A / MODE B / MODE C predictions disagree by more than a
  configurable tolerance.

## Plausibility and consistency checks (coupled solver)

*Consistency, not validation.* They can show the model is in the wrong ballpark; they cannot show it is right.

- **Still plausibility**: predicted productivity vs the 2-5 L/m2/day band published for passive solar stills
  (`run_still_plausibility_check`). Coupled baseline 8.6 L/m2/day (5.1 without the 1.5x reflector): out of band, because
  the model has no side/edge/basin losses and the reflector gain is a user input.
- **Isolated evaporator**: open-surface evaporator under one sun vs the 2.77 kg/m2/h laboratory flux reported by Lal et al.,
  Chem. Eng. J. 522, 167097 (2025) (figure taken from the project brief, not re-checked against the paper)
  (`run_isolated_evaporator_consistency`). The repo's closures give 0.97 kg/m2/h for a flat surface and need ~53x exchange
  area (or ~930 W/m2 from the environment) to reach 2.77: not consistent without an unmeasured 3-D enhancement.
- Linear-scaling arithmetic and mode A/B agreement remain what they were: not evidence.
