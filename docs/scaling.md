# Scaling

How the simulator turns a target water demand into required geometry.

Phase 7 will document:

- Household demand model (drinking + cooking + safety factor).
- Village aggregation (population -> households -> total L/day).
- Sizing equation: `A_required = target_output / predicted_productivity`.
- Module count, glass area, reflector area, footprint.

Productivity input is either the empirical flux the user supplies
(with an explicit source tag) or the value computed by the physics
model — never a hard-coded number.
