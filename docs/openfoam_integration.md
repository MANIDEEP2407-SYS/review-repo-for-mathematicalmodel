# OpenFOAM integration (future)

Not implemented. Placeholder for the parameter-mapping document that
Phase 12 will produce once the mathematical simulator is stable.

## Planned parameter mapping (illustrative, subject to change)

| Python parameter        | OpenFOAM equivalent                              |
|-------------------------|--------------------------------------------------|
| `solar_irradiance`      | heat-flux boundary condition on evaporator wall  |
| `ambient_temperature`   | far-field thermal boundary condition             |
| `water_temperature`     | initial / boundary condition on liquid region    |
| `glass_temperature`     | wall thermal condition on glass patch            |
| `evaporator_area`       | geometry / patch definition                      |
| `water_velocity`        | inlet velocity boundary condition                |
| `venturi_pressure`      | pressure boundary condition at throat            |
| `relative_humidity`     | inlet vapour mass-fraction                       |

An existing OpenFOAM case skeleton lives one level up in
`../openfoam_desalination/` (not part of this Python package). It is
NOT considered validated until Phase 12 runs and compares it against
this mathematical simulator's outputs and against reference data.
