"""System aggregator: SystemConfig + SystemResult + simulate().

Steady-state (per-day) simulation of the passive solar desalination
system. All input defaults come from the parameter registry so that
provenance and allowed ranges are enforceable.

Two solvers, selected by ``SystemConfig.coupling``:

* ``"coupled"`` (default): evaporator, vapour gap and glass condenser solved as one
  balance (physics/coupled.py); venturi/ejector pressure feeds the diffusion law.
  ``eta_thermal``, ``h_fg`` (constant), ``mass_transfer_coefficient``, ``relative_humidity``
  and ``venturi_inlet_pressure`` are NOT used on this path.
* ``"legacy"``: the original decoupled model, kept bit-for-bit so old results reproduce.
  Its evaporation/condensation are capped separately and the venturi never reaches yield
  (README section 8 flaws).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from ..config.defaults import PARAMETER_REGISTRY, default as _d
from . import solar as _solar
from . import evaporation as _evap
from . import heat_transfer as _ht
from . import condensation as _cond
from . import mass_balance as _mb
from . import salinity as _sal
from . import venturi as _vt
from . import coupled as _cpl


EVAPORATION_MODES = ("energy_limited", "empirical", "mass_transfer", "hybrid")
COUPLINGS = ("coupled", "legacy")


@dataclass
class SystemConfig:
    # Solar
    solar_irradiance: float = field(default_factory=lambda: _d("solar_irradiance"))
    sunlight_hours: float = field(default_factory=lambda: _d("sunlight_hours"))
    optical_gain: float = field(default_factory=lambda: _d("optical_gain"))
    glass_transmittance: float = field(default_factory=lambda: _d("glass_transmittance"))
    eta_optical: float = field(default_factory=lambda: _d("eta_optical"))
    eta_thermal: float = field(default_factory=lambda: _d("eta_thermal"))

    # Evaporator geometry
    evaporator_area_geometric: float = field(default_factory=lambda: _d("evaporator_area_geometric"))
    area_utilization_factor: float = field(default_factory=lambda: _d("area_utilization_factor"))

    # Thermal boundary
    ambient_temperature: float = field(default_factory=lambda: _d("ambient_temperature"))
    wind_speed: float = field(default_factory=lambda: _d("wind_speed"))
    relative_humidity: float = field(default_factory=lambda: _d("relative_humidity"))
    insulation_thickness: float = field(default_factory=lambda: _d("insulation_thickness"))
    insulation_k: float = field(default_factory=lambda: _d("insulation_k"))
    evaporator_emissivity: float = field(default_factory=lambda: _d("evaporator_emissivity"))

    # Fluid properties
    h_fg: float = field(default_factory=lambda: _d("h_fg"))
    water_density: float = field(default_factory=lambda: _d("water_density"))

    # Glass
    glass_area: float = field(default_factory=lambda: _d("glass_area"))
    glass_thickness: float = field(default_factory=lambda: _d("glass_thickness"))
    glass_k: float = field(default_factory=lambda: _d("glass_k"))
    glass_emissivity: float = field(default_factory=lambda: _d("glass_emissivity"))
    condensation_coefficient: float = field(default_factory=lambda: _d("condensation_coefficient"))
    collection_efficiency: float = field(default_factory=lambda: _d("collection_efficiency"))

    # Solver selection.  evaporation_mode=None resolves to "energy_limited" for the legacy solver
    # (its historical default) and to "mass_transfer" for the coupled solver: a diffusion mode is
    # needed for chamber pressure to matter, and the empirical-flux cap (a concept-diagram number)
    # should not silently set the baseline.
    coupling: str = "coupled"

    # Evaporation modes
    evaporation_mode: Optional[str] = None
    empirical_flux: float = field(default_factory=lambda: _d("empirical_flux"))
    mass_transfer_coefficient: float = field(default_factory=lambda: _d("mass_transfer_coefficient"))

    # Salinity
    feed_tds: float = field(default_factory=lambda: _d("feed_tds"))
    feed_volume: float = field(default_factory=lambda: _d("feed_volume"))
    salt_saturation_limit: float = field(default_factory=lambda: _d("salt_saturation_limit"))

    # Venturi
    venturi_enabled: bool = False
    motive_flow_rate: float = field(default_factory=lambda: _d("motive_flow_rate"))
    venturi_inlet_diameter: float = field(default_factory=lambda: _d("venturi_inlet_diameter"))
    venturi_throat_diameter: float = field(default_factory=lambda: _d("venturi_throat_diameter"))
    venturi_inlet_pressure: float = field(default_factory=lambda: _d("venturi_inlet_pressure"))
    discharge_coefficient: float = field(default_factory=lambda: _d("discharge_coefficient"))
    venturi_loss_fraction: float = field(default_factory=lambda: _d("venturi_loss_fraction"))

    # Coupled-solver parameters (ASSUMED unless the registry says otherwise)
    gap_convection_coefficient: float = field(default_factory=lambda: _d("gap_convection_coefficient"))
    diffusion_path_length: float = field(default_factory=lambda: _d("diffusion_path_length"))
    glass_absorptance: float = field(default_factory=lambda: _d("glass_absorptance"))
    atmospheric_pressure: float = field(default_factory=lambda: _d("atmospheric_pressure"))
    motive_water_temperature: float = field(default_factory=lambda: _d("motive_water_temperature"))
    chamber_leak_pressure_floor: float = field(default_factory=lambda: _d("chamber_leak_pressure_floor"))
    pump_efficiency: float = field(default_factory=lambda: _d("pump_efficiency"))
    glass_allowable_stress: float = field(default_factory=lambda: _d("glass_allowable_stress"))
    evaporation_derating: float = field(default_factory=lambda: _d("evaporation_derating"))
    wick_supply_flux_limit: float = field(default_factory=lambda: _d("wick_supply_flux_limit"))
    water_specific_heat: float = field(default_factory=lambda: _d("water_specific_heat"))
    sensible_heat: bool = True      # energy-consistent feed/condensate bookkeeping (coupled.py, note (e))

    def __post_init__(self):
        if self.evaporation_mode is None:
            self.evaporation_mode = "energy_limited" if self.coupling == "legacy" else "mass_transfer"

    def validate(self) -> List[str]:
        """Range-check every value that has a registered parameter. Returns a list of violation messages."""
        violations = []
        mapping = {
            "solar_irradiance": self.solar_irradiance,
            "sunlight_hours": self.sunlight_hours,
            "optical_gain": self.optical_gain,
            "glass_transmittance": self.glass_transmittance,
            "eta_optical": self.eta_optical,
            "eta_thermal": self.eta_thermal,
            "evaporator_area_geometric": self.evaporator_area_geometric,
            "area_utilization_factor": self.area_utilization_factor,
            "ambient_temperature": self.ambient_temperature,
            "wind_speed": self.wind_speed,
            "relative_humidity": self.relative_humidity,
            "insulation_thickness": self.insulation_thickness,
            "insulation_k": self.insulation_k,
            "evaporator_emissivity": self.evaporator_emissivity,
            "h_fg": self.h_fg,
            "water_density": self.water_density,
            "glass_area": self.glass_area,
            "glass_thickness": self.glass_thickness,
            "glass_k": self.glass_k,
            "glass_emissivity": self.glass_emissivity,
            "condensation_coefficient": self.condensation_coefficient,
            "collection_efficiency": self.collection_efficiency,
            "empirical_flux": self.empirical_flux,
            "mass_transfer_coefficient": self.mass_transfer_coefficient,
            "feed_tds": self.feed_tds,
            "feed_volume": self.feed_volume,
            "salt_saturation_limit": self.salt_saturation_limit,
            "motive_flow_rate": self.motive_flow_rate,
            "venturi_inlet_diameter": self.venturi_inlet_diameter,
            "venturi_throat_diameter": self.venturi_throat_diameter,
            "venturi_inlet_pressure": self.venturi_inlet_pressure,
            "discharge_coefficient": self.discharge_coefficient,
            "venturi_loss_fraction": self.venturi_loss_fraction,
            "gap_convection_coefficient": self.gap_convection_coefficient,
            "diffusion_path_length": self.diffusion_path_length,
            "glass_absorptance": self.glass_absorptance,
            "atmospheric_pressure": self.atmospheric_pressure,
            "motive_water_temperature": self.motive_water_temperature,
            "chamber_leak_pressure_floor": self.chamber_leak_pressure_floor,
            "pump_efficiency": self.pump_efficiency,
            "glass_allowable_stress": self.glass_allowable_stress,
            "evaporation_derating": self.evaporation_derating,
            "wick_supply_flux_limit": self.wick_supply_flux_limit,
            "water_specific_heat": self.water_specific_heat,
        }
        for name, value in mapping.items():
            if name in PARAMETER_REGISTRY:
                try:
                    PARAMETER_REGISTRY[name].validate(value)
                except ValueError as e:
                    violations.append(str(e))
        if self.evaporation_mode not in EVAPORATION_MODES:
            violations.append(f"evaporation_mode='{self.evaporation_mode}' not in {EVAPORATION_MODES}.")
        if self.coupling not in COUPLINGS:
            violations.append(f"coupling='{self.coupling}' not in {COUPLINGS}.")
        return violations


@dataclass
class SystemResult:
    # Areas
    A_geometric: float
    A_effective: float

    # Solar energy
    G_effective: float
    Q_solar_incident_J: float
    Q_absorbed_J: float
    Q_useful_J: float

    # Temperatures (lumped)
    T_ambient_C: float
    T_sky_C: float
    T_evaporator_C: float
    T_glass_C: float

    # Loss breakdown (average W over sunlight_hours)
    Q_convection_W: float
    Q_radiation_W: float
    Q_conduction_W: float

    # Evaporation (kg per session)
    m_evap_energy_limited_kg: float
    m_evap_empirical_kg: float
    m_evap_mass_transfer_kg: float
    m_evap_selected_kg: float
    evaporation_mode_used: str
    limited_by: str

    # Condensation & collection
    m_condensation_capacity_kg: float
    m_freshwater_kg: float
    condensation_limited_by: str
    U_glass: float

    # Mass balance
    feed_kg: float
    brine_kg: float
    recovery_ratio: float
    mass_closure_error_kg: float

    # Salinity
    initial_tds_mg_per_L: float
    brine_tds_mg_per_L: float
    concentration_factor: float
    salt_saturated: bool

    # Venturi
    venturi_enabled: bool
    P_chamber_Pa: float
    saturation_T_shift_C: float

    # Efficiencies
    energy_efficiency: float                 # freshwater latent / incident
    thermal_efficiency_actual: float         # Q_useful / Q_absorbed

    # Productivity
    daily_freshwater_kg: float
    daily_freshwater_L: float
    productivity_L_per_m2_day: float

    # Diagnostics
    warnings: List[str] = field(default_factory=list)
    mode_predictions_kg: Dict[str, float] = field(default_factory=dict)

    # --- coupled-solver outputs (filled with neutral values on the legacy path) ---
    coupling: str = "legacy"
    T_condensate_C: float = 0.0            # condensate film surface T_c (legacy: = T_glass)
    mdot_kg_s: float = 0.0                 # steady vapour flow during sunlight
    P_chamber_set_Pa: float = 0.0          # pressure the ejector/atmosphere sets
    boiling_limited: bool = False          # chamber pressure floats to Psat(T_c): vacuum cannot help
    near_boiling: bool = False
    ejector_power_W: float = 0.0           # parasitic pump electrical power
    ejector_kWh_per_m3: float = 0.0
    ejector_binding_floor: str = "none"
    vacuum_load_kN: float = 0.0            # (P_atm - P_chamber) * A_glass
    vacuum_load_tonnes: float = 0.0
    cover_plate_stress_MPa: float = 0.0    # screening estimate only
    cover_vacuum_implausible: bool = False
    Q_gap_W: float = 0.0
    Q_ideal_cover_W: float = 0.0
    P_absorbed_W: float = 0.0
    solver_converged: bool = True
    solver_residual_max: float = 0.0
    solver_residuals: Dict[str, float] = field(default_factory=dict)
    resistance_share: Dict[str, float] = field(default_factory=dict)


def _select_evaporation(
    cfg: SystemConfig,
    Q_useful: float,
    T_evap_C: float,
    A_eff: float,
):
    energy = _evap.evaporate_energy_limited(Q_useful, cfg.h_fg, A_eff, cfg.sunlight_hours)
    emp = _evap.evaporate_empirical(cfg.empirical_flux, A_eff, cfg.sunlight_hours)
    mt = _evap.evaporate_mass_transfer(
        T_evap_C, cfg.ambient_temperature, cfg.relative_humidity,
        cfg.mass_transfer_coefficient, A_eff, cfg.sunlight_hours,
    )
    hybrid = _evap.evaporate_hybrid(
        Q_useful, cfg.h_fg, T_evap_C, cfg.ambient_temperature,
        cfg.relative_humidity, cfg.mass_transfer_coefficient,
        A_eff, cfg.sunlight_hours,
    )
    catalog = {
        "energy_limited": energy,
        "empirical": emp,
        "mass_transfer": mt,
        "hybrid": hybrid,
    }
    return catalog[cfg.evaporation_mode], catalog


def simulate(cfg: Optional[SystemConfig] = None) -> SystemResult:
    """Run one sunlight session.  Dispatches on cfg.coupling ("coupled" default | "legacy")."""
    if cfg is None:
        cfg = SystemConfig()
    if cfg.coupling == "legacy":
        return _simulate_legacy(cfg)
    return _simulate_coupled(cfg)


def _simulate_legacy(cfg: SystemConfig) -> SystemResult:
    violations = cfg.validate()

    # Areas
    A_geo = cfg.evaporator_area_geometric
    A_eff = A_geo * cfg.area_utilization_factor

    # Solar energy
    solar_in = _solar.compute_solar_input(
        irradiance=cfg.solar_irradiance,
        area_geometric=A_geo,
        sunlight_hours=cfg.sunlight_hours,
        optical_gain=cfg.optical_gain,
        glass_transmittance=cfg.glass_transmittance,
        eta_optical=cfg.eta_optical,
    )
    Q_absorbed = solar_in.Q_absorbed
    Q_useful = Q_absorbed * cfg.eta_thermal

    # Convert energies to average power over sunlight_hours for thermal solve.
    duration_s = cfg.sunlight_hours * 3600.0
    P_absorbed_W = Q_absorbed / duration_s
    P_useful_W = Q_useful / duration_s
    # Effective absorbed power ACTS on A_eff for the lumped balance loss terms.
    thermal = _ht.solve_evaporator_temperature(
        P_absorbed_W=P_absorbed_W,
        P_evaporation_W=P_useful_W,
        T_amb_C=cfg.ambient_temperature,
        area=A_eff,
        emissivity=cfg.evaporator_emissivity,
        wind_speed=cfg.wind_speed,
        k_ins=cfg.insulation_k,
        d_ins=cfg.insulation_thickness,
    )

    # Evaporation predictions across all modes
    selected, catalog = _select_evaporation(cfg, Q_useful, thermal.T_evap_C, A_eff)

    # Condensation
    cond = _cond.compute_condensation(
        m_evap_kg=selected.m_evap_kg,
        T_vapor_C=thermal.T_evap_C,
        T_glass_C=thermal.T_glass_C,
        T_ambient_C=cfg.ambient_temperature,
        glass_area=cfg.glass_area,
        glass_thickness=cfg.glass_thickness,
        glass_k=cfg.glass_k,
        glass_emissivity=cfg.glass_emissivity,
        wind_speed=cfg.wind_speed,
        h_cond_inner=cfg.condensation_coefficient,
        h_fg=cfg.h_fg,
        sunlight_hours=cfg.sunlight_hours,
        collection_efficiency=cfg.collection_efficiency,
    )

    # Mass balance
    feed_kg = cfg.feed_volume * (cfg.water_density / 1000.0)  # L -> kg
    mb = _mb.compute_mass_balance(
        feed_kg=feed_kg,
        evaporated_kg=selected.m_evap_kg,
        condensed_capacity_kg=cond.m_cond_capacity_kg,
        freshwater_collected_kg=cond.m_freshwater_collected_kg,
    )

    # Salinity
    sal = _sal.compute_salinity(
        feed_volume_L=cfg.feed_volume,
        feed_tds_mg_per_L=cfg.feed_tds,
        m_evap_kg=selected.m_evap_kg,
        water_density_kg_per_m3=cfg.water_density,
        saturation_limit_mg_per_L=cfg.salt_saturation_limit,
    )

    # Venturi
    vt = _vt.compute_venturi(
        enabled=cfg.venturi_enabled,
        motive_flow_m3_s=cfg.motive_flow_rate,
        inlet_diameter_m=cfg.venturi_inlet_diameter,
        throat_diameter_m=cfg.venturi_throat_diameter,
        inlet_pressure_Pa=cfg.venturi_inlet_pressure,
        fluid_density=cfg.water_density,
        discharge_coefficient=cfg.discharge_coefficient,
        loss_fraction=cfg.venturi_loss_fraction,
    )

    # Efficiencies
    Q_fresh_latent = mb.freshwater_collected_kg * cfg.h_fg
    energy_eff = Q_fresh_latent / solar_in.Q_incident if solar_in.Q_incident > 0 else 0.0
    thermal_eff_actual = Q_useful / Q_absorbed if Q_absorbed > 0 else 0.0

    daily_kg = mb.freshwater_collected_kg
    daily_L = daily_kg / (cfg.water_density / 1000.0)
    productivity = daily_L / A_geo if A_geo > 0 else 0.0

    # Warnings
    warnings = list(violations)
    if selected.m_evap_kg > 0 and cond.limited_by != "evaporation":
        warnings.append(
            f"Condensation is limiting production (limited_by={cond.limited_by}). "
            f"Increase glass area, lower T_glass, or accept lower yield."
        )
    if energy_eff > 1.0:
        warnings.append("Energy efficiency > 100 %: model inconsistency, check inputs.")
    if sal.saturated:
        warnings.append(
            f"Brine TDS {sal.brine_tds_mg_per_L:.0f} mg/L exceeds saturation "
            f"{cfg.salt_saturation_limit:.0f} mg/L. Salt crystallization risk."
        )
    if abs(mb.closure_error_kg) > 1e-3:
        warnings.append(f"Mass closure error {mb.closure_error_kg:.3e} kg/day.")
    if vt.enabled and vt.warning:
        warnings.append(vt.warning)

    return SystemResult(
        A_geometric=A_geo,
        A_effective=A_eff,
        G_effective=solar_in.G_effective,
        Q_solar_incident_J=solar_in.Q_incident,
        Q_absorbed_J=Q_absorbed,
        Q_useful_J=Q_useful,
        T_ambient_C=cfg.ambient_temperature,
        T_sky_C=thermal.T_sky_C,
        T_evaporator_C=thermal.T_evap_C,
        T_glass_C=thermal.T_glass_C,
        Q_convection_W=thermal.Q_convection_W,
        Q_radiation_W=thermal.Q_radiation_W,
        Q_conduction_W=thermal.Q_conduction_W,
        m_evap_energy_limited_kg=catalog["energy_limited"].m_evap_kg,
        m_evap_empirical_kg=catalog["empirical"].m_evap_kg,
        m_evap_mass_transfer_kg=catalog["mass_transfer"].m_evap_kg,
        m_evap_selected_kg=selected.m_evap_kg,
        evaporation_mode_used=cfg.evaporation_mode,
        limited_by=selected.limited_by,
        m_condensation_capacity_kg=cond.m_cond_capacity_kg,
        m_freshwater_kg=mb.freshwater_collected_kg,
        condensation_limited_by=cond.limited_by,
        U_glass=cond.U_glass,
        feed_kg=feed_kg,
        brine_kg=mb.brine_kg,
        recovery_ratio=mb.recovery_ratio,
        mass_closure_error_kg=mb.closure_error_kg,
        initial_tds_mg_per_L=sal.initial_tds_mg_per_L,
        brine_tds_mg_per_L=sal.brine_tds_mg_per_L,
        concentration_factor=sal.concentration_factor,
        salt_saturated=sal.saturated,
        venturi_enabled=vt.enabled,
        P_chamber_Pa=vt.P_chamber_effective_Pa,
        saturation_T_shift_C=vt.saturation_temperature_shift_C,
        energy_efficiency=energy_eff,
        thermal_efficiency_actual=thermal_eff_actual,
        daily_freshwater_kg=daily_kg,
        daily_freshwater_L=daily_L,
        productivity_L_per_m2_day=productivity,
        warnings=warnings,
        mode_predictions_kg={k: v.m_evap_kg for k, v in catalog.items()},
        coupling="legacy",
        T_condensate_C=thermal.T_glass_C,
        mdot_kg_s=selected.m_evap_kg / duration_s,
        P_chamber_set_Pa=vt.P_chamber_effective_Pa,
        ejector_power_W=(cfg.motive_flow_rate * vt.dP_theoretical_Pa / max(cfg.pump_efficiency, 1e-3)
                         if vt.enabled else 0.0),
        vacuum_load_kN=(max(cfg.atmospheric_pressure - vt.P_chamber_effective_Pa, 0.0)
                        * cfg.glass_area / 1e3),
        P_absorbed_W=P_absorbed_W,
    )


def _simulate_coupled(cfg: SystemConfig) -> SystemResult:
    violations = cfg.validate()
    t_s = cfg.sunlight_hours * 3600.0

    A_geo = cfg.evaporator_area_geometric
    A_eff = A_geo * cfg.area_utilization_factor
    solar_in = _solar.compute_solar_input(
        irradiance=cfg.solar_irradiance, area_geometric=A_geo, sunlight_hours=cfg.sunlight_hours,
        optical_gain=cfg.optical_gain, glass_transmittance=cfg.glass_transmittance,
        eta_optical=cfg.eta_optical,
    )
    P_abs = solar_in.Q_absorbed / t_s
    # light absorbed in the cover: reflector-boosted irradiance crossing the cover area
    P_glass_abs = cfg.glass_absorptance * cfg.solar_irradiance * cfg.optical_gain * cfg.glass_area

    ej = _vt.compute_ejector(
        enabled=cfg.venturi_enabled, motive_flow_m3_s=cfg.motive_flow_rate,
        inlet_diameter_m=cfg.venturi_inlet_diameter, throat_diameter_m=cfg.venturi_throat_diameter,
        P_atm_Pa=cfg.atmospheric_pressure, fluid_density=cfg.water_density,
        discharge_coefficient=cfg.discharge_coefficient, loss_fraction=cfg.venturi_loss_fraction,
        motive_water_T_C=cfg.motive_water_temperature, leak_floor_Pa=cfg.chamber_leak_pressure_floor,
        pump_efficiency=cfg.pump_efficiency,
    )
    T_sky = _ht.T_sky_celsius(cfg.ambient_temperature)
    params = _cpl.CoupledParams(
        P_abs_W=P_abs, P_glass_abs_W=P_glass_abs, A_eff=A_eff, A_loss=A_geo, A_glass=cfg.glass_area,
        T_amb_C=cfg.ambient_temperature, T_sky_C=T_sky,
        h_conv_out=_ht.h_convection_wind(cfg.wind_speed),
        eps_e=cfg.evaporator_emissivity, eps_g=cfg.glass_emissivity,
        k_ins=cfg.insulation_k, d_ins=cfg.insulation_thickness,
        h_gap_conv=cfg.gap_convection_coefficient, delta=cfg.diffusion_path_length,
        d_g=cfg.glass_thickness, k_g=cfg.glass_k, h_cond=cfg.condensation_coefficient,
        P_chamber_Pa=ej.P_chamber_set_Pa, J_emp_kg_m2_s=cfg.empirical_flux / 3600.0,
        f_der=cfg.evaporation_derating, supply_kg_m2_s=cfg.wick_supply_flux_limit,
        cp_w=cfg.water_specific_heat, sensible=cfg.sensible_heat,
    )
    sol = _cpl.solve(params, cfg.evaporation_mode)
    st, sA = sol.state, sol.state_ideal

    m_kg = {k: v * t_s for k, v in sol.mdot_modes.items()}
    m_sel = m_kg[cfg.evaporation_mode]
    m_cap = sA.mdot * t_s                    # what this cover can condense with an ideal evaporator

    feed_kg = cfg.feed_volume * (cfg.water_density / 1000.0)
    mb = _mb.compute_mass_balance(
        feed_kg=feed_kg, evaporated_kg=m_sel, condensed_capacity_kg=m_cap,
        freshwater_collected_kg=m_sel * cfg.collection_efficiency,   # yield = mdot * t * eta_coll
    )
    sal = _sal.compute_salinity(
        feed_volume_L=cfg.feed_volume, feed_tds_mg_per_L=cfg.feed_tds, m_evap_kg=m_sel,
        water_density_kg_per_m3=cfg.water_density, saturation_limit_mg_per_L=cfg.salt_saturation_limit,
    )

    load = _vt.cover_vacuum_load(cfg.atmospheric_pressure, sol.P_eff_Pa, cfg.glass_area,
                                 cfg.glass_thickness, cfg.glass_allowable_stress)
    Q_latent = st.mdot * _cpl.hfg(st.T_e) * t_s           # latent heat actually delivered as vapour
    Q_fresh_latent = mb.freshwater_collected_kg * _cpl.hfg(st.T_e)
    energy_eff = Q_fresh_latent / solar_in.Q_incident if solar_in.Q_incident > 0 else 0.0
    thermal_eff = Q_latent / solar_in.Q_absorbed if solar_in.Q_absorbed > 0 else 0.0
    daily_kg = mb.freshwater_collected_kg
    daily_L = daily_kg / (cfg.water_density / 1000.0)
    productivity = daily_L / A_geo if A_geo > 0 else 0.0
    U_total = 1.0 / (1.0 / cfg.condensation_coefficient + 1.0 / st.U_out)
    res_max = max(st.residuals.values()) if st.residuals else 0.0
    cond_limited = sol.limited_by if sol.limited_by in ("film", "glass_uvalue", "boiling") else "evaporation"

    warnings = list(violations) + list(sol.warnings)
    if cond_limited in ("film", "glass_uvalue"):
        warnings.append(
            f"Cover rejection takes {sol.resistance_share['glass_uvalue']*100:.0f} % of the temperature budget "
            f"(T_gi = {st.T_gi:.1f} C). Diagnostic only: in the coupled model more glass can raise OR lower "
            "yield depending on the vapour-gap resistance (see docs/physics.md).")
    if cfg.glass_transmittance + cfg.glass_absorptance > 1.0 + 1e-9:
        warnings.append(
            f"glass_transmittance ({cfg.glass_transmittance:.2f}) + glass_absorptance ({cfg.glass_absorptance:.2f}) "
            "> 1: the cover would create energy (reflectance < 0). Lower one of them.")
    if st.T_e >= st.T_gi and Q_latent > solar_in.Q_absorbed * (1.0 + 1e-9) and st.T_e >= cfg.ambient_temperature:
        warnings.append("Latent energy exceeds absorbed energy: model inconsistency, check inputs.")
    if sal.saturated:
        warnings.append(f"Brine TDS {sal.brine_tds_mg_per_L:.0f} mg/L exceeds saturation "
                        f"{cfg.salt_saturation_limit:.0f} mg/L. Salt crystallization risk.")
    if abs(mb.closure_error_kg) > 1e-3:
        warnings.append(f"Mass closure error {mb.closure_error_kg:.3e} kg/day.")
    if ej.enabled and ej.warning:
        warnings.append(ej.warning)
    if load.message and cfg.venturi_enabled:
        warnings.append(load.message)
    if sol.P_eff_Pa > cfg.atmospheric_pressure * (1.0 + 1e-9):
        warnings.append(f"Chamber would sit at {sol.P_eff_Pa/1e3:.0f} kPa, above ambient: steam would vent. "
                        "Cover too small for the absorbed power; results not meaningful.")

    L_prod = max(daily_L, 1e-12)
    ej_kwh = ej.pump_power_W * t_s / 3.6e6
    return SystemResult(
        A_geometric=A_geo, A_effective=A_eff, G_effective=solar_in.G_effective,
        Q_solar_incident_J=solar_in.Q_incident, Q_absorbed_J=solar_in.Q_absorbed,
        Q_useful_J=Q_latent,
        T_ambient_C=cfg.ambient_temperature, T_sky_C=T_sky,
        T_evaporator_C=st.T_e, T_glass_C=st.T_gi,
        Q_convection_W=st.Q_gap_conv, Q_radiation_W=st.Q_gap_rad, Q_conduction_W=st.Q_ins,
        m_evap_energy_limited_kg=m_kg["energy_limited"], m_evap_empirical_kg=m_kg["empirical"],
        m_evap_mass_transfer_kg=m_kg["mass_transfer"], m_evap_selected_kg=m_sel,
        evaporation_mode_used=cfg.evaporation_mode, limited_by=sol.limited_by,
        m_condensation_capacity_kg=m_cap, m_freshwater_kg=mb.freshwater_collected_kg,
        condensation_limited_by=cond_limited, U_glass=U_total,
        feed_kg=feed_kg, brine_kg=mb.brine_kg, recovery_ratio=mb.recovery_ratio,
        mass_closure_error_kg=mb.closure_error_kg,
        initial_tds_mg_per_L=sal.initial_tds_mg_per_L, brine_tds_mg_per_L=sal.brine_tds_mg_per_L,
        concentration_factor=sal.concentration_factor, salt_saturated=sal.saturated,
        venturi_enabled=cfg.venturi_enabled, P_chamber_Pa=sol.P_eff_Pa,
        saturation_T_shift_C=_cpl.t_sat_C(cfg.atmospheric_pressure) - _cpl.t_sat_C(sol.P_eff_Pa),
        energy_efficiency=energy_eff, thermal_efficiency_actual=thermal_eff,
        daily_freshwater_kg=daily_kg, daily_freshwater_L=daily_L, productivity_L_per_m2_day=productivity,
        warnings=warnings, mode_predictions_kg=dict(m_kg),
        coupling="coupled", T_condensate_C=st.T_c, mdot_kg_s=st.mdot,
        P_chamber_set_Pa=ej.P_chamber_set_Pa, boiling_limited=sol.boiling_limited,
        near_boiling=sol.near_boiling, ejector_power_W=ej.pump_power_W,
        ejector_kWh_per_m3=(ej_kwh / (L_prod / 1000.0) if cfg.venturi_enabled else 0.0),
        ejector_binding_floor=ej.binding_floor,
        vacuum_load_kN=load.load_kN, vacuum_load_tonnes=load.equivalent_mass_t,
        cover_plate_stress_MPa=load.plate_stress_Pa / 1e6, cover_vacuum_implausible=load.implausible,
        Q_gap_W=st.Q_gap, Q_ideal_cover_W=sA.mdot * _cpl.hfg(sA.T_e), P_absorbed_W=P_abs,
        solver_converged=st.converged and sA.converged, solver_residual_max=res_max,
        solver_residuals=dict(st.residuals), resistance_share=dict(sol.resistance_share),
    )
