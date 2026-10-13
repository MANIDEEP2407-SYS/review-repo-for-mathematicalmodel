"""Parameter registry with full provenance.

Every parameter carries name, symbol, unit, default, allowed range and a
source tier (measured/literature/assumed/calculated). Values here are the
scientifically defensible defaults; SystemConfig lets the user override
any of them at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class ParameterSource(str, Enum):
    MEASURED = "measured"
    LITERATURE = "literature"
    ASSUMED = "assumed"
    CALCULATED = "calculated"


@dataclass(frozen=True)
class Parameter:
    name: str
    symbol: str
    unit: str
    default: float
    allowed_range: Tuple[float, float]
    source: ParameterSource
    notes: str = ""

    def validate(self, value: float) -> float:
        lo, hi = self.allowed_range
        if not (lo <= value <= hi):
            raise ValueError(
                f"{self.name} = {value} {self.unit} is outside allowed "
                f"range [{lo}, {hi}] {self.unit}."
            )
        return float(value)


PARAMETER_REGISTRY: Dict[str, Parameter] = {}


def register(param: Parameter) -> Parameter:
    if param.name in PARAMETER_REGISTRY:
        raise KeyError(f"Parameter '{param.name}' already registered.")
    PARAMETER_REGISTRY[param.name] = param
    return param


def get(name: str) -> Parameter:
    if name not in PARAMETER_REGISTRY:
        raise KeyError(f"Parameter '{name}' is not registered.")
    return PARAMETER_REGISTRY[name]


def default(name: str) -> float:
    return get(name).default


_L = ParameterSource.LITERATURE
_A = ParameterSource.ASSUMED
_M = ParameterSource.MEASURED
_C = ParameterSource.CALCULATED


_STANDARD_PARAMETERS = [
    # ---------- Solar ----------
    Parameter("solar_irradiance", "G", "W/m^2", 800.0, (100.0, 1361.0), _L,
              "Peak clear-sky ~1000 W/m^2; 800 is a realistic tropical average during solar hours."),
    Parameter("sunlight_hours", "t_sun", "h", 8.0, (1.0, 14.0), _A,
              "Effective peak sunlight hours per day (Kalpar-like site)."),
    Parameter("optical_gain", "G_opt", "-", 1.5, (1.0, 3.0), _A,
              "Multiplicative reflector + concentration gain. 1.0 = no reflector."),
    Parameter("glass_transmittance", "tau_g", "-", 0.90, (0.5, 0.95), _L,
              "Solar transmittance of low-iron toughened glass."),
    Parameter("reflector_reflectivity", "rho_r", "-", 0.85, (0.0, 0.98), _L,
              "Aluminum reflector reflectivity."),

    # ---------- Evaporator geometry ----------
    Parameter("evaporator_area_geometric", "A_geo", "m^2", 1.0, (0.05, 100.0), _A,
              "Base (raft) area of the evaporator."),
    Parameter("area_utilization_factor", "f_util", "-", 0.85, (0.1, 1.0), _A,
              "Effective evaporating fraction of the geometric area (3D cotton wet fraction)."),
    Parameter("number_of_flowers", "N_f", "-", 25.0, (1.0, 10000.0), _A,
              "Number of FeS-cotton flower structures on the raft."),
    Parameter("flower_diameter", "d_f", "m", 0.07, (0.01, 0.30), _A,
              "Diameter of a single 3D cotton flower (approx 7 cm)."),

    # ---------- Thermal / efficiencies ----------
    Parameter("eta_optical", "eta_opt", "-", 0.90, (0.3, 1.0), _A,
              "Optical efficiency: transmittance x absorptance x reflectorlosses."),
    Parameter("eta_thermal", "eta_th", "-", 0.65, (0.1, 0.95), _A,
              "Fraction of absorbed energy that drives evaporation vs. lost."),
    Parameter("ambient_temperature", "T_amb", "C", 30.0, (-10.0, 55.0), _A,
              "Kalpar-area mean ambient temperature during operation."),
    Parameter("wind_speed", "v_w", "m/s", 2.0, (0.0, 20.0), _A,
              "Wind speed at glass cover."),
    Parameter("relative_humidity", "RH", "-", 0.50, (0.0, 1.0), _A,
              "Ambient relative humidity."),
    Parameter("insulation_thickness", "d_ins", "m", 0.05, (0.005, 0.30), _A,
              "Bottom / side insulation thickness."),
    Parameter("insulation_k", "k_ins", "W/(m K)", 0.04, (0.02, 0.20), _L,
              "Thermal conductivity of polystyrene foam."),

    # ---------- Water properties ----------
    Parameter("h_fg", "h_fg", "J/kg", 2.40e6, (2.20e6, 2.55e6), _L,
              "Latent heat of vaporization near 50-70 C (temperature-varying)."),
    Parameter("water_density", "rho_w", "kg/m^3", 1000.0, (960.0, 1030.0), _L,
              "Density of freshwater; assumed constant."),
    Parameter("water_specific_heat", "c_p", "J/(kg K)", 4186.0, (4100.0, 4220.0), _L,
              "Specific heat of water."),

    # ---------- Glass cover ----------
    Parameter("glass_area", "A_g", "m^2", 1.1, (0.05, 120.0), _A,
              "Glass cover area, slightly larger than the evaporator due to slope."),
    Parameter("glass_angle_deg", "theta_g", "deg", 17.5, (5.0, 60.0), _A,
              "Cover tilt from horizontal (matches diagram range 15-20 deg)."),
    Parameter("glass_thickness", "d_g", "m", 0.010, (0.003, 0.020), _A,
              "10 mm toughened glass sheet (matches diagram)."),
    Parameter("glass_k", "k_g", "W/(m K)", 1.0, (0.7, 1.4), _L,
              "Thermal conductivity of soda-lime glass."),
    Parameter("glass_emissivity", "eps_g", "-", 0.88, (0.7, 0.95), _L,
              "Long-wave emissivity of glass."),
    Parameter("evaporator_emissivity", "eps_e", "-", 0.95, (0.7, 1.0), _L,
              "High absorptance/emissivity of carbon-black-like FeS cotton."),

    # ---------- Mass transfer / evaporation ----------
    Parameter("mass_transfer_coefficient", "k_m", "kg/(m^2 s Pa)", 3.0e-8,
              (1.0e-9, 1.0e-6), _A,
              "Effective vapour mass-transfer coefficient (order of magnitude for solar stills)."),
    Parameter("empirical_flux", "J_emp", "kg/(m^2 h)", 1.05, (0.1, 10.0), _L,
              "Empirical evaporation flux. 8.4 kg/m^2/day over 8 h -> 1.05 kg/m^2/h. Cite 396."),

    # ---------- Condensation ----------
    Parameter("condensation_coefficient", "h_cond", "W/(m^2 K)", 8000.0, (500.0, 20000.0), _A,
              "Inner-side film-condensation coefficient on glass (order-of-magnitude, not looked up; re-tagged "
              "from 'literature' to ASSUMED when the coupled solver made it a real resistance)."),
    Parameter("collection_efficiency", "eta_coll", "-", 0.90, (0.5, 1.0), _A,
              "Fraction of condensate that reaches collection (drip losses excluded)."),

    # ---------- Salinity / feed ----------
    Parameter("feed_tds", "C_0", "mg/L", 5000.0, (100.0, 60000.0), _M,
              "Kalpar-area brackish groundwater TDS (representative)."),
    Parameter("feed_volume", "V_0", "L", 20.0, (0.5, 100000.0), _A,
              "Feed water charge per module."),
    Parameter("salt_saturation_limit", "C_sat", "mg/L", 250000.0, (100000.0, 360000.0), _L,
              "Approx solubility limit for NaCl in brine."),

    # ---------- Venturi ----------
    Parameter("motive_flow_rate", "Q_m", "m^3/s", 1.0e-3, (1.0e-6, 1.0e-1), _A,
              "Motive water flow rate through Venturi."),
    Parameter("venturi_inlet_diameter", "D1", "m", 0.025, (0.005, 0.100), _A,
              "Venturi inlet diameter."),
    Parameter("venturi_throat_diameter", "D2", "m", 0.010, (0.002, 0.050), _A,
              "Venturi throat diameter."),
    Parameter("venturi_inlet_pressure", "P1", "Pa", 101325.0, (10000.0, 500000.0), _A,
              "Inlet static pressure (atmospheric by default)."),
    Parameter("discharge_coefficient", "C_d", "-", 0.95, (0.5, 1.0), _L,
              "Discharge coefficient for a smooth Venturi."),
    Parameter("venturi_loss_fraction", "eta_v", "-", 0.30, (0.0, 0.9), _A,
              "Fraction of theoretical vacuum lost to real-world irreversibilities."),

    # ---------- Coupled solver (added with coupling="coupled"; see docs/physics.md) ----------
    # Everything below is ASSUMED unless stated: none of it has been measured on a prototype.
    Parameter("gap_convection_coefficient", "h_c,gap", "W/(m^2 K)", 2.0, (0.1, 15.0), _A,
              "Convective (sensible, air/vapour) coupling evaporator -> glass across the vapour gap. "
              "Natural convection in a hot still is typically ~1-4 W/m2K; the radiative part is added "
              "separately from the emissivities. Calibrate on prototype."),
    Parameter("diffusion_path_length", "delta", "m", 0.02, (0.002, 0.30), _A,
              "EFFECTIVE vapour diffusion path through air in the gap, natural-convection enhancement "
              "included. Geometric gap is 0.1-0.3 m, but convection shortens the effective path to "
              "~1-2 cm (Dunkle-equivalent). Strongest unknown in the coupled model: CALIBRATE ON PROTOTYPE."),
    Parameter("glass_absorptance", "alpha_g", "-", 0.05, (0.0, 0.30), _A,
              "Fraction of the light crossing the cover that is absorbed in the glass (heats the cover). "
              "Low-iron glass is a few percent; thickness dependent, not looked up."),
    Parameter("atmospheric_pressure", "P_atm", "Pa", 101325.0, (80000.0, 105000.0), _L,
              "Ambient pressure (standard atmosphere; lower at altitude/weather)."),
    Parameter("motive_water_temperature", "T_motive", "C", 30.0, (5.0, 60.0), _A,
              "Temperature of the ejector motive water; its vapour pressure is the hard floor of the "
              "ejector (cavitation). Set to the tank/ambient temperature."),
    Parameter("chamber_leak_pressure_floor", "P_leak", "Pa", 10000.0, (1000.0, 101325.0), _A,
              "Lowest absolute pressure the sealed chamber can hold given leaks / air ingress. "
              "Measure with a vacuum-decay test; 10 kPa is a guess for a glass-and-gasket cover."),
    Parameter("pump_efficiency", "eta_pump", "-", 0.50, (0.1, 0.9), _A,
              "Wire-to-water efficiency of the motive pump (small centrifugal pump)."),
    Parameter("glass_allowable_stress", "sigma_allow", "Pa", 2.0e7, (5.0e6, 1.0e8), _A,
              "Allowable bending stress used ONLY for the cover-vacuum warning. Order of magnitude for "
              "annealed glass design loads; NOT a structural design value."),
    Parameter("evaporation_derating", "f_der", "-", 1.0, (0.01, 1.0), ParameterSource.CALCULATED,
              "Multiplier on the evaporator's vapour-side conductance / flux caps (fouling, ageing). "
              "1 = clean. Set by physics.realworld, acts INSIDE the energy balance."),
    Parameter("wick_supply_flux_limit", "J_wick", "kg/(m^2 s)", 1.0, (1.0e-9, 1.0), ParameterSource.CALCULATED,
              "Water the wick can deliver per m2 of evaporator. 1.0 = no practical limit. Set by "
              "physics.realworld (capillary model), acts INSIDE the energy balance."),

    # ---------- Household / village demand ----------
    Parameter("drinking_per_person", "q_d", "L/(person day)", 4.0, (1.0, 10.0), _L,
              "WHO minimum drinking water."),
    Parameter("cooking_per_person", "q_c", "L/(person day)", 2.0, (0.5, 5.0), _A,
              "Cooking water allowance."),
    Parameter("household_size", "N_p", "-", 4.0, (1.0, 20.0), _A,
              "Persons per household (Kalpar mean)."),
    Parameter("safety_factor", "SF", "-", 1.20, (1.0, 2.0), _A,
              "Overdesign multiplier."),
]

for _p in _STANDARD_PARAMETERS:
    register(_p)
