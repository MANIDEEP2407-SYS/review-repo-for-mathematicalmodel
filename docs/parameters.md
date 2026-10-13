# Parameters

Generated from `src/config/defaults.py::PARAMETER_REGISTRY` and `src/physics/realworld.py::REALWORLD_REGISTRY`
(sustainability parameters live in `src/sustainability/lifecycle.py`). Source tier is one of `measured`, `literature`, `assumed`, `calculated`.
Parameters marked `coupled` are used by the coupled solver only; `legacy` by the legacy solver only.

| name | symbol | unit | default | range | source | solver | notes |
|---|---|---|---|---|---|---|---|
| solar_irradiance | G | W/m^2 | 800 | [100, 1361] | literature | both | Peak clear-sky ~1000 W/m^2; 800 is a realistic tropical average during solar hours. |
| sunlight_hours | t_sun | h | 8 | [1, 14] | assumed | both | Effective peak sunlight hours per day (Kalpar-like site). |
| optical_gain | G_opt | - | 1.5 | [1, 3] | assumed | both | Multiplicative reflector + concentration gain. 1.0 = no reflector. |
| glass_transmittance | tau_g | - | 0.9 | [0.5, 0.95] | literature | both | Solar transmittance of low-iron toughened glass. |
| reflector_reflectivity | rho_r | - | 0.85 | [0, 0.98] | literature | both | Aluminum reflector reflectivity. |
| evaporator_area_geometric | A_geo | m^2 | 1 | [0.05, 100] | assumed | both | Base (raft) area of the evaporator. |
| area_utilization_factor | f_util | - | 0.85 | [0.1, 1] | assumed | both | Effective evaporating fraction of the geometric area (3D cotton wet fraction). |
| number_of_flowers | N_f | - | 25 | [1, 10000] | assumed | both | Number of FeS-cotton flower structures on the raft. |
| flower_diameter | d_f | m | 0.07 | [0.01, 0.3] | assumed | both | Diameter of a single 3D cotton flower (approx 7 cm). |
| eta_optical | eta_opt | - | 0.9 | [0.3, 1] | assumed | both | Optical efficiency: transmittance x absorptance x reflectorlosses. |
| eta_thermal | eta_th | - | 0.65 | [0.1, 0.95] | assumed | legacy | Fraction of absorbed energy that drives evaporation vs. lost. |
| ambient_temperature | T_amb | C | 30 | [-10, 55] | assumed | both | Kalpar-area mean ambient temperature during operation. |
| wind_speed | v_w | m/s | 2 | [0, 20] | assumed | both | Wind speed at glass cover. |
| relative_humidity | RH | - | 0.5 | [0, 1] | assumed | legacy | Ambient relative humidity. |
| insulation_thickness | d_ins | m | 0.05 | [0.005, 0.3] | assumed | both | Bottom / side insulation thickness. |
| insulation_k | k_ins | W/(m K) | 0.04 | [0.02, 0.2] | literature | both | Thermal conductivity of polystyrene foam. |
| h_fg | h_fg | J/kg | 2.4e+06 | [2.2e+06, 2.55e+06] | literature | legacy | Latent heat of vaporization near 50-70 C (temperature-varying). |
| water_density | rho_w | kg/m^3 | 1000 | [960, 1030] | literature | both | Density of freshwater; assumed constant. |
| water_specific_heat | c_p | J/(kg K) | 4186 | [4100, 4220] | literature | both | Specific heat of water. |
| glass_area | A_g | m^2 | 1.1 | [0.05, 120] | assumed | both | Glass cover area, slightly larger than the evaporator due to slope. |
| glass_angle_deg | theta_g | deg | 17.5 | [5, 60] | assumed | both | Cover tilt from horizontal (matches diagram range 15-20 deg). |
| glass_thickness | d_g | m | 0.01 | [0.003, 0.02] | assumed | both | 10 mm toughened glass sheet (matches diagram). |
| glass_k | k_g | W/(m K) | 1 | [0.7, 1.4] | literature | both | Thermal conductivity of soda-lime glass. |
| glass_emissivity | eps_g | - | 0.88 | [0.7, 0.95] | literature | both | Long-wave emissivity of glass. |
| evaporator_emissivity | eps_e | - | 0.95 | [0.7, 1] | literature | both | High absorptance/emissivity of carbon-black-like FeS cotton. |
| mass_transfer_coefficient | k_m | kg/(m^2 s Pa) | 3e-08 | [1e-09, 1e-06] | assumed | legacy | Effective vapour mass-transfer coefficient (order of magnitude for solar stills). |
| empirical_flux | J_emp | kg/(m^2 h) | 1.05 | [0.1, 10] | literature | both (mode B/D cap in coupled) | Empirical evaporation flux. 8.4 kg/m^2/day over 8 h -> 1.05 kg/m^2/h. Cite 396. |
| condensation_coefficient | h_cond | W/(m^2 K) | 8000 | [500, 20000] | assumed | both | Inner-side film-condensation coefficient on glass (order-of-magnitude, not looked up; re-tagged from 'literature' to ASSUMED when the coupled solver made it a real resistance). |
| collection_efficiency | eta_coll | - | 0.9 | [0.5, 1] | assumed | both | Fraction of condensate that reaches collection (drip losses excluded). |
| feed_tds | C_0 | mg/L | 5000 | [100, 60000] | measured | both | Kalpar-area brackish groundwater TDS (representative). |
| feed_volume | V_0 | L | 20 | [0.5, 100000] | assumed | both | Feed water charge per module. |
| salt_saturation_limit | C_sat | mg/L | 250000 | [100000, 360000] | literature | both | Approx solubility limit for NaCl in brine. |
| motive_flow_rate | Q_m | m^3/s | 0.001 | [1e-06, 0.1] | assumed | both | Motive water flow rate through Venturi. |
| venturi_inlet_diameter | D1 | m | 0.025 | [0.005, 0.1] | assumed | both | Venturi inlet diameter. |
| venturi_throat_diameter | D2 | m | 0.01 | [0.002, 0.05] | assumed | both | Venturi throat diameter. |
| venturi_inlet_pressure | P1 | Pa | 101325 | [10000, 500000] | assumed | legacy | Inlet static pressure (atmospheric by default). |
| discharge_coefficient | C_d | - | 0.95 | [0.5, 1] | literature | both | Discharge coefficient for a smooth Venturi. |
| venturi_loss_fraction | eta_v | - | 0.3 | [0, 0.9] | assumed | both | Fraction of theoretical vacuum lost to real-world irreversibilities. |
| gap_convection_coefficient | h_c,gap | W/(m^2 K) | 2 | [0.1, 15] | assumed | coupled | Convective (sensible, air/vapour) coupling evaporator -> glass across the vapour gap. Natural convection in a hot still is typically ~1-4 W/m2K; the radiative part is added separately from the emissivities. Calibrate on prototype. |
| diffusion_path_length | delta | m | 0.02 | [0.002, 0.3] | assumed | coupled | EFFECTIVE vapour diffusion path through air in the gap, natural-convection enhancement included. Geometric gap is 0.1-0.3 m, but convection shortens the effective path to ~1-2 cm (Dunkle-equivalent). Strongest unknown in the coupled model: CALIBRATE ON PROTOTYPE. |
| glass_absorptance | alpha_g | - | 0.05 | [0, 0.3] | assumed | coupled | Fraction of the light crossing the cover that is absorbed in the glass (heats the cover). Low-iron glass is a few percent; thickness dependent, not looked up. |
| atmospheric_pressure | P_atm | Pa | 101325 | [80000, 105000] | literature | coupled | Ambient pressure (standard atmosphere; lower at altitude/weather). |
| motive_water_temperature | T_motive | C | 30 | [5, 60] | assumed | coupled | Temperature of the ejector motive water; its vapour pressure is the hard floor of the ejector (cavitation). Set to the tank/ambient temperature. |
| chamber_leak_pressure_floor | P_leak | Pa | 10000 | [1000, 101325] | assumed | coupled | Lowest absolute pressure the sealed chamber can hold given leaks / air ingress. Measure with a vacuum-decay test; 10 kPa is a guess for a glass-and-gasket cover. |
| pump_efficiency | eta_pump | - | 0.5 | [0.1, 0.9] | assumed | coupled | Wire-to-water efficiency of the motive pump (small centrifugal pump). |
| glass_allowable_stress | sigma_allow | Pa | 2e+07 | [5e+06, 1e+08] | assumed | coupled | Allowable bending stress used ONLY for the cover-vacuum warning. Order of magnitude for annealed glass design loads; NOT a structural design value. |
| evaporation_derating | f_der | - | 1 | [0.01, 1] | calculated | coupled | Multiplier on the evaporator's vapour-side conductance / flux caps (fouling, ageing). 1 = clean. Set by physics.realworld, acts INSIDE the energy balance. |
| wick_supply_flux_limit | J_wick | kg/(m^2 s) | 1 | [1e-09, 1] | calculated | coupled | Water the wick can deliver per m2 of evaporator. 1.0 = no practical limit. Set by physics.realworld (capillary model), acts INSIDE the energy balance. |
| drinking_per_person | q_d | L/(person day) | 4 | [1, 10] | literature | both | WHO minimum drinking water. |
| cooking_per_person | q_c | L/(person day) | 2 | [0.5, 5] | assumed | both | Cooking water allowance. |
| household_size | N_p | - | 4 | [1, 20] | assumed | both | Persons per household (Kalpar mean). |
| safety_factor | SF | - | 1.2 | [1, 2] | assumed | both | Overdesign multiplier. |

## Real-world derating parameters (all assumed)

| name | symbol | unit | default | range | source | notes |
|---|---|---|---|---|---|---|
| pore_radius | r_p | m | 2e-05 | [1e-06, 0.0002] | assumed | Effective capillary radius of the cotton wick. Measure with a wicking-rise test. |
| contact_angle | theta | deg | 30 | [0, 85] | assumed | Water-cotton contact angle; rises with FeS coating hydrophobicity/ageing. |
| wick_porosity | eps_w | - | 0.8 | [0.3, 0.95] | assumed | Void fraction of the wick. |
| wick_tortuosity | tau_w | - | 1.5 | [1, 4] | assumed | Pore path tortuosity. |
| wick_length | L_w | m | 0.05 | [0.005, 1] | assumed | Path length (= lift height) from the water surface to the evaporating tip. |
| wick_area_fraction | phi_w | - | 0.0025 | [1e-05, 0.5] | assumed | Wick flow cross-section per m^2 of evaporator plan area (stems/strands, not the flower surface). |
| day_length_h | D | h | 12 | [6, 14] | assumed | Sunrise-to-sunset length for the half-sine profile. |
| cleaning_interval_days | T_clean | d | 14 | [1, 180] | assumed | Days between glass/wick cleaning. |
| soiling_rate | s | 1/d | 0.003 | [0, 0.05] | assumed | Fractional glass-transmittance loss per day (dust). |
| fouling_timescale_days | tau_f | d | 60 | [5, 1000] | assumed | e-folding time of salt/biofilm yield loss; cleaning resets it. |
| flower_degradation | d_fl | - | 0.15 | [0, 0.8] | assumed | Fractional yield loss by end of flower life (linear); the average over life is half. |
