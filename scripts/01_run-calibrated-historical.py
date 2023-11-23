#!/usr/bin/env python

"""Run calibrated historical"""

# Here we run one scenario - the historical - from 1750 to 2014 using the
# CMIP6 emissions and the most recent fair calibration. We then dump this out
# including the GHG states as a restart, so we don't need to keep rerunning
# the historical period when comparing scenarios.

# Internal variability is OFF, since we currently don't have the prescribed
# internal variability inputs.

import os

import numpy as np
import pandas as pd
import pooch
import xarray as xr
from dotenv import load_dotenv
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

load_dotenv()
datadir = os.getenv("DATADIR")

scenarios = ['historical']

df_solar = pd.read_csv(
    "../data/solar_erf_timebounds.csv", index_col="year"
)
df_volcanic = pd.read_csv(
    "../data/volcanic_ERF_1750-2101_timebounds.csv", index_col='timebounds'
)

volcanic_forcing = df_volcanic["erf"].loc[1750:2015].values
solar_forcing = df_solar["erf"].loc[1750:2015].values

df_methane = pd.read_csv(
    "../calibrations/v1.3.0/CH4_lifetime.csv",
    index_col=0,
)

df_configs = pd.read_csv(
    "../calibrations/v1.3.0/calibrated_constrained_parameters.csv",
    index_col=0,
)
valid_all = df_configs.index

trend_shape = np.linspace(0, 1, 271)[:266]

f = FAIR(ch4_method="Thornhill2021")
f.define_time(1750, 2015, 1)
f.define_scenarios(scenarios)
f.define_configs(valid_all)
species, properties = read_properties()
species.remove("NOx aviation")
species.remove("Contrails")
f.define_species(species, properties)
f.allocate()

f.fill_from_rcmip()

# solar and volcanic forcing
fill(
    f.forcing,
    volcanic_forcing[:, None, None] * df_configs["fscale_Volcanic"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    solar_forcing[:, None, None] * df_configs["fscale_solar_amplitude"].values.squeeze()
    + trend_shape[:, None, None] * df_configs["fscale_solar_trend"].values.squeeze(),
    specie="Solar",
)

# climate response
fill(f.climate_configs["ocean_heat_capacity"], df_configs.loc[:, "clim_c1":"clim_c3"].values)
fill(
    f.climate_configs["ocean_heat_transfer"],
    df_configs.loc[:, "clim_kappa1":"clim_kappa3"].values,
)  # not massively robust, since relies on kappa1, kappa2, kappa3 being in adjacent columns
fill(f.climate_configs["deep_ocean_efficacy"], df_configs["clim_epsilon"].values.squeeze())
fill(f.climate_configs["gamma_autocorrelation"], df_configs["clim_gamma"].values.squeeze())
fill(f.climate_configs["sigma_eta"], df_configs["clim_sigma_eta"].values.squeeze())
fill(f.climate_configs["sigma_xi"], df_configs["clim_sigma_xi"].values.squeeze())
fill(f.climate_configs["seed"], df_configs["seed"])
fill(f.climate_configs["stochastic_run"], False)
fill(f.climate_configs["use_seed"], True)
fill(f.climate_configs["forcing_4co2"], df_configs["clim_F_4xCO2"])

# species level
f.fill_species_configs()

# carbon cycle
fill(f.species_configs["iirf_0"], df_configs["cc_r0"].values.squeeze(), specie="CO2")
fill(
    f.species_configs["iirf_airborne"], df_configs["cc_rA"].values.squeeze(), specie="CO2"
)
fill(f.species_configs["iirf_uptake"], df_configs["cc_rU"].values.squeeze(), specie="CO2")
fill(
    f.species_configs["iirf_temperature"],
    df_configs["cc_rT"].values.squeeze(),
    specie="CO2",
)

# aerosol indirect
fill(f.species_configs["aci_scale"], df_configs["aci_beta"].values.squeeze())
fill(
    f.species_configs["aci_shape"],
    df_configs["aci_shape_so2"].values.squeeze(),
    specie="Sulfur",
)
fill(
    f.species_configs["aci_shape"], df_configs["aci_shape_bc"].values.squeeze(), specie="BC"
)
fill(
    f.species_configs["aci_shape"], df_configs["aci_shape_oc"].values.squeeze(), specie="OC"
)

# methane lifetime baseline and sensitivity
fill(
    f.species_configs["unperturbed_lifetime"],
    df_methane.loc["historical_best", "base"],
    specie="CH4",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "CH4"],
    specie="CH4",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "N2O"],
    specie="N2O",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "VOC"],
    specie="VOC",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "NOx"],
    specie="NOx",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "HC"],
    specie="Equivalent effective stratospheric chlorine",
)
fill(
    f.species_configs["lifetime_temperature_sensitivity"],
    df_methane.loc["historical_best", "temp"],
)

# emissions adjustments for N2O and CH4 (we don't want to make these defaults as people
# might wanna run pulse expts with these gases)
fill(f.species_configs["baseline_emissions"], 19.019783117809567, specie="CH4")
fill(f.species_configs["baseline_emissions"], 0.08602230754, specie="N2O")

# aerosol direct
for specie in [
    "BC",
    "CH4",
    "N2O",
    "NH3",
    "NOx",
    "OC",
    "Sulfur",
    "VOC",
    "Equivalent effective stratospheric chlorine",
]:
    fill(
        f.species_configs["erfari_radiative_efficiency"],
        df_configs[f"ari_{specie}"],
        specie=specie,
    )

# forcing scaling
for specie in [
    "CO2",
    "CH4",
    "N2O",
    "Stratospheric water vapour",
    "Light absorbing particles on snow and ice",
    "Land use",
]:
    fill(
        f.species_configs["forcing_scale"],
        df_configs[f"fscale_{specie}"].values.squeeze(),
        specie=specie,
    )

for specie in [
    "CFC-11",
    "CFC-12",
    "CFC-113",
    "CFC-114",
    "CFC-115",
    "HCFC-22",
    "HCFC-141b",
    "HCFC-142b",
    "CCl4",
    "CHCl3",
    "CH2Cl2",
    "CH3Cl",
    "CH3CCl3",
    "CH3Br",
    "Halon-1211",
    "Halon-1301",
    "Halon-2402",
    "CF4",
    "C2F6",
    "C3F8",
    "c-C4F8",
    "C4F10",
    "C5F12",
    "C6F14",
    "C7F16",
    "C8F18",
    "NF3",
    "SF6",
    "SO2F2",
    "HFC-125",
    "HFC-134a",
    "HFC-143a",
    "HFC-152a",
    "HFC-227ea",
    "HFC-23",
    "HFC-236fa",
    "HFC-245fa",
    "HFC-32",
    "HFC-365mfc",
    "HFC-4310mee",
]:
    fill(
        f.species_configs["forcing_scale"],
        df_configs["fscale_minorGHG"].values.squeeze(),
        specie=specie,
    )

# ozone
for specie in [
    "CH4",
    "N2O",
    "Equivalent effective stratospheric chlorine",
    "CO",
    "VOC",
    "NOx",
]:
    fill(
        f.species_configs["ozone_radiative_efficiency"],
        df_configs[f"o3_{specie}"],
        specie=specie,
    )

# tune down volcanic efficacy
fill(f.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")


# initial condition of CO2 concentration (but not baseline for forcing calculations)
fill(
    f.species_configs["baseline_concentration"],
    df_configs["cc_co2_concentration_1750"].values.squeeze(),
    specie="CO2",
)

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()
os.makedirs(os.path.join(datadir, "output"), exist_ok=True)
os.makedirs(os.path.join(datadir, "restarts"), exist_ok=True)

f.temperature.loc[dict(scenario=scenarios, layer=0)].to_netcdf(
    os.path.join(datadir, "output", "historical_temperature.nc")
)
f.forcing_sum.loc[dict(scenario=scenarios)].to_netcdf(
    os.path.join(datadir, "output", "historical_forcing_sum.nc")
)
f.concentration.loc[dict(scenario=scenarios)].to_netcdf(
    os.path.join(datadir, "output", "historical_concentration.nc")
)
f.ocean_heat_content_change.loc[dict(scenario=scenarios)].to_netcdf(
    os.path.join(datadir, "output", "historical_ocean_heat_content_change.nc")
)
f.toa_imbalance.loc[dict(scenario=scenarios)].to_netcdf(
    os.path.join(datadir, "output", "historical_toa_imbalance.nc")
)
f.forcing.loc[dict(scenario=scenarios)].to_netcdf(
    os.path.join(datadir, "output", "historical_forcing.nc")
)
f.forcing_sum.loc[dict(scenario=scenarios)].to_netcdf(
    os.path.join(datadir, "output", "historical_forcing_sum.nc")
)


# dump restarts
f.temperature.loc[dict(scenario=scenarios, timebounds=2015)].to_netcdf(
    os.path.join(datadir, "restarts", "temperature_2015.nc")
)
f.alpha_lifetime.loc[dict(scenario=scenarios, timebounds=2015)].to_netcdf(
    os.path.join(datadir, "restarts", "alpha_lifetime_2015.nc")
)
f.airborne_emissions.loc[dict(scenario=scenarios, timebounds=2015)].to_netcdf(
    os.path.join(datadir, "restarts", "airborne_emissions_2015.nc")
)
f.cumulative_emissions.loc[dict(scenario=scenarios, timebounds=2015)].to_netcdf(
    os.path.join(datadir, "restarts", "cumulative_emissions_2015.nc")
)
f.concentration.loc[dict(scenario=scenarios, timebounds=2015)].to_netcdf(
    os.path.join(datadir, "restarts", "concentration_2015.nc")
)
f.forcing.loc[dict(scenario=scenarios, timebounds=2015)].to_netcdf(
    os.path.join(datadir, "restarts", "forcing_2015.nc")
)
f.gas_partitions.loc[dict(scenario=scenarios)].to_netcdf(
    os.path.join(datadir, "restarts", "gas_partitions_2015.nc")
)
f.ocean_heat_content_change.loc[dict(scenario=scenarios, timebounds=2015)].to_netcdf(
    os.path.join(datadir, "restarts", "ocean_heat_content_change_2015.nc")
)
