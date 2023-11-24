#!/usr/bin/env python

"""Run calibrated scenario using restart"""

# Here we run one scenario - the historical - from 1750 to 2014 using the
# CMIP6 emissions and the most recent fair calibration. We then dump this out
# including the GHG states as a restart, so we don't need to keep rerunning
# the historical period when comparing scenarios.

# Internal variability is OFF, since we currently don't have the prescribed
# internal variability inputs.

import copy
import os

import numpy as np
import pandas as pd
import pooch
import xarray as xr
from dotenv import load_dotenv
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
import matplotlib.pyplot as pl

load_dotenv()
datadir = os.getenv("DATADIR")

scenarios = ['ssp119', 'ssp245', 'ssp585']

df_solar = pd.read_csv(
    "../data/solar_erf_timebounds.csv", index_col="year"
)
df_volcanic = pd.read_csv(
    "../data/volcanic_ERF_1750-2101_timebounds.csv", index_col='timebounds'
)

volcanic_forcing = df_volcanic["erf"].loc[2015:2101].values
solar_forcing = df_solar["erf"].loc[2015:2101].values
solar_forcing_decliner = np.zeros(87)
solar_forcing_decliner[:11] = np.linspace(1, 0, 11)
solar_forcing = solar_forcing * solar_forcing_decliner

df_methane = pd.read_csv(
    "../calibrations/v1.3.0/CH4_lifetime.csv",
    index_col=0,
)

df_configs = pd.read_csv(
    "../calibrations/v1.3.0/calibrated_constrained_parameters.csv",
    index_col=0,
)
valid_all = df_configs.index

trend_shape = np.ones(87)
trend_shape[:6] = np.linspace(0, 1, 271)[265:271]

f = FAIR(ch4_method="Thornhill2021")
f.define_time(2015, 2101, 1)
f.define_scenarios(scenarios)
f.define_configs(valid_all)
species, properties = read_properties()
species.remove("NOx aviation")
species.remove("Contrails")
f.define_species(species, properties)
f.allocate()

fair_to_rcmip = {
    "CO2 FFI": "AR6 climate diagnostics|Infilled|Emissions|CO2|Energy and Industrial Processes",
    "CO2 AFOLU": "AR6 climate diagnostics|Infilled|Emissions|CO2|AFOLU",
    "CH4": "AR6 climate diagnostics|Infilled|Emissions|CH4",
    "N2O": "AR6 climate diagnostics|Infilled|Emissions|N2O",
    "Sulfur": "AR6 climate diagnostics|Infilled|Emissions|Sulfur",
    "BC": "AR6 climate diagnostics|Infilled|Emissions|BC",
    "OC": "AR6 climate diagnostics|Infilled|Emissions|OC",
    "NH3": "AR6 climate diagnostics|Infilled|Emissions|NH3",
    "NOx": "AR6 climate diagnostics|Infilled|Emissions|NOx",
    "VOC": "AR6 climate diagnostics|Infilled|Emissions|VOC",
    "CO": "AR6 climate diagnostics|Infilled|Emissions|CO",
    "CFC-11": "AR6 climate diagnostics|Infilled|Emissions|CFC11",
    "CFC-12": "AR6 climate diagnostics|Infilled|Emissions|CFC12",
    "CFC-113": "AR6 climate diagnostics|Infilled|Emissions|CFC113",
    "CFC-114": "AR6 climate diagnostics|Infilled|Emissions|CFC114",
    "CFC-115": "AR6 climate diagnostics|Infilled|Emissions|CFC115",
    "HCFC-22": "AR6 climate diagnostics|Infilled|Emissions|HCFC22",
    "HCFC-141b": "AR6 climate diagnostics|Infilled|Emissions|HCFC141b",
    "HCFC-142b": "AR6 climate diagnostics|Infilled|Emissions|HCFC142b",
    "CCl4": "AR6 climate diagnostics|Infilled|Emissions|CCl4",
    "CHCl3": "AR6 climate diagnostics|Infilled|Emissions|CHCl3",
    "CH2Cl2": "AR6 climate diagnostics|Infilled|Emissions|CH2Cl2",
    "CH3Cl": "AR6 climate diagnostics|Infilled|Emissions|CH3Cl",
    "CH3CCl3": "AR6 climate diagnostics|Infilled|Emissions|CH3CCl3",
    "CH3Br": "AR6 climate diagnostics|Infilled|Emissions|CH3Br",
    "Halon-1211": "AR6 climate diagnostics|Infilled|Emissions|Halon1211",
    "Halon-1301": "AR6 climate diagnostics|Infilled|Emissions|Halon1301",
    "Halon-2402": "AR6 climate diagnostics|Infilled|Emissions|Halon2402",
    "CF4": "AR6 climate diagnostics|Infilled|Emissions|PFC|CF4",
    "C2F6": "AR6 climate diagnostics|Infilled|Emissions|PFC|C2F6",
    "C3F8": "AR6 climate diagnostics|Infilled|Emissions|PFC|C3F8",
    "c-C4F8": "AR6 climate diagnostics|Infilled|Emissions|PFC|cC4F8",
    "C4F10": "AR6 climate diagnostics|Infilled|Emissions|PFC|C4F10",
    "C5F12": "AR6 climate diagnostics|Infilled|Emissions|PFC|C5F12",
    "C6F14": "AR6 climate diagnostics|Infilled|Emissions|PFC|C6F14",
    "C7F16": "AR6 climate diagnostics|Infilled|Emissions|PFC|C7F16",
    "C8F18": "AR6 climate diagnostics|Infilled|Emissions|PFC|C8F18",
    "NF3": "AR6 climate diagnostics|Infilled|Emissions|NF3",
    "SF6": "AR6 climate diagnostics|Infilled|Emissions|SF6",
    "SO2F2": "AR6 climate diagnostics|Infilled|Emissions|SO2F2",
    "HFC-125": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC125",
    "HFC-134a": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC134a",
    "HFC-143a": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC143a",
    "HFC-152a": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC152a",
    "HFC-227ea": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC227ea",
    "HFC-23": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC23",
    "HFC-236fa": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC236fa",
    "HFC-245fa": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC245ca",  # is this an error in the harmonized infilled data?
    "HFC-32": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC32",
    "HFC-365mfc": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC365mfc",
    "HFC-4310mee": "AR6 climate diagnostics|Infilled|Emissions|HFC|HFC43-10",
}

unit_convert = {specie: 1 for specie in fair_to_rcmip.keys()}
unit_convert["CO2 FFI"] = 0.001
unit_convert["CO2 AFOLU"] = 0.001
unit_convert["N2O"] = 0.001

ar6_wg3 = pd.read_csv(os.path.join(datadir, "emissions", "20220314_ar6emissions_harmonized_infilled.csv"))
filt = ar6_wg3.loc[
    (ar6_wg3["Model"]=="IMAGE 3.0.1") &
    (ar6_wg3["Scenario"]=="SSP1-19")
]
for fairname, rcmipname in fair_to_rcmip.items():
    f.emissions.loc[dict(specie=fairname, scenario='ssp119')] = filt.loc[
        (filt["Variable"]==rcmipname),
        "2015":"2100"
    ].values.squeeze()[:, None] * unit_convert[fairname]
filt = ar6_wg3.loc[
    (ar6_wg3["Model"]=="MESSAGE-GLOBIOM 1.0") &
    (ar6_wg3["Scenario"]=="SSP2-45")
]
for fairname, rcmipname in fair_to_rcmip.items():
    f.emissions.loc[dict(specie=fairname, scenario='ssp245')] = filt.loc[
        (filt["Variable"]==rcmipname),
        "2015":"2100"
    ].values.squeeze()[:, None] * unit_convert[fairname]
filt = ar6_wg3.loc[
    (ar6_wg3["Model"]=="REMIND-MAgPIE 1.5") &
    (ar6_wg3["Scenario"]=="SSP5-Baseline")
]
for fairname, rcmipname in fair_to_rcmip.items():
    f.emissions.loc[dict(specie=fairname, scenario='ssp585')] = filt.loc[
        (filt["Variable"]==rcmipname),
        "2015":"2100"
    ].values.squeeze()[:, None] * unit_convert[fairname]

# add in missing emissions
f.emissions.loc[dict(specie="Halon-1202")] = 0

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

# load restarts and overwrite scenario name
concentration_2015 = xr.load_dataarray(os.path.join(datadir, "restarts", "concentration_2015.nc"))
forcing_2015 = xr.load_dataarray(os.path.join(datadir, "restarts", "forcing_2015.nc"))
temperature_2015 = xr.load_dataarray(os.path.join(datadir, "restarts", "temperature_2015.nc"))
airborne_emissions_2015 = xr.load_dataarray(os.path.join(datadir, "restarts", "airborne_emissions_2015.nc"))
cumulative_emissions_2015 = xr.load_dataarray(os.path.join(datadir, "restarts", "cumulative_emissions_2015.nc"))
alpha_lifetime_2015 = xr.load_dataarray(os.path.join(datadir, "restarts", "alpha_lifetime_2015.nc"))
ocean_heat_content_change_2015 = xr.load_dataarray(os.path.join(datadir, "restarts", "ocean_heat_content_change_2015.nc"))
gas_partitions_2015 = xr.load_dataarray(os.path.join(datadir, "restarts", "gas_partitions_2015.nc"))

# These are the magic lines
initialise(f.concentration, concentration_2015[0, ...])
initialise(f.forcing, forcing_2015[0, ...])
initialise(f.temperature, temperature_2015[0, ...])
initialise(f.airborne_emissions, airborne_emissions_2015[0, ...])
initialise(f.cumulative_emissions, cumulative_emissions_2015[0, ...])
initialise(f.ocean_heat_content_change, ocean_heat_content_change_2015[0, ...])
f.gas_partitions[:, ...]=copy.deepcopy(gas_partitions_2015[0:1, ...].data)

f.run()

# load historical
historical_concentration = xr.load_dataarray(os.path.join(datadir, "output", "historical_concentration.nc"))
historical_forcing = xr.load_dataarray(os.path.join(datadir, "output", "historical_forcing.nc"))
historical_temperature = xr.load_dataarray(os.path.join(datadir, "output", "historical_temperature.nc"))
historical_ocean_heat_content_change = xr.load_dataarray(os.path.join(datadir, "output", "historical_ocean_heat_content_change.nc"))
historical_toa_imbalance = xr.load_dataarray(os.path.join(datadir, "output", "historical_toa_imbalance.nc"))

twenty = np.ones(21)
twenty[0] = 0.5
twenty[-1] = 0.5
temperature_baseliner = np.average(historical_temperature.loc[dict(timebounds=np.arange(1995, 2016))], weights=twenty, axis=0)

# pl.plot(np.arange(1750, 2016), 0.85 + historical_temperature.loc[dict(scenario="historical", config=valid_all)] - temperature_baseliner, color='k')
# pl.plot(f.timebounds, 0.85 + f.temperature.loc[dict(scenario="ssp119", layer=0, config=valid_all)] - temperature_baseliner, color='b', alpha=0.01)
# pl.plot(f.timebounds, 0.85 + f.temperature.loc[dict(scenario="ssp245", layer=0, config=valid_all)] - temperature_baseliner, color='orange', alpha=0.01)
# pl.plot(f.timebounds, 0.85 + f.temperature.loc[dict(scenario="ssp585", layer=0, config=valid_all)] - temperature_baseliner, color='r', alpha=0.01)
# pl.show()
#
# pl.plot(np.arange(1750, 2016), historical_concentration.loc[dict(specie="CO2", scenario="historical", config=valid_all)], color='k')
# pl.plot(f.timebounds, f.concentration.loc[dict(scenario="ssp119", specie="CO2", config=valid_all)], color='b', alpha=0.01)
# pl.plot(f.timebounds, f.concentration.loc[dict(scenario="ssp245", specie="CO2", config=valid_all)], color='orange', alpha=0.01)
# pl.plot(f.timebounds, f.concentration.loc[dict(scenario="ssp585", specie="CO2", config=valid_all)], color='r', alpha=0.01)
# pl.show()
#
# pl.plot(np.arange(1750, 2016), historical_ocean_heat_content_change.loc[dict(scenario="historical", config=valid_all)], color='k')
# pl.plot(f.timebounds, f.ocean_heat_content_change.loc[dict(scenario="ssp119", config=valid_all)], color='b', alpha=0.01)
# pl.plot(f.timebounds, f.ocean_heat_content_change.loc[dict(scenario="ssp585", config=valid_all)], color='r', alpha=0.01)
# pl.show()
#
# pl.plot(np.arange(1750, 2016), historical_toa_imbalance.loc[dict(scenario="historical", config=valid_all)], color='k')
# pl.plot(f.timebounds, f.toa_imbalance.loc[dict(scenario="ssp119", config=valid_all)], color='b')
# pl.plot(f.timebounds, f.toa_imbalance.loc[dict(scenario="ssp585", config=valid_all)], color='r')
# pl.show()

# what outputs do I actually want: temperature, CO2 conc
# forcing: CO2, CH4, N2O, aerosol, other
temp = (f.temperature.loc[dict(layer=0)] - temperature_baseliner + 0.85).data
cco2 = f.concentration.loc[dict(specie="CO2")].data
fco2 = f.forcing.loc[dict(specie="CO2")].data
fch4 = f.forcing.loc[dict(specie="CH4")].data
fn2o = f.forcing.loc[dict(specie="N2O")].data
faer = (f.forcing.loc[dict(specie="Aerosol-radiation interactions")] + f.forcing.loc[dict(specie="Aerosol-cloud interactions")]).data
fsum = f.forcing_sum.data

temp = 0.5*(temp[1:, ...] + temp[:-1, ...])
cco2 = 0.5*(cco2[1:, ...] + cco2[:-1, ...])
fco2 = 0.5*(fco2[1:, ...] + fco2[:-1, ...])
fch4 = 0.5*(fch4[1:, ...] + fch4[:-1, ...])
fn2o = 0.5*(fn2o[1:, ...] + fn2o[:-1, ...])
faer = 0.5*(faer[1:, ...] + faer[:-1, ...])
fsum = 0.5*(fsum[1:, ...] + fsum[:-1, ...])
foth = fsum - fco2 - fch4 - fn2o - faer - fsum
fnon = fsum - fco2

pl.scatter(f.emissions.loc[dict(specie='CO2', scenario='ssp119')], fnon[:, 0, :])
pl.scatter(f.emissions.loc[dict(specie='CO2', scenario='ssp245')], fnon[:, 1, :])
pl.scatter(f.emissions.loc[dict(specie='CO2', scenario='ssp585')], fnon[:, 2, :])
pl.show()

ds = xr.Dataset(
    data_vars=dict(
        temperature=(["timepoints", "scenario", "config"], temp),
        concentration=(["timepoints", "scenario", "config"], cco2),
        forcing = (["timepoints", "scenario", "config"], fsum),
        forcing_co2 = (["timepoints", "scenario", "config"], fco2),
        forcing_aerosol = (["timepoints", "scenario", "config"], faer),
        forcing_nonco2ch4n2oaer = (["timepoints", "scenario", "config"], foth),
    ),
    coords=dict(
        timepoints=np.arange(2015.5, 2101),
        scenario=scenarios,
        config=valid_all,
    ),
)

ds.to_netcdf(os.path.join(datadir, "output", "batch_0001.nc"))
