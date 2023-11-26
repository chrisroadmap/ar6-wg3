#!/usr/bin/env python

"""Analyse a batch"""

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()
datadir = os.getenv("DATADIR")
batch_size = int(os.getenv("BATCH_SIZE"))

ar6_wg3 = pd.read_csv(os.path.join(datadir, "emissions", "20220314_ar6emissions_harmonized_infilled.csv"))
vetted = pd.read_csv(os.path.join(datadir, "emissions", "passing-vetting.csv"))

# get CO2 FFI emissions as the predictor
batch_start = 0
batch_end = batch_start + batch_size

scenarios = []
co2_emissions = np.zeros((86, batch_size))
so2_emissions = np.zeros((86, batch_size))
for iscen in range(batch_start, batch_end):
    scenarios.append(f"{vetted.loc[iscen, 'Model']}___{vetted.loc[iscen, 'Scenario']}")

    filt = ar6_wg3.loc[
        (ar6_wg3["Model"]==vetted.loc[iscen, 'Model']) &
        (ar6_wg3["Scenario"]==vetted.loc[iscen, 'Scenario'])
    ]
    co2_emissions[:, iscen] = filt.loc[
        (filt["Variable"]=="AR6 climate diagnostics|Infilled|Emissions|CO2|Energy and Industrial Processes"),
        "2015":"2100"
    ].values.squeeze() * 0.001
    so2_emissions[:, iscen] = filt.loc[
        (filt["Variable"]=="AR6 climate diagnostics|Infilled|Emissions|Sulfur"),
        "2015":"2100"
    ].values.squeeze()

pl.scatter(co2_emissions.ravel(), so2_emissions.ravel())
pl.show()

# print(co2_emissions)
# pl.plot(co2_emissions)
# pl.show()

# pl.scatter(f.emissions.loc[dict(specie='CO2', scenario='ssp119')], fnon[:, 0, :])
# pl.scatter(f.emissions.loc[dict(specie='CO2', scenario='ssp245')], fnon[:, 1, :])
# pl.scatter(f.emissions.loc[dict(specie='CO2', scenario='ssp585')], fnon[:, 2, :])
# pl.show()

ds = xr.load_dataset(os.path.join(datadir, "output", f"batch_0001.nc"))
print(ds)
    #
    # ds = xr.Dataset(
    #     data_vars=dict(
    #         temperature=(["timepoints", "scenario", "config"], temp),
    #         concentration=(["timepoints", "scenario", "config"], cco2),
    #         forcing = (["timepoints", "scenario", "config"], fsum),
    #         forcing_co2 = (["timepoints", "scenario", "config"], fco2),
    #         forcing_aerosol = (["timepoints", "scenario", "config"], faer),
    #         forcing_nonco2ch4n2oaer = (["timepoints", "scenario", "config"], foth),
    #     ),
    #     coords=dict(
    #         timepoints=np.arange(2015.5, 2101),
    #         scenario=scenarios,
    #         config=valid_all,
    #     ),
    # )
    #
    # ds.to_netcdf(os.path.join(datadir, "output", f"batch_{batch_num:0{4}d}.nc"))
    # batch_num = batch_num + 1

# # across scenarios
# pl.plot(ds.forcing[:, :, 0] - ds.forcing_co2[:, :, 0])
# pl.show()
#
# # across configs
# pl.plot(ds.forcing[:, 0, :] - ds.forcing_co2[:, 0, :])
# pl.show()

# create a design matrix - non-CO2 forcing is the predictant
# ordering is time, scenario, config
y = np.ravel(ds.forcing - ds.forcing_co2, order="F")

# to start with, use only CO2 emissions, PD aerosol forcing, and year as the predictors
x = np.zeros((len(y), 3))

x[:,0] = np.tile(np.ravel(co2_emissions, order="F"), 1001)
x[:,1] = np.tile(np.arange(2015, 2101), 10*1001)
x[:,2] = np.repeat(np.ravel(ds.forcing_aerosol.loc[dict(timepoints=2015.5)]), 86)

print(x[:87, :])

# I am not expecting this to do a great job
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(x, y)
print(reg.intercept_, reg.coef_)
print(reg.score(x, y))

print(reg.predict(np.array([np.linspace(35, 0, 86), np.arange(2015, 2101), np.ones(86)*(-2.0)]).T))

pl.plot((ds.forcing - ds.forcing_co2).loc[dict(scenario=ds.scenario[0])])
pl.show()
