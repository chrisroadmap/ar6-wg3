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
here = os.path.dirname(os.path.realpath(__file__))

ar6_wg3 = pd.read_csv(os.path.join(datadir, "emissions", "20220314_ar6emissions_harmonized_infilled.csv"))
vetted = pd.read_csv(os.path.join(datadir, "emissions", "passing-vetting.csv"))
params = pd.read_csv(os.path.join(here, "..", "calibrations", "v1.3.0", "calibrated_constrained_parameters.csv"), index_col=0)

params.aci_beta = np.log(-params.aci_beta)
params.aci_shape_so2 = np.log(params.aci_shape_so2)
params.aci_shape_bc = np.log(params.aci_shape_bc)
params.aci_shape_oc = np.log(params.aci_shape_oc)

params.drop(columns=['seed'], inplace=True)

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
#     so2_emissions[:, iscen] = filt.loc[
#         (filt["Variable"]=="AR6 climate diagnostics|Infilled|Emissions|Sulfur"),
#         "2015":"2100"
#     ].values.squeeze()
#
# pl.scatter(co2_emissions.ravel(), so2_emissions.ravel())
# pl.show()


ds = xr.load_dataset(os.path.join(datadir, "output", f"batch_0001.nc"))

# non-CO2 forcing is the predictant
# ordering is time, scenario, config
y = np.ravel(ds.forcing - ds.forcing_co2, order="F")

# create a design matrix of predictors
#x = np.zeros((len(y), 47))
x = np.zeros((len(y), 3))

x[:,0] = np.tile(np.ravel(co2_emissions, order="F"), 1001)
x[:,1] = np.tile(np.arange(2015, 2101), 10*1001)
#x[:,2:] = np.repeat(params, 86*10, axis=0)
x[:,2] = np.repeat(np.ravel(ds.forcing_aerosol.loc[dict(timepoints=2015.5)]), 86)

# I am not expecting this to do a great job
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
x2 = poly.fit_transform(x)
print(x2.shape)

reg = linear_model.LinearRegression()
reg.fit(x2, y)

print(reg.score(x2, y))

# pred_x = np.concatenate(
#     [
#         #np.linspace(35, 0, 86) * np.ones((1, 86)),
#         x[:86, 0] * np.ones((1, 86)),
#         np.arange(2015, 2101) * np.ones((1, 86)),
#         (np.ones((86, 45)) * params.iloc[0:1, ...].values).T
#     ],
#     axis=0
# ).T
#
# pl.plot((ds.forcing - ds.forcing_co2).loc[dict(scenario=ds.scenario[0], config=ds.config[0])])
# pl.plot(reg.predict(pred_x))
#
# pred_x = np.concatenate(
#     [
#         #np.linspace(35, 0, 86) * np.ones((1, 86)),
#         x[:86, 0] * np.ones((1, 86)),
#         np.arange(2015, 2101) * np.ones((1, 86)),
#         (np.ones((86, 45)) * params.iloc[5:6, ...].values).T
#     ],
#     axis=0
# ).T
# pl.plot((ds.forcing - ds.forcing_co2).loc[dict(scenario=ds.scenario[0], config=ds.config[5])])
# pl.plot(reg.predict(pred_x))
#
# pl.show()
