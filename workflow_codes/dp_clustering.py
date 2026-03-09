import polars as pl
import numpy as np
from Algorithms.DIRICHLET.DP import (DirichletProcessMixtureRegression, plot_results)
from data_analysis import df_signals
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
import warnings


df_filt1 = df_signals.filter(
(pl.col("int_water") > 0.0002) | (pl.col("int_deu") > 0.0002))

df_filt2 = df_filt1.filter(
(pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))

# Tus datos
x = df_filt2['int_water'].to_numpy() # Features
y = df_filt2['int_deu'].to_numpy()  

x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())

# --- Fit the model ---
model = DirichletProcessMixtureRegression(
    alpha=1.0,
    sigma=1.0,
    n_iter=5000,
    n_burnin=2500,
    use_angular_distance=True,
    slope_prior_kappa=3,
    seed=7,
)
model.fit(x, y)

fig = plot_results(x, y, model)
fig.write_html("dpmr_results.html")
fig.show()