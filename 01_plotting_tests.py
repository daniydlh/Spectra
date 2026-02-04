from 01_data_analysis import df_all
from 01_data_analysis import df_signals
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plot(output, x_base1, y_base1, x_base2, y_base2):


    fig = go.Figure()
    # Base spectrum (line)
    fig.add_trace(go.Scatter(
        x=x_base1,
        y=y_base1,
        mode="lines",
        name="base",
        opacity=1.,
        line=dict(color="blue", width=1),
    ))

      # Base spectrum (line)
    fig.add_trace(go.Scatter(
        x=x_base2,
        y=y_base2,
        mode="lines",
        name="base",
        opacity=1.,
        line=dict(color="red", width=1),
    ))
    """
     # Base spectrum (line)
    fig.add_trace(go.Scatter(
        x=x_base3,
        y=y_base3,
        mode="lines",
        name="base",
        opacity=1.,
        line=dict(color="orange", width=1),
    ))
    """
    #fig.write_html(output, include_plotlyjs="cdn")  # archivo interactivo
    fig.show()

plot('df_all_and_signals.html', df_all['freq'], df_all['int_water'], df_signals['freq'], df_signals['int_water'])

"""
df = df_signals.filter(
    (pl.col("int_water") > 0.0002) | (pl.col("int_deu") > 0.00015))
df.height
df = df.filter(
    (pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))
df.height

n_zeros = df.select(
    (pl.col("int_deu") == 0.).sum()
).item()
n_zeros
"""
sigma = (1.796410137e-05)/3
fig = go.Figure()
# Base spectrum (line)
fig.add_trace(go.Scatter(
    x=df_signals["int_water"],
    y=df_signals["int_deu"],
    mode="markers",
    name="SO2 + D2O spectrum",
    opacity=1,
    line=dict(color="blue", width=1),
))

fig.write_html("plot_overlap.html", include_plotlyjs="cdn")  # archivo interactivo
fig.show()
