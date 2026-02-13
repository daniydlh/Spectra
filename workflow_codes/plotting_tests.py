from data_analysis import df_all
from data_analysis import df_signals
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils import plot_overlapped_spectra


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
"""