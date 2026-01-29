from data_analysis import df_int_unique
from data_analysis import df_signals, df_int_unique, df_TTT
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plot(output, x_base1, y_base1, x_base2, y_base2, x_base3, y_base3):



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

     # Base spectrum (line)
    fig.add_trace(go.Scatter(
        x=x_base3,
        y=y_base3,
        mode="lines",
        name="base",
        opacity=1.,
        line=dict(color="orange", width=1),
    ))

    fig.write_html("three_spectra.html", include_plotlyjs="cdn")  # archivo interactivo
    fig.show()
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

fig.write_html("plot_filtered_IvsI.html", include_plotlyjs="cdn")  # archivo interactivo
fig.show()
"""
data = np.arctan2(df_int["int_water"],df_int["int_deu"])
data
# Create histogram
fig = px.histogram(
    data,
    nbins=500,
    labels={'value': 'x'},
    title='Histogram Int ratio'
)

# Show plot
fig.show()
"""
