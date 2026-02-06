from data_analysis import df_all
from data_analysis import df_signals
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plot_overlapped_spectra(
    output,
    df,
    x,
    y1,
    y2,
    y3,
    xlims=None,
    ylims=None,
    save_html=False,
    save_pdf=False,
):

    xvals = df[x].to_numpy()
    y1vals = df[y1].to_numpy() * 1000.
    y2vals = df[y2].to_numpy() * 1000.
    y3vals = df[y3].to_numpy() * 1000.

    fig = go.Figure()

    # Spectra
    fig.add_trace(go.Scatter(
        x=xvals,
        y=y1vals,
        mode="lines",
        name="SO$_2$",
        line=dict(color="royalblue", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=xvals,
        y=y2vals,
        mode="lines",
        name="SO$_2$ + H$_2$O",
        line=dict(color="red", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=xvals,
        y=y3vals,
        mode="lines",
        name="SO$_2$ + D$_2$O",
        line=dict(color="orange", width=2),
    ))

    # Axis limits
    if xlims is not None:
        fig.update_xaxes(range=list(xlims))
    if ylims is not None:
        fig.update_yaxes(range=list(ylims))

    # Layout (publication style)
    fig.update_layout(
        width=900,
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Times New Roman",
            size=20,
            color="black"
        ),
        xaxis=dict(
            title=dict(text="Frequency (MHz)", font=dict(size=26)),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            gridwidth=1,
            ticks="outside",
            tickwidth=2,
            ticklen=8,
            showline=True,
            linewidth=2,
            linecolor="black",
            tickfont=dict(size=22)
        ),
        yaxis=dict(
            title=dict(text="Intensity (µV)", font=dict(size=26)),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            gridwidth=1,
            ticks="outside",
            tickwidth=2,
            ticklen=8,
            showline=True,
            linewidth=2,
            linecolor="black",
            tickfont=dict(size=22)
        ),
        legend=dict(
            font=dict(size=20),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0
        ),
        margin=dict(l=90, r=30, t=40, b=80)
    )

    # Save
    if save_pdf:
        fig.write_image(
            f"spectra/{output}.pdf",
            format="pdf",
            scale=3
        )

    if save_html:
        fig.write_html(
            f"spectra{output}.html",
            include_plotlyjs="cdn"
        )

    fig.show()

plot_overlapped_spectra('overlapped_spectra', df_all, 'freq', 'int_so2', 'int_water', 'int_deu', xlims=[7223,7230], ylims=[-0.5,6.9], save_pdf=True)

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