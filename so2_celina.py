import numpy as np
import polars as pl
import pandas as pd
from utils import matching_peaks_pair, matching_peaks_three
from peak_finding import freq
import plotly.express as px
import plotly.graph_objects as go


peaks_w = np.loadtxt('Spectra_signals/max_water.fft', delimiter=',', skiprows=15)
freq_peaks_w = peaks_w[:, 0]
int_peaks_w = peaks_w[:, 1]
len(freq_peaks_w)

peaks_d = np.loadtxt('Spectra_signals/max_deu.fft', delimiter=',', skiprows=15)
freq_peaks_d = peaks_d[:, 0]
int_peaks_d = peaks_d[:, 1]

peaks_so2 = np.loadtxt('Spectra_signals/max_so2.fft', delimiter=',', skiprows=15)
freq_peaks_so2 = peaks_so2[:, 0]
int_peaks_so2 = peaks_so2[:, 1]

#matching between the three spectra
matched_freqs_3, matched_int_so2_3, matched_int_w_3, matched_int_d_3 = matching_peaks_three(freq_peaks_so2, int_peaks_so2, freq_peaks_w, int_peaks_w, freq_peaks_d, int_peaks_d, freq[1]-freq[0])

#ONLY PEAKS WITH SO2 dominance or only SO2
mask = np.ones(len(matched_freqs_3), dtype=bool)

for i, val in enumerate(matched_freqs_3):
    if matched_int_w_3[i] > matched_int_so2_3[i] or matched_int_d_3[i] > matched_int_so2_3[i]:
        mask[i] = False

so2_over_matched_freqs_3 = matched_freqs_3[mask]
so2_over_matched_int_so2_3 = matched_int_so2_3[mask]
so2_over_matched_int_w_3 = matched_int_w_3[mask]
so2_over_matched_int_d_3 = matched_int_d_3[mask]

full_int_so2 = np.zeros_like(freq)
idx = np.searchsorted(freq, so2_over_matched_freqs_3)
full_int_so2[idx] = so2_over_matched_int_so2_3

full_int_w = np.zeros_like(freq)
idx_w = np.searchsorted(freq, so2_over_matched_int_w_3)
full_int_w[idx_w] = so2_over_matched_int_w_3

full_int_d = np.zeros_like(freq)
idx_d = np.searchsorted(freq, so2_over_matched_int_d_3)
full_int_d[idx_d] = so2_over_matched_int_d_3


#widths, width_heights, left_ips, right_ips = peak_widths(int_peaks_so2, peaks_so2, rel_height=0.9)

fig = go.Figure()
# Base spectrum (line)
fig.add_trace(go.Scatter(
    x=freq,
    y=full_int_so2,
    mode="lines",
    name="SO2",
    opacity=1.0,
    line=dict(color="blue", width=1),
))

# +H2O peaks
fig.add_trace(go.Scatter(
    x=so2_over_matched_freqs_3,
    y=so2_over_matched_int_w_3,
    mode="markers",
    name="+H2O",
    opacity=1.,
    marker=dict(
        color="red",
        symbol="circle",
        size=4,
    ),
))

# +D2O peaks
fig.add_trace(go.Scatter(
    x=so2_over_matched_freqs_3,
    y=so2_over_matched_int_d_3,
    mode="markers",
    name="+D2O",
    opacity=1.,
    marker=dict(
        color="yellow",
        symbol="diamond",
        size=4,
    ),
))

fig.update_layout(xaxis_title='Frequency', yaxis_title='Intensity')

fig.show(renderer="browser")

SO2_csv = pd.DataFrame({
    "Frequency": pd.Series(freq),
    "Peak intensity SO2": pd.Series(full_int_so2),
    "Peak frequency +H2O": pd.Series(so2_over_matched_freqs_3),
    "Peak intensity +H2O": pd.Series(so2_over_matched_int_w_3),
    "Peak frequency +D2O": pd.Series(so2_over_matched_freqs_3),
    "Peak intensity +D2O": pd.Series(so2_over_matched_int_d_3),
})

SO2_csv.to_csv("SO2_peaks.csv")
