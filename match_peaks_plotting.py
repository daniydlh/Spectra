import numpy as np
import polars as pl
from utils import matching_peaks_pair, matching_peaks_three
from peak_finding import freq
import plotly.express as px


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

#MATCH PEAKS BETWEEN SPECTRA
matched_freqs1, matched_int_so2, matched_int_d1 = matching_peaks_pair(freq_peaks_so2, int_peaks_so2, freq_peaks_d, int_peaks_d, freq[1]-freq[0])
matched_freqs2, matched_int_w2, matched_int_so2 = matching_peaks_pair(freq_peaks_w, int_peaks_w, freq_peaks_so2, int_peaks_so2, freq[1]-freq[0])

matched_freqs_3, matched_int_so2_3, matched_int_w_3, matched_int_d_3 = matching_peaks_three(freq_peaks_w, int_peaks_w, freq_peaks_so2, int_peaks_so2, freq_peaks_d, int_peaks_d, freq[1]-freq[0])

mask = np.ones(len(matched_freqs_3), dtype=bool)

for i, val in enumerate(matched_freqs_3):
    if matched_int_w_3[i] > matched_int_so2_3[i] or matched_int_d_3[i] > matched_int_so2_3[i]:
        mask[i] = False

so2_over_matched_freqs_3 = matched_freqs_3[mask]
so2_over_matched_int_so2_3 = matched_int_so2_3[mask]
so2_over_matched_int_w_3 = matched_int_w_3[mask]
so2_over_matched_int_d_3 = matched_int_d_3[mask]


#CSV file
df_matched_peaks1 = pl.DataFrame({
    "Frequency": matched_freqs1,
    "Intensity_water": matched_int_w1,
    "Intensity_deu": matched_int_d1
})

df_matched_peaks2 = pl.DataFrame({
    "Frequency": matched_freqs2,
    "Intensity_water": matched_int_w2,
    "Intensity_so2": matched_int_so2
})

df_matched_peaks1 = df_matched_peaks1.with_columns([(pl.col('Intensity_deu')/pl.col('Intensity_water')).alias("ratio_d/w")])
df_matched_peaks2 = df_matched_peaks2.with_columns([(pl.col('Intensity_water')/pl.col('Intensity_so2')).alias("ratio_w/so2")])

df_matched_peaks1.write_csv('peaks_water+deu.csv', include_header=True)
df_matched_peaks2.write_csv('peaks_water+so2.csv', include_header=True)

#print(df_matched_peaks)
x = df_matched_peaks1['Intensity_water'].to_numpy()
y = df_matched_peaks1['Intensity_deu'].to_numpy()
z = np.arctan2(x,y)
w = x/y
f = df_matched_peaks1['Frequency']


fig = px.scatter(
    x=x,
    y=y,
    opacity=0.5,
)
fig.show(renderer='browser')  # opens in external browser
