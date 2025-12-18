import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from utils import get_noise, fft_df, matching_peaks, fft_arr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks



spectra_water = pl.read_csv("2025-10-16-SO2-W_2200k.fft", 
                        has_header=True, 
                        skip_rows=14)

spectra_deu = pl.read_csv("2025-10-16-SO2-D-W_2000k.fft", 
                        has_header=True, 
                        skip_rows=14)

spectra_so2 = pl.read_csv("2025-10-14-SO2_3000kavg.fft", 
                        has_header=True, 
                        skip_rows=14)


only_noise_regions = [[2210,2250],[2660,2672],[2814,2830]] #input for function
sigma_water = get_noise(spectra_water, only_noise_regions)
sigma_deu = get_noise(spectra_deu, only_noise_regions)
sigma_so2 = get_noise(spectra_so2, only_noise_regions)
sigma = 3e-6

signals_water = spectra_water.with_columns(
                pl.when(pl.col('intensity') < 3*sigma_water)
                .then(0)
                .otherwise(pl.col('intensity'))
                .alias('intensity')
)

signals_deu = spectra_deu.with_columns(
                pl.when(pl.col('intensity') < 3*sigma_water)
                .then(0)
                .otherwise(pl.col('intensity'))
                .alias('intensity')
)

signals_so2 = spectra_so2.with_columns(
                pl.when(pl.col('intensity') < 3*sigma_water)
                .then(0)
                .otherwise(pl.col('intensity'))
                .alias('intensity')
)

#Negative spectra, only noise with blanks
negative_water = spectra_water.filter(pl.col('intensity') < 3 * sigma_water)
negative_deu = spectra_deu.filter(pl.col('intensity') < 3 * sigma_deu)
negative_so2 = spectra_so2.filter(pl.col('intensity') < 3 * sigma_so2)

#Spectra only with signals to FFT
fft_df('Spectra_signals/signals_water.fft', signals_water, sep=',')
fft_df('Spectra_signals/signals_deu.fft', signals_deu, sep=',')
fft_df('Spectra_signals/signals_so2.fft', signals_so2, sep=',')

#FIND PEAKS NAD TURN INTO FFT
freq = signals_water['freq'].to_numpy()
int_w = signals_water['intensity'].to_numpy()
int_d = signals_deu['intensity'].to_numpy()
int_so2 = signals_so2['intensity'].to_numpy()

peaks_w, props_w = find_peaks(int_w, prominence = 0.0)
peaks_d, props_d = find_peaks(int_d, prominence = 0.0)
peaks_so2, props_so2 = find_peaks(int_so2, prominence = 0.0)

len(peaks_so2)

freq_peaks_w = freq[peaks_w]
freq_peaks_d = freq[peaks_d]
freq_peaks_so2 = freq[peaks_so2]
int_peaks_w = int_w[peaks_w]
int_peaks_d = int_d[peaks_d]
int_peaks_so2= int_so2[peaks_so2]
fft_arr('Spectra_signals/max_water.fft', freq_peaks_w, int_peaks_w, sep=',')
fft_arr('Spectra_signals/max_deu.fft', freq_peaks_d, int_peaks_d, sep=',')
fft_arr('Spectra_signals/max_so2.fft', freq_peaks_so2, int_peaks_so2, sep=',')


#MATCH PEAKS BETWEEN SPECTRA
matched_freqs1, matched_int_w, matched_int_d = matching_peaks(freq_peaks_w, int_peaks_w, freq_peaks_d, int_peaks_d, freq[1]-freq[0])
matched_freqs2, matched_int_w, matched_int_so2 = matching_peaks(freq_peaks_w, int_peaks_w, freq_peaks_so2, int_peaks_so2, freq[1]-freq[0])
len(matched_freqs2)


#CSV file
df_matched_peaks = pl.DataFrame({
    "Frequency": matched_freqs2,
    "Intensity_water": matched_int_w,
    "Intensity_deu": matched_int_so2
})

df__matched_peaks = df_matched_peaks.with_columns([(pl.col('Intensity_deu')/pl.col('Intensity_water')).alias("ratio_d/w")])

df_matched_peaks.write_csv('peaks_water+so2.csv', include_header=True)
print(df_matched_peaks)

plt.figure()  # start a new figure

# Plot first dataset as scatter (balls)
plt.scatter(df_peaks['Frequency'], df_peaks['ratio_d/w'], color='red', label='ratio', s=10)  # s=size of markers

# Plot second dataset as line
#plt.plot(df_peaks['Frequency'],df_peaks['Intensity_deu'], color='blue', label='Deuterated', linewidth=2)

# Labels and title
plt.xlabel("Frequency")
plt.ylabel("ratio")
#plt.ylim(0, 2)
plt.title("Rotational Spectra Comparison")
plt.legend()
plt.grid(True)

plt.show()







