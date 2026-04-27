import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (concat_cols_on_freq, detect_peaks, combine_unique_freqs, 
                peaks_dict_to_arrays, get_int_at_peaks_AIopt, plot_3d_int,
                ratio_arc_cols, plot_spectra, plot_2d_ratio_int,
                plot_2d_int, int_is_peak, groups_ispeak, unique_by_freq_keep_max3,
                increase_or_decrease, plot_histogram_array, 
                plot_xy_by_ratio_ranges, make_ratio_ranges, groups_incr_decr,
                how_much_decr_ref, plot_overlapped_spectra, set_baseline_at_zero,
                apply_detection_limits, compute_sigma, only_noise, overwrite_from_peaks,
                l2_normalization)





molecule = 'so2'
spectra = ['so2', 'h2o', 'd2o']
i1, i2, i3 = f'int_{spectra[0]}', f'int_{spectra[1]}', f'int_{spectra[2]}'
cols = [i1, i2, i3]
"""
data1 = "DFM_H2O.csv"
data2 = "DFM_DOH.csv"
data3 = "DFM_D2O.csv"
sep = " "
"""
data1 = "2025-10-19-SO2_2300k.csv"
data2 = "2025-10-16-SO2-W_2200k.csv"
data3 = "2025-10-16-SO2-D-W_2000k.csv"
sep = ","

#IMPORTANT: Use same order as in spectra list for spectra csv reading spectra[0] = spectra 1

df_spectra1 = pl.read_csv(f"data/{molecule}/{data1}", 
                        has_header=False, 
                        skip_rows=0,
                        separator=sep)

df_spectra2 = pl.read_csv(f"data/{molecule}/{data2}", 
                        has_header=False, 
                        skip_rows=0,
                        separator=sep)

df_spectra3 = pl.read_csv(f"data/{molecule}/{data3}", 
                        has_header=False, 
                        skip_rows=0,
                        separator=sep)

# Data construction: 
# --- df_all: all data
# --- df_signals: data above noise
# --- df_int: all peaks and the respective intensity at each spectrum

#SPECTRA PROCESSING - NOISE REMOVAL
#sigma_list = [10e-6, 20e-6, 20e-6]
df_all = concat_cols_on_freq([df_spectra1, df_spectra2, df_spectra3], cols)
df_all_set0 = set_baseline_at_zero(df_all) #computes median and sets base line (median) at 0 (median in noise is very very similar)
noise = only_noise(df_all_set0, 1) #noise region over 5x mean (mean always positive, 0 and negative gives errors)
sigma_list = compute_sigma(noise) #computes sigma (std) from noise region
df_signals, detection_limits = apply_detection_limits(df_all_set0, sigma_list, detection_mult=3) #removes noise
print(detection_limits)

#FIND PEAKS
peak_dict = detect_peaks(df_signals) #gets freq of each peak above noise
peak_array = peaks_dict_to_arrays(peak_dict) # N arrays of [freq, int] pairs
all_peaks = combine_unique_freqs(peak_dict)
df_int = get_int_at_peaks_AIopt(all_peaks, df_signals, return_df=True) #using df_signals so freq is 0 if peak is bloew signal
#df_int_ext = get_int_at_peaks_AIopt(all_peaks, df_all_set0, return_df=True) #using df_all_set0 so freq peaks always have a value in all spectra
df_int = unique_by_freq_keep_max3(df_int, "freq", cols, tol=0.05)
df_int = df_int.sort("freq")
#df_signals_ext = overwrite_from_peaks(df_signals, df_int, key="freq")

# PRINT A FREQUENCY

#Add FEATURES ratios and arctg2 features
df_int = ratio_arc_cols(df_int, ratio=True, arctan2=True)
df_signals = ratio_arc_cols(df_signals, ratio=True, arctan2=True)
df_int, df_int_bool = int_is_peak(df_int, peak_array, 0.05)
df_int, df_groups_incr_decrs_overall = increase_or_decrease(df_int, cols, 0.1)
df_int.height
df_int
# Temporarily show all rows of every DataFrame
"""
with pl.Config():
    pl.Config.set_tbl_rows(-1)  # -1 means show all rows
    print(df_h2o_dec)# GROUP BY
"""

df_groups_ispeak = groups_ispeak(df_int) #groupped by True or False, signal or not signal at all spectra
df_groups_incr_decrs = groups_incr_decr(df_int, i2, i3) #groupped by increase or decrease of singal when varying composition
df_TTF = df_groups_ispeak["TTF"] #EXAMPLE
df_dec_dec = df_groups_incr_decrs["--"] #EXAMPLE (SO" signal decreases for both cases)
#common = df_TTT.join(df_dec_dec, on="freq", how="semi") #all -- freqs belong to TTT (checked)
df_two_third_decr = how_much_decr_ref(df_dec_dec, f"{i1}/{i2}", f"{i1}/{i3}", 0.39, 0.05) #all signals that decrease an specific quantity

# Isolate all lines that decrease with water and create echo.acs
df_h2o_dec = pl.concat([df_groups_incr_decrs["--"], df_groups_incr_decrs["-0"], df_groups_incr_decrs["-+"], df_groups_incr_decrs["-="]]).select(df_groups_incr_decrs["--"].columns[:4])
df_h2o_dec_inv = pl.concat([df_groups_incr_decrs["=="], 
                            df_groups_incr_decrs["=0"], 
                            df_groups_incr_decrs["+0"], 
                            df_groups_incr_decrs["+-"], 
                            df_groups_incr_decrs["++"]]).select(df_groups_incr_decrs["--"].columns[:4])

df_h2o_dec = df_h2o_dec.sort("freq")
df_h2o_dec_inv = df_h2o_dec_inv.sort("freq")
df_h2o_dec.write_csv(f"data/{molecule}/lines_decreased_with_h2o.csv", float_precision=4, include_header=True)
df_h2o_dec_inv.write_csv(f"data/{molecule}/echo.acs", float_precision=4, include_header=True)
df_h2o_dec_inv
# Check single intensity with a tolerance
print(df_int.filter((pl.col("freq") - 7532.6041).abs() < 0.1)) #FIND A FREQ
print(df_signals.filter((pl.col(f"int_{spectra[0]}") - 4.46240e-3).abs() < 0.00000001)) #FIND A FREQ by INT

#print(df_int.filter((pl.col("freq") - 5057.1).abs() < 0.1))

#L2 normalization and plot
"""
df_filt1 = df_int.filter(
(pl.col("int_water") > 0.0002) | (pl.col("int_deu") > 0.0002) | (pl.col("int_so2") > 0.0002))
df_filt2 = df_filt1.filter(
(pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))
df_int_l2 = l2_normalization(df_filt2, cols=['int_so2', 'int_water', 'int_deu'], plot_2d=True, plot_3d=True) 
"""

#plot_spectra(f"plots/spectra/{molecule}/spectra_{spectra[0]}", df_signals, peak_array, 'freq', i1, detection_limits[0], show_peaks=True, show_threshold=True, save_html=True)
#plot_spectra(f"plots/spectra/{molecule}/spectra_{spectra[1]}", df_signals, peak_array, 'freq', i2, detection_limits[1], show_peaks=True, show_threshold=True, save_html=True)
#plot_spectra(f"plots/spectra/{molecule}/spectra_{spectra[2]}", df_signals, peak_array, 'freq', i3, detection_limits[2], show_peaks=True, show_threshold=True, save_html=True)
plot_overlapped_spectra(f"plots/spectra/{molecule}/overlapped_spectra_{spectra[0]}_{spectra[1]}_{spectra[2]}", df_signals, 'freq', i1, i2, i3, vline_at=None, save_pdf=False, save_html=True)

#plot_2d_int(f"plots/intensity_rays/{molecule}/plot_2d_{spectra[0]}_{spectra[1]}", df_int, i1, i2, peaks=df_int, save_html=True, save_pdf=False)
#plot_2d_int(f"plots/intensity_rays/{molecule}/plot_2d_{spectra[0]}_{spectra[1]}_zoom", df_signals, cols=[spectra[0], spectra[1]], peaks=df_int, save_html=True, save_pdf=True, lims=[[-1,60],[-1,46]], zoom_lims=[[-0.01,1.],[-0.01,1.]], width=600, height=600)
#plot_3d_int(f"plots/intensity_rays/{molecule}/plot_3d_{spectra[0]}_{spectra[1]}_{spectra[2]}", df_signals, i1, i2, i3, save_html=True)

"""
df_int_wd = df_int.select([pl.col("int_water"), pl.col("int_deu")])
df_int_wd_clean = df_int_wd.filter((pl.col("int_water") != 0) & (pl.col("int_deu") != 0))

df = df_signals.filter((pl.col("int_water") > 0.00017) | (pl.col("int_deu") > 0.00012))
df = df.filter(
    (pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))

plot_histogram_array(df["int_water/int_deu"])

ranges_extra, labels = make_ratio_ranges(0., df["int_water/int_deu"].max(), 0.5)
ratio_ranges = [
        (0.0, 0.9),
        (0.9, 1.1),
        (1.1, None)
    ]
fig = plot_xy_by_ratio_ranges(df, "int_water", "int_deu", "int_water/int_deu", ratio_ranges)
fig.write_html("plot_IvI_groupped.html", include_plotlyjs="cdn")  # archivo interactivo

df.filter((pl.col("int_water/int_deu") >1.5)).height


# Data plotting
plot_base_peaks("plot_so2_peaks.html", df_all, peak_array, 'freq', 'int_so2', sigma3_list[0])
plot_base_peaks("plot_h2o_peaks.html", df_all, peak_array, 'freq', 'int_water', sigma3_list[1])
plot_base_peaks("plot_d2o_peaks.html", df_all, peak_array, 'freq', 'int_deu', sigma3_list[2])

plot_3d("plot_3d_int.html", df_int[:,1], df_int[:,2], df_int[:,3], min=0.0, max=0.001)
plot_3d("plot_3d_rat.html", df_int[:,4], df_int[:,5], df_int[:,6])
plot_3d("plot_3d_arc.html", df_int[:,7], df_int[:,8], df_int[:,9])

plot_2d_ratio_int("plot_2d_ratio12_i1.html", df_int["int_so2"], df_int["int_so2/int_water"])
plot_2d_ratio_int("plot_2d_ratio12_i2.html", df_int["int_water"], df_int["int_so2/int_water"])
plot_2d_ratio_int("plot_2d_ratio23_i2.html", df_int["int_water"], df_int["int_water/int_deu"])
plot_2d_ratio_int("plot_2d_ratio23_i3.html", df_int["int_deu"], df_int["int_water/int_deu"])
plot_2d_ratio_int("plot_2d_ratio32_i3.html", df_int["int_deu"], df_int["int_deu"]/df_int["int_water"])



plot_2d_int("plot_2d_water_deu_FFF.html", df_FFF['int_water'], df_FFF['int_deu'])
plot_2d_ratio_int("plot_2d_ratio23_i2_FFF.html", df_FFF_rat["int_deu"], df_FFF_rat["int_water/int_deu"])
"""