import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (noise_rm_all, concat_cols_on_freq, detect_peaks, combine_unique_freqs, 
                peaks_dict_to_arrays, get_int_at_peaks_AIopt, plot_3d,
                ratio_arc_cols, plot_base_peaks, plot_2d_ratio_int,
                plot_2d_int, int_is_peak, groups_ispeak, unique_by_freq_keep_max3, 
                fft_arr, fft_df, increase_or_decrease, plot_histogram_array, 
                plot_xy_by_ratio_ranges, make_ratio_ranges, groups_incr_decr,
                how_much_decr_ref)


spectra_water = pl.read_csv("data/2025-10-16-SO2-W_2200k.fft", 
                        has_header=True, 
                        skip_rows=14)

spectra_deu = pl.read_csv("data/2025-10-16-SO2-D-W_2000k.fft", 
                        has_header=True, 
                        skip_rows=14)

spectra_so2 = pl.read_csv("data/2025-10-19-SO2_2300k.fft", 
                        has_header=True, 
                        skip_rows=14)


sigma_list = [4e-6, 4e-6, 4e-6]

# Data construction: 
# --- df_signals: all data above noise
# --- df_int: all peaks and the respective intensity in the others spectra
df_all = concat_cols_on_freq([spectra_so2, spectra_water, spectra_deu],["so2", "water", "deu"])
df_signals, sigma3_list = noise_rm_all(df_all)
peak_dict = detect_peaks(df_signals)
peak_array = peaks_dict_to_arrays(peak_dict) # N arrays of [freq, int] pairs
all_peaks = combine_unique_freqs(peak_dict)
df_int = get_int_at_peaks_AIopt(all_peaks,df_signals,return_df=True)

#Add ratios and arctg2 features
df_int = ratio_arc_cols(df_int, ratio=True, arctan2=True)
df_signals = ratio_arc_cols(df_signals, ratio=True, arctan2=True)

#Remove duplicated peaks between spectra
df_int = unique_by_freq_keep_max3(df_int, "freq", "int_so2", "int_water", "int_deu", tol=0.05)

#Grouping by 
df_int, df_int_bool = int_is_peak(df_int, peak_array, 0.05)
df_int, df_int_groups_incr_decr = increase_or_decrease(df_int, 0.1)
df_groups_ispeak = groups_ispeak(df_int)
df_groups_incr_decrs = groups_incr_decr(df_int)
df_TTT = df_groups_ispeak["TTT"]
df_dec_dec = df_groups_incr_decrs["--"]
common = df_TTT.join(df_dec_dec, on="freq", how="semi") #all -- freqs belong to TTT (checked)
df_two_third_decr = how_much_decr_ref(df_dec_dec, "int_so2/int_water", "int_so2/int_deu", 0.39, 0.05)
df_two_third_decr
#print(df_two_third_decr.filter((pl.col("freq") - 5555.7).abs() < 0.1))
df_TTT, df_TTT_count_incr_decr = increase_or_decrease(df_TTT, 0.05)
df_h2o_dec = pl.concat([df_groups_incr_decrs["--"], df_groups_incr_decrs["-0"], df_groups_incr_decrs["-+"], df_groups_incr_decrs["-="]]).select(df_groups_incr_decrs["--"].columns[:4])
df_h2o_dec_inv = pl.concat([df_groups_incr_decrs["=="], df_groups_incr_decrs["=0"], df_groups_incr_decrs["+0"], 
                            df_groups_incr_decrs["+-"], df_groups_incr_decrs["00"], df_groups_incr_decrs["0-"], 
                            df_groups_incr_decrs["0+"], df_groups_incr_decrs["++"]]).select(df_groups_incr_decrs["--"].columns[:4])
df_h2o_dec_inv.height
df_h2o_dec.height
fft_df("lines_to_remove.dat", df_h2o_dec_inv, sep="\t", decimals=5)

# Temporarily show all rows
with pl.Config():
    pl.Config.set_tbl_rows(-1)  # -1 means show all rows
    print(df_int_groups_incr_decr)

df_int_groups_incr_decr
# Check single intensity with a tolerance
lines = df_int.filter((pl.col("freq") - 5555.7).abs() < 0.005)
lines

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
plot_base_peaks("plot_so2_peaks.html", df_all['freq'], df_all['int_so2'], peak_array['int_so2'][:,0], peak_array['int_so2'][:,1], sigma_list[0])
plot_base_peaks("plot_h2o_peaks.html", df_all['freq'], df_all['int_water'], peak_array['int_water'][:,0], peak_array['int_water'][:,1], sigma_list[0])
plot_base_peaks("plot_d2o_peaks.html", df_all['freq'], df_all['int_deu'], peak_array['int_deu'][:,0], peak_array['int_deu'][:,1], sigma_list[0])

plot_3d("plot_3d_int.html", df_int[:,1], df_int[:,2], df_int[:,3], min=0.0, max=0.001)
plot_3d("plot_3d_rat.html", df_int[:,4], df_int[:,5], df_int[:,6])
plot_3d("plot_3d_arc.html", df_int[:,7], df_int[:,8], df_int[:,9])

plot_2d_ratio_int("plot_2d_ratio12_i1.html", df_int["int_so2"], df_int["int_so2/int_water"])
plot_2d_ratio_int("plot_2d_ratio12_i2.html", df_int["int_water"], df_int["int_so2/int_water"])
plot_2d_ratio_int("plot_2d_ratio23_i2.html", df_int["int_water"], df_int["int_water/int_deu"])
plot_2d_ratio_int("plot_2d_ratio23_i3.html", df_int["int_deu"], df_int["int_water/int_deu"])
plot_2d_ratio_int("plot_2d_ratio32_i3.html", df_int["int_deu"], df_int["int_deu"]/df_int["int_water"])

plot_2d_int("plot_2d_so2_water.html", df_int['int_so2'], df_int['int_water'])
plot_2d_int("plot_2d_so2_deu.html", df_int['int_so2'], df_int['int_deu'])
plot_2d_int("plot_2d_water_deu.html", df_int['int_water'], df_int['int_deu'])

plot_2d_int("plot_2d_water_deu_FFF.html", df_FFF['int_water'], df_FFF['int_deu'])
plot_2d_ratio_int("plot_2d_ratio23_i2_FFF.html", df_FFF_rat["int_deu"], df_FFF_rat["int_water/int_deu"])
"""
df_signals
df_int.height
