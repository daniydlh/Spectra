from polars import Null
from data_analysis import df_signals, df_int
from Algorithms.RANSAC.algorithm_utils import save_clustering_from_input_lines, precluster_and_cluster_RANSAC, write_model_info_and_plots
import polars as pl
import numpy as np
import plotly.graph_objects as go

molecule = 'so2'
spectra = ['so2', 'h2o', 'd2o']
i1, i2, i3 = f'int_{spectra[0]}', f'int_{spectra[1]}', f'int_{spectra[2]}'
cols = [i1, i2, i3]

# NO preclustering
euc_threshold_list = [8e-6]
ang_threshold_list = [0.04] #rads
angle_growth_list=[0.0]
angle_max = ang_threshold_list[0]
max_it_list = [1000]
min_samples_list = [2]
max_clusters_list = [30]
cols_to_fit = [i2, i3]
ref_col = f"{i2}/{i3}"
origin_cleaning_limits = [0.0, 0.0]
ratio_ranges = [(0.0, None)]

#df_int = df_int.filter((pl.col(i2) != 0.) & (pl.col(i3) != 0.))

"""
# PRECULSTERING
ratio_ranges = [
        (0.0, 0.9),
        (0.9, 1.1),
        (1.1, None)
    ]

euc_threshold_list = [8e-6, 1.0e-5,2e-5]
ang_threshold_list = [0.04, 0.03, 0.03] #rads
angle_growth_list=[0., 0., 0.]
angle_max = 0.03
max_it_list = [10000,10000,10000]
min_samples_list = [2,2,2]
max_clusters_list = [15,4,15]
cols_to_fit = ["int_water", "int_deu"]
origin_cleaning_limits = [0., 0.]
"""

# RUNNING MODEL
models, X_dict = precluster_and_cluster_RANSAC(df_int, cols_to_fit, 1, ref_col, ratio_ranges, 
                                         max_it_list, min_samples_list, max_clusters_list, euc_threshold_list, ang_threshold_list, 
                                         angle_growth_list, angle_max, origin_cleaning_limits=origin_cleaning_limits, force_origin=True, 
                                         distance_type='angular', remove_zeros=False, seed=42, sklearn_bool=False)


# CREATING OUTPUT AND PLOTS
#plot_lims = [[-1,60],[-1,46]]
#zoom_plot_lims = [[-0.01,1.],[-0.01,1.]]
zoom_plot_lims = None
df_output_dict = write_model_info_and_plots(models, X_dict, df_int, cols_to_fit, rltv_path=f"models/{molecule}/RANSAC",
                                            sort_by_arctan=True, interactive_plot=True, plot_lims_tuple=None, 
                                            zoom_lims=zoom_plot_lims)

# MODEL INFO
#models["model_df_0.0_inf"].interactive_distance_histogram(5,bins=100,xlims=None, histnorm=None, save_pdf=False, save_html=False, height=300, output="histogram_model_df_0.0_inf")
#models["model_df_0.0_inf"].global_hist(nbins=80, save_pdf=False, save_html=True)
print("Unassigned datapoints:", len(models["model_df_0.0_inf"].unassigned))
print("Number of clusters:", len(models["model_df_0.0_inf"].clusters_))

# CHECK SPECIFIC FREQ
#print(df_output_dict["model_df_0.0_inf"].filter((pl.col("freq") -  9108.2720 ).abs() < 0.1))

"""
# CHECK CLUSTER OF A SET OF LINES
input_file = "lines/4so2/4so2_freqs.csv"
lines = np.loadtxt(input_file)
save_clustering_from_input_lines(df_output_dict["model_df_0.0_inf"], lines, input_file, model="0.04_ang", save_csv=True)
"""
