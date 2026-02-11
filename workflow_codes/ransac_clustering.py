from data_analysis import df_signals, df_int
from Algorithms.RANSAC.algorithm_utils import precluster_and_cluster_RANSAC, write_model_info_and_plots
import polars as pl
import numpy as np


# NO preclustering
euc_threshold_list = [8e-6]
ang_threshold_list = [0.02] #rads
max_it_list = [10000]
min_samples_list = [2]
max_clusters_list = [22]
cols_to_fit = ["int_water", "int_deu"]
origin_cleaning_limits = [0.0, 0.0]
ratio_ranges = [
        (0.0, None)
    ]
angle_growth_list=[0.3]
angle_max = 0.04
"""
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
models, X_dict = precluster_and_cluster_RANSAC(df_signals, cols_to_fit, 1, "int_water/int_deu", ratio_ranges, 
                                         max_it_list, min_samples_list, max_clusters_list, euc_threshold_list, ang_threshold_list, 
                                         angle_growth_list, angle_max, origin_cleaning_limits=origin_cleaning_limits, force_origin=True, 
                                         distance_type='angular', remove_zeros=False)

#df_int_filt = df_int.filter((pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))

plot_lims = [[-1,60],[-1,46]]
zoom_plot_lims = [[-0.01,1.],[-0.01,1.]]
df_output_dict = write_model_info_and_plots(models, X_dict, df_int, cols_to_fit, rltv_path="models/RANSAC", interactive_plot=True, plot_lims_tuple=plot_lims, zoom_lims=zoom_plot_lims)

models["model_df_0.0_inf"].interactive_distance_histogram(19, bins=35)
len(models["model_df_0.0_inf"].unassigned)
models["model_df_0.0_inf"].clusters_


#lines = df_output_dict["model_df_0.0_inf"].filter((pl.col("freq") - 2437.60347).abs() < 0.01)
#lines
"""
selected_cols = ['freq', 'int_water', 'int_deu', 'cluster']  # choose columns you want
df_output_dict['model_df_1.1_inf'].select(selected_cols).filter(pl.col("cluster").is_not_null()).sort("freq").write_csv("output_model_df_1.1_inf.csv",float_precision=8)
"""
