from data_analysis import df_signals, df_int
from Algorithms.RANSAC.algorithm_utils import precluster_and_cluster_RANSAC, write_model_info_and_plots
import polars as pl
import numpy as np



# NO preclustering
euc_threshold_list = [8e-6]
ang_threshold_list = [0.04] #rads
max_it_list = [10000]
min_samples_list = [2]
max_clusters_list = [40]
cols_to_fit = ["int_water", "int_deu"]
origin_cleaning_limits = [0.000, 0.000]
ratio_ranges = [
        (0.0, None)
    ]
angle_growth_list=[0.]
angle_max = 0.05
"""
ratio_ranges = [
        (0.0, 0.9),
        (0.9, 1.1),
        (1.1, None)
    ]

euc_threshold_list = [8e-6, 1.0e-5,2e-5]
ang_threshold_list = [0.025, 0.025, 0.025] #rads
angle_growth_list=[0., 0., 0.]
angle_max = 0.03
max_it_list = [3000,3000,10000]
min_samples_list = [2,2,2]
max_clusters_list = [15,4,20]
cols_to_fit = ["int_water", "int_deu"]
origin_cleaning_limits = [0.00017, 0.00012]
"""


models, X = precluster_and_cluster_RANSAC(df_signals, cols_to_fit, 1, "int_water/int_deu", ratio_ranges, 
                                         max_it_list, min_samples_list, max_clusters_list, euc_threshold_list, ang_threshold_list, 
                                         angle_growth_list, angle_max, origin_cleaning_limits, force_origin=True, 
                                         distance_type='angular')

plot_lims = [[0,0.0014],[0,5e-4]]
write_model_info_and_plots(models, X, "models/RANSAC")
models
models["model_df_0.0_inf"].get_cluster_info()
n_unassigned = int(np.sum(models["model_df_0.0_inf"].labels_ == -1))
print(n_unassigned)
len(models["model_df_0.0_inf"].clusters_)
