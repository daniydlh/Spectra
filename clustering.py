from data_analysis import df_signals, df_int
from Algorithms.RANSAC.algorithm_utils import precluster_and_cluster_RANSAC, write_model_info_and_plots
import polars as pl
import numpy as np

"""
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
angle_growth_list=[1]
angle_max = 0.06
"""
ratio_ranges = [
        (0.0, 0.9),
        (0.9, 1.1),
        (1.1, None)
    ]

euc_threshold_list = [8e-6, 1.0e-5,2e-5]
ang_threshold_list = [0.0285, 0.0015, 0.1] #rads
angle_growth_list=[0., 0., 0.]
angle_max = 0.03
max_it_list = [5000,5000,5000]
min_samples_list = [2,2,2]
max_clusters_list = [15,4,20]
cols_to_fit = ["int_water", "int_deu"]
origin_cleaning_limits = [0, 0]


models, X_dict = precluster_and_cluster_RANSAC(df_signals, cols_to_fit, 3, "int_water/int_deu", ratio_ranges, 
                                         max_it_list, min_samples_list, max_clusters_list, euc_threshold_list, ang_threshold_list, 
                                         angle_growth_list, angle_max, origin_cleaning_limits, force_origin=True, 
                                         distance_type='angular')


df_int = df_int.filter((pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))
df_int.height
plot_lims = [[0,0.0014],[0,5e-4]]
df_output_dict = write_model_info_and_plots(models, X_dict, df_int, cols_to_fit, "models/RANSAC",)

models["model_df_0.9_1.1"].get_cluster_info()
n_unassigned = int(np.sum(models["model_df_0.9_1.1"].labels_ == -1))
print(n_unassigned)
models["model_df_0.9_1.1"].clusters_
df_int.height
df_output_dict
"""
selected_cols = ['freq', 'cluster']  # choose columns you want
df_output_dict["model_df_0.0_inf"].select(selected_cols).write_csv("output_model_df_0.0_inf.csv")
"""