
import polars as pl
from Algorithms.RANSAC.RANSAC import LinearClusterer
import numpy as np
import os
import shutil

def precluster_and_cluster_RANSAC(df, cols_to_fit, n_preclusters,
                                 ref_col_precluster, range_list, max_it_list,
                                 min_samples_list, max_clusters_list, euc_threshold_list, ang_threshold_list,
                                 angle_growth_list, angle_max, origin_cleaning_limits=None, remove_zeros=True, 
                                 force_origin=False, distance_type='euclidean'):
    
    # Check if arguments are coherent
    checks = {"Number of ranges": range_list, "Number of euclidean thresholds": euc_threshold_list,
              "Number of maximum iterations": max_it_list, "Number of minimum samples": min_samples_list}

    for name, value in checks.items():
        if n_preclusters != len(value):
            print(f"Number of preclusters is no equal to {name} (Number of preclusters = {n_preclusters}, {name} = {len(value)})")
            break
    if cols_to_fit is not None:
        if len(cols_to_fit) != 2:
            print(f"Fitting must be 2D for this function: Number of columns of to fit = {len(cols_to_fit)} != 2")
            return

    # General filtering of the messy origin areas (2D) (optional)
    if origin_cleaning_limits is not None:
        if len(cols_to_fit) != 2:
            print(f"Cleaning must be 2D for this function: Number of columns of reference = {len(cols_to_fit)} != 2")
            return
        df = df.filter((pl.col(cols_to_fit[0]) > origin_cleaning_limits[0]) | (pl.col(cols_to_fit[1]) > origin_cleaning_limits[1]))
    
    # Removing rows with a 0 in columns to fit (optional)
    if remove_zeros is True:
        df = df.filter((pl.col(cols_to_fit[0]) != 0.) & (pl.col(cols_to_fit[1]) != 0.)) 

    # Creates a dictionary of precluster depending on range_list values. None = infinite.
    preclusters_dict = {}
    
    for low, high in range_list:
        if high is None:
            preclusters_dict[f"df_{low}_inf"] = df.filter((pl.col(ref_col_precluster) >= low))
        else:
            preclusters_dict[f"df_{low}_{high}"] = df.filter((pl.col(ref_col_precluster) >= low) & (pl.col(ref_col_precluster) < high))
    
    # Run RANSAC model for each cluster
    models_preclusters_dict = {}
    X_dict = {}

    for j, (precluster_key , precluster_values) in enumerate(preclusters_dict.items()):
        x = precluster_values[cols_to_fit[0]].to_numpy()
        y = precluster_values[cols_to_fit[1]].to_numpy()
        X = np.column_stack((x,y))

        # Create, fit clusterer and store the returned object in a dictionary

        clusterer = LinearClusterer(
            distance_threshold=euc_threshold_list[j],
            angle_threshold=ang_threshold_list[j],
            angle_growth=angle_growth_list[j],
            angle_max=angle_max,
            min_samples=min_samples_list[j],
            max_clusters=max_clusters_list[j],
            max_iterations=max_it_list[j],
            force_origin=force_origin,
            distance_type=distance_type
        )
        models_preclusters_dict[f"model_{precluster_key}"] = clusterer.fit(X) # .fit() returns self so clusterer is equal to model = clusterer.fit()
        X_dict[f"model_{precluster_key}"] = X 

    return models_preclusters_dict, X_dict


################################################################################################

def write_model_info_and_plots(models, X, df_ref, cols_to_fit, dir, plot_lims_tuple=None):
    
    df_output_dict = {}

    for m in models:

        model_dir = f"{dir}/{m}"
        print(models)
        # Create the directory (and any missing parent dirs) if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        models[m].plot_interactive(X[m], lims=plot_lims_tuple, dir=f"{model_dir}/plot_{m}.html",)
        df_output_dict[m] = models[m].write_df_output(df_ref, cols_to_fit)
        
        for cluster in models[m].clusters_:
            models[m].write_cluster_dat(f"{model_dir}/{m}_cluster_{cluster['id']}.dat", cluster_id=cluster['id'])
    return df_output_dict
