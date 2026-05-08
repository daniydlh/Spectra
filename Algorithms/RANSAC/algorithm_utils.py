import shutil
import polars as pl
from Algorithms.RANSAC.RANSAC import LinearClusterer
import numpy as np
import os
from pathlib import Path


def precluster_and_cluster_RANSAC(df, cols_to_fit, n_preclusters,
                                 ref_col_precluster, range_list, max_it_list,
                                 min_samples_list, max_clusters_list, euc_threshold_list, ang_threshold_list,
                                 angle_growth_list, angle_max, origin_cleaning_limits=None, remove_zeros=False, 
                                 force_origin=False, distance_type='angular', seed=42, sklearn_bool=False,
                                 reassign=False):
    
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
            distance_type=distance_type,
            random_state=seed,
            use_sklearn_ransac=sklearn_bool
        )
        
        print(f"Clustering algorithm iniciated for model_{precluster_key}")
        clusterer.fit(X)

        # Reassign all points (including unassigned) to nearest cluster line
        if reassign:
            clusterer.reassign_by_angular_proximity(X, distance_mode='angular')

        models_preclusters_dict[f"model_{precluster_key}"] = clusterer
        X_dict[f"model_{precluster_key}"] = X

    return models_preclusters_dict, X_dict


################################################################################################


def create_output(df, model, cols):

        output_file = f"output_{model}.csv"
        df.select(cols).filter(pl.col("cluster").is_not_null()).write_csv(output_file)
        
        info_lines = [
            f"# Model: {model}",
            f"# Number of clusters: {len(model.clusters_)}",
            "# Clusters: " + ", f".join([f"{c['id']} ({len(c['points'])} transitions)" for c in model.clusters_]),
            f"# Unassigned transitions: {len(model.unassigned)}"
        ]

        with open(output_file, "r") as f:
            csv_content = f.read()
        
        with open(output_file, "w") as f:
            for line in info_lines:
                f.write(line + "\n")
            f.write(csv_content)


################################################################################################


################################################################################################


def write_model_info_and_plots(models, X, df_ref_peaks, cols_to_fit, rltv_path, plot_lims_tuple=None, 
                              interactive_plot=False, cluster_info=True, create_full_output_file=True, 
                              sort_by_arctan=False, zoom_lims=None):
    
    df_output_dict = {}
    
    for m in models:

        out_path = Path(rltv_path) / m

        # If directory exists, ask user before overwriting
        if out_path.exists():
            answer = input(f"The directory '{out_path}' already exists. Overwrite? [y/N]: ").strip().lower()
            if answer == 'y':
                shutil.rmtree(out_path)
                print(f"Directory '{out_path}' has been overwritten.")
            else:
                answer = input(f"Rename '{out_path}'. For example: {rltv_path}/model_test_01. Enter to skip model: ").strip().lower()
                if answer != "":
                    out_path = Path(answer) / m
                else:
                    print(f"Skipping model '{m}' because directory exists and it was not renamed")
                    continue  # skip this model

        out_path.mkdir(parents=True, exist_ok=True)

        print(f"Writing model (only maxima) clusterization for {m}...")
        cols = ['freq', cols_to_fit[0], cols_to_fit[1], 'cluster']  # choose columns you want
        df_output_dict[m] = models[m].write_df_output(df_ref_peaks, 
                                    cols_to_fit, create_file=create_full_output_file, 
                                    selected_cols=cols, sort_by_arctan=sort_by_arctan, 
                                    model_name=m, model_path=f"{out_path}/full_output_{m}.csv")

        if interactive_plot is True:
            print(f"Interactive plot for {m}...")
            peak_cluster = df_output_dict[m].select(cols).filter(pl.col("cluster").is_not_null())
            models[m].plot_interactive(X[m], lims=plot_lims_tuple, cols=cols_to_fit, peaks=peak_cluster,
                                         model_path=f"{out_path}/plot_{m}", save_html=True,
                                          save_pdf=True, zoom_lims=zoom_lims, width=600,
                                           height=600, sort_by_arctan=sort_by_arctan)
        else:
            print(f"Plot for {m}...")
            models[m].plot(X[m])
        
        if cluster_info is True:

            print(f"Writing cluster information for {m}...")

            if sort_by_arctan is True:
                sorted_clusters = sorted(models[m].clusters_, key=lambda c: float(c["arctan"]))
                cluster_id_to_index = {c["id"]: idx for idx, c in enumerate(sorted_clusters)}
            else:
                sorted_clusters = sorted(models[m].clusters_, key=lambda c: float(c["id"]))
                cluster_id_to_index = {c["id"]: idx for idx, c in enumerate(sorted_clusters)}

            for cluster in models[m].clusters_:
                cluster_index = cluster_id_to_index[cluster["id"]]
                models[m].write_cluster_dat(f"{out_path}/{m}_cluster_{cluster_index}.csv", cluster_id=cluster_index)


    return df_output_dict

##############

def save_clustering_from_input_lines(df, lines, input_file, model=None, save_csv=False, tol=0.01):

    rows = []

    for freq in lines:
        closest = (
            df.filter((pl.col("freq") - freq).abs() < tol)
            .drop_nulls("cluster")
            .sort((pl.col("freq") - freq).abs())
            .select(["freq", "cluster"])
            .head(1)
        )

        if closest.height:
            f, c = closest.row(0)
            rows.append((round(float(f), 4), int(c)))

    result_df = pl.DataFrame(rows, schema=["freq", "cluster"])

    if save_csv:
        input_path = Path(input_file)
        output_path = input_path.with_name(input_path.stem + f"_clustering_{model}.csv")

        result_df.write_csv(output_path)

        print(f"Saved clustering file to: {output_path}")

    return result_df


