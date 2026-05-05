import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
from gui_RANSAC import LinearClusterer

if "df_peaks" not in st.session_state:
    st.warning("Upload spectra in Spectra Analysis page.")
    st.stop()

if "mix_list_stored" not in st.session_state:
    st.warning("Select a name for each mixture spectra in Spectra Analisys page")
    st.stop()

df_peaks = st.session_state["df_peaks"]
st.info(f"Loaded {len(df_peaks)} peaks — rerun model if you changed Spectral Analisys parameters.")
mix_list = st.session_state["mix_list_stored"]
col_names = [f"int_{mix_list[0]}", f"int_{mix_list[1]}"]

if df_peaks.empty:
    st.warning("No peaks found. Check page 1 parameters.")
    st.stop()

# --- SIDEBAR ---
ang = st.sidebar.number_input("Angular threshold", step=0.005, key="ang", value=0.04, format="%.3f")
ang_growth = st.sidebar.number_input("Angular threshold growth per cluster", step=0.005, key="ang_growth", value=0.0, format="%.3f")
ang_max = st.sidebar.number_input("Maximum angle possible after growth", step=0.005, key="ang_max", value=0.6, format="%.3f")
max_clusters = st.sidebar.number_input("Maximum clusters", step=1, key="max_clust", value=30)
iterations = st.sidebar.number_input("RANSAC iterations", step=500, key="it", value=1000)

# --- RUN ---
if st.button("Run model", key="run_model_btn"):
    x = df_peaks[col_names[0]].to_numpy()
    y = df_peaks[col_names[1]].to_numpy()
    X = np.column_stack((x, y))
    

    st.write(f"Fitting {X.shape[0]} points...")

    with st.spinner("Running model..."):
        clusterer = LinearClusterer(
            angle_threshold=ang,
            angle_growth=ang_growth,
            angle_max=ang_max,
            min_samples=2,
            max_clusters=max_clusters,
            max_iterations=iterations,
            force_origin=True,
            distance_type="angular"
        )
        model = clusterer.fit(X)

    cols = ['freq', col_names[0], col_names[1], 'cluster']
    df_output = model.write_df_output(
        pl.from_pandas(df_peaks),
        cols_to_fit=col_names,
        create_file=False,
        selected_cols=cols,
        sort_by_arctan=True,
        model_name=None,
        model_path=None
    )

    peak_cluster = df_output.select(cols).filter(pl.col("cluster").is_not_null())

    freqs = df_peaks["freq"].to_numpy()
    with st.spinner("Plotting model"):
        fig = model.plot_interactive(
            X, width=800, height=600,
            lims=None, cols=col_names, zoom_lims=None,
            peaks=peak_cluster,
            freqs=freqs,
            show_fig=False,
            sort_by_arctan=True,
            save_html=None, save_pdf=None, model_path=None
        )

    # Store everything in session state
    st.session_state["model_fig"] = fig
    st.session_state["model_info"] = (len(model.clusters_), len(model.unassigned))
    st.session_state["peak_cluster_csv"] = peak_cluster.to_pandas().to_csv(index=False)

# --- DISPLAY RESULTS (persists across reruns) ---
if "model_fig" in st.session_state:
    n_clusters, n_unassigned = st.session_state["model_info"]
    st.success("Model ran successfully!")
    st.info(f"**Clusters found:** {n_clusters} | **Unassigned points:** {n_unassigned} | **ang:** {ang} | **ang_growth:** {ang_growth} | **ang_max:** {ang_max} | **max_clusters:** {max_clusters} | **iterations:** {iterations}")
    st.download_button(
        "Download cluster assignation (CSV)",
        st.session_state["peak_cluster_csv"],
        "cluster_output.csv",
        key="download_cluster"
    )
    st.plotly_chart(st.session_state["model_fig"], use_container_width=True)

