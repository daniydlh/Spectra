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

DEFAULTS = {
    "ang": 0.04,
    "ang_growth": 0.0,
    "ang_max": 0.0,
    "max_clust": 30,
    "it": 1000,
}

# Reset flag must be handled BEFORE widgets are created
if st.session_state.get("_reset_params"):
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.session_state["_reset_params"] = False

# --- SIDEBAR ---
ang = st.sidebar.number_input("Angular threshold", step=0.005, key="ang", value=0.04, format="%.4f")
max_clusters = st.sidebar.number_input("Maximum clusters", step=1, key="max_clust", value=30)

with st.sidebar.expander("Advanced"):
    ang_growth = st.number_input("Angular threshold growth per cluster", step=0.005, key="ang_growth", value=0.0, format="%.4f")
    ang_max = st.number_input("Maximum angle possible after growth", step=0.005, key="ang_max", value=0.6, format="%.4f")
    iterations = st.number_input("RANSAC iterations", step=500, key="it", value=1000)

if st.sidebar.button("Restore default parameters"):
    st.session_state["_reset_params"] = True
    st.rerun()


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
    st.session_state["clusterer"] = model   # ← add this


# --- DISPLAY RESULTS (persists across reruns) ---
if "model_fig" in st.session_state:
    n_clusters, n_unassigned = st.session_state["model_info"]
    st.success("Model ran successfully!")
    st.info(f"**Clusters found:** {n_clusters} | **Unassigned points:** {n_unassigned}")
    st.download_button(
        "Download cluster assignation (CSV)",
        st.session_state["peak_cluster_csv"],
        "cluster_output.csv",
        key="download_cluster"
    )
    st.plotly_chart(st.session_state["model_fig"], use_container_width=True)

# --- CLUSTER INSPECTOR (add after the plotly_chart display) ---

if "model_fig" in st.session_state and "clusterer" not in st.session_state:
    # Store clusterer too — add this inside your "Run model" button block:
    # st.session_state["clusterer"] = model   ← add this line in your button block
    pass

if "clusterer" in st.session_state:
    clusterer = st.session_state["clusterer"]
    clusters = clusterer.clusters_

    st.divider()
    st.subheader("Cluster inspector")

    # Build display options: sorted by arctan
    sorted_clusters = sorted(clusters, key=lambda c: float(c["arctan"]))
    cluster_options = {
        f"Cluster {i}  |  arctan = {c['arctan']:.4f}  |  slope = {c['slope']:.4f}  |  N = {c['n_points']}": i
        for i, c in enumerate(sorted_clusters)
    }

    selected_labels = st.multiselect(
        "Add clusters to inspect (sorted by arctan):",
        options=list(cluster_options.keys()),
        default=[],
        placeholder="Select one or more clusters..."
    )

    for label in selected_labels:
        display_idx = cluster_options[label]
        c = sorted_clusters[display_idx]

        with st.expander(f"Cluster {display_idx} — arctan {c['arctan']:.4f}", expanded=True):
            # --- Characteristics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("N points", c["n_points"])
            col2.metric("Slope", f"{c['slope']:.5f}")
            col3.metric("Intercept", f"{c['intercept']:.5f}")
            col4.metric("arctan", f"{c['arctan']:.5f}")
            col5.metric("MAD", f"{c['mad']:.4f}")

            # --- Points table
            pts = c["points"]
            df_pts = pd.DataFrame(pts, columns=[col_names[0], col_names[1]])
            # Join freq from df_peaks by matching on the two intensity columns
            df_pts = df_pts.merge(
                df_peaks[[col_names[0], col_names[1], "freq"]],
                on=[col_names[0], col_names[1]],
                how="left"
            )
            df_pts = df_pts[["freq", col_names[0], col_names[1]]]  # reorder columns
            df_pts.index.name = "point"
            st.dataframe(df_pts, use_container_width=True, height=300)

            # --- Angular distance histogram

            fig_hist = clusterer.interactive_distance_histogram(
                cluster_id=c["id"],
                bins=50,
                width=700,
                height=300,
                show_fig=False,
            )

            fig_hist.update_layout(
                title="Angular distance to cluster ray"
            )
            st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{display_idx}")


            # --- Download
            csv_pts = df_pts.to_csv(index=False)
            st.download_button(
                label=f"Download cluster {display_idx} points (CSV)",
                data=csv_pts,
                file_name=f"cluster_{display_idx}_arctan{c['arctan']:.4f}.csv",
                mime="text/csv",
                key=f"dl_cluster_{display_idx}"
            )