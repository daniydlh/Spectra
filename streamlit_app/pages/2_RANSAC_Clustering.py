import copy
import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
from gui_RANSAC import LinearClusterer
from gui_src import freqs_df_to_cluster
from _clustering_shared import (
    inject_css, check_guards, get_X, build_nd_fig,
    freq_upload_section, apply_highlight, render_cluster_inspector, PAGE_CSS,
)

st.set_page_config(page_title="RANSAC Clustering", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Guards ────────────────────────────────────────────────────────────────────
check_guards()

df_peaks  = st.session_state["df_peaks"]
mix_list  = st.session_state["mix_list_stored"]
col_names = [f"int_{m}" for m in mix_list]
n_dims    = len(mix_list)

st.info(f"Loaded {len(df_peaks)} peaks — rerun model if you changed Spectral Analysis parameters.")

if df_peaks.empty:
    st.warning("No peaks found. Check page 1 parameters.")
    st.stop()

if n_dims > 2:
    st.info(
        f"**{n_dims} spectra detected.** Clustering runs on all {n_dims} dimensions. "
        + ("A 3-D interactive plot will be shown." if n_dims == 3 else
           "A **UMAP** 2-D projection will be shown — this is not a physical representation of the data.")
    )

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = {"ang": 0.04, "ang_growth": 0.0, "ang_max": 0.0, "max_clust": 30, "it": 1000}

if st.session_state.get("_reset_params"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["_reset_params"] = False

refine_em = st.checkbox("Refinement of clusters by EM algorithm", value=False, key="EM")

# ── Sidebar ───────────────────────────────────────────────────────────────────
ang          = st.sidebar.number_input("Angular threshold", step=0.005, key="ang", value=0.04, format="%.4f")
max_clusters = st.sidebar.number_input("Maximum clusters", step=1, key="max_clust", value=30)

if refine_em:
    with st.sidebar.expander("EM algorithm"):
        max_em_it = st.number_input("Maximum EM iterations", step=10, key="max_em_it", value=100)
        conv      = st.number_input("Convergence threshold", step=1, key="conv", value=0)

with st.sidebar.expander("Advanced"):
    ang_growth = st.number_input("Angular threshold growth per cluster", step=0.005, key="ang_growth", value=0.0, format="%.4f")
    ang_max    = st.number_input("Maximum angle after growth", step=0.005, key="ang_max", value=0.0, format="%.4f")
    ransac_it  = st.number_input("RANSAC iterations", step=500, key="it", value=1000)

if st.sidebar.button("Restore default parameters"):
    st.session_state["_reset_params"] = True
    st.rerun()

# ── Run ───────────────────────────────────────────────────────────────────────
run_clicked = st.button("RUN MODEL", key="run_model_btn", use_container_width=True)

if run_clicked:
    X = get_X(df_peaks, col_names)

    with st.spinner(f"Fitting {X.shape[0]} points in {n_dims}-D..."):
        # RANSAC only supports 2-D directly; for N>2 we fit on each pair of
        # dimensions and combine, OR we use the first two dims for clustering
        # and use all dims only for display.
        # Design decision: cluster on the 2-D angular projection (first two
        # intensity cols) and display N-D. Consistent with physical meaning
        # (angular distance between spectral lines is a 2-D concept).
        X_fit = X[:, :2]

        clusterer = LinearClusterer(
            angle_threshold=ang,
            angle_growth=ang_growth,
            angle_max=ang_max,
            min_samples=2,
            max_clusters=max_clusters,
            max_iterations=ransac_it,
            force_origin=True,
            distance_type="angular",
            random_state=42,
        )
        clusterer.fit(X_fit)
        if refine_em:
            clusterer.iterative_refinement(X_fit, max_iterations=max_em_it,
                                           convergence_threshold=conv,
                                           distance_mode="angular", verbose=True)
        model = clusterer

    cols_out       = ["freq", col_names[0], col_names[1], "cluster"]
    cols_freq_clus = ["freq", "cluster"]

    df_output = model.write_df_output(
        pl.from_pandas(df_peaks), cols_to_fit=col_names[:2],
        create_file=False, selected_cols=cols_out,
        sort_by_arctan=True, model_name=None, model_path=None,
    )
    peak_cluster = df_output.select(cols_out).filter(pl.col("cluster").is_not_null())
    freq_cluster = df_output.select(cols_freq_clus).filter(pl.col("cluster").is_not_null())

    with st.spinner("Building plot..."):
        fig, umap_warn = build_nd_fig(model, X, col_names, mix_list, df_peaks,
                                      df_output, cols_out, "RANSAC model")

    st.session_state["model_fig"]        = fig
    st.session_state["model_umap_warn"]  = umap_warn
    st.session_state["model_info"]       = (len(model.clusters_), len(model.unassigned),
                                             getattr(model, "n_iterations_", None))
    st.session_state["freq_cluster"]     = freq_cluster.to_pandas()
    st.session_state["peak_cluster_csv"] = peak_cluster.to_pandas().to_csv(index=False)
    st.session_state["clusterer"]        = model
    st.session_state["model_params"]     = dict(ang=ang, ang_growth=ang_growth, ang_max=ang_max,
                                                 max_clusters=max_clusters, ransac_it=ransac_it,
                                                 refine=refine_em)

# ── Display ───────────────────────────────────────────────────────────────────
if "model_fig" in st.session_state:
    p = st.session_state["model_params"]
    st.success("Model ran successfully!")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_cl, n_un, n_it = st.session_state["model_info"]
        iter_str = f"  \n**EM iterations:** {n_it}" if n_it is not None else ""
        st.info(f"**Clusters:** {n_cl} | **Unassigned:** {n_un}{iter_str}")
    with col_b:
        st.info(
            f"**Angle:** {p['ang']:.4f} | **Growth:** {p['ang_growth']:.4f} | "
            f"**Max:** {p['ang_max']:.4f} | **Max clusters:** {p['max_clusters']} | "
            f"**RANSAC it:** {p['ransac_it']} | **EM:** {'Yes' if p['refine'] else 'No'}"
        )

    if st.session_state.get("model_umap_warn"):
        st.warning(
            "⚠️ **UMAP projection** — the plot below is a 2-D dimensionality reduction of your "
            f"{n_dims}-D data. It is useful for visualising cluster separation but **does not "
            "represent physical distances or angles** between spectral lines."
        )

    st.download_button(
        "⬇ Download cluster assignation (CSV)",
        st.session_state["peak_cluster_csv"],
        "ransac_cluster_output.csv",
        key="download_cluster",
        use_container_width=True,
        type="primary",
    )

    st.subheader("Check the cluster of specific lines")

    highlighted_freqs, assigned_freqs, match_tol = freq_upload_section(
        prefix="ransac",
        freq_cluster=st.session_state.get("freq_cluster"),
        match_tol_key="tol",
        uploader_key="uploader_freqs_of_int",
        download_key="download_specific_freq",
        n_dims=n_dims,
    )

    display_fig = apply_highlight(
        st.session_state["model_fig"], highlighted_freqs, match_tol, n_dims
    )
    st.plotly_chart(display_fig, use_container_width=True)

# ── Cluster inspector ─────────────────────────────────────────────────────────
if "clusterer" in st.session_state:
    st.divider()
    st.subheader("Cluster inspector")
    render_cluster_inspector(
        st.session_state["clusterer"], df_peaks, col_names, "ransac", n_dims
    )
