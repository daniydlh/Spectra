import copy
import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
from gui_KMeans import KMeansAngularClusterer
from gui_src import freqs_df_to_cluster


st.set_page_config(
    page_title="KMeans Clustering",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 1600px !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* ── Run button ── */
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}

.run-btn-wrapper {
    margin: 1.2rem 0 1.5rem 0;
}

.run-btn-wrapper [data-testid="stButton"] > button {
    width: 100% !important;
    background: #0f172a !important;
    background-image: linear-gradient(
        105deg,
        transparent 30%,
        rgba(255,255,255,0.07) 48%,
        rgba(255,255,255,0.13) 50%,
        rgba(255,255,255,0.07) 52%,
        transparent 70%
    ) !important;
    background-size: 200% 100% !important;
    animation: shimmer 2.8s linear infinite !important;
    color: #e2e8f0 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 1rem !important;
    border-radius: 6px !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 2px 12px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.run-btn-wrapper [data-testid="stButton"] > button:hover {
    border-color: rgba(148,163,184,0.45) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 4px 20px rgba(0,0,0,0.4) !important;
    color: #ffffff !important;
}

.run-btn-wrapper [data-testid="stButton"] > button:active {
    transform: scale(0.995) !important;
}

/* Plotly chart card */
[data-testid="stPlotlyChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 6px 32px rgba(0,0,0,0.22), 0 1.5px 6px rgba(0,0,0,0.14) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    margin-bottom: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Guards ────────────────────────────────────────────────────────────────────
if "df_peaks" not in st.session_state:
    st.warning("Upload spectra in Spectra Analysis page.")
    st.stop()

if "mix_list_stored" not in st.session_state:
    st.warning("Select a name for each mixture spectra in Spectra Analysis page.")
    st.stop()

df_peaks = st.session_state["df_peaks"]
st.info(f"Loaded {len(df_peaks)} peaks — rerun model if you changed Spectral Analysis parameters.")
mix_list = st.session_state["mix_list_stored"]
col_names = [f"int_{mix_list[0]}", f"int_{mix_list[1]}"]

if df_peaks.empty:
    st.warning("No peaks found. Check page 1 parameters.")
    st.stop()

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS_KMEANS = {
    "km_n_clusters_mode": "Auto (elbow)",
    "km_n_clusters": 10,
    "km_max_clusters": 30,
    "km_outlier_thresh": 0.0,
    "km_n_init": 10,
    "km_max_iter": 300,
}

if st.session_state.get("_reset_params_kmeans"):
    for key, val in DEFAULTS_KMEANS.items():
        st.session_state[key] = val
    st.session_state["_reset_params_kmeans"] = False

refine_em = st.checkbox("Refinement of clusters by EM algorithm", value=False, key="km_EM")

# ── Sidebar ───────────────────────────────────────────────────────────────────
n_clusters_mode = st.sidebar.radio(
    "Number of clusters",
    options=["Auto (elbow)", "Fixed"],
    key="km_n_clusters_mode",
)

if n_clusters_mode == "Fixed":
    n_clusters = st.sidebar.number_input("Fixed k", step=1, key="km_n_clusters", value=10, min_value=1)
    max_clusters_val = None
else:
    n_clusters = None
    max_clusters_val = st.sidebar.number_input("Max clusters (elbow search)", step=1, key="km_max_clusters", value=30, min_value=2)

if refine_em:
    with st.sidebar.expander("EM algorithm"):
        max_em_it = st.number_input("Maximum EM iterations", step=10, key="km_max_em_it", value=100)
        conv = st.number_input("Convergence threshold", step=1, key="km_conv", value=0)

with st.sidebar.expander("Advanced"):
    outlier_thresh_raw = st.number_input(
        "Outlier angular threshold (0 = disabled)",
        step=0.005, key="km_outlier_thresh", value=0.0, format="%.4f",
    )
    outlier_thresh = outlier_thresh_raw if outlier_thresh_raw > 0.0 else None
    n_init = st.number_input("K-Means restarts (n_init)", step=1, key="km_n_init", value=10, min_value=1)
    max_iter = st.number_input("K-Means max iterations", step=50, key="km_max_iter", value=300, min_value=10)

if st.sidebar.button("Restore default parameters", key="km_restore"):
    st.session_state["_reset_params_kmeans"] = True
    st.rerun()

# ── Run controls ──────────────────────────────────────────────────────────────
run_clicked = st.button("RUN MODEL", key="km_run_model_btn", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if run_clicked:
    x = df_peaks[col_names[0]].to_numpy()
    y = df_peaks[col_names[1]].to_numpy()
    X = np.column_stack((x, y))

    with st.spinner(f"Fitting {X.shape[0]} points..."):
        clusterer = KMeansAngularClusterer(
            n_clusters=n_clusters,
            max_clusters=max_clusters_val if max_clusters_val is not None else 30,
            min_samples=2,
            outlier_threshold=outlier_thresh,
            max_iter=max_iter,
            n_init=n_init,
            random_state=42,
        )
        clusterer.fit(X)
        if refine_em:
            clusterer.iterative_refinement(
                X,
                max_iterations=max_em_it,
                convergence_threshold=conv,
                distance_mode="angular",
                verbose=True,
            )
        model = clusterer

    cols = ["freq", col_names[0], col_names[1], "cluster"]
    cols_freq_cluster = ["freq", "cluster"]

    df_output = model.write_df_output(
        pl.from_pandas(df_peaks),
        cols_to_fit=col_names,
        create_file=False,
        selected_cols=cols,
        sort_by_arctan=True,
        model_name=None,
        model_path=None,
    )

    peak_cluster = df_output.select(cols).filter(pl.col("cluster").is_not_null())
    freq_cluster = df_output.select(cols_freq_cluster).filter(pl.col("cluster").is_not_null())

    freqs = df_peaks["freq"].to_numpy()
    with st.spinner("Plotting model"):
        fig = model.plot_interactive(
            X, width=800, height=700,
            lims=None, cols=col_names, zoom_lims=None,
            peaks=peak_cluster,
            freqs=freqs,
            show_fig=False,
            sort_by_arctan=True,
            save_html=None, save_pdf=None, model_path=None,
            axis=mix_list,
        )

    # Elbow figure (only in auto mode)
    elbow_fig = None
    if n_clusters is None and model.inertias_:
        elbow_fig = model.plot_elbow(width=700, height=350, show_fig=False)

    st.session_state["km_model_fig"] = fig
    st.session_state["km_elbow_fig"] = elbow_fig
    st.session_state["km_model_info"] = (
        len(model.clusters_),
        len(model.unassigned),
        getattr(model, "n_iterations_", None),
        model.n_clusters_selected_,
    )
    st.session_state["km_freq_cluster"] = freq_cluster.to_pandas()
    st.session_state["km_peak_cluster_csv"] = peak_cluster.to_pandas().to_csv(index=False)
    st.session_state["km_clusterer"] = model
    st.session_state["km_model_params"] = {
        "n_clusters_mode": n_clusters_mode,
        "n_clusters": n_clusters,
        "max_clusters": max_clusters_val,
        "outlier_thresh": outlier_thresh,
        "n_init": n_init,
        "max_iter": max_iter,
        "refine": refine_em,
    }

# ── Display results ───────────────────────────────────────────────────────────
if "km_model_fig" in st.session_state:

    p = st.session_state["km_model_params"]

    st.success("Model ran successfully!")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_clusters_found, n_unassigned, n_iter, k_selected = st.session_state["km_model_info"]
        iter_str = f"  \n**EM iterations:** {n_iter}" if n_iter is not None else ""
        k_str = f"  \n**k selected:** {k_selected}" if k_selected is not None else ""
        st.info(f"**Clusters found:** {n_clusters_found} | **Unassigned:** {n_unassigned}{k_str}{iter_str}")
    with col_b:
        mode_str = p["n_clusters_mode"]
        k_info = (
            f"k = {p['n_clusters']}" if p["n_clusters"] is not None
            else f"max k = {p['max_clusters']}"
        )
        outlier_info = f"{p['outlier_thresh']:.4f}" if p["outlier_thresh"] else "off"
        param_str = (
            f"**Mode:** {mode_str} ({k_info}) | "
            f"**Outlier threshold:** {outlier_info} | "
            f"**n_init:** {p['n_init']} | "
            f"**Refine by EM:** {'Yes' if p['refine'] else 'No'}"
        )
        st.info(param_str)

    st.download_button(
        "⬇ Download cluster assignation (CSV)",
        st.session_state["km_peak_cluster_csv"],
        "kmeans_cluster_output.csv",
        key="km_download_cluster",
        use_container_width=True,
        type="primary",
    )

    # ── Tabs: Cluster plot | Elbow plot ───────────────────────────────────────
    has_elbow = st.session_state.get("km_elbow_fig") is not None
    tab_labels = ["Cluster plot", "Elbow plot"] if has_elbow else ["Cluster plot"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.subheader("Check the cluster of specific lines")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_freqs = st.file_uploader(
                "Upload list of frequencies of interest (one column)",
                accept_multiple_files=False,
                key="km_uploader_freqs_of_int",
            )
        with col2:
            match_tol = st.number_input(
                "Matching tolerance (MHz)", step=0.01, key="km_tol", value=0.01, format="%.3f"
            )

        # ── Parse uploaded frequencies and build highlight set ────────────────
        highlighted_freqs = set()
        assigned_freqs = None

        if uploaded_freqs:
            freq_cluster = st.session_state.get("km_freq_cluster")
            if freq_cluster is None:
                st.warning("Run the model first before uploading frequencies.")
            else:
                df_raw_input = pd.read_csv(uploaded_freqs, header=None)
                if isinstance(df_raw_input.iloc[0, 0], str) and "freq" in df_raw_input.iloc[0, 0].lower():
                    df_input = df_raw_input.iloc[1:].reset_index(drop=True)
                else:
                    df_input = df_raw_input
                df_input.columns = ["freq"] + list(df_input.columns[1:])
                df_input["freq"] = pd.to_numeric(df_input["freq"])

                assigned_freqs = freqs_df_to_cluster(df_input, freq_cluster, tol=match_tol)

                all_model_freqs = freq_cluster["freq"].to_numpy()
                for f_query in df_input["freq"].to_numpy():
                    matches = all_model_freqs[np.abs(all_model_freqs - f_query) <= match_tol]
                    highlighted_freqs.update(matches.tolist())

                st.download_button(
                    "⬇ Download assigned clusters (CSV)",
                    assigned_freqs.to_csv(index=False),
                    "kmeans_freqs_to_cluster.csv",
                    key="km_download_specific_freq",
                    use_container_width=True,
                    type="primary",
                )

        # ── Build display figure (highlight or normal) ────────────────────────
        DIM_OPACITY = 0.06

        def _hex_to_rgba(hex_color, alpha):
            hex_color = hex_color.lstrip("#")
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        def _parse_freq_from_hovertext(text):
            try:
                return float(str(text).split("Freq: ")[1].split("<br>")[0])
            except (IndexError, ValueError):
                return None

        if highlighted_freqs:
            display_fig = copy.deepcopy(st.session_state["km_model_fig"])

            for trace in display_fig.data:
                texts = getattr(trace, "text", None)
                if texts is None:
                    continue
                if isinstance(texts, str):
                    texts = [texts]
                if not any("Freq:" in str(t) for t in texts):
                    continue

                base_color = None
                if hasattr(trace, "marker") and trace.marker is not None:
                    bc = trace.marker.color
                    if isinstance(bc, str) and bc.startswith("#"):
                        base_color = bc
                    elif bc == "lightgray":
                        base_color = "#c0c0c0"

                rgba_colors = []
                rgba_line_colors = []
                for t in texts:
                    f = _parse_freq_from_hovertext(t)
                    is_match = f is not None and any(abs(f - hf) <= match_tol for hf in highlighted_freqs)
                    alpha = 1.0 if is_match else DIM_OPACITY
                    rgba_colors.append(
                        _hex_to_rgba(base_color, alpha) if base_color
                        else f"rgba(136,136,136,{alpha})"
                    )
                    rgba_line_colors.append(f"rgba(0,0,0,{alpha})")

                if rgba_colors:
                    trace.marker.color = rgba_colors
                    trace.marker.opacity = None
                    if trace.marker.line is not None:
                        trace.marker.line.color = rgba_line_colors

            st.plotly_chart(display_fig, use_container_width=True)
        else:
            st.plotly_chart(st.session_state["km_model_fig"], use_container_width=True)

    if has_elbow:
        with tabs[1]:
            st.subheader("Elbow plot — inertia vs number of clusters")
            st.caption(
                "The vertical red line marks the k selected automatically by the elbow criterion "
                "(largest second difference of the inertia curve). "
                "If the elbow looks wrong, switch to **Fixed** mode in the sidebar and set k manually."
            )
            st.plotly_chart(
                st.session_state["km_elbow_fig"],
                use_container_width=True,
                key="km_elbow_chart",
            )

# ── Cluster inspector ─────────────────────────────────────────────────────────
if "km_clusterer" in st.session_state:
    clusterer = st.session_state["km_clusterer"]
    clusters = clusterer.clusters_

    st.divider()
    st.subheader("Cluster inspector")

    sorted_clusters = sorted(clusters, key=lambda c: float(c["arctan"]))
    orig_id_to_display_idx = {c["id"]: i for i, c in enumerate(sorted_clusters)}

    cluster_options = {
        f"Cluster {i}  |  arctan = {c['arctan']:.4f}  |  slope = {c['slope']:.4f}  |  N = {c['n_points']}": i
        for i, c in enumerate(sorted_clusters)
    }

    selected_labels = st.multiselect(
        "Add clusters to inspect (sorted by arctan):",
        options=list(cluster_options.keys()),
        default=[],
        placeholder="Select one or more clusters...",
    )

    for label in selected_labels:
        display_idx = cluster_options[label]
        c = sorted_clusters[display_idx]

        with st.expander(f"Cluster {display_idx} — arctan {c['arctan']:.4f}", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("N points", c["n_points"])
            col2.metric("Slope", f"{c['slope']:.5f}")
            col3.metric("Intercept", f"{c['intercept']:.5f}")
            col4.metric("arctan", f"{c['arctan']:.5f}")
            col5.metric("MAD", f"{c['mad']:.6f}")

            pts = c["points"]
            df_pts = pd.DataFrame(pts, columns=[col_names[0], col_names[1]])

            ROUND = 8
            df_pts["_k0"] = df_pts[col_names[0]].round(ROUND)
            df_pts["_k1"] = df_pts[col_names[1]].round(ROUND)
            df_ref = df_peaks[[col_names[0], col_names[1], "freq"]].copy()
            df_ref["_k0"] = df_ref[col_names[0]].round(ROUND)
            df_ref["_k1"] = df_ref[col_names[1]].round(ROUND)

            df_pts = df_pts.merge(
                df_ref[["_k0", "_k1", "freq"]],
                on=["_k0", "_k1"],
                how="left",
            ).drop(columns=["_k0", "_k1"])

            df_pts = df_pts[["freq", col_names[0], col_names[1]]]
            df_pts.index.name = "point"

            st.dataframe(df_pts, use_container_width=True, height=300)

            fig_hist = clusterer.interactive_distance_histogram(
                cluster_id=c["id"],
                bins=50,
                width=700,
                height=300,
                show_fig=False,
            )
            fig_hist.update_layout(
                title=f"Angular distance histogram of each point to the cluster {display_idx} ray"
            )
            st.plotly_chart(fig_hist, use_container_width=True, key=f"km_hist_{display_idx}")

            st.download_button(
                label=f"⬇ Download cluster {display_idx} points (CSV)",
                data=df_pts.to_csv(index=False),
                file_name=f"kmeans_cluster_{display_idx}_arctan{c['arctan']:.4f}.csv",
                mime="text/csv",
                key=f"km_dl_cluster_{display_idx}",
                use_container_width=True,
            )

    # ── Echo file ─────────────────────────────────────────────────────────────
    if selected_labels:
        selected_orig_ids = set()
        for label in selected_labels:
            display_idx = cluster_options[label]
            c = sorted_clusters[display_idx]
            selected_orig_ids.add(c["id"])

        echo_mask = (
            np.isin(clusterer.labels_, list(selected_orig_ids), invert=True)
            & (clusterer.labels_ != -1)
        )
        echo_indices = np.where(echo_mask)[0]

        if len(echo_indices) > 0:
            X_full = np.column_stack((
                df_peaks[col_names[0]].to_numpy(),
                df_peaks[col_names[1]].to_numpy(),
            ))
            echo_xy = X_full[echo_indices]
            df_echo = pd.DataFrame(echo_xy, columns=[col_names[0], col_names[1]])

            df_echo["cluster"] = [
                orig_id_to_display_idx.get(clusterer.labels_[i], -1)
                for i in echo_indices
            ]

            ROUND = 8
            df_echo["_k0"] = df_echo[col_names[0]].round(ROUND)
            df_echo["_k1"] = df_echo[col_names[1]].round(ROUND)
            df_ref_echo = df_peaks[[col_names[0], col_names[1], "freq"]].copy()
            df_ref_echo["_k0"] = df_ref_echo[col_names[0]].round(ROUND)
            df_ref_echo["_k1"] = df_ref_echo[col_names[1]].round(ROUND)

            df_echo = df_echo.merge(
                df_ref_echo[["_k0", "_k1", "freq"]],
                on=["_k0", "_k1"],
                how="left",
            ).drop(columns=["_k0", "_k1"])

            df_echo = df_echo[["freq"]].sort_values("freq")

            st.divider()
            st.download_button(
                label="⬇ Download echo.acs",
                data=df_echo.to_csv(index=False, header=False),
                file_name="echo.acs",
                mime="text/csv",
                key="km_download_echo",
                use_container_width=True,
                type="primary",
            )
        else:
            st.info("No points remain for the echo file after excluding selected clusters.")
