"""
Shared helpers for all clustering pages (RANSAC, DBSCAN, KMeans).
Handles N-D aware plotting, frequency highlighting and cluster inspector.
"""
import copy
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from gui_src import plot_nd_interactive, freqs_df_to_cluster, col_names_from_mix

# ── CSS (injected once per page) ─────────────────────────────────────────────
PAGE_CSS = """
<style>
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 1600px !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
.run-btn-wrapper [data-testid="stButton"] > button {
    width: 100% !important;
    background: #0f172a !important;
    background-image: linear-gradient(
        105deg, transparent 30%,
        rgba(255,255,255,0.07) 48%, rgba(255,255,255,0.13) 50%,
        rgba(255,255,255,0.07) 52%, transparent 70%) !important;
    background-size: 200% 100% !important;
    animation: shimmer 2.8s linear infinite !important;
    color: #e2e8f0 !important; font-size: 1.1rem !important;
    font-weight: 600 !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; padding: 0.85rem 1rem !important;
    border-radius: 6px !important; border: 1px solid rgba(148,163,184,0.2) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 2px 12px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.run-btn-wrapper [data-testid="stButton"] > button:hover {
    border-color: rgba(148,163,184,0.45) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 4px 20px rgba(0,0,0,0.4) !important;
    color: #ffffff !important;
}
[data-testid="stPlotlyChart"] {
    border-radius: 12px !important; overflow: hidden !important;
    box-shadow: 0 6px 32px rgba(0,0,0,0.22), 0 1.5px 6px rgba(0,0,0,0.14) !important;
    border: 1px solid rgba(255,255,255,0.07) !important; margin-bottom: 0.5rem !important;
}
</style>
"""

DIM_OPACITY = 0.06


def inject_css():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def check_guards():
    """Stop the page early if spectra haven't been uploaded yet."""
    if "df_peaks" not in st.session_state:
        st.warning("Upload spectra in Spectra Analysis page.")
        st.stop()
    if "mix_list_stored" not in st.session_state:
        st.warning("Select a name for each mixture in Spectra Analysis page.")
        st.stop()


def get_X(df_peaks, col_names):
    return np.column_stack([df_peaks[c].to_numpy() for c in col_names])


def build_nd_fig(model, X, col_names, mix_list, df_peaks, df_output, cols, model_title):
    """
    Build the correct N-D figure and return (fig, umap_warning).
    Works for 2-D, 3-D and 4-D+ (UMAP).
    """
    freqs = df_peaks["freq"].to_numpy()
    peak_cluster = df_output.select(cols).filter(pl.col("cluster").is_not_null())

    # Sort labels to match arctan order (same as write_df_output sort_by_arctan=True)
    sorted_clusters = sorted(model.clusters_, key=lambda c: float(c["arctan"]))
    sorted_ids      = [c["id"] for c in sorted_clusters]
    label_remap     = {old: new for new, old in enumerate(sorted_ids)}
    label_remap[-1] = -1
    display_labels  = np.array([label_remap.get(l, -1) for l in model.labels_])

    # Remap cluster list ids for the display
    display_clusters = []
    for c in sorted_clusters:
        dc = dict(c)
        dc["id"] = label_remap[c["id"]]
        display_clusters.append(dc)

    fig, umap_warn = plot_nd_interactive(
        X=X,
        labels=display_labels,
        clusters=display_clusters,
        unassigned=model.unassigned,
        mix_list=mix_list,
        col_names=col_names,
        freqs=freqs,
        peaks_df=peak_cluster.to_pandas(),
        model_title=model_title,
        width=900,
        height=700,
    )
    return fig, umap_warn


# ── Frequency-highlight helpers (2-D only — no-op for 3-D+) ──────────────────

def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _parse_freq_from_hovertext(text):
    try:
        return float(str(text).split("Freq: ")[1].split("<br>")[0])
    except (IndexError, ValueError):
        return None


def apply_highlight(fig, highlighted_freqs, match_tol, n_dims):
    """
    Dim non-highlighted points.  Only meaningful for 2-D Scattergl plots.
    Returns a (possibly deep-copied) figure.
    """
    if not highlighted_freqs or n_dims != 2:
        return fig  # no-op for 3-D / UMAP

    display_fig = copy.deepcopy(fig)
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

        rgba_colors, rgba_line_colors = [], []
        for t in texts:
            f = _parse_freq_from_hovertext(t)
            is_match = f is not None and any(abs(f - hf) <= match_tol for hf in highlighted_freqs)
            alpha = 1.0 if is_match else DIM_OPACITY
            rgba_colors.append(_hex_to_rgba(base_color, alpha) if base_color else f"rgba(136,136,136,{alpha})")
            rgba_line_colors.append(f"rgba(0,0,0,{alpha})")

        if rgba_colors:
            trace.marker.color = rgba_colors
            trace.marker.opacity = None
            if trace.marker.line is not None:
                trace.marker.line.color = rgba_line_colors

    return display_fig


def freq_upload_section(prefix, freq_cluster, match_tol_key, uploader_key, download_key, n_dims):
    """
    Renders file uploader + tolerance input, parses highlights.
    Returns (highlighted_freqs: set, assigned_freqs: DataFrame|None, match_tol: float).
    """
    col1, col2 = st.columns(2)
    with col1:
        uploaded_freqs = st.file_uploader(
            "Upload list of frequencies of interest (one column)",
            accept_multiple_files=False,
            key=uploader_key,
        )
    with col2:
        match_tol = st.number_input(
            "Matching tolerance (MHz)", step=0.01, key=match_tol_key, value=0.01, format="%.3f"
        )

    highlighted_freqs = set()
    assigned_freqs = None

    if uploaded_freqs:
        if freq_cluster is None:
            st.warning("Run the model first before uploading frequencies.")
        else:
            df_raw = pd.read_csv(uploaded_freqs, header=None)
            if isinstance(df_raw.iloc[0, 0], str) and "freq" in df_raw.iloc[0, 0].lower():
                df_input = df_raw.iloc[1:].reset_index(drop=True)
            else:
                df_input = df_raw
            df_input.columns = ["freq"] + list(df_input.columns[1:])
            df_input["freq"] = pd.to_numeric(df_input["freq"])

            assigned_freqs = freqs_df_to_cluster(df_input, freq_cluster, tol=match_tol)

            all_model_freqs = freq_cluster["freq"].to_numpy()
            for fq in df_input["freq"].to_numpy():
                matches = all_model_freqs[np.abs(all_model_freqs - fq) <= match_tol]
                highlighted_freqs.update(matches.tolist())

            if n_dims > 2:
                st.info("Frequency highlighting is shown in the table below (plot dimming is 2-D only).")

            st.download_button(
                "⬇ Download assigned clusters (CSV)",
                assigned_freqs.to_csv(index=False),
                f"{prefix}_freqs_to_cluster.csv",
                key=download_key,
                use_container_width=True,
                type="primary",
            )

    return highlighted_freqs, assigned_freqs, match_tol


def render_cluster_inspector(
    clusterer, df_peaks, col_names, prefix, n_dims
):
    """
    Full cluster inspector: metrics, dataframe, histogram, download, echo file.
    """
    clusters = clusterer.clusters_
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

            pts = c["points"]  # stored in original (non-*1000) scale
            df_pts = pd.DataFrame(pts, columns=col_names[:2] if n_dims > 2 else col_names)

            if n_dims > 2:
                # clusters_ only store 2-D projected points for angular algos;
                # show all dims by matching back on the two fitting cols
                pass

            ROUND = 8
            df_pts["_k0"] = df_pts[col_names[0]].round(ROUND)
            df_pts["_k1"] = df_pts[col_names[1]].round(ROUND)
            df_ref = df_peaks[[col_names[0], col_names[1], "freq"]].copy()
            df_ref["_k0"] = df_ref[col_names[0]].round(ROUND)
            df_ref["_k1"] = df_ref[col_names[1]].round(ROUND)

            df_pts = df_pts.merge(
                df_ref[["_k0", "_k1", "freq"]], on=["_k0", "_k1"], how="left"
            ).drop(columns=["_k0", "_k1"])

            display_cols = ["freq", col_names[0], col_names[1]]
            df_pts = df_pts[display_cols]
            df_pts.index.name = "point"
            st.dataframe(df_pts, use_container_width=True, height=300)

            fig_hist = clusterer.interactive_distance_histogram(
                cluster_id=c["id"], bins=50, width=700, height=300, show_fig=False,
            )
            fig_hist.update_layout(title=f"Angular distance histogram — cluster {display_idx}")
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{prefix}_hist_{display_idx}")

            st.download_button(
                label=f"⬇ Download cluster {display_idx} points (CSV)",
                data=df_pts.to_csv(index=False),
                file_name=f"{prefix}_cluster_{display_idx}_arctan{c['arctan']:.4f}.csv",
                mime="text/csv",
                key=f"{prefix}_dl_cluster_{display_idx}",
                use_container_width=True,
            )

    # Echo file
    if selected_labels:
        selected_orig_ids = {sorted_clusters[cluster_options[lbl]]["id"] for lbl in selected_labels}

        echo_mask = np.isin(clusterer.labels_, list(selected_orig_ids), invert=True) & (clusterer.labels_ != -1)
        echo_indices = np.where(echo_mask)[0]

        if len(echo_indices) > 0:
            X_full = np.column_stack([df_peaks[c].to_numpy() for c in col_names])
            echo_xy = X_full[echo_indices]
            df_echo = pd.DataFrame(echo_xy, columns=col_names)
            df_echo["cluster"] = [orig_id_to_display_idx.get(clusterer.labels_[i], -1) for i in echo_indices]

            ROUND = 8
            df_echo["_k0"] = df_echo[col_names[0]].round(ROUND)
            df_echo["_k1"] = df_echo[col_names[1]].round(ROUND)
            df_ref_echo = df_peaks[[col_names[0], col_names[1], "freq"]].copy()
            df_ref_echo["_k0"] = df_ref_echo[col_names[0]].round(ROUND)
            df_ref_echo["_k1"] = df_ref_echo[col_names[1]].round(ROUND)

            df_echo = df_echo.merge(
                df_ref_echo[["_k0", "_k1", "freq"]], on=["_k0", "_k1"], how="left"
            ).drop(columns=["_k0", "_k1"])
            df_echo = df_echo[["freq"]].sort_values("freq")

            st.divider()
            st.download_button(
                label="⬇ Download echo.acs",
                data=df_echo.to_csv(index=False, header=False),
                file_name="echo.acs",
                mime="text/csv",
                key=f"{prefix}_download_echo",
                use_container_width=True,
                type="primary",
            )
        else:
            st.info("No points remain for the echo file after excluding selected clusters.")
