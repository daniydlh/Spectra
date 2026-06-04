import polars as pl
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import Iterable, Tuple, Sequence
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── N-spectra pipeline ────────────────────────────────────────────────────────

def pipeline_spectra_GUI(df, sigma=None, multiplier=None, freq_col='freq', cols=None, remove_zeros=False):
    df = pl.from_pandas(df)
    df_clean, detection_limits = apply_detection_limits(df, sigma_list=sigma, detection_mult=multiplier)
    peak_dict = detect_peaks(df_clean)
    peak_array = peaks_dict_to_arrays(peak_dict)
    all_peaks = combine_unique_freqs(peak_dict)
    df_peaks = get_int_at_peaks_AIopt(all_peaks, df_clean, return_df=True)
    df_peaks = unique_by_freq_keep_maxN(df_peaks, "freq", cols, tol=0.05)
    df_peaks = df_peaks.sort("freq")

    if remove_zeros is True:
        # remove rows where ANY intensity column is 0
        mask = pl.lit(True)
        for c in cols:
            mask = mask & (pl.col(c) != 0.0)
        df_peaks = df_peaks.filter(mask)

    df = df.to_pandas()
    df_clean = df_clean.to_pandas()
    df_peaks = df_peaks.to_pandas()

    return df, df_clean, df_peaks, detection_limits, peak_array


def pipeline_sigma_GUI(df):
    df = pl.from_pandas(df)
    df = set_baseline_at_zero(df)
    noise = only_noise(df, 1)
    sigma_list = compute_sigma(noise)
    return sigma_list


# ── Core signal processing ────────────────────────────────────────────────────

def apply_detection_limits(df: pl.DataFrame, sigma_list: list = None, detection_mult=3) -> pl.DataFrame:
    detection_limits = []
    new_cols = []
    for i, col in enumerate(df.columns[1:]):
        if sigma_list:
            sigma = sigma_list[i]
            detection_limits.append(detection_mult * sigma)
        else:
            print("No sigma values found, computing standard deviation...")
            sigma = df[col].std()
            detection_limits.append(detection_mult * sigma)

        new_col = pl.when(pl.col(col) < detection_mult * sigma).then(0).otherwise(pl.col(col)).alias(col)
        new_cols.append(new_col)

    return df.select([df.columns[0]] + new_cols), detection_limits


def detect_peaks(df: pl.DataFrame, freq_col: str = "freq", prominence: float = 0.0):
    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)

    freq = df.get_column(freq_col).to_numpy()
    peak_dict = {}

    for col in df.columns:
        if col == freq_col:
            continue
        signal = df.get_column(col).to_numpy()
        peaks, _ = find_peaks(signal, prominence=prominence)
        peak_dict[col] = {"peak_freq": freq[peaks], "peak_int": signal[peaks]}

    return peak_dict


def peaks_dict_to_arrays(peak_dict):
    arrays = {}
    for key, val in peak_dict.items():
        arrays[key] = np.stack([val['peak_freq'], val['peak_int']], axis=1)
    return arrays


def set_baseline_at_zero(df: pl.DataFrame) -> pl.DataFrame:
    cols = df.columns[1:]
    df = df.with_columns([(pl.col(col) - pl.col(col).median()).alias(col) for col in cols])
    return df


def only_noise(df, mult):
    cols = df.columns[1:]
    return df.with_columns([
        pl.when(abs(pl.col(col)) < (pl.col(col).mean() * mult))
        .then(pl.col(col)).otherwise(None).alias(col)
        for col in cols
    ])


def compute_sigma(df: pl.DataFrame) -> list:
    cols = df.columns[1:]
    sigma_df = df.select([pl.col(col).std().alias(f"std_{col}") for col in cols])
    return list(sigma_df.row(0))


def concat_cols_on_freq(dfs: Iterable[pl.DataFrame], names) -> pl.DataFrame:
    """Horizontally join N dataframes on their shared frequency column."""
    new_dfs = []
    for i, (df, col_name) in enumerate(zip(dfs, names)):
        if i == 0:
            new_df = df.rename({df.columns[0]: 'freq', df.columns[1]: col_name})
            new_dfs.append(new_df)
            continue
        new_df = df.drop(df.columns[0]).rename({df.columns[1]: col_name})
        new_dfs.append(new_df)

    return pl.concat(new_dfs, how="horizontal")


def unique_by_freq_keep_maxN(df, freq_col, cols, tol):
    """Keep the row with the highest max-intensity per frequency bin. Works for any N columns."""
    df_binned = df.with_columns(
        ((pl.col(freq_col) / tol).round(0) * tol).alias("f_bin")
    )
    df_scored = df_binned.with_columns(
        pl.max_horizontal(*cols).alias("imax")
    )
    df_unique = (
        df_scored
        .sort("imax", descending=True)
        .unique(subset="f_bin", keep="first")
        .drop(["f_bin", "imax"])
    )
    return df_unique


# Keep the old 2-col name as alias for backwards compat
def unique_by_freq_keep_max3(df, freq_col, cols, tol):
    return unique_by_freq_keep_maxN(df, freq_col, cols, tol)


def combine_unique_freqs(peak_dict):
    all_freqs = []
    for vals in peak_dict.values():
        all_freqs.append(vals['peak_freq'])
    combined = np.concatenate(all_freqs)
    return np.unique(combined)


def get_int_at_peaks_AIopt(peak_freqs, df, return_df=False):
    freqs = df.select(df.columns[0]).to_numpy().ravel()
    cols = df.columns[1:]
    data = df.select(cols).to_numpy()

    idxs = np.searchsorted(freqs, peak_freqs)
    idxs = np.clip(idxs, 0, len(freqs) - 1)
    left = np.maximum(idxs - 1, 0)
    right = np.minimum(idxs, len(freqs) - 1)
    nearest = np.where(
        np.abs(freqs[left] - peak_freqs) <= np.abs(freqs[right] - peak_freqs),
        left, right
    )

    intensities = {col: data[nearest, i] for i, col in enumerate(cols)}

    if return_df:
        freq_col = pl.DataFrame({"freq": peak_freqs})
        intensity_df = pl.DataFrame(intensities)
        return pl.concat([freq_col, intensity_df], how="horizontal")

    return intensities


def freqs_df_to_cluster(df_input_freqs, df_model_freqs, tol):
    freq_col = df_input_freqs.columns[0]
    model_freq_col = df_model_freqs.columns[0]
    cluster_col = df_model_freqs.columns[1]

    model_freqs_arr = df_model_freqs[model_freq_col].to_numpy()
    cluster_arr = df_model_freqs[cluster_col].to_numpy()

    def find_cluster(freq):
        diffs = np.abs(model_freqs_arr - freq)
        min_diff = diffs.min()
        if min_diff <= tol:
            return cluster_arr[diffs.argmin()]
        return "not_found"

    result_df = df_input_freqs.copy()
    result_df["cluster"] = result_df[freq_col].apply(find_cluster)
    return result_df


# ── N-D plotting helpers (used by clustering pages) ──────────────────────────

GLASBEY = [
    "#0000FF", "#FF0000", "#00FF00", "#FF00B6",
    "#005300", "#FFD300", "#009FFF", "#9A4D42", "#00FFBE",
    "#783FC1", "#1F9698", "#FFACFD", "#B1CC71", "#F1085C",
    "#FE8F42", "#DD00FF", "#201A01", "#720055", "#766C95",
    "#02AD24", "#C8FF00", "#886C00", "#FFB79F", "#858567",
    "#A10300", "#14F9FF", "#00479E", "#DC5E93", "#93D4FF",
]


def plot_nd_interactive(
    X,
    labels,
    clusters,
    unassigned,
    mix_list,
    col_names,
    freqs=None,
    peaks_df=None,
    model_title="Model",
    width=900,
    height=700,
):
    """
    Unified N-D interactive plot dispatcher.

    • 2-D  → 2-D Scattergl  (identical to existing plot_interactive output)
    • 3-D  → Scatter3d
    • 4-D+ → UMAP 2-D projection with a prominent warning banner

    Parameters
    ----------
    X          : ndarray (N, D) – raw (non-scaled) intensity values
    labels     : ndarray (N,)   – cluster labels (-1 = unassigned)
    clusters   : list of cluster dicts (same schema as LinearClusterer)
    unassigned : ndarray        – indices of unassigned points
    mix_list   : list[str]      – mixture names (length D)
    col_names  : list[str]      – column names (length D)
    freqs      : ndarray or None – frequency per point
    peaks_df   : Polars/Pandas DataFrame or None – peaks with 'cluster' col
    model_title: str
    width, height: int

    Returns
    -------
    fig : plotly Figure
    umap_warning : bool  – True if UMAP was used
    """
    D = X.shape[1]

    if D == 2:
        fig = _plot_2d(X, labels, clusters, mix_list, freqs, peaks_df, model_title, width, height)
        return fig, False
    elif D == 3:
        fig = _plot_3d(X, labels, clusters, mix_list, freqs, peaks_df, model_title, width, height)
        return fig, False
    else:
        fig, ok = _plot_umap(X, labels, clusters, mix_list, freqs, peaks_df, model_title, width, height)
        return fig, True


def _cluster_colors(labels):
    unique = np.unique(labels)
    return {lbl: GLASBEY[i % len(GLASBEY)] for i, lbl in enumerate(unique)}


def _hover_base(label, freq, coords, coord_names, cluster_info=None):
    lines = []
    if label == -1:
        lines.append("Unassigned")
    else:
        lines.append(f"Cluster {label}")
    if freq is not None:
        lines.append(f"Freq: {freq:.4f}")
    for name, val in zip(coord_names, coords):
        lines.append(f"{name}: {val:.5f}")
    if cluster_info is not None:
        arctan = cluster_info.get("arctan", None)
        if arctan is not None:
            lines.append(f"arctan: {arctan:.5f}")
    return "<br>".join(lines)


def _plot_2d(X, labels, clusters, mix_list, freqs, peaks_df, title, width, height):
    Xs = X * 1000
    unique_labels = np.unique(labels)
    colors = _cluster_colors(labels)

    # Build cluster lookup
    cluster_by_id = {c["id"]: c for c in clusters}

    x_min, x_max = Xs[:, 0].min(), Xs[:, 0].max()
    y_min, y_max = Xs[:, 1].min(), Xs[:, 1].max()
    xpad = (x_max - x_min) * 0.1
    ypad = (y_max - y_min) * 0.1
    x_range = np.array([x_min - xpad, x_max + xpad])

    fig = go.Figure()

    for lbl in unique_labels:
        mask = labels == lbl
        pts = Xs[mask]
        f_arr = freqs[mask] if freqs is not None else None
        color = colors[lbl]

        if lbl == -1:
            htxt = [
                f"Unassigned<br>{'Freq: ' + f'{f:.4f}<br>' if f_arr is not None else ''}X: {x:.5f}<br>Y: {y:.5f}"
                for (x, y), *rest in [((p[0], p[1]), None) for p in pts]
            ]
            if f_arr is not None:
                htxt = [
                    f"Unassigned<br>Freq: {f:.4f}<br>X: {x:.5f}<br>Y: {y:.5f}"
                    for (x, y), f in zip(pts, f_arr)
                ]
            else:
                htxt = [f"Unassigned<br>X: {x:.5f}<br>Y: {y:.5f}" for x, y in pts]
            fig.add_trace(go.Scattergl(
                x=pts[:, 0], y=pts[:, 1], mode="markers", name="Unassigned",
                marker=dict(size=7, color="lightgray", opacity=0.5, line=dict(width=0.6, color="black")),
                text=htxt, hoverinfo="text",
            ))
        else:
            ci = cluster_by_id.get(lbl, {})
            slope = ci.get("slope", 0)
            intercept = ci.get("intercept", 0)
            arctan = ci.get("arctan", 0)
            eq = f"x = {intercept:.2f}" if np.isinf(slope) else f"y = {slope:.3f}x + {intercept:.2f}"
            if f_arr is not None:
                htxt = [
                    f"Cluster {lbl}<br>Freq: {f:.4f}<br>X: {x:.5f}<br>Y: {y:.5f}<br>{eq}<br>arctan: {arctan:.5f}"
                    for (x, y), f in zip(pts, f_arr)
                ]
            else:
                htxt = [
                    f"Cluster {lbl}<br>X: {x:.5f}<br>Y: {y:.5f}<br>{eq}<br>arctan: {arctan:.5f}"
                    for x, y in pts
                ]
            fig.add_trace(go.Scattergl(
                x=pts[:, 0], y=pts[:, 1], mode="markers", name=f"Cluster {lbl}",
                marker=dict(size=7, color=color, opacity=0.6, line=dict(width=0.0, color="black")),
                text=htxt, hoverinfo="text",
            ))

            # peaks
            if peaks_df is not None:
                try:
                    pk = peaks_df[peaks_df["cluster"] == lbl] if hasattr(peaks_df, "__getitem__") else peaks_df.filter(pl.col("cluster") == lbl).to_pandas()
                    if len(pk) > 0:
                        px_vals = pk[col_names_from_mix(mix_list)[0]].to_numpy() * 1000 if hasattr(pk[col_names_from_mix(mix_list)[0]], 'to_numpy') else pk[col_names_from_mix(mix_list)[0]].values * 1000
                        py_vals = pk[col_names_from_mix(mix_list)[1]].to_numpy() * 1000 if hasattr(pk[col_names_from_mix(mix_list)[1]], 'to_numpy') else pk[col_names_from_mix(mix_list)[1]].values * 1000
                        pf_vals = pk["freq"].to_numpy() if hasattr(pk["freq"], 'to_numpy') else pk["freq"].values
                        ph = [f"Cluster {lbl}<br>Signal Maximum<br>Freq: {f:.4f}" for f in pf_vals]
                        fig.add_trace(go.Scattergl(
                            x=px_vals, y=py_vals, mode="markers", name=f"Cluster {lbl}",
                            marker=dict(size=7, color=color, opacity=1.0, line=dict(width=0.8, color="black")),
                            text=ph, hoverinfo="text",
                        ))
                except Exception:
                    pass

    # regression lines
    for c in clusters:
        slope, intercept = c["slope"], c["intercept"]
        lbl = c["id"]
        color = colors.get(lbl, "#888888")
        if np.isinf(slope):
            fig.add_trace(go.Scattergl(
                x=[intercept, intercept], y=[y_min - ypad, y_max + ypad],
                mode="lines", line=dict(color=color, width=2.5, dash="dash"),
                hoverinfo="skip", showlegend=False,
            ))
        else:
            y_line = slope * x_range + intercept
            fig.add_trace(go.Scattergl(
                x=x_range, y=y_line, mode="lines", opacity=0.4,
                line=dict(color=color, width=2.5, dash="dash"),
                hovertemplate=f"<b>Cluster {lbl}</b><br>y={slope:.3f}x+{intercept:.2f}<br><extra></extra>",
                showlegend=False,
            ))

    axis_kw = dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.15)",
                   zeroline=False, ticks="outside", ticklen=8, tickwidth=2, linewidth=2)
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        xaxis_title=f"Intensity {mix_list[0]} (µV)",
        yaxis_title=f"Intensity {mix_list[1]} (µV)",
        width=width, height=height, hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=20, color="black"),
        showlegend=False,
    )
    fig.update_xaxes(**axis_kw, range=[x_min - xpad, x_max + xpad])
    fig.update_yaxes(**axis_kw, range=[y_min - ypad, y_max + ypad])
    return fig


def col_names_from_mix(mix_list):
    return [f"int_{m}" for m in mix_list]


def _plot_3d(X, labels, clusters, mix_list, freqs, peaks_df, title, width, height):
    Xs = X * 1000
    unique_labels = np.unique(labels)
    colors = _cluster_colors(labels)
    cluster_by_id = {c["id"]: c for c in clusters}

    fig = go.Figure()

    for lbl in unique_labels:
        mask = labels == lbl
        pts = Xs[mask]
        f_arr = freqs[mask] if freqs is not None else None
        color = colors[lbl]

        if lbl == -1:
            htxt = (
                [f"Unassigned<br>Freq: {f:.4f}<br>X:{x:.4f} Y:{y:.4f} Z:{z:.4f}" for (x, y, z), f in zip(pts, f_arr)]
                if f_arr is not None
                else [f"Unassigned<br>X:{x:.4f} Y:{y:.4f} Z:{z:.4f}" for x, y, z in pts]
            )
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", name="Unassigned",
                marker=dict(size=3, color="lightgray", opacity=0.4),
                text=htxt, hoverinfo="text",
            ))
        else:
            ci = cluster_by_id.get(lbl, {})
            arctan = ci.get("arctan", 0)
            if f_arr is not None:
                htxt = [
                    f"Cluster {lbl}<br>Freq: {f:.4f}<br>X:{x:.4f} Y:{y:.4f} Z:{z:.4f}<br>arctan: {arctan:.5f}"
                    for (x, y, z), f in zip(pts, f_arr)
                ]
            else:
                htxt = [
                    f"Cluster {lbl}<br>X:{x:.4f} Y:{y:.4f} Z:{z:.4f}<br>arctan: {arctan:.5f}"
                    for x, y, z in pts
                ]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", name=f"Cluster {lbl}",
                marker=dict(size=3, color=color, opacity=0.6),
                text=htxt, hoverinfo="text",
            ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 20}},
        scene=dict(
            xaxis_title=f"Int {mix_list[0]} (µV)",
            yaxis_title=f"Int {mix_list[1]} (µV)",
            zaxis_title=f"Int {mix_list[2]} (µV)",
            bgcolor="white",
        ),
        width=width, height=height,
        font=dict(family="Times New Roman", size=16, color="black"),
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def _plot_umap(X, labels, clusters, mix_list, freqs, peaks_df, title, width, height):
    """UMAP 2-D projection for 4-D+ data."""
    try:
        import umap
    except ImportError:
        # graceful fallback: first two dims
        fig = _plot_2d(X[:, :2], labels, clusters, mix_list[:2], freqs, None, title + " (first 2 dims — install umap-learn)", width, height)
        return fig, False

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    unique_labels = np.unique(labels)
    colors = _cluster_colors(labels)

    fig = go.Figure()
    for lbl in unique_labels:
        mask = labels == lbl
        pts = embedding[mask]
        f_arr = freqs[mask] if freqs is not None else None
        color = colors[lbl]

        name = "Unassigned" if lbl == -1 else f"Cluster {lbl}"
        htxt = (
            [f"{name}<br>Freq: {f:.4f}<br>UMAP1:{x:.3f} UMAP2:{y:.3f}" for (x, y), f in zip(pts, f_arr)]
            if f_arr is not None
            else [f"{name}<br>UMAP1:{x:.3f} UMAP2:{y:.3f}" for x, y in pts]
        )
        fig.add_trace(go.Scattergl(
            x=pts[:, 0], y=pts[:, 1], mode="markers", name=name,
            marker=dict(
                size=6,
                color="lightgray" if lbl == -1 else color,
                opacity=0.5 if lbl == -1 else 0.7,
                line=dict(width=0.4, color="black"),
            ),
            text=htxt, hoverinfo="text",
        ))

    fig.update_layout(
        title={"text": f"{title} — UMAP projection ({X.shape[1]}D → 2D)", "x": 0.5, "xanchor": "center", "font": {"size": 18}},
        xaxis_title="UMAP dimension 1",
        yaxis_title="UMAP dimension 2",
        width=width, height=height, hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=16, color="black"),
        showlegend=False,
    )
    return fig, True
