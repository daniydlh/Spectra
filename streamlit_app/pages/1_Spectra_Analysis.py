import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
from gui_src import pipeline_spectra_GUI, pipeline_sigma_GUI, concat_cols_on_freq

st.set_page_config(
    page_title="Spectra Analysis - Peak Search",
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
[data-testid="stPlotlyChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 6px 32px rgba(0,0,0,0.22), 0 1.5px 6px rgba(0,0,0,0.14) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    margin-bottom: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Spectra Analysis — Peak search")

DARK_BG      = "#181926"
PLOT_BG      = "#13141f"
GRID_COLOR   = "rgba(255,255,255,0.07)"
AXIS_COLOR   = "rgba(255,255,255,0.25)"
FONT_COLOR   = "#c9d1e0"
LINE_COLORS  = ["#4fc3f7", "#ce93d8", "#a5d6a7", "#ffcc80", "#ef9a9a", "#b39ddb"]
PEAK_COLOR   = "#00e676"
THRESH_COLOR = "rgba(255,82,82,0.7)"

# ── Upload section ─────────────────────────────────────────────────────────────
if "df_stored" not in st.session_state:
    separator = st.text_input("File separator for .fft file", value=",", key="separator")

    n_spectra = st.number_input(
        "How many spectra to upload?",
        min_value=2, max_value=10, value=2, step=1, key="n_spectra_input"
    )

    uploaded_files = []
    mix_names = []

    cols_upload = st.columns(min(int(n_spectra), 3))  # max 3 side-by-side for readability

    for i in range(int(n_spectra)):
        col = cols_upload[i % len(cols_upload)]
        with col:
            f = st.file_uploader(
                f"Spectra #{i+1} (.fft)",
                accept_multiple_files=False,
                key=f"uploader_{i}"
            )
            name = st.text_input(
                f"Name #{i+1} (e.g. SO2, DMF)",
                key=f"mix_{i}"
            )
            uploaded_files.append(f)
            mix_names.append(name)

    all_uploaded = all(f is not None for f in uploaded_files)
    all_named    = all(n.strip() != "" for n in mix_names)

    if all_uploaded and all_named:
        col_names = [f"int_{n}" for n in mix_names]
        st.session_state["file_names_stored"] = [f.name for f in uploaded_files]

        dfs = []
        for f in uploaded_files:
            df = pd.read_csv(f, skiprows=14, header=0, sep=str(separator))
            dfs.append(df)

        df_combined = concat_cols_on_freq(
            [pl.from_pandas(d) for d in dfs], col_names
        ).to_pandas()

        sigma_list = pipeline_sigma_GUI(df_combined)

        st.session_state["dfs_stored"]       = dfs
        st.session_state["df_stored"]        = df_combined
        st.session_state["mix_list_stored"]  = mix_names
        st.session_state["sigma_init"]       = sigma_list
        st.rerun()

else:
    names = st.session_state.get("file_names_stored", [])
    mix_list = st.session_state["mix_list_stored"]
    n = len(mix_list)

    name_lines = "  \n".join([f"**Spectra #{i+1}:** {names[i]}  ({mix_list[i]})" for i in range(n)])
    st.info(name_lines)

    if st.button("Re-upload files"):
        for key in ["dfs_stored", "sigma_init", "mix_list_stored", "mult_stored",
                    "file_names_stored", "df_stored"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── Sidebar + analysis ─────────────────────────────────────────────────────────
if "dfs_stored" in st.session_state:
    df        = st.session_state["df_stored"]
    mix_list  = st.session_state["mix_list_stored"]
    sigma_init = st.session_state["sigma_init"]
    col_names  = [f"int_{m}" for m in mix_list]
    n_spectra  = len(mix_list)

    # Reset handler
    if st.session_state.get("_reset_spectra_params"):
        for i in range(n_spectra):
            st.session_state[f"sigma{i+1}"] = float(sigma_init[i])
        st.session_state["mult"] = int(st.session_state.get("mult_stored", 3))
        st.session_state["_reset_spectra_params"] = False

    st.sidebar.header("Parameters")
    sigma_list = []
    for i, name in enumerate(mix_list):
        key = f"sigma{i+1}"
        if key not in st.session_state:
            st.session_state[key] = float(sigma_init[i])
        s = st.sidebar.number_input(
            f"Sigma for {name}", step=0.000001, key=key, format="%.8f"
        )
        sigma_list.append(s)

    if "mult" not in st.session_state:
        st.session_state["mult"] = int(st.session_state.get("mult_stored", 3))
    mult = st.sidebar.number_input("Sigma multiplier", step=1, key="mult")
    st.session_state["mult_stored"] = mult

    if st.sidebar.button("Restore initial parameters"):
        st.session_state["_reset_spectra_params"] = True
        st.rerun()

    remove_zeros = st.checkbox(
        "Remove frequency if it is below the detection limit in any spectra",
        value=False, key="remove_zeros"
    )
    st.session_state["removes_zeros"] = remove_zeros

    # ── Run pipeline ──────────────────────────────────────────────────────────
    df, df_clean, df_peaks, detection_limits, peaks_array = pipeline_spectra_GUI(
        df, sigma=sigma_list, multiplier=mult,
        freq_col="freq", cols=col_names, remove_zeros=remove_zeros
    )

    st.session_state["df_peaks"] = df_peaks
    st.info(
        f"Current peaks found: **{len(df_peaks)}** | "
        f"Sigma: {[f'{s:.6f}' for s in sigma_list]} | "
        f"Multiplier: {mult}"
    )

    st.download_button(
        "⬇ Download maximum intensities (CSV)",
        df_peaks.to_csv(index=False),
        "freq_maxint_output.csv",
        key="download_df_peaks"
    )

    # ── Plots (one per spectra) ───────────────────────────────────────────────
    for i, col in enumerate(col_names):
        fig = px.line()

        fig.add_scatter(
            x=df["freq"], y=df[col], mode="lines", name=mix_list[i],
            line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=1.2)
        )

        fig.add_hline(y=sigma_list[i] * mult, line_dash="dash", line_color=THRESH_COLOR)

        if col in peaks_array:
            fig.add_scatter(
                x=peaks_array[col][:, 0],
                y=peaks_array[col][:, 1],
                mode="markers", name="Peak maximum",
                marker=dict(color=PEAK_COLOR, symbol="circle", size=5, line=dict(width=0))
            )

        fig.update_layout(
            height=550, uirevision=f"fixed_{i}",
            paper_bgcolor=DARK_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COLOR, size=15),
            legend=dict(
                bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.12)",
                borderwidth=1, font=dict(color=FONT_COLOR, size=17),
                x=0.01, y=0.99, xanchor="left", yanchor="top",
            ),
            xaxis=dict(
                gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, tickcolor=AXIS_COLOR,
                tickfont=dict(color=FONT_COLOR, size=17),
                title="Frequency (MHz)", title_font=dict(color=FONT_COLOR, size=18),
                zerolinecolor=GRID_COLOR,
            ),
            yaxis=dict(
                gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, tickcolor=AXIS_COLOR,
                tickfont=dict(color=FONT_COLOR, size=17),
                title="Intensity (μV)", title_font=dict(color=FONT_COLOR, size=18),
                zerolinecolor=GRID_COLOR,
            ),
        )

        st.plotly_chart(fig, width="stretch")
