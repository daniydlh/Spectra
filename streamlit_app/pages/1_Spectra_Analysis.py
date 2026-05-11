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

st.title("Spectra analisys - Peak search")

# --- FILE UPLOAD OR SHOW LOADED FILES ---

if 'df_stored' not in st.session_state:
    separator = st.text_input("File separator for .fft file", value=",", key='separator')

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file1 = st.file_uploader(
            "Upload FIRST spectra in .fft format",
            accept_multiple_files=False,
            key="uploader1"
        )
        mix1 = st.text_input("Name the FIRST mixture. Examples: SO2, DMF, 100CO2", key='mix1')

    with col2:
        uploaded_file2 = st.file_uploader(
            "Upload SECOND spectra in .fft format",
            accept_multiple_files=False,
            key="uploader2"
        )
        mix2 = st.text_input("Name the SECOND mixture. Examples: SO2_H2O, 1DMF_2H2O, 50DMF_50CO2", key='mix2')

    if uploaded_file1 and uploaded_file2 and mix1 and mix2:
        uploaded_files = [uploaded_file1, uploaded_file2]
        mix_list = [mix1, mix2]
        col_names = [f"int_{mix1}", f"int_{mix2}"]

        st.session_state['file_names_stored'] = [uploaded_file1.name, uploaded_file2.name]

        dfs = []
        for file in uploaded_files:
            df = pd.read_csv(file, skiprows=14, header=0, sep=str(separator))
            dfs.append(df)

        df = concat_cols_on_freq(
            [pl.from_pandas(d) for d in dfs], col_names
        ).to_pandas()

        sigma_list = pipeline_sigma_GUI(df)

        st.session_state['dfs_stored'] = dfs
        st.session_state['df_stored'] = df
        st.session_state['mix_list_stored'] = mix_list
        st.session_state['sigma_init'] = sigma_list
        st.rerun()

else:
    names = st.session_state.get('file_names_stored', ['?', '?'])
    st.info(f"**First file:** {names[0]}  \n**Second file:** {names[1]}")
    if st.button("Re-uploads files"):
        for key in ['dfs_stored', 'sigma_init', 'mix_list_stored', 'mult_stored', 'file_names_stored', 'df_stored']:
            st.session_state.pop(key, None)
        st.rerun()


# --- SIDEBAR PARAMETERS + MAIN PLOT ---
if 'dfs_stored' in st.session_state:
    df        = st.session_state['df_stored']
    mix_list  = st.session_state['mix_list_stored']
    sigma_init = st.session_state['sigma_init']
    mix1, mix2 = mix_list[0], mix_list[1]
    col_names = [f"int_{mix1}", f"int_{mix2}"]

    # --- Handle reset BEFORE widgets are instantiated
    if st.session_state.get("_reset_spectra_params"):
        for i in range(len(mix_list)):
            st.session_state[f"sigma{i+1}"] = float(sigma_init[i])
        st.session_state["mult"] = int(st.session_state.get('mult_stored', 3))
        st.session_state["_reset_spectra_params"] = False

    st.sidebar.header("Parameters")
    sigma_list = []
    for i, name in enumerate(mix_list):
        sigma_key = f"sigma{i+1}"
        if sigma_key not in st.session_state:
            st.session_state[sigma_key] = float(sigma_init[i])
        s = st.sidebar.number_input(
            f"Sigma for {name}",
            step=0.000001,
            key=f"sigma{i+1}",
            format="%.8f"
        )
        sigma_list.append(s)

    if "mult" not in st.session_state:
        st.session_state["mult"] = int(st.session_state.get('mult_stored', 3))
    mult = st.sidebar.number_input(
        "Sigma multiplier",
        step=1,
        key="mult",
    )
    st.session_state['mult_stored'] = mult


    if st.sidebar.button("Restore initial parameters"):
        st.session_state["_reset_spectra_params"] = True
        st.rerun()

    remove_zeros = st.checkbox("Remove frequency if it is below the detection limit in one of the two spectra",
            value=False,
            key="remove_zeros")
                
    st.session_state['removes_zeros'] = remove_zeros

    # --- RUN PIPELINE ---
    df, df_clean, df_peaks, detection_limits, peaks_array = pipeline_spectra_GUI(
        df,
        sigma=sigma_list,
        multiplier=mult,
        freq_col='freq',
        cols=col_names,
        remove_zeros=remove_zeros
    )

    st.session_state['df_peaks'] = df_peaks
    st.info(f"Current peaks found: **{len(df_peaks)}** | Sigma: {[f'{s:.6f}' for s in sigma_list]} | Multiplier: {mult}")

    
    st.download_button(
        "⬇ Download maximum intensities (CSV)",
        df_peaks.to_csv(index=False),
        "freq_maxint_output.csv",
        key="download_df_peaks"
    )

    # --- PLOT LOOP ---
    DARK_BG      = "#181926"
    PLOT_BG      = "#13141f"
    GRID_COLOR   = "rgba(255,255,255,0.07)"
    AXIS_COLOR   = "rgba(255,255,255,0.25)"
    FONT_COLOR   = "#c9d1e0"
    LINE_COLORS  = ["#4fc3f7", "#ce93d8"]   # light-blue, lilac for the two spectra
    PEAK_COLOR   = "#00e676"                 # vivid green
    THRESH_COLOR = "rgba(255,82,82,0.7)"     # soft red

    for i in range(len(col_names)):
        fig = px.line()

        fig.add_scatter(
            x=df['freq'],
            y=df[col_names[i]],
            mode="lines",
            name=mix_list[i],
            line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=1.2)
        )

        fig.add_hline(
            y=sigma_list[i] * mult,
            line_dash="dash",
            line_color=THRESH_COLOR,
        )

        fig.add_scatter(
            x=peaks_array[col_names[i]][:, 0],
            y=peaks_array[col_names[i]][:, 1],
            mode="markers",
            name="Maximum intensity of each peak",
            marker=dict(
                color=PEAK_COLOR,
                symbol="circle",
                size=5,
                line=dict(width=0)
            )
        )

        fig.update_layout(
            height=550,
            uirevision=f"fixed_{i}",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COLOR, size=15),
            legend=dict(
                bgcolor="rgba(255,255,255,0.05)",
                bordercolor="rgba(255,255,255,0.12)",
                borderwidth=1,
                font=dict(color=FONT_COLOR, size=17),
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
            ),
            xaxis=dict(
                gridcolor=GRID_COLOR,
                linecolor=AXIS_COLOR,
                tickcolor=AXIS_COLOR,
                tickfont=dict(color=FONT_COLOR, size=17),
                title="Frequency (MHz)",
                title_font=dict(color=FONT_COLOR, size=18),
                zerolinecolor=GRID_COLOR,
            ),
            yaxis=dict(
                gridcolor=GRID_COLOR,
                linecolor=AXIS_COLOR,
                tickcolor=AXIS_COLOR,
                tickfont=dict(color=FONT_COLOR, size=17),
                title="Intensity (μV)",
                title_font=dict(color=FONT_COLOR, size=18),
                zerolinecolor=GRID_COLOR,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)