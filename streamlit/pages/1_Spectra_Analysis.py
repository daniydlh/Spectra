import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
from gui_src import pipeline_spectra_GUI, pipeline_sigma_GUI, concat_cols_on_freq

st.title("Spectra analisys - first filtering")

# --- FILE UPLOAD OR SHOW LOADED FILES ---

if 'df_stored' not in st.session_state:
    separator = st.text_input("File separator for .fft file", value=",", key='separator')
    uploaded_file1 = st.file_uploader(
        "Upload FIRST spectra in .fft format",
        accept_multiple_files=False
    )
    uploaded_file2 = st.file_uploader(
        "Upload SECOND spectra in .fft format",
        accept_multiple_files=False
    )

    if uploaded_file1 and uploaded_file2:
        uploaded_files = [uploaded_file1, uploaded_file2]
        st.session_state['file_names_stored'] = [uploaded_file1.name, uploaded_file2.name]

else:
    names = st.session_state.get('file_names_stored', ['?', '?'])
    st.info(f"**First file:** {names[0]}  \n**Second file:** {names[1]}")
    if st.button("Re-uploads files"):
        for key in ['dfs_stored', 'sigma_init', 'mix_list_stored', 'mult_stored', 'file_names_stored', 'df_stored']:
            st.session_state.pop(key, None)
        st.rerun()
    uploaded_file1, uploaded_file2 = None, None


# --- LOAD FILES ---

if 'df_stored' not in st.session_state and uploaded_file1 and uploaded_file2:

    mix1 = st.text_input("Name the FIRST mixture. Examples: SO2, DMF, 100CO2", key='mix1')
    mix2 = st.text_input("Name the SECOND mixture. Examples: SO2_H2O, 1DMF_2H2O, 50DMF_50CO2", key='mix2')

    if mix1 and mix2:
        mix_list = [mix1, mix2]
        col_names = [f"int_{mix1}", f"int_{mix2}"]

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


# --- SIDEBAR PARAMETERS + MAIN PLOT ---

if 'dfs_stored' in st.session_state:
    df        = st.session_state['df_stored']
    mix_list  = st.session_state['mix_list_stored']
    sigma_init = st.session_state['sigma_init']
    mix1, mix2 = mix_list[0], mix_list[1]
    col_names = [f"int_{mix1}", f"int_{mix2}"]

    st.sidebar.header("Parameters")

    sigma_list = []
    for i, name in enumerate(mix_list):

        # Only set default value if not already in session state
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

    # --- RUN PIPELINE ---
    df, df_clean, df_peaks, detection_limits, peaks_array = pipeline_spectra_GUI(
        df,
        sigma=sigma_list,
        multiplier=mult,
        freq_col='freq',
        cols=col_names
    )

    st.session_state['df_peaks'] = df_peaks  # ← save BEFORE reading
    st.info(f"Current peaks found: **{len(df_peaks)}** | Sigma: {[f'{s:.6f}' for s in sigma_list]} | Multiplier: {mult}")
    st.download_button(
        "Download maximum intensities (CSV)",
        df_peaks.to_csv(index=False),
        "freq_maxint_output.csv",
        key="download_df_peaks"
    )

    # --- PLOT LOOP ---
    for i in range(len(col_names)):
        fig = px.line()

        fig.add_scatter(
            x=df['freq'],
            y=df[col_names[i]],
            mode="lines",
            name=mix_list[i]
        )

        fig.add_hline(
            y=sigma_list[i] * mult,
            line_dash="dash",
            line_color="red"
        )

        fig.add_scatter(
            x=peaks_array[col_names[i]][:, 0],
            y=peaks_array[col_names[i]][:, 1],
            mode="markers",
            name="Maximum intensity of each peak",
            marker=dict(color="red", symbol="circle", size=3, line=dict(width=0))
        )

        fig.update_layout(uirevision=f"fixed_{i}")

        st.plotly_chart(fig, use_container_width=True)