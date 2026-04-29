import streamlit as st
import pandas as pd
import plotly.express as px
from utils import pipeline_spectra_GUI, pipeline_sigma_GUI

st.title("My First Scientific App")


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

if uploaded_file1 and uploaded_file2:
    if uploaded_files:
        dfs = []
        separator = st.text_input("File separator for .fft file", value=",", key='separator')
        for file in uploaded_files:
            df = pd.read_csv(file, skiprows=14, header=0, sep=str(separator))
            st.write(f"Loaded: {file.name}")
            dfs.append(df)


# --- DETECTION LIMIT CONTROL (SYNCED) ---
if uploaded_file1 and uploaded_file2:

    if uploaded_files:
        sigma_list = []
        for df in dfs:
            sigma = pipeline_sigma_GUI(df)
            sigma_list.extend(sigma)

        st.sidebar.header("Parameters")
        mix1 = st.text_input("Name the FIRST mixture. Examples: SO2, DMF, 100CO2", key='mix1')
        mix2 = st.text_input("Name the SECOND mixture. Examples: SO2_H2O, 1DMF_2H2O, 50DMF_50CO2", key='mix2')
        mix_list = [mix1, mix2]

        sigma1 = st.sidebar.number_input(
            f"Sigma for {mix1}",
            step=0.000001,
            key="sigma1",
            value=sigma_list[0],
            format="%.8f"
        )

        sigma2 = st.sidebar.number_input(
            f"Sigma for {mix2}",
            step=0.000001,
            key="sigma2",
            value=sigma_list[1],
            format="%.8f"
        )

        sigma_list = [sigma1, sigma2]

        mult = st.sidebar.number_input(
            "Sigma multiplier",
            step=1,
            key="mult",
            value=3      
        )

# --- MAIN APP ---

if uploaded_file1 and uploaded_file2:
    if uploaded_files and mix1 and mix2:
        for i, df in enumerate(dfs):
            
            df, dl, peaks, df_clean = pipeline_spectra_GUI(
                df,
                sigma=[sigma_list[i]],
                multiplier=mult,
                freq_col='freq'
            )

            x = df['freq']
            y = df['intensity']

            fig = px.line()

            fig.add_scatter(x=x, y=y, mode="lines", name=f"{mix_list[i]}")

            fig.add_hline(
                y=sigma_list[i] * mult,
                line_dash="dash",
                line_color="red"
            )

            fig.add_scatter(
                x=peaks['intensity'][:, 0],
                y=peaks['intensity'][:, 1],
                mode="markers",
                name="Maximum intensity of each peak",
                marker=dict(
                    color="red",
                    symbol="circle",
                    size=3,
                    line=dict(width=0)
                )
            )

            fig.update_layout(uirevision=f"fixed_{i}")

            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                f"{mix_list[i]}_output.csv",
                key=f"download_{i}"   # ← add this
            )