import streamlit as st
import pandas as pd
import plotly.express as px
from gui_src import pipeline_spectra_GUI, pipeline_sigma_GUI

st.title("Spectra analisys - first filtering")

# --- FILE UPLOAD OR SHOW LOADED FILES ---

if 'dfs_stored' not in st.session_state:
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
        st.session_state['file_names_stored'] = [uploaded_file1.name, uploaded_file2.name]  # ← must be here

else:
    names = st.session_state.get('file_names_stored', ['?', '?'])
    st.info(f"**First file:** {names[0]}  \n**Second file:** {names[1]}")
    if st.button("Re-upload files"):
        for key in ['dfs_stored', 'sigma_init', 'mix_list_stored', 'mult_stored', 'file_names_stored']:
            st.session_state.pop(key, None)
        st.rerun()
    uploaded_file1, uploaded_file2 = None, None


if uploaded_file1 and uploaded_file2:
    if uploaded_files:
        dfs = []
        separator = st.text_input("File separator for .fft file", value=",", key='separator')
        for file in uploaded_files:
            df = pd.read_csv(file, skiprows=14, header=0, sep=str(separator))
            st.write(f"Loaded: {file.name}")
            dfs.append(df)
        st.session_state['dfs_stored'] = dfs  # ← store
        


# --- DETECTION LIMIT CONTROL (SYNCED) ---
if uploaded_file1 and uploaded_file2:

    if uploaded_files:
        sigma_list = []
        for df in dfs:
            sigma = pipeline_sigma_GUI(df)
            sigma_list.extend(sigma)
        st.session_state['sigma_init'] = sigma_list  # ← store

        st.sidebar.header("Parameters")
        mix1 = st.text_input("Name the FIRST mixture. Examples: SO2, DMF, 100CO2", key='mix1')
        mix2 = st.text_input("Name the SECOND mixture. Examples: SO2_H2O, 1DMF_2H2O, 50DMF_50CO2", key='mix2')
        mix_list = [mix1, mix2]
        st.session_state['mix_list_stored'] = mix_list  # ← store

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
        st.session_state['mult_stored'] = mult  # ← store

# --- RESTORE & PLOT IF RETURNING FROM ANOTHER PAGE ---

elif 'dfs_stored' in st.session_state:
    dfs       = st.session_state['dfs_stored']
    mix_list  = st.session_state.get('mix_list_stored', ['', ''])
    mult      = st.session_state.get('mult_stored', 3)
    mix1, mix2 = mix_list[0], mix_list[1]

    sigma_init = st.session_state.get('sigma_init', [0.0, 0.0])

    st.sidebar.header("Parameters")
    mix1 = st.text_input("Name the FIRST mixture. Examples: SO2, DMF, 100CO2", key='mix1', value=mix1)
    mix2 = st.text_input("Name the SECOND mixture. Examples: SO2_H2O, 1DMF_2H2O, 50DMF_50CO2", key='mix2', value=mix2)
    mix_list = [mix1, mix2]
    st.session_state['mix_list_stored'] = mix_list

    sigma1 = st.sidebar.number_input(
        f"Sigma for {mix1}", step=0.000001, key="sigma1",
        value=float(sigma_init[0]), format="%.8f"
    )
    sigma2 = st.sidebar.number_input(
        f"Sigma for {mix2}", step=0.000001, key="sigma2",
        value=float(sigma_init[1]), format="%.8f"
    )
    sigma_list = [sigma1, sigma2]

    mult = st.sidebar.number_input("Sigma multiplier", step=1, key="mult", value=int(mult))
    st.session_state['mult_stored'] = mult

# --- MAIN APP ---

if uploaded_file1 and uploaded_file2:
    if uploaded_files and mix1 and mix2:
        dfs_clean=[]
        for i, df in enumerate(dfs):
            
            df, dl, peaks, df_clean = pipeline_spectra_GUI(
                df,
                sigma=[sigma_list[i]],
                multiplier=mult,
                freq_col='freq'
            )

            dfs_clean.append(df_clean)

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
                key=f"download_{i}"
            )
            
        st.session_state["dfs_clean"] = dfs_clean

# ← plot on return from another page
elif 'dfs_stored' in st.session_state and mix1 and mix2:
    dfs_clean = []
    for i, df in enumerate(dfs):

        df, dl, peaks, df_clean = pipeline_spectra_GUI(
            df,
            sigma=[sigma_list[i]],
            multiplier=mult,
            freq_col='freq'
        )

        dfs_clean.append(df_clean)

        x = df['freq']
        y = df['intensity']

        fig = px.line()
        fig.add_scatter(x=x, y=y, mode="lines", name=f"{mix_list[i]}")
        fig.add_hline(y=sigma_list[i] * mult, line_dash="dash", line_color="red")
        fig.add_scatter(
            x=peaks['intensity'][:, 0],
            y=peaks['intensity'][:, 1],
            mode="markers",
            name="Maximum intensity of each peak",
            marker=dict(color="red", symbol="circle", size=3, line=dict(width=0))
        )
        fig.update_layout(uirevision=f"fixed_{i}")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            f"{mix_list[i]}_output.csv",
            key=f"download_{i}"
        )

    st.session_state["dfs_clean"] = dfs_clean