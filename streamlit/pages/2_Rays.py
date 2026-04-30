import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
from gui_src import concat_cols_on_freq
from RANSAC import LinearClusterer

if "df_peaks" in st.session_state:
    df_peaks = st.session_state["df_peaks"]
else:
    st.write("Not found")

if "mix_list_stored" not in st.session_state:
    st.warning("Run page 1 first.")
    st.stop()

mix_list = st.session_state["mix_list_stored"]
col_names = [f"int_{mix_list[0]}", f"int_{mix_list[1]}" ]

st.write(df_peaks)

if not df_peaks.empty:

    ang = st.sidebar.number_input(
    "Angular threshold", step=0.005, key="ang",
    value=float(0.04), format="%.3f"
    )

    ang_growth = st.sidebar.number_input(
    "Angular threshold grwoth per cluster", step=0.005, key="ang_growth",
    value=float(0.0), format="%.3f"
    )

    ang_max = st.sidebar.number_input(
    "Angular threshold growth per cluster", step=0.005, key="ang_max",
    value=float(0.6), format="%.3f"
    )

    max_clusters = st.sidebar.number_input(
    "Maximum angle able to reach", step=1, key="max_clust",
    value=30
    )

    iterations = st.sidebar.number_input(
    "RANSAC algorithm iterations", step=500, key="it",
    value=1000
    )


if st.button("Run model"):
    st.write("Running model...")
        
    # Create, fit clusterer and store the returned object in a dictionary
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

    x = df_peaks[col_names[0]].to_numpy()
    y = df_peaks[col_names[1]].to_numpy()
    X = np.column_stack((x,y))

    model = clusterer.fit(X)
    st.write(model)

   





