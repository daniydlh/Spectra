import numpy as np
from workflow_codes.data_analysis import df_signals
import polars as pl
from Algorithms.FLEXMIX.flexmix import FlexMix

df_filt1 = df_signals.filter(
(pl.col("int_water") > 0.00017) | (pl.col("int_deu") > 0.00012))

df_filt2 = df_filt1.filter(
(pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))

# Tus datos
X = df_filt2['int_water'].to_numpy().reshape(-1,1) # Features
y = df_filt2['int_deu'].to_numpy()        # Target

# Crear y ajustar modelo
model = FlexMix(n_components=35, random_state=42, max_iter=10000, tol=1e-8, fit_intercept=False)
model.fit(X, y)

# Ver los componentes encontrados
for info in model.get_component_info():
    print(f"Componente {info['Component']}: {info['N_points']} puntos")

# Visualizar
xlims=[]
ylims=[]
model.plot_interactive(X, y, save_pdf=False, save_html=True)

info = model.get_component_info()
info
