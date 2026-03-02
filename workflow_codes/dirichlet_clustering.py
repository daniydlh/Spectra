from data_analysis import df_signals, df_int
from Algorithms.DIRICHLET.dirichlet import DirichletProcessMixtureRegression
import polars as pl
import numpy as np
import plotly.graph_objects as go

df_filt1 = df_signals.filter(
(pl.col("int_water") > 0.00017) | (pl.col("int_deu") > 0.00012))

df_filt2 = df_filt1.filter(
(pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))



# Tus datos
X = df_filt2['int_water'].to_numpy().reshape(-1,1) # Features
y = df_filt2['int_deu'].to_numpy()  

X = (X - X.mean(0)) / X.std(0)
y = (y - y.mean()) / y.std()


# Crear modelo DPMR (sin intercepto)
model = DirichletProcessMixtureRegression(
    alpha=10,           # Ajusta esto: mayor valor = más componentes
    max_components=50,   # Límite superior
    n_iter=3000,          # Iteraciones de Gibbs
    burn_in=1000,          # Burn-in
    fit_intercept=False, # SIN intercepto (líneas por el origen)
    random_state=42
)

model.fit(X, y)

# Ver componentes descubiertos
for info in model.get_component_info():
    print(f"Componente {info['Component']}: {info['N_points']} puntos")
    print(f"  Ecuación: y = {info['Coefficients'][0]:.4f}x")

# Visualizar
model.plot_interactive(X, y)

# Ver convergencia
model.plot_trace()