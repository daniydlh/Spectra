import polars as pl
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time
from data_analysis import df_signals


ti = time.time()


# Intensidades
y1 = df_signals["int_water"].to_numpy()  # señal 1
y2 = df_signals["int_deu"].to_numpy()    # señal 2

# Construimos la matriz para clustering
Y = np.column_stack([y1, y2])  # cada fila = (int_water, int_deu)

# --------------------------------------------
# 2 Ajustar Gaussian Mixture Model (GMM)
# --------------------------------------------
n_comp = 40  # número de clusters
gmm = GaussianMixture(
    n_components=n_comp,          # K, número de Gaussianas
    covariance_type='full',  # tipo de covarianza: 'full', 'diag', 'tied', 'spherical'
    max_iter=10000,            # máximo número de iteraciones EM
    tol=1e-4,                # tolerancia de convergencia
    init_params='kmeans',    # inicialización: 'kmeans' o 'random'
    reg_covar=1e-6           # regularización para estabilidad
)
gmm.fit(Y)



# Etiquetas de cluster y probabilidades
labels = gmm.predict(Y)
probs = gmm.predict_proba(Y)

# --------------------------------------------
# 3 Graficar clusters
# --------------------------------------------
plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'cyan', 'magenta']

for i in range(n_comp):
    mask = labels == i
    plt.scatter(y1[mask], y2[mask], color=colors[i], alpha=0.6, label=f"Cluster {i}")

plt.xlabel("Intensidad Water")
plt.ylabel("Intensidad Deuterio")
plt.title("Clusters de Intensidades (int_water vs int_deu)")
plt.legend()
plt.grid(True)
plt.show()

tf = time.time()
duration = tf - ti
print(f"Running time: {duration:.4f} s.")




