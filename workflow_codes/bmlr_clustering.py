from data_analysis import df_signals
import polars as pl
import numpy as np
from Algorithms.BMLR.BMLR import (BayesianMixtureLinearRegression, plot_results)

df_filt1 = df_signals.filter(
(pl.col("int_water") > 0.0003) | (pl.col("int_deu") > 0.0003))

df_filt2 = df_filt1.filter(
(pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))

# Tus datos
x = df_filt2['int_water'].to_numpy() # Features
y = df_filt2['int_deu'].to_numpy()  

#x = (x - x.min()) / (x.max() - x.min())
#y = (y - y.min()) / (y.max() - y.min())


model = BayesianMixtureLinearRegression(
        K=15,                      # number of mixture components (fixed, unlike DP)
        n_iter=250,                # total Gibbs sweeps
        n_burnin=100,              # discard first 250 as burn-in
        mu_beta=1.0,               # prior mean slope (arctan(1) = 45°)
        tau=10.0,                  # prior std on slope (weakly informative)
        a0=10.0,                    # InvGamma shape (weakly informative)
        b0=0.000001,                # InvGamma scale → E[σ²] = 0.5
        alpha_dir=0.1,             # Dirichlet concentration (uniform-ish weights)
        use_angular_distance=True, # apply angular penalty during assignment
        kappa=20,                  # sharpness of angular penalty
        seed=7,
    )
model.fit(x, y)
fig = plot_results(x, y, model)
fig.show()
fig.write_html("bmlr_results.html")
print("\nPlot saved → bmlr_results.html")

# Print posterior precision ratio for each cluster
# If likelihood_precision / prior_precision << 1, prior is dominating
print("\nPrecision diagnosis:")
print(f"Prior precision (1/τ²): {1/model.tau**2:.4f}")
for k in range(model.K):
    idx = np.where(model.z_map_ == k)[0]
    if len(idx) == 0:
        continue
    xk = x[idx]
    sigma2_k = model.sigma2_mean_[k]
    lik_prec = np.sum(xk**2) / sigma2_k
    prior_prec = 1 / model.tau**2
    ratio = lik_prec / prior_prec
    beta_post = np.mean(model.beta_samples, axis=0)[k]
    print(f"  k={k:2d} | n={len(idx):5d} | "
          f"lik_prec={lik_prec:.3f} | prior_prec={prior_prec:.3f} | "
          f"ratio={ratio:.2f} | β̄={beta_post:.4f}")
