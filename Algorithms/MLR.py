import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from data_analysis import (df_signals, df_int_wd_clean)
import plotly.graph_objects as go


# ----------------------------
# Filter Data
# ----------------------------
df = df_signals.filter(
    (pl.col("int_water") > 0.00025) & (pl.col("int_deu") > 0.0003))



# ----------------------------
# Mixture Linear Regression for 1 feature, multiple components
# ----------------------------
nc=10
def mixture_linear_regression_1d_multi(df, feature_col, target_col, n_components=nc, n_iter=100):
    X = df[feature_col].to_numpy().reshape(-1, 1)
    y = df[target_col].to_numpy()
    n_samples = X.shape[0]

    # Initialize parameters
    weights = np.random.randn(n_components)
    biases = np.random.randn(n_components)
    mixing_coeffs = np.ones(n_components) / n_components
    sigma = np.ones(n_components)

    for iteration in range(n_iter):
        # E-step: responsibilities
        resp = np.zeros((n_samples, n_components))
        for k in range(n_components):
            mean = X[:, 0] * weights[k] + biases[k]
            resp[:, k] = mixing_coeffs[k] * (1 / np.sqrt(2 * np.pi * sigma[k]**2)) * \
                         np.exp(-(y - mean)**2 / (2 * sigma[k]**2))
        resp /= resp.sum(axis=1, keepdims=True)

        # M-step: update parameters
        Nk = resp.sum(axis=0)
        mixing_coeffs = Nk / n_samples
        for k in range(n_components):
            r = resp[:, k]
            weights[k] = np.sum(r * X[:, 0] * (y - biases[k])) / np.sum(r * X[:, 0]**2)
            biases[k] = np.sum(r * (y - X[:, 0] * weights[k])) / Nk[k]
            sigma[k] = np.sqrt(np.sum(r * (y - (X[:, 0] * weights[k] + biases[k]))**2) / Nk[k])

    return weights, biases, mixing_coeffs, sigma

# ----------------------------
# Fit model with 6 regressions
# ----------------------------
weights, biases, mixing_coeffs, sigma = mixture_linear_regression_1d_multi(
    df, feature_col="int_water", target_col="int_deu", n_components=nc
)

print("Weights:", weights)
print("Biases:", biases)
print("Mixing coefficients:", mixing_coeffs)
print("Sigmas:", sigma)

#----------------------------
# Prepare Plotly figure
# ----------------------------

X_plot = df["int_water"].to_numpy()
y_plot = df["int_deu"].to_numpy()
x_range = np.linspace(X_plot.min(), X_plot.max(), 200)

fig = go.Figure()

# Scatter data
fig.add_trace(go.Scatter(
    x=X_plot, y=y_plot, mode="markers",
    name="Data", marker=dict(color="black", size=5, opacity=0.6)
))

# Regression lines
#colors = ["red", "blue", "green", "orange", "purple", "brown"]
#colors = colors * ((6 // len(colors)) + 1)  # repeat colors if needed
for k in range(len(weights)):  # safer: loop over actual weights
    y_pred = weights[k] * x_range + biases[k]
    fig.add_trace(go.Scatter(
        x=x_range, y=y_pred, mode="lines",
        name=f"Regression {k+1}",
        line=dict(color="blue", width=2)
    ))

# Layout
fig.update_layout(
    title="Mixture of 6 Linear Regressions (1 Feature)",
    xaxis_title="int_deu",
    yaxis_title="int_water",
    legend=dict(itemsizing='constant')
)
# Set x and y limits
fig.update_xaxes(range=[X_plot.min()-0.05, X_plot.max()+0.05])
fig.update_yaxes(range=[y_plot.min()-0.05, y_plot.max()+0.05])

fig.show()