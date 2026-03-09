"""
Dirichlet Process Mixture Regression (DPMR) with Angular Distance
==================================================================

THEORETICAL BACKGROUND
-----------------------
Standard Gaussian Mixture Models (GMM) require fixing K (number of clusters) a priori.
The Dirichlet Process (DP) is a non-parametric Bayesian prior that allows the number of
clusters to grow with data — "letting the data decide K."

A Dirichlet Process Mixture Model places a DP prior over cluster assignments:
    G ~ DP(alpha, G0)         <- G is a random probability measure
    theta_i ~ G               <- each point draws its parameter from G
    x_i ~ F(theta_i)          <- likelihood

Here we use the "Stick-Breaking" construction (Sethuraman, 1994):
    beta_k ~ Beta(1, alpha)   for k = 1, ..., K_max
    pi_k = beta_k * prod_{j<k}(1 - beta_j)   (stick weights, sum to 1)
    z_i ~ Categorical(pi)     (cluster assignment)

The concentration parameter `alpha` controls how many clusters are expected:
    E[number of clusters] ~ alpha * log(N)
    - Small alpha → fewer, larger clusters
    - Large alpha → more, smaller clusters

REGRESSION MODEL
----------------
We model each cluster as a LINE through the ORIGIN (intercept=0):
    y_i = slope_k * x_i + epsilon_i,  epsilon_i ~ N(0, sigma^2)

Slopes are constrained to (0, pi) to represent angles. This is meaningful when
data represent directions (e.g., gradients, orientations, velocity components).

ANGULAR (CIRCULAR) DISTANCE
-----------------------------
When slopes represent angles theta ∈ (0, π), Euclidean distance on slope space
is inadequate near the boundaries. The angular distance between angles θ_a, θ_b is:
    d_angular(θ_a, θ_b) = min(|θ_a - θ_b|, π - |θ_a - θ_b|)
This is the geodesic on the half-circle S^1/2, symmetric and periodic.

INFERENCE
---------
We use Collapsed Gibbs Sampling (Neal, 2000, Algorithm 3):
  - Marginalize out the mixture weights pi analytically via CRP (Chinese Restaurant Process)
  - Sample cluster assignments z_i one at a time using:
      P(z_i = k | z_{-i}, data) ∝ n_{-i,k} * p(y_i | x_i, theta_k)   [existing cluster]
                                  ∝ alpha * p(y_i | x_i, G0)           [new cluster]
  - Update cluster parameters (slopes) via MAP/MLE on assigned points
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------------------------

def generate_data(
    true_slopes: list[float],
    n_points_per_cluster: int = 40,
    noise_std: float = 0.3,
    x_range: tuple = (0.1, 5.0),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data: lines through origin with given slopes (angles in rad).
    Returns x, y arrays and ground-truth cluster labels.
    """
    rng = np.random.default_rng(seed)
    xs, ys, labels = [], [], []
    for k, slope in enumerate(true_slopes):
        x = rng.uniform(*x_range, n_points_per_cluster)
        # y = slope * x + noise (intercept = 0)
        noise = rng.normal(0, noise_std, n_points_per_cluster)
        y = slope * x + noise
        xs.append(x)
        ys.append(y)
        labels.append(np.full(n_points_per_cluster, k))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(labels)


# ---------------------------------------------------------------------------
# ANGULAR UTILITIES
# ---------------------------------------------------------------------------

def slope_to_angle(slope: float) -> float:
    """Convert a slope (rise/run) to its angle in radians: arctan(slope) ∈ (-π/2, π/2).
    We remap to (0, π) by treating negative slopes as their supplement."""
    angle = np.arctan(slope)              # maps slope → (-π/2, π/2)
    if angle < 0:
        angle += np.pi                    # remap to (0, π) half-circle
    return angle


def angular_distance(theta1: float, theta2: float) -> float:
    """
    Geodesic distance on the half-circle [0, π].
    For undirected lines (y = slope*x through origin), we only care about
    angle modulo π, so the distance is min(|Δ|, π - |Δ|).
    """
    diff = abs(theta1 - theta2)
    return min(diff, np.pi - diff)


def circular_mean_slopes(slopes: np.ndarray) -> float:
    """
    Compute the circular mean of a set of slopes-as-angles.
    Uses complex number averaging on the unit circle (doubled angles trick
    for π-periodic data), then halves back.
    Theoretical basis: for π-periodic data, map θ → 2θ on S^1, average, halve.
    """
    if len(slopes) == 0:
        return np.pi / 4  # fallback: 45°

    # Convert slopes to angles in [0, π)
    angles = np.array([slope_to_angle(float(s)) for s in slopes])

    # Double-angle trick: embed π-periodic angles onto full circle
    doubled = 2 * angles
    mean_x = np.mean(np.cos(doubled))
    mean_y = np.mean(np.sin(doubled))
    mean_doubled = np.arctan2(mean_y, mean_x)
    if mean_doubled < 0:
        mean_doubled += 2 * np.pi

    mean_angle = mean_doubled / 2  # back to [0, π)
    return np.tan(mean_angle)      # convert angle back to slope


# ---------------------------------------------------------------------------
# DIRICHLET PROCESS MIXTURE REGRESSION
# ---------------------------------------------------------------------------

@dataclass
class ClusterParams:
    """
    Parameters for a single regression cluster.

    THEORETICAL NOTE:
    Each cluster is a line y = slope * x through the origin.
    The slope is constrained to represent an angle in (0, π),
    enforced via arctan-based remapping.
    """
    slope: float = np.pi / 4          # initial slope = tan(45°) = 1.0
    sigma: float = 0.5                # noise std (shared or per-cluster)
    points: list = field(default_factory=list)  # indices of assigned points


class DirichletProcessMixtureRegression:
    """
    Non-parametric Bayesian linear regression via Dirichlet Process mixture.

    ALGORITHM: Collapsed Gibbs Sampler (Neal 2000, Algorithm 3)
    - The DP weights π are integrated out analytically.
    - Cluster assignments z_i are sampled from their full conditional.
    - Cluster slopes are updated via MLE (or MAP with a vague prior) after each sweep.

    ANGULAR DISTANCE:
    When reassigning points, we use angular distance on slope space instead of
    Euclidean distance. This is critical near slope=0 and slope→∞ where Euclidean
    distance breaks down for directional data.
    """

    def __init__(
        self,
        alpha: float = 1.0,        # DP concentration: controls cluster proliferation
        sigma: float = 0.3,        # noise std (assumed known; extend with prior if needed)
        K_max: int = 20,           # truncation level for stick-breaking
        n_iter: int = 200,         # Gibbs sampling iterations
        n_burnin: int = 100,       # burn-in iterations (discarded)
        slope_prior_mean: float = np.pi / 4,   # prior mean angle (45°) → slope=1
        slope_prior_kappa: float = 0.5,        # concentration of von Mises prior on angle
        use_angular_distance: bool = True,     # use angular vs Euclidean on slope space
        seed: int = 0,
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.K_max = K_max
        self.n_iter = n_iter
        self.n_burnin = n_burnin
        self.slope_prior_mean = slope_prior_mean
        self.slope_prior_kappa = slope_prior_kappa
        self.use_angular_distance = use_angular_distance
        self.rng = np.random.default_rng(seed)

        # Will be set during fit()
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.N: int = 0
        self.assignments: Optional[np.ndarray] = None   # z_i ∈ {0,...,K_max-1}
        self.cluster_slopes: dict[int, float] = {}       # k → slope
        self.history: list[dict] = []                    # trace for diagnostics

    # ------------------------------------------------------------------
    # LIKELIHOOD HELPERS
    # ------------------------------------------------------------------

    def _log_likelihood_point(self, xi: float, yi: float, slope: float) -> float:
        """
        Log-likelihood of (xi, yi) under cluster with given slope:
            log p(yi | xi, slope) = log N(yi; slope*xi, sigma^2)

        THEORETICAL NOTE:
        This is the standard Gaussian regression likelihood. The intercept is
        forced to 0 by design, meaning all lines pass through the origin.
        """
        mu = slope * xi
        return stats.norm.logpdf(yi, loc=mu, scale=self.sigma)

    def _log_likelihood_new_cluster(self, xi: float, yi: float) -> float:
        """
        Log marginal likelihood for a NEW cluster under the base measure G0.
        We integrate out the slope parameter analytically (or approximate).

        THEORETICAL NOTE:
        For a new cluster in the CRP, we need:
            p(yi | xi, G0) = ∫ p(yi | xi, theta) * G0(theta) d(theta)
        We approximate this by evaluating at the prior mean slope (weakly informative).
        A full treatment would use a conjugate prior (e.g., Normal-Normal for slope).
        """
        slope_prior = np.tan(self.slope_prior_mean)
        return self._log_likelihood_point(xi, yi, slope_prior)

    # ------------------------------------------------------------------
    # SLOPE UPDATE (MLE / MAP)
    # ------------------------------------------------------------------

    def _update_slope_mle(self, cluster_id: int) -> None:
        """
        Maximum Likelihood Estimate of slope for a cluster, given its assigned points.

        For y_i = slope * x_i + eps (intercept=0), the MLE is:
            slope_hat = sum(x_i * y_i) / sum(x_i^2)

        This is the OLS solution for regression through the origin.
        We then CLAMP the resulting angle to (0, π) using arctan remapping.
        """
        idx = np.where(self.assignments == cluster_id)[0]
        if len(idx) == 0:
            return

        xi = self.x[idx]
        yi = self.y[idx]

        # OLS through origin: minimizes sum((y - slope*x)^2)
        denom = np.sum(xi ** 2)
        if denom < 1e-10:
            return

        slope_mle = np.sum(xi * yi) / denom

        # ANGULAR CONSTRAINT: remap slope to angle in (0, π), then back to slope
        # This ensures no cluster has a "backwards" or out-of-range direction
        angle = np.arctan(slope_mle)
        # Clamp to (0.01, π - 0.01) to avoid degenerate horizontal/vertical lines
        angle = np.clip(angle, 0.01, np.pi - 0.01)
        self.cluster_slopes[cluster_id] = np.tan(angle)

    # ------------------------------------------------------------------
    # GIBBS SAMPLING — CORE
    # ------------------------------------------------------------------

    def _gibbs_sweep(self) -> None:
        """
        One full Gibbs sweep: reassign each point z_i by sampling from:

            P(z_i = k | z_{-i}, x_i, y_i) ∝
                n_{-i,k} * p(y_i | x_i, slope_k)    [existing cluster k]
                alpha    * p(y_i | x_i, G0)          [new cluster]

        This is the Chinese Restaurant Process (CRP) predictive:
        - A customer (data point) sits at an existing table (cluster) with
          probability proportional to how many people are already there × likelihood.
        - Or they start a new table with probability ∝ alpha × base likelihood.

        ANGULAR DISTANCE MODIFICATION:
        When computing the "effective likelihood" for cluster assignment, we can
        optionally penalize clusters that are angularly far from the data point's
        implied angle (y_i/x_i). This biases the sampler toward geometrically
        coherent clusters without fully overriding the probabilistic framework.
        """
        active_clusters = list(np.unique(self.assignments))

        for i in range(self.N):
            xi, yi = self.x[i], self.y[i]
            current_k = self.assignments[i]

            # --- Temporarily remove point i from its cluster ---
            self.assignments[i] = -1  # sentinel

            # Recompute active clusters after removal
            counts = {}
            for k in np.unique(self.assignments[self.assignments >= 0]):
                counts[k] = np.sum(self.assignments == k)

            # --- Build log-probability vector ---
            log_probs = []
            candidate_ids = []

            # 1) Existing clusters
            for k, n_k in counts.items():
                if n_k == 0:
                    continue
                slope_k = self.cluster_slopes.get(k, np.tan(self.slope_prior_mean))
                log_p = np.log(n_k) + self._log_likelihood_point(xi, yi, slope_k)

                # ANGULAR DISTANCE PENALTY (optional):
                # If the data point implies a direction theta_data = arctan(yi/xi),
                # penalize clusters whose slope is angularly far from that direction.
                if self.use_angular_distance and xi > 1e-6:
                    theta_data = slope_to_angle(yi / xi)
                    theta_k = slope_to_angle(slope_k)
                    ang_dist = angular_distance(theta_data, theta_k)
                    # Soft penalty: Gaussian kernel in angular space
                    # kappa controls sharpness; higher kappa = harder assignment
                    log_p -= self.slope_prior_kappa * (ang_dist ** 2)

                log_probs.append(log_p)
                candidate_ids.append(k)

            # 2) New cluster (CRP "new table" term)
            log_p_new = np.log(self.alpha) + self._log_likelihood_new_cluster(xi, yi)
            log_probs.append(log_p_new)
            candidate_ids.append("new")

            # --- Normalize and sample (log-sum-exp trick for numerical stability) ---
            log_probs = np.array(log_probs)
            log_probs -= np.max(log_probs)           # shift for stability
            probs = np.exp(log_probs)
            probs /= probs.sum()

            chosen_idx = self.rng.choice(len(candidate_ids), p=probs)
            chosen = candidate_ids[chosen_idx]

            if chosen == "new":
                # Assign to a fresh cluster ID (first unused integer)
                new_k = max(self.cluster_slopes.keys(), default=-1) + 1
                self.assignments[i] = new_k
                self.cluster_slopes[new_k] = np.tan(self.slope_prior_mean)
            else:
                self.assignments[i] = chosen

        # --- Update slopes for all active clusters (MLE step) ---
        for k in np.unique(self.assignments):
            self._update_slope_mle(k)

        # --- Prune empty clusters (housekeeping) ---
        active = set(np.unique(self.assignments))
        for k in list(self.cluster_slopes.keys()):
            if k not in active:
                del self.cluster_slopes[k]

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DirichletProcessMixtureRegression":
        """
        Run the Collapsed Gibbs Sampler on data (x, y).

        INITIALIZATION:
        All points start in a single cluster (conservative initialization).
        Alternatively, random initialization can speed up convergence.
        """
        self.x = x.copy()
        self.y = y.copy()
        self.N = len(x)

        # Initialize: all points in cluster 0
        self.assignments = np.zeros(self.N, dtype=int)
        slope_init = np.tan(self.slope_prior_mean)
        self.cluster_slopes = {0: slope_init}
        self._update_slope_mle(0)

        print(f"Starting Gibbs sampling: {self.n_iter} iterations, {self.n_burnin} burn-in")

        for it in range(self.n_iter):
            self._gibbs_sweep()

            n_clusters = len(np.unique(self.assignments))

            # Store diagnostics every 10 steps
            if it % 10 == 0:
                self.history.append({
                    "iter": it,
                    "n_clusters": n_clusters,
                    "slopes": list(self.cluster_slopes.values()),
                })
                print(f"  Iter {it:4d} | Active clusters: {n_clusters} | "
                      f"Slopes: {[f'{s:.3f}' for s in self.cluster_slopes.values()]}")

        # Post-burn-in: final assignments are used for inference
        # (For full Bayesian inference, average over post-burn-in samples)
        print(f"\nFinal: {len(np.unique(self.assignments))} clusters discovered.")
        return self

    # ------------------------------------------------------------------
    # RELABEL (cosmetic: sort clusters by slope)
    # ------------------------------------------------------------------

    def relabel_by_slope(self) -> None:
        """Renumber cluster IDs in order of increasing slope for clean plotting."""
        old_ids = sorted(self.cluster_slopes.keys(), key=lambda k: self.cluster_slopes[k])
        id_map = {old: new for new, old in enumerate(old_ids)}
        self.assignments = np.array([id_map[z] for z in self.assignments])
        self.cluster_slopes = {id_map[k]: v for k, v in self.cluster_slopes.items()}


# ---------------------------------------------------------------------------
# PLOTLY VISUALIZATION
# ---------------------------------------------------------------------------

def plot_results(
    x: np.ndarray,
    y: np.ndarray,
    model: DirichletProcessMixtureRegression,
    true_labels: Optional[np.ndarray] = None,
    true_slopes: Optional[list[float]] = None,
) -> go.Figure:
    """
    Multi-panel Plotly figure:
      Panel 1: Scatter of data colored by discovered cluster + fitted lines
      Panel 2: Slope angles on a half-circle (angular space visualization)
      Panel 3: Gibbs sampling trace (# clusters over iterations)
    """
    model.relabel_by_slope()
    assignments = model.assignments
    cluster_slopes = model.cluster_slopes
    n_clusters = len(cluster_slopes)

    # Color palette
    colors = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
        "#9b5de5", "#00b4d8", "#90be6d", "#f3722c", "#577590"
    ]
    cluster_colors = {k: colors[i % len(colors)] for i, k in enumerate(sorted(cluster_slopes))}

    # ----------------------------------------------------------------
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "DPMR Clustering: Data + Fitted Lines",
            "Angular Space (Slope Directions)",
            "Gibbs Sampling Trace: # Active Clusters",
            "Residuals per Cluster",
        ),
        specs=[[{"colspan": 1}, {"type": "polar"}],
               [{"colspan": 1}, {"colspan": 1}]],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    # ----------------------------------------------------------------
    # PANEL 1: Scatter + fitted lines
    x_line = np.linspace(0, x.max() * 1.05, 100)

    for k in sorted(cluster_slopes.keys()):
        idx = np.where(assignments == k)[0]
        slope = cluster_slopes[k]
        angle_deg = np.degrees(np.arctan(slope))
        color = cluster_colors[k]

        # Scatter
        fig.add_trace(
            go.Scatter(
                x=x[idx], y=y[idx],
                mode="markers",
                name=f"Cluster {k} (θ≈{angle_deg:.1f}°)",
                marker=dict(color=color, size=7, opacity=0.75,
                            line=dict(width=0.5, color="white")),
                legendgroup=f"c{k}",
            ),
            row=1, col=1
        )

        # Fitted line y = slope * x
        fig.add_trace(
            go.Scatter(
                x=x_line, y=slope * x_line,
                mode="lines",
                name=f"Line {k}",
                line=dict(color=color, width=2.5, dash="solid"),
                legendgroup=f"c{k}",
                showlegend=False,
            ),
            row=1, col=1
        )

    # True lines (if provided)
    if true_slopes is not None:
        for slope in true_slopes:
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=slope * x_line,
                    mode="lines",
                    name=f"True slope={slope:.2f}",
                    line=dict(color="black", width=1.5, dash="dot"),
                    showlegend=True,
                ),
                row=1, col=1
            )

    # Origin marker
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers",
                   marker=dict(symbol="cross", size=12, color="black"),
                   name="Origin (fixed intercept)", showlegend=True),
        row=1, col=1
    )

    # ----------------------------------------------------------------
    # PANEL 2: Polar plot — slopes as directions on half-circle
    for k in sorted(cluster_slopes.keys()):
        idx = np.where(assignments == k)[0]
        slope = cluster_slopes[k]
        angle_rad = np.arctan(slope)
        angle_deg = np.degrees(angle_rad)
        color = cluster_colors[k]
        n_k = len(idx)

        # Arrow (r=1) for fitted cluster direction
        fig.add_trace(
            go.Scatterpolar(
                r=[0, 1],
                theta=[angle_deg, angle_deg],
                mode="lines+markers",
                name=f"Cluster {k} dir",
                line=dict(color=color, width=3),
                marker=dict(size=[4, 12], color=color, symbol=["circle", "arrow"]),
                legendgroup=f"c{k}_polar",
                showlegend=False,
            ),
            row=1, col=2
        )

        # Points as small radial dots (r = count normalized)
        point_angles = [np.degrees(np.arctan(y[i] / x[i]))
                        for i in idx if x[i] > 0.01]
        if point_angles:
            fig.add_trace(
                go.Scatterpolar(
                    r=np.ones(len(point_angles)) * 0.7,
                    theta=point_angles,
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.5),
                    legendgroup=f"c{k}_polar",
                    showlegend=False,
                ),
                row=1, col=2
            )

    # True directions
    if true_slopes is not None:
        for slope in true_slopes:
            a = np.degrees(np.arctan(slope))
            fig.add_trace(
                go.Scatterpolar(
                    r=[0, 1.15], theta=[a, a],
                    mode="lines",
                    line=dict(color="black", dash="dot", width=2),
                    showlegend=False,
                ),
                row=1, col=2
            )

    # ----------------------------------------------------------------
    # PANEL 3: Gibbs trace
    iters = [h["iter"] for h in model.history]
    n_cls_trace = [h["n_clusters"] for h in model.history]

    fig.add_trace(
        go.Scatter(
            x=iters, y=n_cls_trace,
            mode="lines+markers",
            line=dict(color="#457b9d", width=2),
            marker=dict(size=5),
            name="# Clusters",
            fill="tozeroy", fillcolor="rgba(69,123,157,0.15)",
        ),
        row=2, col=1
    )
    # FIX: add_vline with row/col also fails in mixed polar+cartesian figures.
    # Use add_shape with explicit axis references. Row=2,col=1 → x2/y2 axes.
    fig.add_shape(
        type="line",
        x0=model.n_burnin, x1=model.n_burnin, xref="x2",
        y0=0, y1=1, yref="y2 domain",
        line=dict(dash="dash", color="red", width=1.5),
    )
    fig.add_annotation(
        x=model.n_burnin, y=1,
        xref="x2", yref="y2 domain",
        text="Burn-in", showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(color="red", size=11),
    )

    # ----------------------------------------------------------------
    # PANEL 4: Residuals
    for k in sorted(cluster_slopes.keys()):
        idx = np.where(assignments == k)[0]
        slope = cluster_slopes[k]
        residuals = y[idx] - slope * x[idx]
        color = cluster_colors[k]

        fig.add_trace(
            go.Scatter(
                x=x[idx], y=residuals,
                mode="markers",
                name=f"Residuals {k}",
                marker=dict(color=color, size=6, opacity=0.7),
                showlegend=False,
            ),
            row=2, col=2
        )

    # FIX: add_hline with row/col doesn't work in figures with mixed subplot types
    # (polar + cartesian). We use add_shape with explicit yref/xref instead.
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="x3 domain",   # x3 = third cartesian axis (row=2, col=2)
        y0=0, y1=0, yref="y3",
        line=dict(dash="dash", color="gray", width=1.5),
    )

    # ----------------------------------------------------------------
    # LAYOUT
    fig.update_layout(
        title=dict(
            text="<b>Dirichlet Process Mixture Regression</b><br>"
                 "<sup>Non-parametric clustering of linear patterns | "
                 "Intercept=0 | Angular distance | Gibbs sampling</sup>",
            font=dict(size=16),
            x=0.5,
        ),
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="v",
            x=1.02, y=1,
            bordercolor="lightgray", borderwidth=1,
        ),
        font=dict(family="Arial", size=12),
        paper_bgcolor="#fafafa",
    )

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="# Active Clusters", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=2)
    fig.update_yaxes(title_text="Residual (y - slope·x)", row=2, col=2)

    # FIX: angularaxis has no 'range' property.
    # Use 'sector' on the polar object to restrict the visible arc to [0, 90] degrees
    # (first quadrant: arctan of positive slopes falls in 0°–90°).
    # radialaxis 'range' IS valid and sets the radial [min, max].
    fig.update_polars(
        sector=[0, 90],                                       # clip polar plot to first quadrant
        angularaxis=dict(direction="counterclockwise"),       # angles increase counter-clockwise
        radialaxis=dict(showticklabels=False, range=[0, 1.3]),
    )

    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ----- 1. Generate synthetic data -----
    # Three clusters with slopes corresponding to angles ~30°, ~60°, ~80°
    true_slopes = [np.tan(np.radians(30)), np.tan(np.radians(60)), np.tan(np.radians(80))]
    print(f"True slopes: {[f'{s:.3f}' for s in true_slopes]}")
    print(f"True angles: 30°, 60°, 80°")

    x, y, true_labels = generate_data(
        true_slopes=true_slopes,
        n_points_per_cluster=50,
        noise_std=0.25,
        seed=42,
    )

    # ----- 2. Fit DPMR -----
    model = DirichletProcessMixtureRegression(
        alpha=1.5,               # moderate concentration: expect ~3 clusters
        sigma=0.3,               # noise level (set close to data generation)
        n_iter=300,              # total Gibbs iterations
        n_burnin=150,            # discard first 150 as burn-in
        use_angular_distance=True,   # use angular geometry on slope space
        slope_prior_kappa=1.0,       # sharpness of angular penalty
        seed=7,
    )
    model.fit(x, y)

    # ----- 3. Plot -----
    fig = plot_results(x, y, model, true_labels=true_labels, true_slopes=true_slopes)
    fig.write_html("dpmr_results.html")
    fig.show()
    print("\nPlot saved to dpmr_results.html")