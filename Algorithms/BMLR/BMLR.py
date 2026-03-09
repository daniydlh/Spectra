"""
Bayesian Mixture of Linear Regressions (BMLR) — also known as Switching Regression Model
==========================================================================================

THEORETICAL BACKGROUND
-----------------------
A standard Gaussian Mixture Model clusters points by proximity in (x,y) space.
Here instead, each cluster is a LINEAR pattern: y_i = β_k * x_i + ε_i

This is the BMLR / Switching Regression model:

    GENERATIVE PROCESS:
    ───────────────────
    1. π ~ Dirichlet(α/K, ..., α/K)     ← mixture weights (symmetric prior)
    2. For each cluster k = 1,...,K:
           β_k  ~ TruncatedNormal(μ_β, τ²; 0, ∞)   ← slope prior (positive slopes only)
           σ²_k ~ InvGamma(a₀, b₀)                 ← variance prior (conjugate)
    3. For each data point i = 1,...,N:
           z_i  ~ Categorical(π)                   ← cluster assignment
           y_i  ~ Normal(β_{z_i} * x_i,  σ²_{z_i}) ← likelihood (intercept forced = 0)

CONSTRAINTS (angular reasoning)
---------------------------------
- Intercept = 0: all lines pass through the origin. This is appropriate for
  signals where the response at x=0 must be zero (e.g. spectra, gradients).

- Slope β_k ∈ (0, π): slopes are treated as ANGLES θ = arctan(β) ∈ (0°, 90°).
  This is natural for directional data. The truncated normal prior enforces positivity.

- Angular distance: when comparing slopes, Euclidean distance on β-space is
  misleading for large slopes. We use:
      d(θ_a, θ_b) = min(|θ_a - θ_b|, π - |θ_a - θ_b|)
  which is the geodesic on the half-circle S¹/₂.

BAYESIAN PRIORS
---------------
    β_k  ~ N(μ_β, τ²)  truncated to (0, ∞)
         → Normal prior on slope, conjugate-like, reflecting our belief that
           slopes cluster around μ_β with spread τ.
    σ²_k ~ InvGamma(a₀, b₀)
         → Conjugate prior for Gaussian variance.
           E[σ²] = b₀/(a₀-1).  Weakly informative: a₀=2, b₀=0.5.
    π    ~ Dirichlet(α/K, ..., α/K)
         → Symmetric prior on mixture weights.
           α small → sparse mixtures; α=K → uniform.

INFERENCE: COLLAPSED GIBBS SAMPLER
------------------------------------
We use a standard Gibbs sampler (not collapsed, to stay pedagogically clear):
  Step 1 — Sample z_i:
      P(z_i=k | rest) ∝ π_k * N(y_i; β_k * x_i, σ²_k)
      Then optionally apply angular distance penalty on β.

  Step 2 — Sample β_k (posterior of slope given assigned points):
      Conjugate normal posterior (with truncation to positives):
          posterior precision:  Λ_k = 1/τ² + Σ_{i:z_i=k} x²_i / σ²_k
          posterior mean:       μ_k = Λ_k⁻¹ * (μ_β/τ² + Σ x_i y_i / σ²_k)
      Sample β_k ~ TruncatedNormal(μ_k, 1/Λ_k; 0, ∞)

  Step 3 — Sample σ²_k (conjugate InvGamma posterior):
          a_k* = a₀ + n_k/2
          b_k* = b₀ + Σ_{i:z_i=k} (y_i - β_k x_i)² / 2
      Sample σ²_k ~ InvGamma(a_k*, b_k*)

  Step 4 — Sample π (conjugate Dirichlet posterior):
          π ~ Dirichlet(α/K + n_1, ..., α/K + n_K)

ANGULAR PENALTY (optional)
---------------------------
During Step 1, we optionally add a von-Mises-like angular penalty:
    log P(z_i=k) -= κ * d_angular(arctan(y_i/x_i), arctan(β_k))²
This softly biases assignments toward geometrically coherent clusters
without being a hard constraint (κ controls sharpness).

References
----------
- Frühwirth-Schnatter (2006). Finite Mixture and Markov Switching Models.
- McLachlan & Peel (2000). Finite Mixture Models.
- Gelman et al. (2013). Bayesian Data Analysis, 3rd ed.
"""

import numpy as np
from scipy import stats
from scipy.stats import truncnorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Optional
import warnings


def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """
    Convert a hex color string (e.g. '#e63946') to an rgba() string.
    Plotly requires proper rgba() for semi-transparent fill — appending
    '1a' to a hex string is NOT a valid Plotly color format.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# ANGULAR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def slope_to_angle(slope: float) -> float:
    """
    Map a slope β to its angle θ = arctan(β) ∈ (-π/2, π/2).
    For positive slopes this gives θ ∈ (0, π/2).
    """
    return float(np.arctan(slope))


def angular_distance(t1: float, t2: float) -> float:
    """
    Geodesic distance on the half-circle [0, π].
    For undirected lines (π-periodic), the metric wraps at π:
        d = min(|t1-t2|, π - |t1-t2|)
    For angles in [0, π/2] (positive slopes), this reduces to |t1-t2|
    but we keep the full formula for generality.
    """
    diff = abs(t1 - t2)
    return min(diff, np.pi - diff)


def sample_truncated_normal(mu: float, sigma: float,
                             lower: float = 0.0, upper: float = np.inf,
                             rng=None) -> float:
    """
    Sample from N(mu, sigma²) truncated to [lower, upper].

    THEORETICAL NOTE:
    The truncated normal arises as the posterior for β_k under a normal prior
    combined with positivity constraint. Using scipy's truncnorm:
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
    """
    if rng is None:
        rng = np.random.default_rng()
    a = (lower - mu) / (sigma + 1e-12)
    b = (upper - mu) / (sigma + 1e-12)
    # scipy truncnorm uses standardized bounds
    sample = truncnorm.rvs(a, b, loc=mu, scale=sigma,
                           random_state=rng.integers(0, 2**31))
    return float(sample)


def sample_inv_gamma(a: float, b: float, rng=None) -> float:
    """
    Sample σ² ~ InvGamma(a, b).
    Equivalent to 1 / Gamma(a, 1/b).

    THEORETICAL NOTE:
    InvGamma is the conjugate prior for the variance of a Gaussian.
    E[σ²] = b/(a-1) for a>1.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Gamma(a, scale=1/b) then invert
    g = rng.gamma(shape=a, scale=1.0 / (b + 1e-12))
    return float(1.0 / (g + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(
    true_slopes: list[float],
    n_per_cluster: int = 50,
    noise_std: float = 0.25,
    x_range: tuple = (0.2, 5.0),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic data: K lines through origin y = β_k * x + ε, ε ~ N(0, noise_std²).
    Returns x (N,), y (N,), true_labels (N,).
    """
    rng = np.random.default_rng(seed)
    xs, ys, labels = [], [], []
    for k, slope in enumerate(true_slopes):
        x = rng.uniform(*x_range, n_per_cluster)
        y = slope * x + rng.normal(0, noise_std, n_per_cluster)
        xs.append(x)
        ys.append(y)
        labels.append(np.full(n_per_cluster, k))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(labels)


# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN MIXTURE OF LINEAR REGRESSIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterState:
    """
    Mutable state for one cluster at a single Gibbs iteration.

    β     : current slope sample
    σ²    : current variance sample
    n     : number of points currently assigned
    """
    beta: float        # slope β_k
    sigma2: float      # variance σ²_k
    n: int = 0         # count of assigned points


class BayesianMixtureLinearRegression:
    """
    Bayesian Mixture of K Linear Regressions (Switching Regression Model).

    K is FIXED (unlike the DP version). Choose K via BIC/AIC or domain knowledge.
    Each cluster is y_i = β_k * x_i + ε_i, with:
        - Intercept = 0 (hardcoded)
        - β_k > 0      (truncated normal prior)
        - β_k treated as angle in (0, π/2) for angular distance computations

    INFERENCE: Full Gibbs sampler over (z, β, σ², π).
    """

    def __init__(
        self,
        K: int = 3,                        # number of mixture components
        n_iter: int = 500,                 # total Gibbs iterations
        n_burnin: int = 250,               # burn-in (discarded samples)
        # ── Slope prior β_k ~ TruncNormal(mu_beta, tau²; 0, ∞) ──
        mu_beta: float = 1.0,              # prior mean slope (≈45° → slope=1)
        tau: float = 2.0,                  # prior std on slope (weakly informative)
        # ── Variance prior σ²_k ~ InvGamma(a0, b0) ──
        a0: float = 2.0,                   # shape: E[σ²] = b0/(a0-1) = 0.5
        b0: float = 0.5,                   # scale
        # ── Mixture prior π ~ Dirichlet(alpha_dir/K, ...) ──
        alpha_dir: float = 1.0,            # total Dirichlet concentration
        # ── Angular penalty ──
        use_angular_distance: bool = True, # penalize angular mismatch in assignment
        kappa: float = 1.0,                # sharpness of angular penalty
        seed: int = 0,
    ):
        self.K = K
        self.n_iter = n_iter
        self.n_burnin = n_burnin
        self.mu_beta = mu_beta
        self.tau = tau
        self.a0 = a0
        self.b0 = b0
        self.alpha_dir = alpha_dir
        self.use_angular_distance = use_angular_distance
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

        # Set during fit()
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.N: int = 0
        self.assignments: Optional[np.ndarray] = None   # z_i ∈ {0,...,K-1}
        self.pi: Optional[np.ndarray] = None            # mixture weights
        self.clusters: list[ClusterState] = []

        # Posterior samples (post burn-in)
        self.beta_samples: list[np.ndarray] = []        # shape: (n_post, K)
        self.sigma2_samples: list[np.ndarray] = []      # shape: (n_post, K)
        self.pi_samples: list[np.ndarray] = []          # shape: (n_post, K)
        self.z_samples: list[np.ndarray] = []           # shape: (n_post, N)

        # Diagnostics
        self.history: list[dict] = []

    # ──────────────────────────────────────────────────────────────────────
    # GIBBS STEPS
    # ──────────────────────────────────────────────────────────────────────

    def _step_z(self) -> None:
        """
        STEP 1: Sample cluster assignments z_i | β, σ², π, x, y.

        For each point i:
            P(z_i = k | rest) ∝ π_k * N(y_i; β_k * x_i, σ²_k)

        Optionally add angular distance penalty (von-Mises-like kernel):
            log P -= κ * d_angular(θ_data_i, θ_k)²
        where θ_data_i = arctan(y_i/x_i) is the "implied angle" of point i.

        CODING NOTE:
        We work in log-space and use the log-sum-exp trick for numerical
        stability before converting to probabilities and sampling.
        """
        for i in range(self.N):
            xi, yi = self.x[i], self.y[i]
            log_probs = np.zeros(self.K)

            for k, cl in enumerate(self.clusters):
                # Log mixture weight
                log_probs[k] = np.log(self.pi[k] + 1e-300)

                # Log Gaussian likelihood: log N(y_i; β_k*x_i, σ²_k)
                mu_ik = cl.beta * xi
                log_probs[k] += stats.norm.logpdf(yi, loc=mu_ik,
                                                   scale=np.sqrt(cl.sigma2))

                # Optional angular distance penalty
                if self.use_angular_distance and xi > 1e-6:
                    theta_data = slope_to_angle(yi / xi)   # implied angle from data
                    theta_k = slope_to_angle(cl.beta)       # cluster angle
                    ang_d = angular_distance(theta_data, theta_k)
                    # Gaussian kernel in angular space:
                    # large kappa → hard assignment; small kappa → soft
                    log_probs[k] -= self.kappa * ang_d ** 2

            # Log-sum-exp normalization for numerical stability:
            # log(sum(exp(log_p))) = max + log(sum(exp(log_p - max)))
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= probs.sum()

            self.assignments[i] = self.rng.choice(self.K, p=probs)

        # Update cluster counts
        for k in range(self.K):
            self.clusters[k].n = int(np.sum(self.assignments == k))

    def _step_beta(self) -> None:
        """
        STEP 2: Sample β_k | z, σ²_k, x, y  for each cluster k.

        BAYESIAN UPDATE (conjugate normal posterior with truncation):
        ─────────────────────────────────────────────────────────────
        Prior:     β_k ~ N(μ_β, τ²)  [truncated to β>0]
        Likelihood: y_i | β_k ~ N(β_k * x_i, σ²_k)

        Posterior precision (= inverse variance):
            Λ_k = 1/τ² + Σ_{i:z_i=k} x²_i / σ²_k

        Posterior mean:
            μ*_k = Λ_k⁻¹ * (μ_β/τ² + Σ_{i:z_i=k} x_i*y_i / σ²_k)

        This is a standard Normal-Normal conjugate update.
        The truncation to β>0 is enforced by sampling from TruncatedNormal.

        ANGULAR CONSTRAINT:
        After sampling β_k, we clip its angle to (0.01, π/2 - 0.01) to
        avoid degenerate near-zero or near-vertical lines.
        """
        for k, cl in enumerate(self.clusters):
            idx = np.where(self.assignments == k)[0]

            if len(idx) == 0:
                # Empty cluster: sample from prior
                cl.beta = sample_truncated_normal(
                    self.mu_beta, self.tau, lower=0.0, rng=self.rng
                )
                continue

            xi = self.x[idx]
            yi = self.y[idx]

            # Conjugate posterior for slope (Normal prior + Normal likelihood)
            prior_precision = 1.0 / (self.tau ** 2)
            lik_precision = np.sum(xi ** 2) / cl.sigma2

            post_precision = prior_precision + lik_precision          # Λ_k
            post_var = 1.0 / post_precision                           # σ*²_k
            post_mean = post_var * (
                self.mu_beta * prior_precision +
                np.sum(xi * yi) / cl.sigma2
            )                                                         # μ*_k

            # Sample from truncated normal posterior (β_k > 0)
            cl.beta = sample_truncated_normal(
                post_mean, np.sqrt(post_var), lower=0.0, rng=self.rng
            )

            # Enforce angular bound: angle must be in (0.01°, 89.99°)
            angle = np.arctan(cl.beta)
            angle = np.clip(angle, 0.01, np.pi / 2 - 0.01)
            cl.beta = float(np.tan(angle))

    def _step_sigma2(self) -> None:
        """
        STEP 3: Sample σ²_k | z, β_k, x, y  for each cluster k.

        BAYESIAN UPDATE (conjugate InvGamma posterior):
        ────────────────────────────────────────────────
        Prior:     σ²_k ~ InvGamma(a₀, b₀)
        Likelihood: y_i - β_k*x_i ~ N(0, σ²_k)  [residuals]

        Posterior:
            a*_k = a₀ + n_k/2
            b*_k = b₀ + (1/2) * Σ_{i:z_i=k} (y_i - β_k*x_i)²

        This is the standard conjugate update for Gaussian variance.
        """
        for k, cl in enumerate(self.clusters):
            idx = np.where(self.assignments == k)[0]

            if len(idx) == 0:
                # Sample from prior for empty clusters
                cl.sigma2 = sample_inv_gamma(self.a0, self.b0, rng=self.rng)
                continue

            xi = self.x[idx]
            yi = self.y[idx]
            residuals = yi - cl.beta * xi

            # Posterior InvGamma parameters
            a_post = self.a0 + len(idx) / 2.0
            b_post = self.b0 + 0.5 * np.sum(residuals ** 2)

            cl.sigma2 = sample_inv_gamma(a_post, b_post, rng=self.rng)
            # Clamp to avoid degenerate near-zero variance
            cl.sigma2 = max(cl.sigma2, 1e-6)

    def _step_pi(self) -> None:
        """
        STEP 4: Sample mixture weights π | z.

        BAYESIAN UPDATE (conjugate Dirichlet posterior):
        ─────────────────────────────────────────────────
        Prior:     π ~ Dirichlet(α/K, ..., α/K)
        Likelihood: counts n_1, ..., n_K

        Posterior:
            π | z ~ Dirichlet(α/K + n_1, ..., α/K + n_K)

        The Dirichlet is conjugate to the Categorical distribution,
        so this update is exact.
        """
        alpha_post = np.array([
            self.alpha_dir / self.K + self.clusters[k].n
            for k in range(self.K)
        ])
        self.pi = self.rng.dirichlet(alpha_post)

    # ──────────────────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, x: np.ndarray, y: np.ndarray
            ) -> "BayesianMixtureLinearRegression":
        """
        Run the full Gibbs sampler.

        INITIALIZATION:
        ───────────────
        - Assignments: k-means-like random partition (each point to random cluster)
        - β_k: OLS slope of initially assigned points (or prior mean if empty)
        - σ²_k: sample residual variance of initially assigned points
        - π: uniform (1/K each)

        MCMC OUTPUT:
        ────────────
        Post burn-in samples are stored in self.beta_samples, etc.
        Point estimates (posterior means) are available via .beta_mean_, etc.
        """
        self.x = x.copy()
        self.y = y.copy()
        self.N = len(x)

        # ── Initialize cluster states ──
        self.assignments = self.rng.integers(0, self.K, self.N)
        self.pi = np.ones(self.K) / self.K

        self.clusters = []
        for k in range(self.K):
            idx = np.where(self.assignments == k)[0]
            if len(idx) > 1:
                # OLS slope through origin for initial points
                beta_init = (np.sum(x[idx] * y[idx]) /
                             max(np.sum(x[idx] ** 2), 1e-10))
                beta_init = max(beta_init, 0.01)   # enforce positivity
                res = y[idx] - beta_init * x[idx]
                sig2_init = max(np.var(res), 0.01)
            else:
                beta_init = self.mu_beta
                sig2_init = self.b0 / (self.a0 - 1)
            self.clusters.append(ClusterState(
                beta=beta_init, sigma2=sig2_init, n=len(idx)
            ))

        print(f"Gibbs sampler: K={self.K}, N={self.N}, "
              f"{self.n_iter} iters ({self.n_burnin} burn-in)")

        for it in range(self.n_iter):
            # ── One full Gibbs sweep ──
            self._step_z()        # sample assignments
            self._step_beta()     # sample slopes
            self._step_sigma2()   # sample variances
            self._step_pi()       # sample weights

            betas = [cl.beta for cl in self.clusters]
            sigmas = [np.sqrt(cl.sigma2) for cl in self.clusters]
            ns = [cl.n for cl in self.clusters]

            if it % 20 == 0:
                self.history.append({
                    "iter": it,
                    "betas": betas.copy(),
                    "sigmas": sigmas.copy(),
                    "pi": self.pi.copy(),
                })
                print(f"  iter {it:4d} | "
                      f"β: {[f'{b:.3f}' for b in betas]} | "
                      f"σ: {[f'{s:.3f}' for s in sigmas]} | "
                      f"n: {ns} | π: {[f'{p:.2f}' for p in self.pi]}")

            # ── Store post-burn-in samples ──
            if it >= self.n_burnin:
                self.beta_samples.append(np.array(betas))
                self.sigma2_samples.append(np.array([cl.sigma2 for cl in self.clusters]))
                self.pi_samples.append(self.pi.copy())
                self.z_samples.append(self.assignments.copy())

        print(f"\nDone. Posterior samples: {len(self.beta_samples)}")

        # ── Point estimates: posterior mean over post-burn-in samples ──
        self.beta_mean_ = np.mean(self.beta_samples, axis=0)      # (K,)
        self.sigma2_mean_ = np.mean(self.sigma2_samples, axis=0)  # (K,)
        self.pi_mean_ = np.mean(self.pi_samples, axis=0)          # (K,)
        # MAP assignment: most frequent cluster per point
        z_arr = np.array(self.z_samples)                          # (n_post, N)
        self.z_map_ = np.array([
            np.bincount(z_arr[:, i], minlength=self.K).argmax()
            for i in range(self.N)
        ])
        return self

    def relabel_by_slope(self) -> None:
        """Reorder cluster indices by increasing posterior mean slope."""
        order = np.argsort(self.beta_mean_)
        inv = np.zeros(self.K, dtype=int)
        for new, old in enumerate(order):
            inv[old] = new
        self.beta_mean_ = self.beta_mean_[order]
        self.sigma2_mean_ = self.sigma2_mean_[order]
        self.pi_mean_ = self.pi_mean_[order]
        self.z_map_ = np.array([inv[z] for z in self.z_map_])
        self.beta_samples = [s[order] for s in self.beta_samples]
        self.sigma2_samples = [s[order] for s in self.sigma2_samples]
        self.pi_samples = [s[order] for s in self.pi_samples]


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    x: np.ndarray,
    y: np.ndarray,
    model: BayesianMixtureLinearRegression,
    true_slopes: Optional[list[float]] = None,
) -> go.Figure:
    """
    6-panel Plotly figure:
      [1,1] Scatter + fitted lines (posterior mean slopes)
      [1,2] Polar half-circle: cluster directions
      [2,1] Posterior distribution of slopes β_k (violin/KDE)
      [2,2] Gibbs trace: slope evolution over iterations
      [3,1] Residuals per cluster
      [3,2] Mixture weights π_k evolution + posterior mean
    """
    model.relabel_by_slope()
    K = model.K
    z = model.z_map_
    betas = model.beta_mean_
    sigmas = np.sqrt(model.sigma2_mean_)
    pis = model.pi_mean_

    COLORS = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
        "#f4a261", "#9b5de5", "#00b4d8", "#90be6d",
    ]
    ck = {k: COLORS[k % len(COLORS)] for k in range(K)}

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Clustering: Data + Posterior Mean Lines",
            "Angular Space (Slope Directions)",
            "Posterior Distribution of Slopes β_k",
            "Gibbs Trace: Slope β_k over Iterations",
            "Residuals per Cluster",
            "Mixture Weights π_k over Iterations",
        ),
        specs=[
            [{"type": "xy"},     {"type": "polar"}],
            [{"type": "xy"},     {"type": "xy"}],
            [{"type": "xy"},     {"type": "xy"}],
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    x_line = np.linspace(0, x.max() * 1.05, 200)

    # ── Panel [1,1]: Scatter + lines ──────────────────────────────────────
    for k in range(K):
        idx = np.where(z == k)[0]
        ang = np.degrees(np.arctan(betas[k]))
        col = ck[k]

        fig.add_trace(go.Scatter(
            x=x[idx], y=y[idx], mode="markers",
            name=f"Cluster {k} (θ≈{ang:.1f}°, π≈{pis[k]:.2f})",
            marker=dict(color=col, size=7, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            legendgroup=f"c{k}",
        ), row=1, col=1)

        # Posterior mean line
        fig.add_trace(go.Scatter(
            x=x_line, y=betas[k] * x_line, mode="lines",
            name=f"β̄_{k}={betas[k]:.3f}",
            line=dict(color=col, width=2.5),
            opacity=0.5,
            legendgroup=f"c{k}", showlegend=False,
        ), row=1, col=1)

        # ±1σ uncertainty band (shaded)
        y_up = (betas[k] + sigmas[k] / (x_line + 0.1)) * x_line
        y_lo = (betas[k] - sigmas[k] / (x_line + 0.1)) * x_line
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([y_up, y_lo[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(col, alpha=0.03),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=f"c{k}",
        ), row=1, col=1)

    # True lines
    if true_slopes:
        for s in true_slopes:
            fig.add_trace(go.Scatter(
                x=x_line, y=s * x_line, mode="lines",
                line=dict(color="black", width=1.5, dash="dot"),
                name=f"True β={s:.2f}", showlegend=True,
            ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(symbol="cross", size=12, color="black"),
        name="Origin", showlegend=True,
    ), row=1, col=1)

    # ── Panel [1,2]: Polar ────────────────────────────────────────────────
    for k in range(K):
        ang_deg = np.degrees(np.arctan(betas[k]))
        col = ck[k]

        # Cluster direction arrow
        fig.add_trace(go.Scatterpolar(
            r=[0, 1], theta=[ang_deg, ang_deg],
            mode="lines+markers",
            line=dict(color=col, width=3),
            marker=dict(size=[4, 10], color=col),
            legendgroup=f"c{k}", showlegend=False,
        ), row=1, col=2)

        # Individual point directions
        pt_angles = [np.degrees(np.arctan(y[i] / x[i]))
                     for i in np.where(z == k)[0] if x[i] > 0.01]
        if pt_angles:
            fig.add_trace(go.Scatterpolar(
                r=np.ones(len(pt_angles)) * 0.65,
                theta=pt_angles,
                mode="markers",
                marker=dict(size=4, color=col, opacity=0.45),
                showlegend=False,
            ), row=1, col=2)

        # Posterior uncertainty arc (±1 posterior std of β samples)
        beta_post = np.array(model.beta_samples)[:, k]
        ang_std = np.std(np.degrees(np.arctan(beta_post)))
        theta_arc = np.linspace(ang_deg - ang_std, ang_deg + ang_std, 30)
        fig.add_trace(go.Scatterpolar(
            r=np.ones(30) * 0.85, theta=theta_arc,
            mode="lines", line=dict(color=col, width=4, dash="dot"),
            showlegend=False,
        ), row=1, col=2)

    if true_slopes:
        for s in true_slopes:
            a = np.degrees(np.arctan(s))
            fig.add_trace(go.Scatterpolar(
                r=[0, 1.15], theta=[a, a], mode="lines",
                line=dict(color="black", dash="dot", width=2),
                showlegend=False,
            ), row=1, col=2)

    # ── Panel [2,1]: Posterior β distributions (violin) ──────────────────
    for k in range(K):
        beta_post = np.array(model.beta_samples)[:, k]
        ang_post = np.degrees(np.arctan(beta_post))
        col = ck[k]

        fig.add_trace(go.Violin(
            y=ang_post,
            name=f"Cluster {k}",
            fillcolor=col, line_color=col,
            opacity=0.6, box_visible=True,
            meanline_visible=True,
            showlegend=False,
            x0=f"k={k}",
        ), row=2, col=1)

        # True slope marker
        if true_slopes and k < len(true_slopes):
            fig.add_shape(
                type="line",
                x0=-0.5 + k, x1=0.5 + k,
                y0=np.degrees(np.arctan(true_slopes[k])),
                y1=np.degrees(np.arctan(true_slopes[k])),
                xref="x3", yref="y3",
                line=dict(color="black", dash="dash", width=2),
            )

    # ── Panel [2,2]: Gibbs trace of slopes ───────────────────────────────
    iters = [h["iter"] for h in model.history]
    for k in range(K):
        col = ck[k]
        betas_trace = [np.degrees(np.arctan(h["betas"][k]))
                       for h in model.history]
        fig.add_trace(go.Scatter(
            x=iters, y=betas_trace,
            mode="lines", name=f"β_{k} trace",
            line=dict(color=col, width=1.5),
            showlegend=False,
        ), row=2, col=2)

    # Burn-in marker (using add_shape on x4/y4)
    fig.add_shape(
        type="line",
        x0=model.n_burnin, x1=model.n_burnin,
        y0=0, y1=1, yref="y4 domain", xref="x4",
        line=dict(color="red", dash="dash", width=1.5),
    )
    fig.add_annotation(
        x=model.n_burnin, y=1, xref="x4", yref="y4 domain",
        text="Burn-in", showarrow=False, font=dict(color="red", size=10),
        xanchor="left", yanchor="top",
    )

    # ── Panel [3,1]: Residuals ────────────────────────────────────────────
    for k in range(K):
        idx = np.where(z == k)[0]
        residuals = y[idx] - betas[k] * x[idx]
        fig.add_trace(go.Scatter(
            x=x[idx], y=residuals, mode="markers",
            name=f"Resid {k}",
            marker=dict(color=ck[k], size=6, opacity=0.7),
            showlegend=False,
        ), row=3, col=1)

    # Zero line for residuals (x5/y5)
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="x5 domain",
        y0=0, y1=0, yref="y5",
        line=dict(color="gray", dash="dash", width=1.5),
    )

    # ── Panel [3,2]: Mixture weights trace ───────────────────────────────
    pi_arr = np.array(model.pi_samples)    # (n_post, K)
    post_iters = np.arange(len(pi_arr))
    for k in range(K):
        fig.add_trace(go.Scatter(
            x=post_iters, y=pi_arr[:, k],
            mode="lines", name=f"π_{k}",
            line=dict(color=ck[k], width=1.5),
            fill="tozeroy" if k == 0 else "tonexty",
            fillcolor=hex_to_rgba(ck[k], alpha=0.20),
            showlegend=False,
        ), row=3, col=2)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                "<b>Bayesian Mixture of Linear Regressions</b><br>"
                "<sup>Intercept=0 · Slopes as angles · Angular distance · "
                "Full Gibbs sampler (β, σ², π, z)</sup>"
            ),
            x=0.5, font=dict(size=15),
        ),
        height=1050,
        template="plotly_white",
        paper_bgcolor="#fafafa",
        font=dict(family="Arial", size=11),
        legend=dict(x=1.02, y=1, bordercolor="lightgray", borderwidth=1),
    )

    # Axis labels
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_xaxes(title_text="Cluster", row=2, col=1)
    fig.update_yaxes(title_text="Slope angle θ (°)", row=2, col=1)
    fig.update_xaxes(title_text="Gibbs iteration", row=2, col=2)
    fig.update_yaxes(title_text="Slope angle θ (°)", row=2, col=2)
    fig.update_xaxes(title_text="x", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=3, col=1)
    fig.update_xaxes(title_text="Post-burn-in sample", row=3, col=2)
    fig.update_yaxes(title_text="π_k", row=3, col=2)

    # Polar: restrict to first quadrant (positive slopes → angles in 0°–90°)
    fig.update_polars(
        sector=[0, 90],
        angularaxis=dict(direction="counterclockwise"),
        radialaxis=dict(showticklabels=False, range=[0, 1.3]),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# POLARS DATAFRAME INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def fit_from_polars(
    df,                          # polars.DataFrame
    x_col: str,
    y_col: str,
    K: int = 3,
    **model_kwargs,
) -> tuple["BayesianMixtureLinearRegression", np.ndarray, np.ndarray]:
    """
    Convenience wrapper: load x, y from a Polars DataFrame and fit the model.

    Usage:
        import polars as pl
        df = pl.read_csv("data.csv")
        model, x, y = fit_from_polars(df, x_col="wavelength", y_col="intensity", K=3)
        fig = plot_results(x, y, model)
        fig.show()

    Preprocessing applied automatically:
        - Drop nulls in x_col and y_col
        - Filter x > 0 (required for angular distance and OLS through origin)
        - Cast to float64
    """
    # Drop nulls and enforce x > 0
    df_clean = (
        df
        .drop_nulls([x_col, y_col])
        .filter(df[x_col] > 0)
    )
    x = df_clean[x_col].cast(float).to_numpy()
    y = df_clean[y_col].cast(float).to_numpy()

    model = BayesianMixtureLinearRegression(K=K, **model_kwargs)
    model.fit(x, y)
    return model, x, y


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Generate synthetic data ────────────────────────────────────────
    # Three clusters at angles 25°, 55°, 78° (slopes = tan of those angles)
    TRUE_ANGLES_DEG = [25, 55, 78]
    true_slopes = [np.tan(np.radians(a)) for a in TRUE_ANGLES_DEG]
    print(f"True angles : {TRUE_ANGLES_DEG}°")
    print(f"True slopes : {[f'{s:.3f}' for s in true_slopes]}")

    x, y, true_labels = generate_data(
        true_slopes=true_slopes,
        n_per_cluster=60,
        noise_std=0.20,
        seed=42,
    )

    # ── To use with a Polars DataFrame instead, do: ───────────────────────
    # import polars as pl
    # df = pl.DataFrame({"x": x, "y": y})          # or pl.read_csv(...)
    # model, x, y = fit_from_polars(df, "x", "y", K=3,
    #                               n_iter=500, n_burnin=250, seed=7)
    # ──────────────────────────────────────────────────────────────────────


    # ── 2. Fit BMLR ───────────────────────────────────────────────────────
    model = BayesianMixtureLinearRegression(
        K=3,                       # number of mixture components (fixed, unlike DP)
        n_iter=500,                # total Gibbs sweeps
        n_burnin=250,              # discard first 250 as burn-in
        mu_beta=1.0,               # prior mean slope (arctan(1) = 45°)
        tau=2.0,                   # prior std on slope (weakly informative)
        a0=2.0,                    # InvGamma shape (weakly informative)
        b0=0.5,                    # InvGamma scale → E[σ²] = 0.5
        alpha_dir=1.0,             # Dirichlet concentration (uniform-ish weights)
        use_angular_distance=True, # apply angular penalty during assignment
        kappa=1.5,                 # sharpness of angular penalty
        seed=7,
    )
    model.fit(x, y)

    # ── 3. Plot ───────────────────────────────────────────────────────────
    fig = plot_results(x, y, model, true_slopes=true_slopes)
    fig.write_html("bmlr_results.html")
    fig.show()
    print("\nPlot saved → bmlr_results.html")