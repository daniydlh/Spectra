import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import invgamma, multivariate_normal
from scipy.special import logsumexp

class DirichletProcessMixtureRegression:
    """
    Dirichlet Process Mixture of Linear Regressions (DPMR)
    
    Uses a nonparametric Bayesian approach where the number of components
    is learned from the data using a Dirichlet Process prior.
    
    Implementation uses truncated stick-breaking representation and
    Gibbs sampling for inference.
    """
    
    def __init__(self, alpha=1.0, max_components=30, n_iter=200, burn_in=50, 
                 fit_intercept=True, random_state=None):
        """
        Hyperparameters:
        -----------
        alpha : float
            Concentration parameter for Dirichlet Process (higher = more components)
        max_components : int
            Maximum number of components (truncation level)
        n_iter : int
            Number of Gibbs sampling iterations
        burn_in : int
            Number of burn-in iterations to discard
        fit_intercept : bool
            If True, fit intercept. If False, force intercept to 0.
        random_state : int or None
            Random seed for reproducibility
        """
        self.alpha = alpha
        self.max_components = max_components
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        
        # Model parameters (fitted)
        self.weights_ = None  # Stick-breaking weights
        self.components_ = []  # List of component parameters
        self.labels_ = None  # Cluster assignments
        self.n_components_ = 0  # Actual number of components used
        self.cluster_history_ = []  # Track cluster assignments over iterations
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _stick_breaking_weights(self, v):
        """
        Convert stick-breaking parameters to mixture weights
        
        π_k = v_k * ∏_{j<k}(1 - v_j)
        """
        weights = np.zeros(len(v))
        cumulative_prod = 1.0
        
        for k in range(len(v)):
            weights[k] = v[k] * cumulative_prod
            cumulative_prod *= (1 - v[k])
        
        return weights
    
    def _initialize_parameters(self, X, y, n_samples):
        """
        Initialize parameters for Gibbs sampling
        """
        # Initialize stick-breaking variables v_k ~ Beta(1, alpha)
        self.v_ = np.random.beta(1, self.alpha, size=self.max_components)
        self.weights_ = self._stick_breaking_weights(self.v_)
        
        # Initialize cluster assignments randomly
        self.z_ = np.random.choice(self.max_components, size=n_samples)
        
        # Initialize component parameters
        self.components_ = []
        for k in range(self.max_components):
            component = {
                'coef': np.random.randn(X.shape[1]) * 0.1,
                'intercept': np.random.randn() * 0.1 if self.fit_intercept else 0.0,
                'sigma': 1.0
            }
            self.components_.append(component)
        
        # Hyperparameters for Normal-Inverse-Gamma prior
        self.prior_mu = np.zeros(X.shape[1])
        self.prior_sigma_mu = 10.0
        self.prior_a = 2.0  # Shape for inverse gamma
        self.prior_b = 2.0  # Scale for inverse gamma
    
    def _sample_cluster_assignments(self, X, y):
        """
        Sample cluster assignments z_i for each data point
        """
        n_samples = len(X)
        log_prob = np.zeros((n_samples, self.max_components))
        
        for i in range(n_samples):
            for k in range(self.max_components):
                # Log probability of assignment to cluster k
                y_pred = X[i] @ self.components_[k]['coef'] + self.components_[k]['intercept']
                residual = y[i] - y_pred
                sigma = self.components_[k]['sigma']
                
                # Log likelihood
                log_lik = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (residual**2) / (sigma**2)
                
                # Log prior (stick-breaking weights)
                log_prior = np.log(self.weights_[k] + 1e-10)
                
                log_prob[i, k] = log_lik + log_prior
        
        # Sample from categorical distribution
        # Normalize log probabilities
        log_prob_normalized = log_prob - logsumexp(log_prob, axis=1, keepdims=True)
        prob = np.exp(log_prob_normalized)
        
        # Sample assignments
        for i in range(n_samples):
            self.z_[i] = np.random.choice(self.max_components, p=prob[i])
    
    def _sample_component_parameters(self, X, y):
        """
        Sample regression parameters for each component
        """
        for k in range(self.max_components):
            # Get data points assigned to cluster k
            mask = self.z_ == k
            n_k = np.sum(mask)
            
            if n_k == 0:
                # No points assigned - sample from prior
                self.components_[k]['coef'] = np.random.randn(X.shape[1]) * self.prior_sigma_mu
                self.components_[k]['intercept'] = np.random.randn() * self.prior_sigma_mu if self.fit_intercept else 0.0
                self.components_[k]['sigma'] = np.sqrt(invgamma.rvs(self.prior_a, scale=self.prior_b))
                continue
            
            X_k = X[mask]
            y_k = y[mask]
            
            if self.fit_intercept:
                # With intercept - use Bayesian linear regression
                # Posterior for beta = [coef, intercept]
                X_design = np.column_stack([X_k, np.ones(n_k)])
                
                # Prior precision
                prior_precision = np.eye(X.shape[1] + 1) / (self.prior_sigma_mu**2)
                
                # Sample sigma^2 from inverse gamma
                residuals = y_k - X_design @ np.append(self.components_[k]['coef'], 
                                                        self.components_[k]['intercept'])
                sse = np.sum(residuals**2)
                
                a_post = self.prior_a + n_k / 2.0
                b_post = self.prior_b + sse / 2.0
                sigma_sq = invgamma.rvs(a_post, scale=b_post)
                self.components_[k]['sigma'] = np.sqrt(sigma_sq)
                
                # Sample beta from normal
                posterior_precision = prior_precision + X_design.T @ X_design / sigma_sq
                posterior_cov = np.linalg.inv(posterior_precision)
                posterior_mean = posterior_cov @ (X_design.T @ y_k / sigma_sq)
                
                beta_sample = np.random.multivariate_normal(posterior_mean, posterior_cov)
                self.components_[k]['coef'] = beta_sample[:-1]
                self.components_[k]['intercept'] = beta_sample[-1]
                
            else:
                # Without intercept
                # Prior precision
                prior_precision = np.eye(X.shape[1]) / (self.prior_sigma_mu**2)
                
                # Sample sigma^2
                residuals = y_k - X_k @ self.components_[k]['coef']
                sse = np.sum(residuals**2)
                
                a_post = self.prior_a + n_k / 2.0
                b_post = self.prior_b + sse / 2.0
                sigma_sq = invgamma.rvs(a_post, scale=b_post)
                self.components_[k]['sigma'] = np.sqrt(sigma_sq)
                
                # Sample beta
                posterior_precision = prior_precision + X_k.T @ X_k / sigma_sq
                posterior_cov = np.linalg.inv(posterior_precision)
                posterior_mean = posterior_cov @ (X_k.T @ y_k / sigma_sq)
                
                self.components_[k]['coef'] = np.random.multivariate_normal(posterior_mean, posterior_cov)
                self.components_[k]['intercept'] = 0.0
    
    def _sample_stick_breaking_weights(self):
        """
        Sample stick-breaking variables v_k
        """
        n_samples = len(self.z_)
        
        for k in range(self.max_components - 1):
            # Count assignments
            n_k = np.sum(self.z_ == k)
            n_greater = np.sum(self.z_ > k)
            
            # Beta distribution parameters
            a = 1.0 + n_k
            b = self.alpha + n_greater
            
            self.v_[k] = np.random.beta(a, b)
        
        # Last component gets weight 1
        self.v_[-1] = 1.0
        
        # Update mixture weights
        self.weights_ = self._stick_breaking_weights(self.v_)
    
    def fit(self, X, y):
        """
        Fit the Dirichlet Process Mixture of Regressions using Gibbs sampling
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Target values
        
        Returns:
        --------
        self
        """
        X = np.atleast_2d(X)
        y = np.array(y).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = len(X)
        
        # Initialize parameters
        self._initialize_parameters(X, y, n_samples)
        
        # Gibbs sampling
        print(f"Running Gibbs sampling for {self.n_iter} iterations...")
        
        for iteration in range(self.n_iter):
            # Sample cluster assignments
            self._sample_cluster_assignments(X, y)
            
            # Sample component parameters
            self._sample_component_parameters(X, y)
            
            # Sample stick-breaking weights
            self._sample_stick_breaking_weights()
            
            # Track cluster assignments after burn-in
            if iteration >= self.burn_in:
                self.cluster_history_.append(self.z_.copy())
            
            if (iteration + 1) % 50 == 0:
                n_active = len(np.unique(self.z_))
                print(f"Iteration {iteration + 1}/{self.n_iter}: {n_active} active components")
        
        # Use posterior mean of cluster assignments
        self.labels_ = self._compute_posterior_labels()
        
        # Count actual components used
        self.n_components_ = len(np.unique(self.labels_))
        
        print(f"\nFinal number of components: {self.n_components_}")
        
        return self
    
    def _compute_posterior_labels(self):
        """
        Compute posterior mode of cluster assignments
        """
        if len(self.cluster_history_) == 0:
            return self.z_
        
        # Stack all post-burn-in samples
        cluster_samples = np.array(self.cluster_history_)
        
        # For each data point, find the mode
        n_samples = cluster_samples.shape[1]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Get mode of cluster assignments for point i
            unique, counts = np.unique(cluster_samples[:, i], return_counts=True)
            labels[i] = unique[np.argmax(counts)]
        
        # Relabel to be contiguous (0, 1, 2, ...)
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
        
        return labels
    
    def predict(self, X):
        """
        Predict target values for new data
        """
        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Use only active components
        unique_labels = np.unique(self.labels_)
        predictions = np.zeros(len(X))
        
        # Weighted prediction based on component weights
        for k in unique_labels:
            y_pred_k = X @ self.components_[k]['coef'] + self.components_[k]['intercept']
            predictions += self.weights_[k] * y_pred_k
        
        return predictions
    
    def get_component_info(self):
        """
        Get information about active mixture components
        """
        info = []
        unique_labels = np.unique(self.labels_)
        
        for k in unique_labels:
            comp = self.components_[k]
            n_points = np.sum(self.labels_ == k)
            
            info.append({
                'Component': k,
                'Weight': f"{self.weights_[k]:.4f}",
                'Coefficients': comp['coef'],
                'Intercept': f"{comp['intercept']:.4f}",
                'Sigma': f"{comp['sigma']:.4f}",
                'N_points': n_points
            })
        
        return info
    
    def plot_interactive(self, X, y, width=1200, height=700):
        """
        Create interactive Plotly visualization (for 1D case)
        """
        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != 1:
            print("Interactive plot only supports 1D features. Using first feature.")
            X = X[:, 0:1]
        
        X_flat = X.flatten()
        y_flat = np.array(y).flatten()
        
        # Create figure
        fig = go.Figure()
        
        # Color palette
        colors_list = px.colors.qualitative.Plotly + px.colors.qualitative.Set1
        
        # Get active components
        unique_labels = np.unique(self.labels_)
        
        # Plot data points colored by cluster
        for idx, k in enumerate(unique_labels):
            mask = self.labels_ == k
            cluster_points_x = X_flat[mask]
            cluster_points_y = y_flat[mask]
            
            color = colors_list[idx % len(colors_list)]
            
            comp = self.components_[k]
            if self.fit_intercept:
                equation = f'y = {comp["coef"][0]:.4f}x + {comp["intercept"]:.4f}'
            else:
                equation = f'y = {comp["coef"][0]:.4f}x'
            
            fig.add_trace(go.Scatter(
                x=cluster_points_x,
                y=cluster_points_y,
                mode='markers',
                name=f'Component {idx} (n={np.sum(mask)}, w={self.weights_[k]:.3f})',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=[f'Component {idx}<br>X: {float(x):.4f}<br>Y: {float(yval):.4f}<br>{equation}' 
                      for x, yval in zip(cluster_points_x, cluster_points_y)],
                hoverinfo='text'
            ))
        
        # Plot regression lines
        x_range = np.linspace(X_flat.min(), X_flat.max(), 100).reshape(-1, 1)
        
        for idx, k in enumerate(unique_labels):
            comp = self.components_[k]
            y_pred = x_range.flatten() * comp['coef'][0] + comp['intercept']
            color = colors_list[idx % len(colors_list)]
            
            if self.fit_intercept:
                equation = f"y = {comp['coef'][0]:.4f}x + {comp['intercept']:.4f}"
            else:
                equation = f"y = {comp['coef'][0]:.4f}x"
            
            fig.add_trace(go.Scatter(
                x=x_range.flatten(),
                y=y_pred,
                mode='lines',
                name=f'Line {idx}',
                line=dict(color=color, width=3, dash='dash'),
                hovertemplate=f'<b>Component {idx}</b><br>{equation}<br>Weight: {self.weights_[k]:.4f}<extra></extra>',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'DPMR: {self.n_components_} Components (α={self.alpha})',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='X',
            yaxis_title='Y',
            width=width,
            height=height,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
            font=dict(size=12),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="lightgray",
                borderwidth=1
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        fig.show()
    
    def plot_trace(self):
        """
        Plot trace of number of active components over iterations
        """
        if len(self.cluster_history_) == 0:
            print("No trace data available. Run fit() first.")
            return
        
        n_active = [len(np.unique(z)) for z in self.cluster_history_]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.burn_in, self.n_iter), n_active, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Number of Active Components', fontsize=12)
        plt.title('DPMR: Convergence of Component Count', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("DIRICHLET PROCESS MIXTURE OF REGRESSIONS (DPMR)")
    print("=" * 70)
    
    # Generate synthetic data with unknown number of linear patterns
    np.random.seed(42)
    
    X_data = []
    y_data = []
    
    # Component 1
    X1 = np.random.uniform(0, 10, 80)
    y1 = 2.5 * X1 + 5 + np.random.normal(0, 1.5, 80)
    X_data.append(X1)
    y_data.append(y1)
    
    # Component 2
    X2 = np.random.uniform(3, 13, 60)
    y2 = -1.5 * X2 + 25 + np.random.normal(0, 2, 60)
    X_data.append(X2)
    y_data.append(y2)
    
    # Component 3
    X3 = np.random.uniform(5, 15, 90)
    y3 = 0.5 * X3 + 8 + np.random.normal(0, 1, 90)
    X_data.append(X3)
    y_data.append(y3)
    
    # Component 4 (smaller)
    X4 = np.random.uniform(8, 18, 40)
    y4 = 1.8 * X4 - 5 + np.random.normal(0, 1.2, 40)
    X_data.append(X4)
    y_data.append(y4)
    
    # Combine and shuffle
    X = np.concatenate(X_data).reshape(-1, 1)
    y = np.concatenate(y_data)
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"\nGenerated {len(X)} data points from 4 unknown components")
    
    # Fit DPMR model
    print("\nFitting DPMR model with Gibbs sampling...")
    print("(This may take a moment...)\n")
    
    model = DirichletProcessMixtureRegression(
        alpha=1.0,  # Concentration parameter
        max_components=30,
        n_iter=200,
        burn_in=50,
        fit_intercept=True,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Print component information
    print("\n" + "=" * 70)
    print("DISCOVERED COMPONENTS")
    print("=" * 70)
    
    for info in model.get_component_info():
        print(f"\nComponent {info['Component']}:")
        print(f"  Weight (π): {info['Weight']}")
        print(f"  Coefficient (β): {info['Coefficients'][0]:.4f}")
        print(f"  Intercept: {info['Intercept']}")
        print(f"  Residual Std (σ): {info['Sigma']}")
        print(f"  Number of points: {info['N_points']}")
    
    print("\n" + "=" * 70)
    
    # Plot trace
    print("\nPlotting component trace...")
    model.plot_trace()
    
    # Interactive visualization
    print("\nGenerating interactive visualization...")
    model.plot_interactive(X, y)