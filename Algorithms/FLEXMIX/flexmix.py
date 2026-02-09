import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression

class FlexMix:
    """
    Flexible Mixture Modeling for clustering with different component types.
    Inspired by R's FlexMix package.
    
    This implementation focuses on finite mixtures of linear regression models.
    """
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-6, random_state=None, fit_intercept=True):
        """
        Parameters:
        -----------
        n_components : int
            Number of mixture components (clusters)
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance for log-likelihood
        random_state : int or None
            Random seed for reproducibility
        fit_intercept : bool
            If True, fit intercept. If False, force intercept to 0.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        
        # Model parameters (fitted)
        self.weights_ = None  # Mixture weights (prior probabilities)
        self.components_ = []  # List of component models
        self.labels_ = None  # Hard cluster assignments
        self.responsibilities_ = None  # Soft assignments (posterior probabilities)
        self.log_likelihood_history_ = []
        self.converged_ = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, X, y):
        """
        Initialize mixture components using k-means++ strategy
        """
        n_samples = len(X)
        
        # Initialize mixture weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize responsibilities randomly
        self.responsibilities_ = np.random.dirichlet(
            np.ones(self.n_components), size=n_samples
        )
        
        # Initialize component parameters using weighted least squares
        self.components_ = []
        for k in range(self.n_components):
            component = {
                'coef': np.zeros(X.shape[1]),
                'intercept': 0.0,
                'sigma': 1.0  # Residual standard deviation
            }
            self.components_.append(component)
        
        # Initial M-step to get reasonable starting values
        self._m_step(X, y)
    
    def _e_step(self, X, y):
        """
        E-step: Calculate responsibilities (posterior probabilities)
        """
        n_samples = len(X)
        log_responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Predicted values for component k
            y_pred = X @ self.components_[k]['coef'] + self.components_[k]['intercept']
            residuals = y - y_pred
            sigma = self.components_[k]['sigma']
            
            # Log-likelihood contribution from component k
            log_prob = -0.5 * np.log(2 * np.pi * sigma**2) - \
                       0.5 * (residuals**2) / (sigma**2)
            
            # Add log mixture weight
            log_responsibilities[:, k] = np.log(self.weights_[k] + 1e-10) + log_prob
        
        # Convert to probabilities using log-sum-exp trick for numerical stability
        log_sum = np.logaddexp.reduce(log_responsibilities, axis=1, keepdims=True)
        self.responsibilities_ = np.exp(log_responsibilities - log_sum)
        
        # Calculate log-likelihood
        log_likelihood = np.sum(log_sum)
        return log_likelihood
    
    def _m_step(self, X, y):
        """
        M-step: Update component parameters using weighted least squares
        """
        n_samples = len(X)
        
        for k in range(self.n_components):
            # Responsibilities for component k
            resp_k = self.responsibilities_[:, k]
            resp_sum = np.sum(resp_k) + 1e-10
            
            # Update mixture weight
            self.weights_[k] = resp_sum / n_samples
            
            if self.fit_intercept:
                # Weighted least squares WITH intercept
                # Ensure y is a column vector
                y_col = y.reshape(-1, 1) if y.ndim == 1 else y
                
                # Use broadcasting for weighted operations (more efficient than diagonal matrix)
                # XtWX = X.T @ diag(resp_k) @ X = X.T @ (resp_k[:, None] * X)
                XtWX = X.T @ (resp_k[:, None] * X) + 1e-6 * np.eye(X.shape[1])
                XtWy = X.T @ (resp_k * y)
                
                # Solve weighted normal equations
                try:
                    coef = np.linalg.solve(XtWX, XtWy)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    coef = np.linalg.pinv(XtWX) @ XtWy
                
                # Calculate intercept (weighted mean)
                y_pred_no_intercept = X @ coef
                intercept = np.sum(resp_k * (y - y_pred_no_intercept)) / resp_sum
                
                # Update component parameters
                self.components_[k]['coef'] = coef
                self.components_[k]['intercept'] = intercept
                
            else:
                # Weighted least squares WITHOUT intercept (force through origin)
                # XtWX = X.T @ (resp_k[:, None] * X)
                XtWX = X.T @ (resp_k[:, None] * X) + 1e-6 * np.eye(X.shape[1])
                XtWy = X.T @ (resp_k * y)
                
                # Solve weighted normal equations
                try:
                    coef = np.linalg.solve(XtWX, XtWy)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    coef = np.linalg.pinv(XtWX) @ XtWy
                
                # Update component parameters (intercept = 0)
                self.components_[k]['coef'] = coef
                self.components_[k]['intercept'] = 0.0
            
            # Update residual variance
            y_pred = X @ coef + self.components_[k]['intercept']
            residuals = y - y_pred
            sigma_sq = np.sum(resp_k * residuals**2) / resp_sum
            self.components_[k]['sigma'] = np.sqrt(sigma_sq + 1e-10)
    
    def fit(self, X, y):
        """
        Fit the mixture model using EM algorithm
        
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
        
        # Initialize parameters
        self._initialize_parameters(X, y)
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step
            log_likelihood = self._e_step(X, y)
            self.log_likelihood_history_.append(log_likelihood)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                print(f"Converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = log_likelihood
            
            # M-step
            self._m_step(X, y)
        
        if not self.converged_:
            print(f"Warning: Did not converge after {self.max_iter} iterations")
        
        # Final hard cluster assignments
        self.labels_ = np.argmax(self.responsibilities_, axis=1)
        
        return self
    
    def predict(self, X):
        """
        Predict target values using mixture model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted values (weighted by responsibilities)
        cluster_predictions : array, shape (n_samples, n_components)
            Predictions from each component
        """
        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Get predictions from each component
        cluster_predictions = np.zeros((len(X), self.n_components))
        
        for k in range(self.n_components):
            cluster_predictions[:, k] = (
                X @ self.components_[k]['coef'] + 
                self.components_[k]['intercept']
            )
        
        # Assign to most likely component
        # (In practice, you might want to use predict_proba for soft assignments)
        cluster_assignments = self.predict_cluster(X)
        predictions = cluster_predictions[np.arange(len(X)), cluster_assignments]
        
        return predictions, cluster_predictions
    
    def predict_cluster(self, X):
        """
        Predict cluster assignments for new data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        
        Returns:
        --------
        clusters : array, shape (n_samples,)
            Cluster assignments
        """
        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Calculate likelihood for each component (without y, use 0 as dummy)
        n_samples = len(X)
        log_prob = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            log_prob[:, k] = np.log(self.weights_[k] + 1e-10)
        
        # Assign to most probable component
        return np.argmax(log_prob, axis=1)
    
    def get_component_info(self):
        """
        Get information about each mixture component
        """
        info = []
        for k in range(self.n_components):
            comp = self.components_[k]
            info.append({
                'Component': k,
                'Weight': f"{self.weights_[k]:.4f}",
                'Coefficients': comp['coef'],
                'Intercept': f"{comp['intercept']:.4f}",
                'Sigma': f"{comp['sigma']:.4f}",
                'N_points': np.sum(self.labels_ == k)
            })
        return info
    
    def plot_convergence(self):
        """
        Plot log-likelihood convergence
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.log_likelihood_history_, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Log-Likelihood', fontsize=12)
        plt.title('EM Algorithm Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_interactive(self, X, y, width=1200, height=700, save_pdf=False, save_html=False, xlims=None, ylims=None, output='flexmix_plot'):
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
        y_flat = np.array(y).flatten()  # Ensure y is 1D
        
        # Create figure
        fig = go.Figure()
        
        # Color palette
        colors_list = px.colors.qualitative.Plotly
        
        # Plot data points colored by cluster
        for k in range(self.n_components):
            mask = self.labels_ == k
            cluster_points_x = X_flat[mask]
            cluster_points_y = y_flat[mask]
            
            color = colors_list[k % len(colors_list)]
            
            fig.add_trace(go.Scattergl(
                x=cluster_points_x,
                y=cluster_points_y,
                mode='markers',
                name=f'Component {k} (n={np.sum(mask)})',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=[f'Component {k}<br>X: {float(x):.2f}<br>Y: {float(yval):.2f}' 
                      for x, yval in zip(cluster_points_x, cluster_points_y)],
                hoverinfo='text',
                showlegend=False
            ))
        
        # Plot regression lines
        x_range = np.linspace(X_flat.min(), X_flat.max(), 100).reshape(-1, 1)
        
        for k in range(self.n_components):
            comp = self.components_[k]
            y_pred = x_range.flatten() * comp['coef'][0] + comp['intercept']
            color = colors_list[k % len(colors_list)]
            
            equation = f"y = {comp['coef'][0]:.3f}x + {comp['intercept']:.3f}"
            
            fig.add_trace(go.Scattergl(
                x=x_range.flatten(),
                y=y_pred,
                mode='lines',
                name=f'Line {k}',
                line=dict(color=color, width=3, dash='dash'),
                hovertemplate=f'<b>Component {k}</b><br>{equation}<br>Weight: {self.weights_[k]:.3f}<extra></extra>',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'FlexMix: Mixture of {self.n_components} Linear Regressions',
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
            ),
            showlegend=False
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        fig.show()

        if xlims is not None:
            fig.update_xaxes(range=[xlims[0], xlims[1]])

        if ylims is not None:
            fig.update_yaxes(range=[ylims[0], ylims[1]])

        if save_pdf is True:
            fig.write_image(f"{output}.pdf",format="pdf",width=1500,height=550,scale=3)

        if save_html is True:
            fig.write_html(f"{output}.html", include_plotlyjs="cdn")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("FLEXMIX - FLEXIBLE MIXTURE MODELING")
    print("=" * 70)
    
    # Generate synthetic data with 3 linear patterns
    np.random.seed(42)
    n_samples_per_component = [100, 80, 120]
    
    X_data = []
    y_data = []
    true_labels = []
    
    # Component 1: steep positive slope
    X1 = np.random.uniform(0, 10, n_samples_per_component[0])
    y1 = 2.5 * X1 + 5 + np.random.normal(0, 1.5, n_samples_per_component[0])
    X_data.append(X1)
    y_data.append(y1)
    true_labels.extend([0] * n_samples_per_component[0])
    
    # Component 2: negative slope
    X2 = np.random.uniform(3, 13, n_samples_per_component[1])
    y2 = -1.5 * X2 + 25 + np.random.normal(0, 2, n_samples_per_component[1])
    X_data.append(X2)
    y_data.append(y2)
    true_labels.extend([1] * n_samples_per_component[1])
    
    # Component 3: slight positive slope
    X3 = np.random.uniform(5, 15, n_samples_per_component[2])
    y3 = 0.5 * X3 + 8 + np.random.normal(0, 1, n_samples_per_component[2])
    X_data.append(X3)
    y_data.append(y3)
    true_labels.extend([2] * n_samples_per_component[2])
    
    # Combine data
    X = np.concatenate(X_data).reshape(-1, 1)
    y = np.concatenate(y_data)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"\nGenerated {len(X)} data points from 3 mixture components")
    
    # Fit FlexMix model
    print("\nFitting FlexMix model with EM algorithm...")
    model = FlexMix(n_components=3, max_iter=100, tol=1e-6, random_state=42, fit_intercept=True)
    model.fit(X, y)
    
    # Print component information
    print("\n" + "=" * 70)
    print("MIXTURE COMPONENTS")
    print("=" * 70)
    
    for info in model.get_component_info():
        print(f"\nComponent {info['Component']}:")
        print(f"  Weight (π): {info['Weight']}")
        print(f"  Coefficient (β): {info['Coefficients'][0]:.4f}")
        print(f"  Intercept: {info['Intercept']}")
        print(f"  Residual Std (σ): {info['Sigma']}")
        print(f"  Number of points: {info['N_points']}")
    
    print("\n" + "=" * 70)
    
    # Plot convergence
    print("\nPlotting convergence...")
    model.plot_convergence()
    
    # Interactive visualization
    print("\nGenerating interactive visualization...")
    model.plot_interactive(X, y)
    
    # Make predictions on new data
    X_new = np.array([[2.0], [7.0], [12.0]])
    predictions, component_predictions = model.predict(X_new)
    
    print("\nPredictions for new data:")
    for i, (x, pred) in enumerate(zip(X_new.flatten(), predictions)):
        print(f"  X = {x:.1f}: y_pred = {pred:.2f}")