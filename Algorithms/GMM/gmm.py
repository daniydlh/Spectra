from turtle import width
import polars as pl
import numpy as np
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from scipy.stats import chi2


class GMMLinearClusterer:
    """Gaussian Mixture Model for clustering linear patterns with interactive visualization"""
    
    def __init__(self, n_components=10, covariance_type='full', max_iter=1000, 
                 random_state=42, tol=1e-4, reg_covar=1e-6):
        """
        Initialize GMM clusterer
        
        Parameters:
        -----------
        n_components : int
            Number of Gaussian components (clusters)
        covariance_type : str
            Type of covariance: 'full', 'tied', 'diag', 'spherical'
        max_iter : int
            Maximum iterations for EM algorithm
        random_state : int
            Random seed for reproducibility
        tol : float
            Convergence tolerance
        reg_covar : float
            Regularization for covariance stability
        """
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            init_params='kmeans',
            reg_covar=reg_covar,
            random_state=random_state
        )
        self.labels_ = None
        self.probs_ = None
        self.X_ = None
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        
    def fit(self, X):
        """
        Fit GMM to data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Input data (e.g., [int_water, int_deu])
        """
        self.X_ = np.asarray(X)
        self.gmm.fit(self.X_)
        
        # Store results
        self.labels_ = self.gmm.predict(self.X_)
        self.probs_ = self.gmm.predict_proba(self.X_)
        self.means_ = self.gmm.means_
        self.covariances_ = self.gmm.covariances_
        self.weights_ = self.gmm.weights_
        
        # Calculate metrics
        self.bic_ = self.gmm.bic(self.X_)
        self.aic_ = self.gmm.aic(self.X_)
        self.log_likelihood_ = self.gmm.score(self.X_)
        
        return self
    
    def _get_ellipse_points(self, mean, cov, n_std=2, n_points=100):
        """
        Generate points for ellipse representing Gaussian component
        
        Parameters:
        -----------
        mean : array, shape (2,)
            Center of ellipse
        cov : array, shape (2, 2)
            Covariance matrix
        n_std : float
            Number of standard deviations (controls ellipse size)
        n_points : int
            Number of points to generate
        
        Returns:
        --------
        x, y : arrays
            Coordinates of ellipse boundary
        """
        # Chi-square value for confidence level
        # n_std=1 → 39% confidence, n_std=2 → 86%, n_std=3 → 99%
        chi2_val = chi2.ppf(0.95, df=2)  # 95% confidence
        scale = np.sqrt(chi2_val)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Angle of rotation
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # Width and height of ellipse
        width, height = 2 * scale * np.sqrt(eigenvalues)
        
        # Generate ellipse points
        theta = np.linspace(0, 2 * np.pi, n_points)
        ellipse = np.column_stack([
            width * np.cos(theta),
            height * np.sin(theta)
        ])
        
        # Rotate ellipse
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        ellipse_rotated = ellipse @ rotation_matrix.T
        
        # Translate to mean
        ellipse_final = ellipse_rotated + mean
        
        return ellipse_final[:, 0], ellipse_final[:, 1]
    
    def plot_interactive(self, col_names=None, n_std=2, output='gmm_clustering', 
                        show_ellipses=True, show_centers=True, 
                        point_size=3, ellipse_width=2, save_pdf=False, save_html=False,
                        lims=None, zoom_lims=None, width=800, height=600):
        """
        Create interactive Plotly visualization
        
        Parameters:
        -----------
        col_names : list of str, optional
            Names for x and y axes
        n_std : float
            Standard deviations for ellipse size
        save_path : str, optional
            Path to save HTML file
        show_ellipses : bool
            Whether to show Gaussian ellipses
        show_centers : bool
            Whether to show cluster centers
        point_size : int
            Size of scatter points
        ellipse_width : int
            Width of ellipse lines
        """
        if self.X_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if col_names is None:
            col_names = ['X', 'Y']
        
        # Create color palette
        # Color palette (highly differentiable, paper-safe)
        GLASBEY = [
            "#0000FF", "#FF0000", "#00FF00", "#FF00B6",
            "#005300", "#FFD300", "#009FFF", "#9A4D42", "#00FFBE",
            "#783FC1", "#1F9698", "#FFACFD", "#B1CC71", "#F1085C",
            "#FE8F42", "#DD00FF", "#201A01", "#720055", "#766C95",
            "#02AD24", "#C8FF00", "#886C00", "#FFB79F", "#858567",
            "#A10300", "#14F9FF", "#00479E", "#DC5E93", "#93D4FF"
        ]
        colors = GLASBEY

        # Extend if needed
        while len(colors) < self.n_components:
            colors.extend(colors)
        
        fig = go.Figure()
        
        # Plot points for each cluster
        for i in range(self.n_components):
            mask = self.labels_ == i
            
            # Get cluster info
            n_points = np.sum(mask)
            weight = self.weights_[i]
            
            fig.add_trace(go.Scattergl(
                x=self.X_[mask, 0],
                y=self.X_[mask, 1],
                mode='markers',
                name=f'Cluster {i} (n={n_points}, w={weight:.3f})',
                marker=dict(
                    size=point_size,
                    color=colors[i],
                    opacity=0.6,
                    line=dict(width=0)
                ),
                hovertemplate=(
                    f'<b>Cluster {i}</b><br>' +
                    f'{col_names[0]}: %{{x:.6f}}<br>' +
                    f'{col_names[1]}: %{{y:.6f}}<br>' +
                    f'Probability: %{{customdata:.3f}}<br>' +
                    '<extra></extra>'
                ),
                customdata=self.probs_[mask, i],
                showlegend=False
            ))
        
        # Plot Gaussian ellipses
        if show_ellipses:
            for i in range(self.n_components):
                mean = self.means_[i]
                cov = self.covariances_[i]
                
                x_ellipse, y_ellipse = self._get_ellipse_points(mean, cov, n_std=n_std)
                
                fig.add_trace(go.Scattergl(
                    x=x_ellipse,
                    y=y_ellipse,
                    mode='lines',
                    name=f'Gaussian {i} ({n_std}σ)',
                    line=dict(
                        color=colors[i],
                        width=ellipse_width,
                        dash='dash'
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        # Plot cluster centers
        if show_centers:
            fig.add_trace(go.Scattergl(
                x=self.means_[:, 0],
                y=self.means_[:, 1],
                mode='markers',
                name='Cluster Centers',
                marker=dict(
                    size=12,
                    color='black',
                    symbol='x',
                    line=dict(width=2, color='white')
                ),
                hovertemplate=(
                    '<b>Center %{pointNumber}</b><br>' +
                    f'{col_names[0]}: %{{x:.6f}}<br>' +
                    f'{col_names[1]}: %{{y:.6f}}<br>' +
                    'Weight: %{customdata:.3f}<br>' +
                    '<extra></extra>'
                ),
                showlegend=True,
                customdata=self.weights_
            ))
        
        # Update layout
        """
        title=dict(
                text=f'GMM Clustering ({self.n_components} components)<br>' +
                     f'<sub>BIC: {self.bic_:.2f} | AIC: {self.aic_:.2f} | ' +
                     f'Log-Likelihood: {self.log_likelihood_:.2f}</sub>',
                x=0.5,
                xanchor='center'
            )
        """
        fig.update_layout(
            title= 'GMM Model',
            xaxis_title=col_names[0],
            yaxis_title=col_names[1],
            hovermode='closest',
            template='plotly_white',
            width=1000,
            height=800,
            font=dict(
                family='Times New Roman',
                size=20,
                color='black'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        if lims is not None:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=2,
                linewidth=2,
                range=lims[0]
            )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=2,
                linewidth=2,
                range=lims[1]
            )
            fig.update_layout(showlegend=False)

        else:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=2,
                linewidth=2,
            )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=2,
                linewidth=2,
            )
            fig.update_layout(showlegend=False)

        if save_html is True:

            fig.write_html(f"models/GMM/{output}.html",
                include_plotlyjs="cdn",
                full_html=True,
                auto_open=False
            )
        if save_pdf is True:
            fig.write_image(f"models/GMM/{output}.pdf",format="pdf",width=width,height=height,scale=3)
        
        fig.show()

        if zoom_lims is not None:
            fig.update_xaxes(range=zoom_lims[0])
            fig.update_yaxes(range=zoom_lims[1])
            fig.write_image(f"models/GMM/{output}_zoom.pdf",format="pdf",width=width,height=height,scale=3)
        
        return fig
    
    def plot_probabilities(self, output, save_html=False, save_pdf=False, width=800, height=600, lims=None, zoom_lims=None):
        """
        Plot probability heatmap for cluster assignments
        """
        if self.X_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Sort points by maximum probability
        max_probs = np.max(self.probs_, axis=1)
        sorted_idx = np.argsort(max_probs)[::-1]
        
        fig = go.Figure(data=go.Heatmap(
            z=self.probs_[sorted_idx].T,
            x=np.arange(len(sorted_idx)),
            y=np.arange(self.n_components),
            colorscale='Viridis',
            colorbar=dict(title='Probability')
        ))
        
        fig.update_layout(
            title='Cluster Assignment Probabilities',
            xaxis_title='Data Point (sorted by max probability)',
            yaxis_title='Cluster',
            height=600
        )
        
        if lims is not None:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=2,
                linewidth=2,
                range=lims[0]
            )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=2,
                linewidth=2,
                range=lims[1]
            )
            fig.update_layout(showlegend=False)

        else:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=2,
                linewidth=2,
            )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.15)',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=2,
                linewidth=2,
            )
            fig.update_layout(showlegend=False)

        if save_html is True:

            fig.write_html(f"modles/GMM/{output}_probabilities.html",
                include_plotlyjs="cdn",
                full_html=True,
                auto_open=False
            )
        if save_pdf is True:
            fig.write_image(f"models/GMM/{output}_probabilities.pdf",format="pdf",width=width,height=height,scale=3)
        
        fig.show()

        if zoom_lims is not None:
            fig.update_xaxes(range=zoom_lims[0])
            fig.update_yaxes(range=zoom_lims[1])
            fig.write_image(f"models/GMM/{output}_zoom.pdf",format="pdf",width=width,height=height,scale=3)
    
    def get_cluster_stats(self):
        """Get statistics for each cluster"""
        stats = []
        for i in range(self.n_components):
            mask = self.labels_ == i
            stats.append({
                'cluster': i,
                'n_points': np.sum(mask),
                'weight': self.weights_[i],
                'mean_x': self.means_[i, 0],
                'mean_y': self.means_[i, 1],
                'cov_xx': self.covariances_[i, 0, 0],
                'cov_yy': self.covariances_[i, 1, 1],
                'cov_xy': self.covariances_[i, 0, 1],
                'avg_probability': np.mean(self.probs_[mask, i])
            })
        return pl.DataFrame(stats)


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    from data_analysis import df_signals
    """
    df_filt1 = df_signals.filter(
    (pl.col("int_water") > 0.0003) | (pl.col("int_deu") > 0.00025))

    df_filt2 = df_filt1.filter(
    (pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))
    """
    
    ti = time.time()
    
    # Prepare data
    y1 = df_signals["int_water"].to_numpy()*1000
    y2 = df_signals["int_deu"].to_numpy()*1000
    X = np.column_stack([y1, y2])
    
    # Fit GMM
    print("Fitting GMM...")
    gmm_clusterer = GMMLinearClusterer(
        n_components=20,
        covariance_type='full',
        max_iter=10000,
        tol=1e-4,
        reg_covar=1e-10
    )
    gmm_clusterer.fit(X)
    
    # Get statistics
    print("\nCluster Statistics:")
    stats = gmm_clusterer.get_cluster_stats()
    print(stats)
    
    # Create interactive plot
    print("\nGenerating interactive plot...")

    gmm_clusterer.plot_interactive(
        col_names=['Intensity (µV) | SO2 + H2O', 'Intensity (µV) | SO2 + D2O'],
        n_std=2,
        output='gmm_clustering_0_inf',
        show_ellipses=True,
        show_centers=True,
        point_size=6,
        ellipse_width=2,
        save_html=True,
        save_pdf=True,
        lims=[[-2,100],[-2,65]],
        width=600,
        height=600
    )
    
    # Optional: Plot probability heatmap
    #gmm_clusterer.plot_probabilities(output='gmm_probabilities.html')
    
    tf = time.time()
    print(f"\nRunning time: {tf - ti:.4f} s.")
    
    # Access results
    print(f"\nBIC: {gmm_clusterer.bic_:.2f}")
    print(f"AIC: {gmm_clusterer.aic_:.2f}")
    print(f"Log-Likelihood: {gmm_clusterer.log_likelihood_:.2f}")
