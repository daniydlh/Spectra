import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
import plotly.graph_objects as go
import polars as pl


# ============================================================================
# CUSTOM ANGULAR/ARCTAN METRICS FOR DBSCAN
# ============================================================================

def angular_distance(point1, point2):
    """
    Calculate angular distance between two points based on arctan2
    
    This measures the difference in angles from origin to each point.
    Good for clustering radial/linear patterns emanating from origin.
    
    Parameters:
    -----------
    point1, point2 : array-like, shape (2,)
        Points in 2D space [x, y]
    
    Returns:
    --------
    float : Angular distance in radians [0, π]
    """
    angle1 = np.arctan2(point1[1], point1[0])
    angle2 = np.arctan2(point2[1], point2[0])
    
    # Shortest angular distance (accounting for wraparound at ±π)
    diff = np.abs(angle1 - angle2)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    
    return diff


def radial_angular_distance(point1, point2, angular_weight=1.0, radial_weight=1.0):
    """
    Combined angular + radial distance
    
    Useful when you want to cluster by BOTH angle AND distance from origin.
    
    Parameters:
    -----------
    point1, point2 : array-like, shape (2,)
        Points in 2D space [x, y]
    angular_weight : float
        Weight for angular component (default: 1.0)
    radial_weight : float
        Weight for radial component (default: 1.0)
    
    Returns:
    --------
    float : Combined distance
    """
    # Angular distance
    angle1 = np.arctan2(point1[1], point1[0])
    angle2 = np.arctan2(point2[1], point2[0])
    angular_diff = np.abs(angle1 - angle2)
    if angular_diff > np.pi:
        angular_diff = 2 * np.pi - angular_diff
    
    # Radial distance
    r1 = np.sqrt(point1[0]**2 + point1[1]**2)
    r2 = np.sqrt(point2[0]**2 + point2[1]**2)
    radial_diff = np.abs(r1 - r2)
    
    # Combined distance
    return angular_weight * angular_diff + radial_weight * radial_diff


def polar_euclidean_distance(point1, point2):
    """
    Euclidean distance in polar coordinates (r, θ)
    
    Converts points to polar, then computes Euclidean distance.
    """
    # Convert to polar
    r1 = np.sqrt(point1[0]**2 + point1[1]**2)
    theta1 = np.arctan2(point1[1], point1[0])
    
    r2 = np.sqrt(point2[0]**2 + point2[1]**2)
    theta2 = np.arctan2(point2[1], point2[0])
    
    # Handle angle wraparound
    theta_diff = np.abs(theta1 - theta2)
    if theta_diff > np.pi:
        theta_diff = 2 * np.pi - theta_diff
    
    # Euclidean in (r, θ) space
    return np.sqrt((r1 - r2)**2 + (r1 * theta_diff)**2)


def slope_distance(point1, point2, origin=(0, 0)):
    """
    Distance based on difference in slope from origin
    
    Perfect for clustering linear patterns through origin!
    Points on the same line through origin have distance ≈ 0.
    
    Parameters:
    -----------
    point1, point2 : array-like, shape (2,)
        Points in 2D space [x, y]
    origin : tuple
        Origin point (default: (0, 0))
    
    Returns:
    --------
    float : Slope-based distance
    """
    # Shift to origin
    p1 = np.array(point1) - np.array(origin)
    p2 = np.array(point2) - np.array(origin)
    
    # Calculate slopes (use arctan for robustness)
    slope1 = np.arctan2(p1[1], p1[0])
    slope2 = np.arctan2(p2[1], p2[0])
    
    # Angular difference
    diff = np.abs(slope1 - slope2)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    
    return diff


# ============================================================================
# DBSCAN WITH CUSTOM METRICS - THREE APPROACHES
# ============================================================================

# APPROACH 1: Using metric='precomputed' with distance matrix
# ============================================================================

class DBSCANAngular:
    """
    DBSCAN with custom angular metrics using precomputed distances
    
    This is the most flexible approach - you can use ANY custom metric.
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric_type='angular', 
                 angular_weight=1.0, radial_weight=1.0):
        """
        Parameters:
        -----------
        eps : float
            Maximum distance (in your metric) between neighbors
        min_samples : int
            Minimum samples in neighborhood
        metric_type : str
            'angular': Pure angular distance
            'radial_angular': Combined angular + radial
            'polar_euclidean': Euclidean in polar space
            'slope': Slope-based distance
        angular_weight, radial_weight : float
            Weights for combined metric
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric_type = metric_type
        self.angular_weight = angular_weight
        self.radial_weight = radial_weight
        self.labels_ = None
        self.X_ = None
        
    def _compute_distance_matrix(self, X):
        """Compute pairwise distance matrix using custom metric"""
        n = len(X)
        dist_matrix = np.zeros((n, n))
        
        if self.metric_type == 'angular':
            metric_func = angular_distance
        elif self.metric_type == 'radial_angular':
            metric_func = lambda p1, p2: radial_angular_distance(
                p1, p2, self.angular_weight, self.radial_weight
            )
        elif self.metric_type == 'polar_euclidean':
            metric_func = polar_euclidean_distance
        elif self.metric_type == 'slope':
            metric_func = slope_distance
        else:
            raise ValueError(f"Unknown metric_type: {self.metric_type}")
        
        for i in range(n):
            for j in range(i+1, n):
                dist = metric_func(X[i], X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def fit(self, X):
        """Fit DBSCAN with custom metric"""
        self.X_ = np.asarray(X)
        
        print(f"Computing {self.metric_type} distance matrix...")
        dist_matrix = self._compute_distance_matrix(self.X_)
        
        print("Fitting DBSCAN...")
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, 
                       metric='precomputed')
        self.labels_ = dbscan.fit_predict(dist_matrix)
        self.core_sample_indices_ = dbscan.core_sample_indices_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_noise_ = list(self.labels_).count(-1)
        
        return self
    
    def plot_interactive(self, col_names=None, save_path=None):
        """Create interactive plot with angular sectors"""
        if self.X_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if col_names is None:
            col_names = ['X', 'Y']
        
        fig = go.Figure()
        
        # Color palette
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        while len(colors) < self.n_clusters_:
            colors.extend(colors)
        
        # Plot clusters
        unique_labels = set(self.labels_)
        for label in unique_labels:
            if label == -1:
                # Noise
                mask = self.labels_ == label
                fig.add_trace(go.Scatter(
                    x=self.X_[mask, 0],
                    y=self.X_[mask, 1],
                    mode='markers',
                    name=f'Noise (n={np.sum(mask)})',
                    marker=dict(size=4, color='lightgray', opacity=0.4),
                    hovertemplate=(
                        f'<b>Noise</b><br>{col_names[0]}: %{{x:.6f}}<br>'
                        f'{col_names[1]}: %{{y:.6f}}<br>Angle: %{{customdata:.4f}}°<extra></extra>'
                    ),
                    customdata=np.degrees(np.arctan2(self.X_[mask, 1], self.X_[mask, 0]))
                ))
            else:
                # Cluster
                mask = self.labels_ == label
                cluster_points = self.X_[mask]
                
                # Calculate cluster angle range
                angles = np.arctan2(cluster_points[:, 1], cluster_points[:, 0])
                mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
                
                fig.add_trace(go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode='markers',
                    name=f'Cluster {label} (n={np.sum(mask)}, θ={np.degrees(mean_angle):.1f}°)',
                    marker=dict(size=5, color=colors[label % len(colors)], opacity=0.7),
                    hovertemplate=(
                        f'<b>Cluster {label}</b><br>{col_names[0]}: %{{x:.6f}}<br>'
                        f'{col_names[1]}: %{{y:.6f}}<br>Angle: %{{customdata:.4f}}°<extra></extra>'
                    ),
                    customdata=np.degrees(angles)
                ))
                
                # Draw line from origin through cluster center
                if self.metric_type in ['angular', 'slope']:
                    max_r = np.max(np.sqrt(cluster_points[:, 0]**2 + cluster_points[:, 1]**2))
                    line_end_x = max_r * 1.2 * np.cos(mean_angle)
                    line_end_y = max_r * 1.2 * np.sin(mean_angle)
                    
                    fig.add_trace(go.Scatter(
                        x=[0, line_end_x],
                        y=[0, line_end_y],
                        mode='lines',
                        line=dict(color=colors[label % len(colors)], width=2, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add origin marker
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            name='Origin',
            marker=dict(size=12, color='black', symbol='x'),
            showlegend=True
        ))
        
        fig.update_layout(
            title=dict(
                text=f'DBSCAN with {self.metric_type.upper()} metric<br>' +
                     f'<sub>eps={self.eps:.4f}, min_samples={self.min_samples}, ' +
                     f'Clusters: {self.n_clusters_}, Noise: {self.n_noise_}</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=col_names[0],
            yaxis_title=col_names[1],
            hovermode='closest',
            template='plotly_white',
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        return fig


# APPROACH 2: Using callable metric (faster for some cases)
# ============================================================================

def dbscan_with_callable_metric(X, eps, min_samples, metric_func):
    """
    DBSCAN with custom callable metric
    
    Some sklearn versions support passing a callable directly.
    This avoids computing the full distance matrix.
    
    Parameters:
    -----------
    X : array-like
        Data
    eps : float
        Epsilon parameter
    min_samples : int
        Minimum samples
    metric_func : callable
        Function that takes two points and returns distance
    
    Returns:
    --------
    labels : array
        Cluster labels
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric_func)
    labels = dbscan.fit_predict(X)
    return labels


# ============================================================================
# TUNING FOR ANGULAR METRICS
# ============================================================================
def estimate_angular_eps(X, min_samples=5, plot=True):
    """
    Estimate optimal eps for angular metric
    
    Since angular distance is in radians, typical eps values are:
    - Very tight clusters: 0.01 - 0.05 radians (~0.5° - 3°)
    - Moderate clusters: 0.05 - 0.2 radians (~3° - 11°)
    - Loose clusters: 0.2 - 0.5 radians (~11° - 29°)
    """
    # Compute angular distances
    n = len(X)
    angles = np.arctan2(X[:, 1], X[:, 0])
    
    # For each point, find k-nearest angular neighbors
    k = min_samples
    k_distances = []
    
    for i in range(n):
        angle_i = angles[i]
        # Compute angular distance to all other points
        dists = np.abs(angles - angle_i)
        dists = np.minimum(dists, 2*np.pi - dists)  # Handle wraparound
        dists = np.sort(dists)
        if len(dists) > k:
            k_distances.append(dists[k])  # k-th nearest (index k because self is at 0)
        else:
            k_distances.append(dists[-1])
    
    k_distances = np.sort(k_distances)[::-1]
    
    if plot:
        fig = go.Figure()
        
        # Plot in radians
        fig.add_trace(go.Scatter(
            x=np.arange(len(k_distances)),
            y=k_distances,
            mode='lines',
            name='Radians',
            line=dict(width=2, color='blue'),
            hovertemplate='Point: %{x}<br>Distance: %{y:.4f} rad<br>≈ %{customdata:.2f}°<extra></extra>',
            customdata=np.degrees(k_distances)
        ))
        
        fig.update_layout(
            title=f'Angular K-Distance Graph (k={k})<br><sub>Look for elbow to estimate eps</sub>',
            xaxis_title='Points (sorted by angular distance)',
            yaxis_title=f'{k}-th Nearest Angular Neighbor Distance (radians)',
            template='plotly_white',
            height=600,
            hovermode='x unified'
        )
        
        # Add annotations for common thresholds
        thresholds = [
            (0.05, '~3°', 'tight'),
            (0.1, '~6°', 'moderate'),
            (0.2, '~11°', 'loose')
        ]
        
        for threshold, label, desc in thresholds:
            if threshold < k_distances.max():
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"{label} ({desc})",
                    annotation_position="right"
                )
        
        fig.show()
        
        # Suggest eps using simple elbow detection
        # Find point where slope changes most
        if len(k_distances) > 10:
            # Calculate first derivative (slope)
            slopes = np.diff(k_distances)
            # Find elbow as max change in slope
            elbow_idx = np.argmax(np.abs(np.diff(slopes))) + 1
            suggested_eps = k_distances[min(elbow_idx, len(k_distances)-1)]
        else:
            suggested_eps = np.median(k_distances)
        
        print(f"\nSuggested eps: {suggested_eps:.4f} radians ({np.degrees(suggested_eps):.2f}°)")
        print(f"\nGuidelines:")
        print(f"  Tight clusters:    0.01-0.05 rad  (~0.5°-3°)")
        print(f"  Moderate clusters: 0.05-0.2 rad   (~3°-11°)")
        print(f"  Loose clusters:    0.2-0.5 rad    (~11°-29°)")
    
    return k_distances


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import time
    from data_analysis import df_signals
    df_filt1 = df_signals.filter(
    (pl.col("int_water") > 0.00017) | (pl.col("int_deu") > 0.00012))

    df_filt2 = df_filt1.filter(
    (pl.col("int_water") != 0.) & (pl.col("int_deu") != 0.))
    
    ti = time.time()
    
    # Prepare data
    y1 = df_filt1["int_water"].to_numpy()
    y2 = df_filt1["int_deu"].to_numpy()
    X = np.column_stack([y1, y2])
    
    print("="*70)
    print("DBSCAN WITH ANGULAR METRICS")
    print("="*70)
    
    # Step 1: Estimate eps for angular metric
    print("\n1. Estimating optimal eps for angular metric...")
    estimate_angular_eps(X, min_samples=7, plot=True)
    
    # Step 2: Try different angular metrics
    """
    metric_configs = [
        {'metric_type': 'angular', 'eps': 0.002, 'name': 'Pure Angular'},
        {'metric_type': 'slope', 'eps': 0.005, 'name': 'Slope-based'},
        {'metric_type': 'radial_angular', 'eps': 0.01, 'angular_weight': 1.0, 
         'radial_weight': 0.01, 'name': 'Angular + Radial (angular-heavy)'},
    ]
    """
    metric_configs = [
        {'metric_type': 'angular', 'eps': 0.003, 'name': 'Pure Angular'}
    ]
    
    for config in metric_configs:
        print(f"\n2. Testing {config['name']}...")
        name = config.pop('name')
        
        dbscan_angular = DBSCANAngular(min_samples=5, **config)
        dbscan_angular.fit(X)
        
        print(f"   Clusters: {dbscan_angular.n_clusters_}")
        print(f"   Noise points: {dbscan_angular.n_noise_}")
        
        dbscan_angular.plot_interactive(
            col_names=['Intensidad Water', 'Intensidad Deuterio'],
            save_path=f'models/DBSCAN/dbscan_{config["metric_type"]}.html'
        )
    
    tf = time.time()
    print(f"\nTotal running time: {tf - ti:.4f} s.")
"""

## Key Points about Angular Metrics:

### **When to use angular metrics:**
1. **Linear patterns through origin** - Perfect for your case!
2. **Radial/spoke patterns** - Data organized in rays from center
3. **Directional clustering** - When angle matters more than distance

### **Metric Recommendations:**

1. **`angular_distance`** - Pure angle-based
   - Best for: Linear patterns, all radii
   - eps range: 0.01 - 0.5 radians (0.5° - 29°)
   
2. **`slope_distance`** - Identical to angular for origin-centered data
   - Best for: Lines through origin
   - eps range: Same as angular

3. **`radial_angular_distance`** - Combined angle + radius
   - Best for: When you want lines AND distance grouping
   - Tune weights: High angular_weight for linear focus

4. **`polar_euclidean_distance`** - Euclidean in (r, θ) space
   - Best for: Balanced radial/angular clustering

### **eps Guidelines for Angular:**

Tight angular clusters:   0.01 - 0.05 rad  (~0.5° - 3°)
Medium:                   0.05 - 0.2 rad   (~3° - 11°)
Loose:                    0.2 - 0.5 rad    (~11° - 29°)
"""