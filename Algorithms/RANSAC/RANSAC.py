from numpy import arctan
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


class LinearClusterer:
    """
    Clusters data points into multiple linear patterns using RANSAC.
    """
    
    def __init__(self, distance_threshold=0.5, angle_threshold=0.5, angle_growth=0.0015, angle_max=0.03, min_samples=5, max_clusters=15, max_iterations=200, force_origin=False, distance_type='euclidean'):
        """
        Parameters:
        -----------
        distance_threshold : float
            Maximum distance from a point to a line to be considered an inlier
        min_samples : int
            Minimum number of points to form a cluster
        max_clusters : int
            Maximum number of linear clusters to find
        """
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.angle_growth = angle_growth
        self.angle_max = angle_max
        self.min_samples = min_samples
        self.max_clusters = max_clusters
        self.max_iterations = max_iterations
        self.force_origin = force_origin
        self.distance_type = distance_type
        self.clusters_ = []
        self.labels_ = None

        
    def point_to_line_distance(self, points, slope, intercept):
        """
        Calculate perpendicular distance from points to a line y = slope*x + intercept
        """
        # Convert to ax + by + c = 0 form: -slope*x + 1*y - intercept = 0
        a = -slope
        b = 1
        c = -intercept
        
        # Distance formula: |ax + by + c| / sqrt(a^2 + b^2)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(a**2 + b**2)
        return distances
    
    def angular_distance(self, points, slope):
        """
        Angular distance between points and a line through the origin
        """
        theta_points = np.arctan2(points[:, 1], points[:, 0])
        theta_line = np.arctan(slope)
        dtheta = np.abs(theta_points - theta_line)
        return np.minimum(dtheta, np.pi - dtheta)
    
    def current_angle_threshold(self, cluster_id):
        theta = self.angle_threshold * (1 + self.angle_growth * cluster_id)
        return min(theta, self.angle_max)
    
    def fit_line(self, X):
        """
        Fit a line to points using least squares
        Returns slope and intercept
        """
        if len(X) < 2:
            return None, None
        
        # Use least squares
        x = X[:, 0]
        y = X[:, 1]
        
        if self.force_origin:
            # Least squares slope with intercept = 0
            denom = np.sum(x**2)
            if denom < 1e-12:
                return np.inf, 0.0
            slope = np.sum(x * y) / denom
            return slope, 0.0

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        
        if abs(denominator) < 1e-10:
            # Vertical line case
            return np.inf, np.mean(x)
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    def ransac_single_cluster(self, X, angle_threshold=None):
        """
        Find a single linear cluster using RANSAC
        """
        if angle_threshold is None:
                angle_threshold = self.angle_threshold

        if len(X) < self.min_samples:
            return None, None
        
        best_slope = None
        best_intercept = None
        best_inliers = []
        best_inlier_mask = np.zeros(len(X), dtype=bool)
        
        for _ in range(self.max_iterations):
            # Randomly sample 2 points
            sample_indices = np.random.choice(len(X), size=2, replace=False)
            sample = X[sample_indices]
            
            # Fit line through these points
            slope, intercept = self.fit_line(sample)
            
            if slope is None:
                continue
            
            if self.distance_type =='angular':
                distances = self.angular_distance(X, slope)
                inlier_mask = distances < self.angle_threshold
                
            else:
                # Handle vertical lines
                if np.isinf(slope):
                    distances = np.abs(X[:, 0] - intercept)
                else:
                    distances = self.point_to_line_distance(X, slope, intercept)
                # Find inliers
                inlier_mask = distances < self.distance_threshold
            inliers = X[inlier_mask]
            
            # Update best model if this is better
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_inlier_mask = inlier_mask
                best_slope = slope
                best_intercept = intercept
        
        # Refit with all inliers
        if len(best_inliers) >= self.min_samples:
            best_slope, best_intercept = self.fit_line(best_inliers)

            # Compute R_2 with best slope and intercept
            y_pred = best_slope * best_inliers[:,0] + best_intercept
            y_mean = np.mean(best_inliers[:,1])
            ss_res = np.sum((best_inliers[:,1] - y_pred)**2)      # residual sum of squares
            ss_tot = np.sum((best_inliers[:,1] - y_mean)**2)     # total sum of squares
            r2 = 1 - ss_res / ss_tot

            return best_inlier_mask, (best_slope, best_intercept, np.arctan(best_slope), r2)
        
        return None, None
    
    def fit(self, X):
        """
        Fit the linear clusterer to data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Input data points
        
        Returns:
        --------
        self
        """
        X = np.array(X)
        if X.shape[1] != 2:
            raise ValueError("Input data must be 2D (x, y coordinates)")
        
        self.labels_ = -np.ones(len(X), dtype=int)  # -1 means unassigned
        remaining_indices = np.arange(len(X))
        self.clusters_ = []
        
        once = False

        for cluster_id in range(self.max_clusters):

            if self.distance_type == "angular":
                    current_angle = self.current_angle_threshold(cluster_id)

                    # Optional: stop if precision is already meaningless
                    if current_angle >= self.angle_max and once is False:
                        once = True
                        print(f"Stopping: angle threshold reached max at cluster {cluster_id}")

            else:
                current_angle = None
            
            remaining_points = X[remaining_indices]
            
            if len(remaining_points) < self.min_samples:
                break
            
            # Find next cluster
            inlier_mask, line_params = self.ransac_single_cluster(remaining_points, angle_threshold=current_angle)
            
            if inlier_mask is None:
                break
            
            # Get original indices of inliers
            inlier_original_indices = remaining_indices[inlier_mask]
            
            # Assign cluster labels
            self.labels_[inlier_original_indices] = cluster_id
            
            # Store cluster info
            self.clusters_.append({
                'id': cluster_id,
                'slope': line_params[0],
                'intercept': line_params[1],
                'arctan': line_params[2],
                'current_angle_threshold': current_angle,
                'r2': line_params[3],
                'points': X[inlier_original_indices],
                'n_points': len(inlier_original_indices)
            })
            
            # Remove assigned points from remaining
            remaining_indices = remaining_indices[~inlier_mask]
            
            if len(remaining_indices) == 0:
                break
        self.unnassigned = remaining_indices
        
        return self
    
    def predict(self, X):
        """
        Assign new points to existing clusters based on distance to lines
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            New data points to classify
        
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels for each point
        """
        X = np.array(X)
        labels = -np.ones(len(X), dtype=int)
        
        for i, point in enumerate(X):
            min_distance = float('inf')
            best_cluster = -1
            
            for cluster in self.clusters_:
                slope = cluster['slope']
                intercept = cluster['intercept']
                
                if np.isinf(slope):
                    distance = abs(point[0] - intercept)
                else:
                    distance = self.point_to_line_distance(point.reshape(1, -1), slope, intercept)[0]
                
                if distance < min_distance and distance < self.distance_threshold:
                    min_distance = distance
                    best_cluster = cluster['id']
            
            labels[i] = best_cluster
        
        return labels
    
    def plot(self, X, figsize=(12, 8)):
        """
        Visualize the clustered data
        """
        X = np.array(X)
        
        plt.figure(figsize=figsize)
        
        # Plot points colored by cluster
        unique_labels = np.unique(self.labels_)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Outliers in gray
                mask = self.labels_ == label
                plt.scatter(X[mask, 0], X[mask, 1], c='gray', alpha=0.3, 
                           label='Unassigned', s=30)
            else:
                mask = self.labels_ == label
                plt.scatter(X[mask, 0], X[mask, 1], c=[color], alpha=0.7, 
                           label=f'Cluster {label}', s=50)
        
        # Plot fitted lines
        x_range = np.array([X[:, 0].min(), X[:, 0].max()])
        
        for cluster in self.clusters_:
            slope = cluster['slope']
            intercept = cluster['intercept']
            
            if np.isinf(slope):
                # Vertical line
                plt.axvline(x=intercept, color='red', linestyle='--', alpha=0.5, linewidth=2)
            else:
                y_range = slope * x_range + intercept
                plt.plot(x_range, y_range, 'r--', alpha=0.5, linewidth=2)
        
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('Linear Cluster Analysis', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_interactive(self, X, width=1200, height=800, lims=None, dir=None):
            """
            Create an interactive visualization using Plotly
            
            Parameters:
            -----------
            X : array-like, shape (n_samples, 2)
                Input data points
            width : int
                Figure width in pixels
            height : int
                Figure height in pixels
            lims: Tiple
                xrane and yrange limits
                [[x_low,x_high][y_low],[y_high]]
            filename: str
                .html file name. eg: "model1.html"
            """

            X = np.array(X)

            # Create figure
            fig = go.Figure()

            # Color palette
            colors_list = px.colors.qualitative.Plotly + px.colors.qualitative.Set1

            # Axis limits and padding
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            x_range = np.array([x_min - x_padding, x_max + x_padding])

            # Assign colors per cluster
            unique_labels = np.unique(self.labels_)
            cluster_colors = {label: colors_list[i % len(colors_list)] for i, label in enumerate(unique_labels)}

            # Plot points
            for label in unique_labels:
                mask = self.labels_ == label
                cluster_points = X[mask]

                if label == -1:
                    # Unassigned points
                    fig.add_trace(go.Scatter(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode='markers',
                        name='Unassigned',
                        marker=dict(
                            size=8,
                            color='lightgray',
                            opacity=0.5,
                            line=dict(width=1, color='gray')
                        ),
                        text=[f'Unassigned<br>X: {x:.5f}<br>Y: {y:.5f}' for x, y in cluster_points],
                        hoverinfo='text'
                    ))
                else:
                    # Cluster points
                    cluster_info = next(c for c in self.clusters_ if c['id'] == label)
                    slope = cluster_info['slope']
                    intercept = cluster_info['intercept']
                    arctan = cluster_info['arctan']


                    if np.isinf(slope):
                        equation = f'x = {intercept:.2f}'
                    else:
                        equation = f'y = {slope:.3f}x + {intercept:.2f}'

                    color = cluster_colors[label]

                    fig.add_trace(go.Scatter(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode='markers',
                        name=f'Cluster {label}',
                        marker=dict(
                            size=10,
                            color=color,
                            opacity=0.9,
                            line=dict(width=1, color='white')
                        ),
                        text=[f'Cluster {label}<br>X: {x:.5f}<br>Y: {y:.5f}<br>{equation}<br>arctan: {arctan:.5f}' for x, y in cluster_points],
                        hoverinfo='text'
                    ))

            # Plot regression lines
            for cluster in self.clusters_:
                slope = cluster['slope']
                intercept = cluster['intercept']
                color = cluster_colors[cluster['id']]

                if np.isinf(slope):
                    # Vertical line
                    fig.add_trace(go.Scatter(
                        x=[intercept, intercept],
                        y=[y_min - y_padding, y_max + y_padding],
                        mode='lines',
                        name=f'Line {cluster["id"]}',
                        line=dict(color=color, width=2, dash='dash'),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                else:
                    y_line = slope * x_range + intercept
                    equation = f'y = {slope:.3f}x + {intercept:.2f}'

                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_line,
                        mode='lines',
                        name=f'Line {cluster["id"]}',
                        opacity=0.75,
                        line=dict(color=color, width=2, dash='dash'),
                        hovertemplate=f'<b>Cluster {cluster["id"]}</b><br>{equation}<br><extra></extra>',
                        showlegend=False
                    ))

            # Layout and axes
            fig.update_layout(
                title={
                    'text': 'Interactive Linear Cluster Analysis',
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
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1)
            )
            if lims is not None:
                # Optional: set general limits:
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='lightgray',
                    range=lims[0]
                )

                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='lightgray',
                    range=lims[1]
                )
                fig.update_layout(showlegend=False)

            else:
            # Axes with grid and padding
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                                zeroline=True, zerolinewidth=2, zerolinecolor='lightgray',
                                range=[x_min - x_padding, x_max + x_padding])
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                                zeroline=True, zerolinewidth=2, zerolinecolor='lightgray',
                                range=[y_min - y_padding, y_max + y_padding])
                fig.update_layout(showlegend=False)

            # Show and save
            fig.show()
            fig.write_html(dir, include_plotlyjs="cdn")
        
    def get_cluster_info(self):
        """
        Get information about each cluster
        """
        info = []
        arctan_list = []
        r2_sum = 0.
        
        for cluster in self.clusters_:
            info.append({
                'Cluster ID': cluster['id'],
                'N Points': cluster['n_points'],
                'Slope': f"{cluster['slope']:.4f}" if not np.isinf(cluster['slope']) else "Vertical",
                'Intercept': f"{cluster['intercept']:.7f}",
                'R2': f"{cluster['r2']:.4f}",
                'arctan': f"{cluster['arctan']:.4f}"
            })
            r2_sum += cluster["r2"]
            arctan_list.append(float(cluster["arctan"]))

        r2_avg = float(r2_sum/len(self.clusters_))
        info.append({
            'R2 avg': r2_avg,
            'Distances (arctan)': sorted(arctan_list)
        })

        return info

    def write_cluster_dat(self, filename, cluster_id):
        """
        Write cluster dictionary to a .dat file with metadata header
        and x y point list.
        """
        cluster = self.clusters_[cluster_id]

        with open(filename, "w") as f:
            # --- metadata
            f.write(f"id {int(cluster['id'])}\n")
            f.write(f"slope {float(cluster['slope'])}\n")
            f.write(f"intercept {float(cluster['intercept'])}\n")
            f.write(f"r2 {float(cluster['r2'])}\n")
            f.write(f"arctan {float(cluster['arctan'])}\n")
            f.write(f"n_points {int(cluster['n_points'])}\n")

            # --- blank line
            f.write("\n")

            # --- column header
            f.write("x y\n")

            # --- points
            for x, y in cluster["points"]:
                f.write(f"{x:.8e} {y:.8e}\n")
"""
# Example usage and testing
if __name__ == "__main__":
    # Generate sample data with multiple linear patterns
    np.random.seed(42)
    
    data = []
    n_patterns = 12  # Number of linear patterns
    
    for i in range(n_patterns):
        # Random line parameters
        slope = np.random.uniform(-3, 3)
        intercept = np.random.uniform(-30, 30)
        n_points = np.random.randint(20, 50)
        
        # Generate points along the line with noise
        x = np.random.uniform(0, 100, n_points)
        y = slope * x + intercept + np.random.normal(0, 2, n_points)
        
        pattern_data = np.column_stack([x, y])
        data.append(pattern_data)
    
    # Combine all data
    X = np.vstack(data)
    
    # Create and fit the clusterer
    clusterer = LinearClusterer(
        distance_threshold=3.0,
        min_samples=10,
        max_clusters=15
    )
    
    clusterer.fit(X)
    
    # Print cluster information
    print("=" * 60)
    print("LINEAR CLUSTERING RESULTS")
    print("=" * 60)
    print(f"\nTotal points: {len(X)}")
    print(f"Number of clusters found: {len(clusterer.clusters_)}")
    print(f"Unassigned points: {np.sum(clusterer.labels_ == -1)}")
    print("\nCluster Details:")
    print("-" * 60)
    
    for info in clusterer.get_cluster_info():
        print(f"Cluster {info['Cluster ID']}: "
              f"{info['N Points']} points, "
              f"slope={info['Slope']}, "
              f"intercept={info['Intercept']}")
    
    print("\n" + "=" * 60)
    
    # Get cluster assignments
    labels = clusterer.labels_
    print(f"\nCluster assignments shape: {labels.shape}")
    print(f"First 10 assignments: {labels[:10]}")
    
    # Visualize results
    clusterer.plot(X)
    
    # Example of predicting on new points
    new_points = np.array([[50, 25], [75, -10], [30, 40]])
    new_labels = clusterer.predict(new_points)
    print(f"\nNew point predictions: {new_labels}")
"""