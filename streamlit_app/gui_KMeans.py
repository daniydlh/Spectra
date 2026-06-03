import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class KMeansAngularClusterer:
    """
    Clusters data points into linear-through-origin patterns using a
    K-Means algorithm operating on angular distance.

    Algorithm
    ---------
    1. Map every point to its angle from the origin in [0, π/2] (lines
       are undirected, so angles are folded into that half-range).
    2. Run K-Means in that 1-D angular feature space for a range of k
       values (1 … max_clusters); select k by the elbow of the inertia
       curve (largest second derivative) unless n_clusters is given
       explicitly.
    3. For each cluster, fit a least-squares line forced through the
       origin and compute slope, arctan, MAD, residuals — exactly
       mirroring the LinearClusterer data structures so all downstream
       methods (write_df_output, plot_interactive, iterative_refinement,
       …) work without modification.

    Points whose angular distance to their assigned cluster's line
    exceeds ``outlier_threshold`` (radians) are marked unassigned (-1)
    after the initial fit.  Set to None to keep all points assigned.

    Parameters
    ----------
    n_clusters : int or None
        Fixed number of clusters.  If None the elbow method selects k
        automatically between 1 and max_clusters.
    max_clusters : int
        Upper bound for the elbow search (ignored when n_clusters is set).
    min_samples : int
        Minimum cluster size after outlier removal; smaller clusters are
        dissolved and their points become unassigned (-1).
    outlier_threshold : float or None
        Angular distance (radians) above which a point is considered an
        outlier after fitting.  None disables outlier removal.
    max_iter : int
        Maximum K-Means iterations.
    n_init : int
        Number of K-Means restarts; best inertia is kept.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters=None,
        max_clusters=30,
        min_samples=2,
        outlier_threshold=None,
        max_iter=300,
        n_init=10,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.min_samples = min_samples
        self.outlier_threshold = outlier_threshold
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.clusters_ = []
        self.labels_ = None
        self.unassigned = np.array([], dtype=int)
        self.unassigned_vals = []
        self.inertias_ = []          # inertia curve from elbow search
        self.n_clusters_selected_ = None

        # expose so shared helpers work unchanged
        self.distance_type = "angular"

    # ------------------------------------------------------------------
    # Angular geometry helpers  (identical to LinearClusterer)
    # ------------------------------------------------------------------

    def angular_distance(self, points, slope):
        """Unsigned angular distance of each point from a line direction."""
        v = np.array([0.0, 1.0]) if np.isinf(slope) else np.array([1.0, slope])
        v /= np.linalg.norm(v)
        norms = np.linalg.norm(points, axis=1)
        valid = norms > 0
        cos_theta = np.zeros(len(points))
        cos_theta[valid] = np.abs((points[valid] @ v) / norms[valid])
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    def angular_distance_histogram(self, points, slope):
        """Signed angular distance in [-π/2, π/2] for histogram display."""
        v = np.array([0.0, 1.0]) if np.isinf(slope) else np.array([1.0, slope])
        v /= np.linalg.norm(v)
        norms = np.linalg.norm(points, axis=1)
        valid = norms > 0
        cos_theta = np.zeros(len(points))
        sin_theta = np.zeros(len(points))
        cos_theta[valid] = (points[valid] @ v) / norms[valid]
        sin_theta[valid] = (
            points[valid, 0] * v[1] - points[valid, 1] * v[0]
        ) / norms[valid]
        return np.arctan2(sin_theta, cos_theta)

    def point_to_line_distance(self, points, slope, intercept):
        """Perpendicular distance from points to y = slope*x + intercept."""
        a, b, c = -slope, 1, -intercept
        return np.abs(a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(a**2 + b**2)

    # ------------------------------------------------------------------
    # Line fitting (forced through origin, identical to LinearClusterer)
    # ------------------------------------------------------------------

    def fit_line(self, X):
        """Least-squares line forced through the origin."""
        x, y = X[:, 0], X[:, 1]
        denom = np.sum(x ** 2)
        if denom < 1e-12:
            return np.inf, 0.0
        return np.sum(x * y) / denom, 0.0

    # ------------------------------------------------------------------
    # Angular feature mapping
    # ------------------------------------------------------------------

    def _angular_features(self, X):
        """
        Map each 2-D point to its angle from the origin in [0, π/2].
        This is the 1-D feature space K-Means operates on.
        """
        norms = np.linalg.norm(X, axis=1)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        unit_x = np.clip(X[:, 0] / norms, -1.0, 1.0)
        angles = np.arccos(unit_x)          # [0, π]
        return np.where(angles > np.pi / 2, np.pi - angles, angles)  # fold to [0, π/2]

    # ------------------------------------------------------------------
    # 1-D K-Means (no sklearn dependency, runs on angular features)
    # ------------------------------------------------------------------

    def _kmeans_1d(self, angles, k):
        """
        Simple Lloyd's algorithm in 1-D.

        Returns
        -------
        labels : ndarray, shape (n,)
        centroids : ndarray, shape (k,)
        inertia : float
        """
        rng = np.random.RandomState(self.random_state)
        # K-Means++ initialisation in 1-D
        centroids = [angles[rng.randint(len(angles))]]
        for _ in range(k - 1):
            dists = np.min(
                np.abs(angles[:, None] - np.array(centroids)[None, :]), axis=1
            ) ** 2
            probs = dists / dists.sum()
            centroids.append(angles[rng.choice(len(angles), p=probs)])
        centroids = np.array(centroids)

        labels = np.zeros(len(angles), dtype=int)
        for _ in range(self.max_iter):
            # assignment
            dists = np.abs(angles[:, None] - centroids[None, :])
            new_labels = np.argmin(dists, axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            # update
            for j in range(k):
                pts = angles[labels == j]
                if len(pts):
                    centroids[j] = pts.mean()

        inertia = float(np.sum((angles - centroids[labels]) ** 2))
        return labels, centroids, inertia

    def _best_kmeans(self, angles, k):
        """Run n_init restarts of 1-D K-Means and return the best result."""
        best = None
        for _ in range(self.n_init):
            labels, centroids, inertia = self._kmeans_1d(angles, k)
            if best is None or inertia < best[2]:
                best = (labels, centroids, inertia)
        return best

    # ------------------------------------------------------------------
    # Elbow selection
    # ------------------------------------------------------------------

    def _elbow_k(self, angles):
        """
        Select k via the largest second difference of the inertia curve
        (elbow / knee point).  Searches k = 1 … max_clusters.
        """
        k_max = min(self.max_clusters, len(angles) - 1)
        inertias = []
        for k in range(1, k_max + 1):
            _, _, inertia = self._best_kmeans(angles, k)
            inertias.append(inertia)

        self.inertias_ = inertias

        if len(inertias) < 3:
            return len(inertias)

        # Second differences — largest jump signals the elbow
        second_diff = np.diff(np.diff(inertias))
        return int(np.argmax(second_diff)) + 2   # offset: k=1 is index 0

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit the KMeans angular clusterer to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)

        Returns
        -------
        self
        """
        X = np.array(X)
        if X.shape[1] != 2:
            raise ValueError("Input data must be 2-D (x, y coordinates).")

        angles = self._angular_features(X)

        # --- choose k
        if self.n_clusters is not None:
            k = min(self.n_clusters, len(X))
        else:
            k = self._elbow_k(angles)

        self.n_clusters_selected_ = k

        # --- run KMeans
        raw_labels, _, _ = self._best_kmeans(angles, k)

        # --- build initial labels array
        self.labels_ = raw_labels.astype(int)

        # --- fit a line to each raw cluster and optionally remove outliers
        self.clusters_ = []
        self.labels_ = -np.ones(len(X), dtype=int)

        unique_raw = np.unique(raw_labels)
        cluster_id = 0
        for raw_id in unique_raw:
            mask = raw_labels == raw_id
            pts = X[mask]
            orig_indices = np.where(mask)[0]

            slope, intercept = self.fit_line(pts)

            # optional outlier removal
            if self.outlier_threshold is not None:
                ang_dist = self.angular_distance(pts, slope)
                keep = ang_dist <= self.outlier_threshold
                pts = pts[keep]
                orig_indices = orig_indices[keep]
                if len(pts) >= self.min_samples:
                    slope, intercept = self.fit_line(pts)

            if len(pts) < self.min_samples:
                continue   # remains unassigned (-1)

            angles_signed = self.angular_distance_histogram(pts, slope)
            mad = float(np.mean(np.abs(angles_signed)))

            self.labels_[orig_indices] = cluster_id
            self.clusters_.append({
                "id": cluster_id,
                "slope": slope,
                "intercept": intercept,
                "arctan": float(np.arctan(slope)) if not np.isinf(slope) else np.pi / 2,
                "current_angle_threshold": self.outlier_threshold,
                "mad": mad,
                "points": pts,
                "n_points": len(pts),
                "point_distance": angles_signed,
                "residuals": angles_signed ** 2,
            })
            cluster_id += 1

        unassigned_indices = np.where(self.labels_ == -1)[0]
        self.unassigned = unassigned_indices
        self.unassigned_vals = [X[i] for i in unassigned_indices]

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Assign new points to the closest cluster by angular proximity.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)

        Returns
        -------
        labels : ndarray, shape (n_samples,)
        """
        X = np.array(X)
        labels = -np.ones(len(X), dtype=int)
        for i, point in enumerate(X):
            norm_p = np.linalg.norm(point)
            if norm_p < 1e-12:
                continue
            p_hat = point / norm_p
            best_id, best_angle = -1, np.inf
            for c in self.clusters_:
                slope = c["slope"]
                v = np.array([0.0, 1.0]) if np.isinf(slope) else np.array([1.0, slope])
                v /= np.linalg.norm(v)
                angle = np.arccos(np.clip(np.abs(np.dot(p_hat, v)), 0.0, 1.0))
                if angle < best_angle:
                    best_angle = angle
                    best_id = c["id"]
            labels[i] = best_id
        return labels

    # ------------------------------------------------------------------
    # reassign_by_angular_proximity  (identical API to LinearClusterer)
    # ------------------------------------------------------------------

    def reassign_by_angular_proximity(self, X, include_unassigned=True, distance_mode="angular"):
        """
        Reassign all points to clusters based on proximity to each
        cluster's fitted line (angular or orthogonal).
        Identical API to LinearClusterer.
        """
        X = np.array(X)

        if not self.clusters_:
            raise RuntimeError("No clusters found. Run fit() first.")
        if distance_mode not in ("angular", "orthogonal"):
            raise ValueError("distance_mode must be 'angular' or 'orthogonal'.")

        cluster_directions = []
        for c in self.clusters_:
            slope = c["slope"]
            v = np.array([0.0, 1.0]) if np.isinf(slope) else np.array([1.0, slope])
            cluster_directions.append(v / np.linalg.norm(v))

        candidate_indices = (
            np.arange(len(X)) if include_unassigned else np.where(self.labels_ != -1)[0]
        )
        new_labels = self.labels_.copy()

        if distance_mode == "angular":
            for idx in candidate_indices:
                p = X[idx]
                norm_p = np.linalg.norm(p)
                if norm_p < 1e-12:
                    new_labels[idx] = self.clusters_[0]["id"]
                    continue
                p_hat = p / norm_p
                best_id, best_angle = None, np.inf
                for c, v in zip(self.clusters_, cluster_directions):
                    angle = np.arccos(np.clip(np.abs(np.dot(p_hat, v)), 0.0, 1.0))
                    if angle < best_angle:
                        best_angle = angle
                        best_id = c["id"]
                new_labels[idx] = best_id
        else:
            pts = X[candidate_indices]
            dist_matrix = np.full((len(pts), len(self.clusters_)), np.inf)
            for j, c in enumerate(self.clusters_):
                slope, intercept = c["slope"], c["intercept"]
                if np.isinf(slope):
                    dist_matrix[:, j] = np.abs(pts[:, 0] - intercept)
                else:
                    dist_matrix[:, j] = self.point_to_line_distance(pts, slope, intercept)
            best_j = np.argmin(dist_matrix, axis=1)
            for ki, idx in enumerate(candidate_indices):
                new_labels[idx] = self.clusters_[best_j[ki]]["id"]

        self.labels_ = new_labels

        for c, v in zip(self.clusters_, cluster_directions):
            cid = c["id"]
            mask = self.labels_ == cid
            pts = X[mask]
            c["points"] = pts
            c["n_points"] = len(pts)

            if len(pts) == 0:
                c["point_distance"] = np.array([])
                c["residuals"] = np.array([])
                c["mad"] = 0.0
                continue

            slope, intercept = c["slope"], c["intercept"]
            if self.distance_type == "angular":
                angles = self.angular_distance_histogram(pts, slope)
                c["point_distance"] = angles
                c["residuals"] = angles ** 2
                c["mad"] = float(np.mean(np.abs(angles)))
            else:
                if np.isinf(slope):
                    ortho = pts[:, 0] - intercept
                else:
                    a, b, cc = -slope, 1.0, -intercept
                    ortho = (a * pts[:, 0] + b * pts[:, 1] + cc) / np.sqrt(a**2 + b**2)
                c["point_distance"] = ortho
                c["residuals"] = ortho ** 2
                c["mad"] = float(np.mean(np.abs(ortho)))

        unassigned_indices = np.where(self.labels_ == -1)[0]
        self.unassigned = unassigned_indices
        self.unassigned_vals = [X[i] for i in unassigned_indices]

        return self

    # ------------------------------------------------------------------
    # iterative_refinement  (identical API to LinearClusterer)
    # ------------------------------------------------------------------

    def iterative_refinement(
        self,
        X,
        max_iterations=20,
        convergence_threshold=0.0,
        distance_mode="angular",
        verbose=True,
    ):
        """
        Iteratively refine cluster assignments by alternating reassignment
        and line refitting.  Identical API to LinearClusterer.
        """
        X = np.array(X)

        if not self.clusters_:
            raise RuntimeError("No clusters found. Run fit() first.")

        if verbose:
            print(f"{'Iter':>5}  {'Max ΔN':>9}  {'Converged':>10}  {'Populations'}")
            print("-" * 72)

        for iteration in range(1, max_iterations + 1):
            prev_populations = {c["id"]: c["n_points"] for c in self.clusters_}

            self.reassign_by_angular_proximity(
                X, include_unassigned=True, distance_mode=distance_mode
            )

            for c in self.clusters_:
                pts = c["points"]
                if len(pts) < self.min_samples:
                    if verbose:
                        print(
                            f"  [iter {iteration}] cluster {c['id']} has only "
                            f"{len(pts)} pts — skipping refit."
                        )
                    continue

                new_slope, new_intercept = self.fit_line(pts)
                if new_slope is None:
                    continue

                c["slope"] = new_slope
                c["intercept"] = new_intercept
                c["arctan"] = (
                    float(np.arctan(new_slope)) if not np.isinf(new_slope) else np.pi / 2
                )

                angles = self.angular_distance_histogram(pts, new_slope)
                c["point_distance"] = angles
                c["residuals"] = angles ** 2
                c["mad"] = float(np.mean(np.abs(angles)))

            new_populations = {c["id"]: c["n_points"] for c in self.clusters_}
            population_delta = {
                cid: abs(new_populations[cid] - prev_populations[cid])
                for cid in new_populations
            }
            max_delta = max(population_delta.values())

            if verbose:
                pop_str = "  ".join(
                    f"C{cid}:{new_populations[cid]}" for cid in sorted(new_populations)
                )
                converged_str = "YES" if max_delta <= convergence_threshold else "no"
                print(f"{iteration:>5}  {max_delta:>9}  {converged_str:>10}  {pop_str}")

            if max_delta <= convergence_threshold:
                if verbose:
                    print(
                        f"\nConverged after {iteration} iteration(s) "
                        f"(max ΔN = {max_delta} ≤ threshold {convergence_threshold})."
                    )
                break
        else:
            if verbose:
                print(
                    f"\nReached max_iterations={max_iterations} without convergence "
                    f"(last max ΔN = {max_delta})."
                )

        self.n_iterations_ = iteration
        return self

    # ------------------------------------------------------------------
    # plot (matplotlib, quick look)
    # ------------------------------------------------------------------

    def plot(self, X, figsize=(12, 8)):
        """Visualize clustered data using matplotlib."""
        X = np.array(X)
        plt.figure(figsize=figsize)
        unique_labels = np.unique(self.labels_)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = self.labels_ == label
            if label == -1:
                plt.scatter(X[mask, 0], X[mask, 1], c="gray", alpha=0.3, label="Unassigned", s=30)
            else:
                plt.scatter(X[mask, 0], X[mask, 1], c=[color], alpha=0.7, label=f"Cluster {label}", s=50)

        x_range = np.array([X[:, 0].min(), X[:, 0].max()])
        for cluster in self.clusters_:
            slope, intercept = cluster["slope"], cluster["intercept"]
            if np.isinf(slope):
                plt.axvline(x=intercept, color="red", linestyle="--", alpha=0.5, linewidth=2)
            else:
                plt.plot(x_range, slope * x_range + intercept, "r--", alpha=0.5, linewidth=2)

        plt.xlabel("Intensity (µV) 1", fontsize=18)
        plt.ylabel("Intensity (µV) 2", fontsize=18)
        plt.title("KMeans Angular model", fontsize=14, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # plot_interactive  (identical API to LinearClusterer)
    # ------------------------------------------------------------------

    def plot_interactive(
        self,
        X,
        width=800,
        height=600,
        lims=None,
        cols=None,
        zoom_lims=None,
        axis=None,
        peaks=None,
        freqs=None,
        show_fig=True,
        save_html=None,
        save_pdf=None,
        model_path=None,
        sort_by_arctan=False,
    ):
        """Interactive Plotly visualisation — identical API to LinearClusterer.plot_interactive."""

        X = np.array(X) * 1000
        if cols is not None:
            x_peak = peaks[cols[0]].to_numpy() * 1000
            y_peak = peaks[cols[1]].to_numpy() * 1000
        else:
            x_peak = peaks[:, 1].to_numpy() * 1000
            y_peak = peaks[:, 2].to_numpy() * 1000
        X_peak = np.column_stack((x_peak, y_peak))
        peak_labels = peaks["cluster"].to_numpy()
        peak_freqs = peaks["freq"].to_numpy() if "freq" in peaks.columns else None

        if sort_by_arctan and self.clusters_:
            sorted_ids = [c["id"] for c in sorted(self.clusters_, key=lambda c: float(c["arctan"]))]
            label_remap = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
            label_remap[-1] = -1
            display_labels = np.array([label_remap.get(l, -1) for l in self.labels_])
            cluster_id_to_display = label_remap
        else:
            display_labels = self.labels_
            cluster_id_to_display = {c["id"]: c["id"] for c in self.clusters_}
            cluster_id_to_display[-1] = -1
            sorted_ids = [c["id"] for c in sorted(self.clusters_, key=lambda c: float(c["arctan"]))]
            label_remap_back = {new_id: old_id for new_id, old_id in enumerate(sorted_ids)}
            label_remap_back[-1] = -1
            peak_labels = np.array([label_remap_back.get(l, -1) for l in peak_labels])

        fig = go.Figure()

        GLASBEY = [
            "#0000FF", "#FF0000", "#00FF00", "#FF00B6",
            "#005300", "#FFD300", "#009FFF", "#9A4D42", "#00FFBE",
            "#783FC1", "#1F9698", "#FFACFD", "#B1CC71", "#F1085C",
            "#FE8F42", "#DD00FF", "#201A01", "#720055", "#766C95",
            "#02AD24", "#C8FF00", "#886C00", "#FFB79F", "#858567",
            "#A10300", "#14F9FF", "#00479E", "#DC5E93", "#93D4FF",
        ]

        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_range = np.array([x_min - x_padding, x_max + x_padding])

        unique_labels = np.unique(display_labels)
        cluster_colors = {label: GLASBEY[i % len(GLASBEY)] for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            mask = display_labels == label
            cluster_points = X[mask]
            cluster_freqs = freqs[mask] if freqs is not None else None

            mask_peak = peak_labels == label
            cluster_peak_points = X_peak[mask_peak]
            cluster_peak_freqs = peak_freqs[mask_peak] if peak_freqs is not None else None

            if label == -1:
                hover_text = (
                    [
                        f"Unassigned<br>Freq: {f:.4f}<br>X: {x:.5f}<br>Y: {y:.5f}"
                        for (x, y), f in zip(cluster_points, cluster_freqs)
                    ]
                    if cluster_freqs is not None
                    else [
                        f"Unassigned<br>X: {x:.5f}<br>Y: {y:.5f}"
                        for x, y in cluster_points
                    ]
                )
                fig.add_trace(go.Scattergl(
                    x=cluster_points[:, 0], y=cluster_points[:, 1],
                    mode="markers", name="Unassigned",
                    marker=dict(size=7, color="lightgray", opacity=0.5,
                                line=dict(width=0.6, color="black")),
                    text=hover_text, hoverinfo="text",
                ))
            else:
                orig_id = next(
                    k for k, v in cluster_id_to_display.items() if v == label and k != -1
                )
                cluster_info = next(c for c in self.clusters_ if c["id"] == orig_id)
                slope = cluster_info["slope"]
                intercept = cluster_info["intercept"]
                arctan = cluster_info["arctan"]
                equation = (
                    f"x = {intercept:.2f}"
                    if np.isinf(slope)
                    else f"y = {slope:.3f}x + {intercept:.2f}"
                )
                color = cluster_colors[label]

                hover_text = (
                    [
                        f"Cluster {label}<br>Freq: {f:.4f}<br>"
                        f"X: {x:.5f}<br>Y: {y:.5f}<br>{equation}<br>arctan: {arctan:.5f}"
                        for (x, y), f in zip(cluster_points, cluster_freqs)
                    ]
                    if cluster_freqs is not None
                    else [
                        f"Cluster {label}<br>X: {x:.5f}<br>Y: {y:.5f}<br>"
                        f"{equation}<br>arctan: {arctan:.5f}"
                        for x, y in cluster_points
                    ]
                )
                fig.add_trace(go.Scattergl(
                    x=cluster_points[:, 0], y=cluster_points[:, 1],
                    mode="markers", name=f"Cluster {label}",
                    marker=dict(size=7, color=color, opacity=0.6,
                                line=dict(width=0.0, color="black")),
                    text=hover_text, hoverinfo="text",
                ))

                peak_hover_text = (
                    [
                        f"Cluster {label}<br>Signal Maximum<br>Freq: {f:.4f}"
                        for (x, y), f in zip(cluster_peak_points, cluster_peak_freqs)
                    ]
                    if cluster_peak_freqs is not None
                    else [
                        f"Cluster {label}<br>Signal Maximum"
                        for x, y in cluster_peak_points
                    ]
                )
                fig.add_trace(go.Scattergl(
                    x=cluster_peak_points[:, 0], y=cluster_peak_points[:, 1],
                    mode="markers", name=f"Cluster {label}",
                    marker=dict(size=7, color=color, opacity=1.0,
                                line=dict(width=0.8, color="black")),
                    text=peak_hover_text, hoverinfo="text",
                ))

        for cluster in self.clusters_:
            slope, intercept = cluster["slope"], cluster["intercept"]
            display_label = cluster_id_to_display[cluster["id"]]
            color = cluster_colors[display_label]

            if np.isinf(slope):
                fig.add_trace(go.Scattergl(
                    x=[intercept, intercept],
                    y=[y_min - y_padding, y_max + y_padding],
                    mode="lines", name=f"Line {display_label}",
                    line=dict(color=color, width=2.5, dash="dash"),
                    hoverinfo="skip", showlegend=False,
                ))
            else:
                y_line = slope * x_range + intercept
                fig.add_trace(go.Scattergl(
                    x=x_range, y=y_line,
                    mode="lines", name=f"Line {display_label}",
                    opacity=0.4,
                    line=dict(color=color, width=2.5, dash="dash"),
                    hovertemplate=(
                        f"<b>Cluster {display_label}</b><br>"
                        f"y = {slope:.3f}x + {intercept:.2f}<br><extra></extra>"
                    ),
                    showlegend=False,
                ))

        axis_common = dict(
            showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.15)",
            zeroline=False, ticks="outside", ticklen=8, tickwidth=2, linewidth=2,
        )

        fig.update_layout(
            title={"text": "KMeans Angular model", "x": 0.5, "xanchor": "center",
                   "font": {"size": 22}},
            xaxis_title=f"Intensity {axis[0]} (µV)  ",
            yaxis_title=f"Intensity {axis[1]} (µV)",
            width=width, height=height,
            hovermode="closest",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Times New Roman", size=20, color="black"),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray", borderwidth=1,
            ),
        )

        if lims is not None:
            fig.update_xaxes(**axis_common, range=lims[0])
            fig.update_yaxes(**axis_common, range=lims[1])
            fig.update_layout(showlegend=False)
        else:
            fig.update_xaxes(**axis_common, range=[x_min - x_padding, x_max + x_padding])
            fig.update_yaxes(**axis_common, range=[y_min - y_padding, y_max + y_padding])
            fig.update_layout(showlegend=False)

        if save_html:
            fig.write_html(f"{model_path}.html", include_plotlyjs="cdn", full_html=True, auto_open=False)
        if save_pdf:
            fig.write_image(f"{model_path}.pdf", format="pdf", width=width, height=height, scale=3)
        if show_fig:
            fig.show()
        if zoom_lims is not None:
            fig.update_xaxes(range=zoom_lims[0])
            fig.update_yaxes(range=zoom_lims[1])
            fig.write_image(f"{model_path}_zoom.pdf", format="pdf", width=width, height=height, scale=3)

        return fig

    # ------------------------------------------------------------------
    # get_cluster_info  (identical API to LinearClusterer)
    # ------------------------------------------------------------------

    def get_cluster_info(self):
        info = []
        arctan_list, mad_sum = [], 0.0
        for cluster in self.clusters_:
            info.append({
                "Cluster ID": cluster["id"],
                "N Points": cluster["n_points"],
                "Slope": f"{cluster['slope']:.4f}" if not np.isinf(cluster["slope"]) else "Vertical",
                "Intercept": f"{cluster['intercept']:.7f}",
                "mad": f"{cluster['mad']:.4f}",
                "arctan": f"{cluster['arctan']:.4f}",
            })
            mad_sum += cluster["mad"]
            arctan_list.append(float(cluster["arctan"]))
        info.append({
            "mad avg": mad_sum / len(self.clusters_),
            "Distances (arctan)": sorted(arctan_list),
        })
        return info

    # ------------------------------------------------------------------
    # interactive_distance_histogram  (identical API)
    # ------------------------------------------------------------------

    def interactive_distance_histogram(
        self, cluster_id, bins=50, xlabel="Distance to cluster ray",
        ylabel="Counts", title=None, xlims=None, ylims=None, histnorm=None,
        save_pdf=False, save_html=False, width=700, height=300,
        output="histogram", show_fig=True,
    ):
        cluster = self.clusters_[cluster_id]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=cluster["point_distance"], nbinsx=bins, histnorm=histnorm,
            marker=dict(color="royalblue", line=dict(color="black", width=0.8)),
            opacity=0.85,
        ))
        if xlims is not None:
            fig.update_xaxes(range=list(xlims))
        if ylims is not None:
            fig.update_yaxes(range=list(ylims))
        fig.update_layout(
            width=width, height=height,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Times New Roman", size=14, color="black"),
            title=None,
            xaxis=dict(
                title=dict(text="Angular distance to ray (rad)", font=dict(size=14)),
                showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1,
                ticks="outside", tickwidth=1, ticklen=5,
                showline=True, linewidth=1, linecolor="black",
                tickfont=dict(size=12), zeroline=True,
                zerolinecolor="rgba(0,0,0,0.3)", zerolinewidth=1,
            ),
            yaxis=dict(
                title=dict(text="Counts", font=dict(size=14)),
                showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1,
                ticks="outside", tickwidth=1, ticklen=5,
                showline=True, linewidth=1, linecolor="black",
                tickfont=dict(size=12),
            ),
            margin=dict(l=60, r=20, t=20, b=55),
            bargap=0.05,
        )
        if save_pdf:
            fig.write_image(f"histograms/{output}_cluster{cluster_id}.pdf", format="pdf", scale=3)
        if save_html:
            fig.write_html(f"histograms/{output}_cluster{cluster_id}.html", include_plotlyjs="cdn")
        if show_fig:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # plot_elbow  — unique to KMeans: inspect the inertia curve
    # ------------------------------------------------------------------

    def plot_elbow(self, width=700, height=350, show_fig=True, save_pdf=False,
                   save_html=False, model_path=None):
        """
        Plot the K-Means inertia curve (elbow plot) produced during fit.
        Only available when n_clusters=None (elbow mode).
        """
        if not self.inertias_:
            raise RuntimeError("No inertia data. Run fit() with n_clusters=None first.")

        k_values = list(range(1, len(self.inertias_) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=self.inertias_,
            mode="lines+markers",
            line=dict(color="royalblue", width=2),
            marker=dict(size=6, color="royalblue"),
            name="Inertia",
        ))
        if self.n_clusters_selected_ is not None:
            fig.add_vline(
                x=self.n_clusters_selected_,
                line_dash="dash", line_color="red",
                annotation_text=f"k = {self.n_clusters_selected_}",
                annotation_position="top right",
            )
        fig.update_layout(
            width=width, height=height,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Times New Roman", size=14, color="black"),
            xaxis=dict(title="Number of clusters (k)", tickmode="linear",
                       showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
            yaxis=dict(title="Inertia", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
            margin=dict(l=60, r=20, t=30, b=55),
        )
        if save_pdf and model_path:
            fig.write_image(f"{model_path}_elbow.pdf", format="pdf", scale=3)
        if save_html and model_path:
            fig.write_html(f"{model_path}_elbow.html", include_plotlyjs="cdn")
        if show_fig:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # write_df_output  (identical API to LinearClusterer)
    # ------------------------------------------------------------------

    def write_df_output(
        self, df_input: pl.DataFrame, cols_to_fit,
        create_file=False, selected_cols=None,
        sort_by_arctan=False, model_name=None, model_path=None,
    ):
        if sort_by_arctan:
            sorted_clusters = sorted(self.clusters_, key=lambda c: float(c["arctan"]))
        else:
            sorted_clusters = sorted(self.clusters_, key=lambda c: float(c["id"]))
        sorted_ids = [c["id"] for c in sorted_clusters]

        mapping_dict = {}
        for cluster in self.clusters_:
            cluster_index = sorted_ids.index(cluster["id"])
            for x_val, y_val in cluster["points"]:
                key = (round(x_val, 8), round(y_val, 8))
                mapping_dict[key] = {
                    "cluster": cluster_index,
                    "arctan2": np.arctan2(y_val, x_val),
                }
        for x_val_un, y_val_un in self.unassigned_vals:
            key = (round(x_val_un, 8), round(y_val_un, 8))
            mapping_dict[key] = {"cluster": -1, "arctan2": np.arctan2(y_val_un, x_val_un)}

        rows = [
            {"x_key": k[0], "y_key": k[1], "cluster": v["cluster"], "arctan2": v["arctan2"]}
            for k, v in mapping_dict.items()
        ]
        df_mapping = pl.DataFrame(rows)

        df_input = df_input.with_columns([
            pl.col(cols_to_fit[0]).round(8).alias("x_key"),
            pl.col(cols_to_fit[1]).round(8).alias("y_key"),
        ])

        df_output = df_input.join(df_mapping, on=["x_key", "y_key"], how="left")
        df_output = df_output.with_columns([
            pl.col("x_key").cast(pl.Utf8),
            pl.col("y_key").cast(pl.Utf8),
            pl.when(pl.col("arctan2").is_not_null())
              .then(pl.col("arctan2").round(8).cast(pl.Utf8))
              .otherwise(pl.lit(None))
              .alias("arctan2"),
        ])

        if create_file and selected_cols and model_name and model_path:
            (
                df_output.select(selected_cols)
                .filter(pl.col("cluster").is_not_null())
                .sort("freq")
                .write_csv(model_path, float_precision=8)
            )

        return df_output

    # ------------------------------------------------------------------
    # write_cluster_dat  (identical API to LinearClusterer)
    # ------------------------------------------------------------------

    def write_cluster_dat(self, filename, cluster_id):
        cluster = self.clusters_[cluster_id]
        with open(filename, "w") as f:
            f.write("-------- Hyperparameters ---------\n")
            f.write(f"n_clusters {self.n_clusters}\n")
            f.write(f"n_clusters_selected {self.n_clusters_selected_}\n")
            f.write(f"max_clusters {self.max_clusters}\n")
            f.write(f"outlier_threshold {self.outlier_threshold}\n")
            f.write(f"min_samples {self.min_samples}\n")
            f.write(f"n_init {self.n_init}\n")
            f.write(f"max_iter {self.max_iter}\n")
            f.write("-------- Variables ---------\n")
            f.write(f"population ranking {int(cluster['id'])}\n")
            f.write(f"slope {float(cluster['slope'])}\n")
            f.write(f"intercept {float(cluster['intercept'])}\n")
            f.write(f"mad {float(cluster['mad'])}\n")
            f.write(f"arctan {float(cluster['arctan'])}\n")
            f.write(f"n_points {int(cluster['n_points'])}\n\n")
            f.write("x y\n")
            for x, y in cluster["points"]:
                f.write(f"{x:.8e} {y:.8e}\n")