import numpy as np
import polars as pl
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

try:
    import hdbscan as hdbscan_lib
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level metric function (must be importable by sklearn's parallel jobs)
# ---------------------------------------------------------------------------

def _angular_distance_metric(u, v):
    """
    Unsigned angular difference between two 2-D direction vectors (radians).
    Range: [0, π].  Used as a sklearn-compatible callable metric so that
    DBSCAN never allocates an N×N distance matrix.
    """
    angle_u = np.arctan2(u[1], u[0])
    angle_v = np.arctan2(v[1], v[0])
    diff = abs(angle_u - angle_v)
    if diff > np.pi:
        diff = 2.0 * np.pi - diff
    return diff


class DBSCANAngular:
    """
    Clusters data points into angular/directional patterns using DBSCAN
    (or optionally HDBSCAN) with a custom angular distance metric.

    Memory-safe version: the N×N precomputed distance matrix has been
    replaced by a callable metric passed directly to sklearn/hdbscan, so
    memory usage is O(N) rather than O(N²).

    Mirrors the interface of LinearClusterer: same clusters_ dict structure,
    same public methods, same return values.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        eps=0.1,
        min_samples=5,
        max_clusters=15,
        force_origin=False,
        random_state=None,
        use_hdbscan=False,
        hdbscan_min_cluster_size=None,
        hdbscan_cluster_selection_epsilon=0.0,
        hdbscan_cluster_selection_method="eom",
    ):
        """
        Parameters
        ----------
        eps : float
            Maximum angular distance (radians) between neighbours in DBSCAN.
            Typical values: 0.01–0.05 tight, 0.05–0.2 moderate, 0.2–0.5 loose.
            Ignored when use_hdbscan=True.
        min_samples : int
            Minimum number of points to form a core point in DBSCAN.
            Also used as hdbscan min_samples when use_hdbscan=True.
        max_clusters : int
            Hard cap on the number of clusters retained (largest N by population).
        force_origin : bool
            If True, fitted lines are constrained to pass through the origin
            (intercept = 0).
        random_state : int or None
            Unused (kept for API parity with LinearClusterer).
        use_hdbscan : bool
            If True, use HDBSCAN instead of DBSCAN. Requires the `hdbscan`
            package to be installed (`pip install hdbscan`).
        hdbscan_min_cluster_size : int or None
            Minimum cluster size for HDBSCAN. Defaults to min_samples when None.
        hdbscan_cluster_selection_epsilon : float
            A distance threshold below which clusters will not be split further
            in HDBSCAN (analogous to eps in DBSCAN). Default 0.0.
        hdbscan_cluster_selection_method : str
            Cluster selection method for HDBSCAN: 'eom' (excess of mass,
            default) or 'leaf'.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.max_clusters = max_clusters
        self.force_origin = force_origin
        self.random_state = random_state
        self.use_hdbscan = use_hdbscan
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_cluster_selection_epsilon = hdbscan_cluster_selection_epsilon
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method

        # Outputs (populated by fit)
        self.clusters_ = []
        self.labels_ = None
        self.unassigned = np.array([], dtype=int)
        self.unassigned_vals = []

    # ------------------------------------------------------------------
    # Distance helpers  (match LinearClusterer signatures exactly)
    # ------------------------------------------------------------------

    def angular_distance(self, points, slope):
        """
        Unsigned angular distance between point vectors and a line direction.
        Returns values in [0, π/2].
        """
        if np.isinf(slope):
            v = np.array([0.0, 1.0])
        else:
            v = np.array([1.0, slope])
        v /= np.linalg.norm(v)

        norms = np.linalg.norm(points, axis=1)
        valid = norms > 0
        cos_theta = np.zeros(len(points))
        cos_theta[valid] = np.abs((points[valid] @ v) / norms[valid])
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    def angular_distance_histogram(self, points, slope):
        """
        Signed angular distance between point vectors and a line direction.
        Range: [−π/2, π/2].
        """
        if np.isinf(slope):
            v = np.array([0.0, 1.0])
        else:
            v = np.array([1.0, slope])
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
        """
        Perpendicular (orthogonal) distance from points to y = slope·x + intercept.
        """
        a = -slope
        b = 1.0
        c = -intercept
        return np.abs(a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(
            a ** 2 + b ** 2
        )

    # ------------------------------------------------------------------
    # DBSCAN / HDBSCAN backends  — NO precomputed N×N matrix
    # ------------------------------------------------------------------

    def _fit_dbscan(self, X):
        """
        Run DBSCAN using the module-level _angular_distance_metric callable.

        Sklearn's ball_tree algorithm supports arbitrary metrics and queries
        the distance function lazily (O(N log N) lookups, O(N) peak memory)
        instead of materialising the full O(N²) distance matrix.
        """
        db = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=_angular_distance_metric,   # callable — no matrix allocated
            algorithm="ball_tree",             # required for custom callables
            n_jobs=-1,                         # parallelise distance queries
        )
        return db.fit_predict(X)

    def _fit_hdbscan(self, X):
        """
        Run HDBSCAN using the module-level _angular_distance_metric callable.

        Passing a callable metric to hdbscan avoids materialising the N×N
        precomputed distance matrix, keeping peak memory O(N).

        Requires the ``hdbscan`` package::

            pip install hdbscan

        Parameters
        ----------
        X : ndarray, shape (n_samples, 2)

        Returns
        -------
        raw_labels : ndarray, shape (n_samples,)
        """
        if not _HDBSCAN_AVAILABLE:
            raise ImportError(
                "The 'hdbscan' package is required for HDBSCAN mode. "
                "Install it with:  pip install hdbscan"
            )

        min_cluster_size = (
            self.hdbscan_min_cluster_size
            if self.hdbscan_min_cluster_size is not None
            else self.min_samples
        )

        # Pass raw X + callable metric — hdbscan computes distances on-the-fly
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
            cluster_selection_method=self.hdbscan_cluster_selection_method,
            metric=_angular_distance_metric,   # callable — no matrix allocated
        )
        return clusterer.fit_predict(X)        # pass raw X, not a dist matrix

    # ------------------------------------------------------------------
    # Line fitting
    # ------------------------------------------------------------------

    def fit_line(self, X):
        """
        Fit a line to 2-D points using least squares.
        Returns (slope, intercept).
        """
        if len(X) < 2:
            return None, None

        x, y = X[:, 0], X[:, 1]

        if self.force_origin:
            denom = np.sum(x ** 2)
            if denom < 1e-12:
                return np.inf, 0.0
            return np.sum(x * y) / denom, 0.0

        n = len(x)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x ** 2).sum()
        denom = n * sum_x2 - sum_x ** 2

        if abs(denom) < 1e-10:
            return np.inf, float(np.mean(x))

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    # ------------------------------------------------------------------
    # Shared post-processing
    # ------------------------------------------------------------------

    def _build_clusters_from_raw_labels(self, X, raw_labels):
        """
        Shared post-processing used by both DBSCAN and HDBSCAN.

        Keeps only the ``max_clusters`` largest clusters, remaps their IDs
        to consecutive 0-based integers, fits a regression line to each,
        and populates ``labels_``, ``clusters_``, ``unassigned``, and
        ``unassigned_vals``.

        Parameters
        ----------
        X : ndarray, shape (n_samples, 2)
        raw_labels : ndarray, shape (n_samples,)
            Integer labels where -1 denotes noise.
        """
        n = len(X)
        self.labels_ = -np.ones(n, dtype=int)
        self.clusters_ = []

        # Keep only the largest max_clusters clusters
        unique_raw = [l for l in np.unique(raw_labels) if l != -1]
        counts = {l: int(np.sum(raw_labels == l)) for l in unique_raw}
        top_labels = sorted(counts, key=lambda l: counts[l], reverse=True)[
            : self.max_clusters
        ]

        # Remap to consecutive 0-based IDs (largest cluster → 0)
        remap = {old: new for new, old in enumerate(top_labels)}
        for i, rl in enumerate(raw_labels):
            self.labels_[i] = remap.get(rl, -1)

        # Build clusters_ dicts (same schema as LinearClusterer)
        threshold_val = (
            self.hdbscan_cluster_selection_epsilon
            if self.use_hdbscan
            else self.eps
        )
        for new_id in range(len(top_labels)):
            mask = self.labels_ == new_id
            pts = X[mask]

            slope, intercept = self.fit_line(pts)
            if slope is None:
                slope, intercept = 0.0, 0.0

            arctan = float(np.arctan(slope)) if not np.isinf(slope) else np.pi / 2
            angles = self.angular_distance_histogram(pts, slope)
            mad = float(np.mean(np.abs(angles)))

            self.clusters_.append(
                {
                    "id": new_id,
                    "slope": slope,
                    "intercept": intercept,
                    "arctan": arctan,
                    "current_angle_threshold": threshold_val,
                    "mad": mad,
                    "points": pts,
                    "n_points": int(mask.sum()),
                    "point_distance": angles,
                    "residuals": angles ** 2,
                }
            )

        unassigned_indices = np.where(self.labels_ == -1)[0]
        self.unassigned = unassigned_indices
        self.unassigned_vals = [X[i] for i in unassigned_indices]

    # ------------------------------------------------------------------
    # Core fit
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit the angular clusterer to data using DBSCAN or HDBSCAN.

        When ``use_hdbscan=True`` the HDBSCAN backend is used; otherwise
        classic DBSCAN with the angular callable metric (no N×N matrix).

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)

        Returns
        -------
        self
        """
        X = np.asarray(X)
        if X.shape[1] != 2:
            raise ValueError("Input data must be 2-D (x, y coordinates).")

        if self.use_hdbscan:
            raw_labels = self._fit_hdbscan(X)
        else:
            raw_labels = self._fit_dbscan(X)

        self._build_clusters_from_raw_labels(X, raw_labels)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Assign new points to existing clusters by closest angular proximity.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
        """
        X = np.asarray(X)
        labels = -np.ones(len(X), dtype=int)

        if not self.clusters_:
            return labels

        # Precompute unit direction vectors for each cluster line
        directions = []
        for c in self.clusters_:
            slope = c["slope"]
            v = np.array([0.0, 1.0]) if np.isinf(slope) else np.array([1.0, slope])
            directions.append(v / np.linalg.norm(v))

        for i, p in enumerate(X):
            norm_p = np.linalg.norm(p)
            if norm_p < 1e-12:
                labels[i] = self.clusters_[0]["id"]
                continue
            p_hat = p / norm_p
            best_angle = np.inf
            best_id = -1
            for c, v in zip(self.clusters_, directions):
                angle = np.arccos(np.clip(np.abs(np.dot(p_hat, v)), 0.0, 1.0))
                if angle < best_angle:
                    best_angle = angle
                    best_id = c["id"]
            labels[i] = best_id

        return labels

    # ------------------------------------------------------------------
    # Reassign
    # ------------------------------------------------------------------

    def reassign_by_angular_proximity(
        self,
        X,
        include_unassigned=True,
        distance_mode="angular",
    ):
        """
        Reassign all (or only assigned) points to their closest cluster line.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
        include_unassigned : bool
        distance_mode : {'angular', 'orthogonal'}

        Returns
        -------
        self
        """
        X = np.asarray(X)

        if not self.clusters_:
            raise RuntimeError("No clusters found. Run fit() first.")
        if distance_mode not in ("angular", "orthogonal"):
            raise ValueError("distance_mode must be 'angular' or 'orthogonal'.")

        # Precompute cluster direction vectors
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
                best_id = min(
                    self.clusters_,
                    key=lambda c, v=None: np.arccos(
                        np.clip(
                            np.abs(np.dot(p_hat, cluster_directions[self.clusters_.index(c)])),
                            0.0,
                            1.0,
                        )
                    ),
                )["id"]
                new_labels[idx] = best_id
        else:  # orthogonal
            pts = X[candidate_indices]
            dist_matrix = np.full((len(pts), len(self.clusters_)), np.inf)
            for j, c in enumerate(self.clusters_):
                slope, intercept = c["slope"], c["intercept"]
                if np.isinf(slope):
                    dist_matrix[:, j] = np.abs(pts[:, 0] - intercept)
                else:
                    dist_matrix[:, j] = self.point_to_line_distance(
                        pts, slope, intercept
                    )
            best_j = np.argmin(dist_matrix, axis=1)
            for k, idx in enumerate(candidate_indices):
                new_labels[idx] = self.clusters_[best_j[k]]["id"]

        self.labels_ = new_labels

        # Rebuild per-cluster stats
        for c, v in zip(self.clusters_, cluster_directions):
            cid = c["id"]
            mask = self.labels_ == cid
            pts = X[mask]
            c["points"] = pts
            c["n_points"] = int(mask.sum())

            if len(pts) == 0:
                c["point_distance"] = np.array([])
                c["residuals"] = np.array([])
                c["mad"] = 0.0
                continue

            slope, intercept = c["slope"], c["intercept"]
            angles = self.angular_distance_histogram(pts, slope)
            c["point_distance"] = angles
            c["residuals"] = angles ** 2
            c["mad"] = float(np.mean(np.abs(angles)))

        unassigned_indices = np.where(self.labels_ == -1)[0]
        self.unassigned = unassigned_indices
        self.unassigned_vals = [X[i] for i in unassigned_indices]

        return self

    # ------------------------------------------------------------------
    # Iterative refinement  (identical logic to LinearClusterer)
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
        Alternates reassignment and line refitting until convergence.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
        max_iterations : int
        convergence_threshold : float
            Stop when max |ΔN| across clusters ≤ this value.
        distance_mode : {'angular', 'orthogonal'}
        verbose : bool

        Returns
        -------
        self
        """
        X = np.asarray(X)

        if not self.clusters_:
            raise RuntimeError("No clusters found. Run fit() first.")

        if verbose:
            print(
                f"{'Iter':>5}  {'Max ΔN':>9}  {'Converged':>10}  Populations"
            )
            print("-" * 72)

        for iteration in range(1, max_iterations + 1):
            prev_populations = {c["id"]: c["n_points"] for c in self.clusters_}

            # Step 1: reassign
            self.reassign_by_angular_proximity(
                X, include_unassigned=True, distance_mode=distance_mode
            )

            # Step 2: refit lines
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
                    float(np.arctan(new_slope))
                    if not np.isinf(new_slope)
                    else np.pi / 2
                )

                angles = self.angular_distance_histogram(pts, new_slope)
                c["point_distance"] = angles
                c["residuals"] = angles ** 2
                c["mad"] = float(np.mean(np.abs(angles)))

            # Step 3: convergence check
            new_populations = {c["id"]: c["n_points"] for c in self.clusters_}
            max_delta = max(
                abs(new_populations[cid] - prev_populations[cid])
                for cid in new_populations
            )

            if verbose:
                pop_str = "  ".join(
                    f"C{cid}:{new_populations[cid]}" for cid in sorted(new_populations)
                )
                print(
                    f"{iteration:>5}  {max_delta:>9}  "
                    f"{'YES' if max_delta <= convergence_threshold else 'no':>10}  {pop_str}"
                )

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
                    f"\nReached max_iterations={max_iterations} without "
                    f"convergence (last max ΔN = {max_delta})."
                )

        self.n_iterations_ = iteration
        return self

    # ------------------------------------------------------------------
    # Interactive plot  (mirrors LinearClusterer.plot_interactive)
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
        """
        Interactive Plotly scatter + fitted-line visualisation.
        Signature and behaviour identical to LinearClusterer.plot_interactive.
        """
        X = np.asarray(X) * 1000

        if cols is not None:
            x_peak = peaks[cols[0]].to_numpy() * 1000
            y_peak = peaks[cols[1]].to_numpy() * 1000
        else:
            x_peak = peaks[:, 1].to_numpy() * 1000
            y_peak = peaks[:, 2].to_numpy() * 1000

        X_peak = np.column_stack((x_peak, y_peak))
        peak_labels = peaks["cluster"].to_numpy()
        peak_freqs = peaks["freq"].to_numpy() if "freq" in peaks.columns else None

        # Optional arctan sorting
        if sort_by_arctan and self.clusters_:
            sorted_ids = [
                c["id"] for c in sorted(self.clusters_, key=lambda c: float(c["arctan"]))
            ]
            label_remap = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
            label_remap[-1] = -1
            display_labels = np.array([label_remap.get(l, -1) for l in self.labels_])
            cluster_id_to_display = label_remap
        else:
            display_labels = self.labels_
            cluster_id_to_display = {c["id"]: c["id"] for c in self.clusters_}
            cluster_id_to_display[-1] = -1
            sorted_ids = [
                c["id"] for c in sorted(self.clusters_, key=lambda c: float(c["arctan"]))
            ]
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
        cluster_colors = {
            label: GLASBEY[i % len(GLASBEY)] for i, label in enumerate(unique_labels)
        }

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
                fig.add_trace(
                    go.Scattergl(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode="markers",
                        name="Unassigned",
                        marker=dict(
                            size=7,
                            color="lightgray",
                            opacity=0.5,
                            line=dict(width=0.6, color="black"),
                        ),
                        text=hover_text,
                        hoverinfo="text",
                    )
                )
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
                        f"Cluster {label}<br>Freq: {f:.4f}<br>X: {x:.5f}<br>"
                        f"Y: {y:.5f}<br>{equation}<br>arctan: {arctan:.5f}"
                        for (x, y), f in zip(cluster_points, cluster_freqs)
                    ]
                    if cluster_freqs is not None
                    else [
                        f"Cluster {label}<br>X: {x:.5f}<br>Y: {y:.5f}<br>"
                        f"{equation}<br>arctan: {arctan:.5f}"
                        for x, y in cluster_points
                    ]
                )

                fig.add_trace(
                    go.Scattergl(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode="markers",
                        name=f"Cluster {label}",
                        marker=dict(
                            size=7, color=color, opacity=0.6,
                            line=dict(width=0.0, color="black"),
                        ),
                        text=hover_text,
                        hoverinfo="text",
                    )
                )

                if cluster_peak_points.shape[0] > 0:
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
                    fig.add_trace(
                        go.Scattergl(
                            x=cluster_peak_points[:, 0],
                            y=cluster_peak_points[:, 1],
                            mode="markers",
                            name=f"Cluster {label}",
                            marker=dict(
                                size=7, color=color, opacity=1.0,
                                line=dict(width=0.8, color="black"),
                            ),
                            text=peak_hover_text,
                            hoverinfo="text",
                        )
                    )

        # Regression lines
        for cluster in self.clusters_:
            slope = cluster["slope"]
            intercept = cluster["intercept"]
            display_label = cluster_id_to_display[cluster["id"]]
            color = cluster_colors[display_label]

            if np.isinf(slope):
                fig.add_trace(
                    go.Scattergl(
                        x=[intercept, intercept],
                        y=[y_min - y_padding, y_max + y_padding],
                        mode="lines",
                        name=f"Line {display_label}",
                        line=dict(color=color, width=2.5, dash="dash"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            else:
                y_line = slope * x_range + intercept
                equation = f"y = {slope:.3f}x + {intercept:.2f}"
                fig.add_trace(
                    go.Scattergl(
                        x=x_range,
                        y=y_line,
                        mode="lines",
                        name=f"Line {display_label}",
                        opacity=0.4,
                        line=dict(color=color, width=2.5, dash="dash"),
                        hovertemplate=f"<b>Cluster {display_label}</b><br>{equation}<br><extra></extra>",
                        showlegend=False,
                    )
                )

        ax = axis if axis is not None else ["1", "2"]
        fig.update_layout(
            title={"text": "DBSCAN Angular model", "x": 0.5, "xanchor": "center",
                   "font": {"size": 22}},
            xaxis_title=f"Intensity {ax[0]} (µV)",
            yaxis_title=f"Intensity {ax[1]} (µV)",
            width=width,
            height=height,
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Times New Roman", size=20, color="black"),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray", borderwidth=1,
            ),
        )

        axis_kwargs = dict(
            showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.15)",
            zeroline=False, ticks="outside", ticklen=8, tickwidth=2, linewidth=2,
        )
        if lims is not None:
            fig.update_xaxes(**axis_kwargs, range=lims[0])
            fig.update_yaxes(**axis_kwargs, range=lims[1])
            fig.update_layout(showlegend=False)
        else:
            fig.update_xaxes(**axis_kwargs, range=[x_min - x_padding, x_max + x_padding])
            fig.update_yaxes(**axis_kwargs, range=[y_min - y_padding, y_max + y_padding])
            fig.update_layout(showlegend=False)

        if save_html:
            fig.write_html(f"{model_path}.html", include_plotlyjs="cdn",
                           full_html=True, auto_open=False)
        if save_pdf:
            fig.write_image(f"{model_path}.pdf", format="pdf",
                            width=width, height=height, scale=3)
        if show_fig:
            fig.show()

        if zoom_lims is not None:
            fig.update_xaxes(range=zoom_lims[0])
            fig.update_yaxes(range=zoom_lims[1])
            fig.write_image(f"{model_path}_zoom.pdf", format="pdf",
                            width=width, height=height, scale=3)

        return fig

    # ------------------------------------------------------------------
    # Cluster info
    # ------------------------------------------------------------------

    def get_cluster_info(self):
        """
        Return a list of dicts with per-cluster statistics, plus a summary
        dict — identical schema to LinearClusterer.get_cluster_info.
        """
        info = []
        arctan_list = []
        mad_sum = 0.0

        for cluster in self.clusters_:
            info.append(
                {
                    "Cluster ID": cluster["id"],
                    "N Points": cluster["n_points"],
                    "Slope": (
                        f"{cluster['slope']:.4f}"
                        if not np.isinf(cluster["slope"])
                        else "Vertical"
                    ),
                    "Intercept": f"{cluster['intercept']:.7f}",
                    "mad": f"{cluster['mad']:.4f}",
                    "arctan": f"{cluster['arctan']:.4f}",
                }
            )
            mad_sum += cluster["mad"]
            arctan_list.append(float(cluster["arctan"]))

        info.append(
            {
                "mad avg": float(mad_sum / len(self.clusters_)) if self.clusters_ else 0.0,
                "Distances (arctan)": sorted(arctan_list),
            }
        )
        return info

    # ------------------------------------------------------------------
    # Write cluster .dat file
    # ------------------------------------------------------------------

    def write_cluster_dat(self, filename, cluster_id):
        """
        Write cluster metadata and point list to a .dat file.
        Identical format to LinearClusterer.write_cluster_dat.
        """
        cluster = self.clusters_[cluster_id]

        with open(filename, "w") as f:
            f.write("-------- Hyperparameters ---------\n")
            f.write(f"eps {self.eps}\n")
            f.write(f"Maximum n of clusters {self.max_clusters}\n")
            f.write(f"Min samples {self.min_samples}\n")
            f.write(f"force_origin {self.force_origin}\n")

            f.write("-------- Variables ---------\n")
            f.write(f"population ranking {int(cluster['id'])}\n")
            f.write(f"slope {float(cluster['slope'])}\n")
            f.write(f"intercept {float(cluster['intercept'])}\n")
            f.write(f"mad {float(cluster['mad'])}\n")
            f.write(f"arctan {float(cluster['arctan'])}\n")
            f.write(f"n_points {int(cluster['n_points'])}\n")

            f.write("\n")
            f.write("x y\n")

            for x, y in cluster["points"]:
                f.write(f"{x:.8e} {y:.8e}\n")

    # ------------------------------------------------------------------
    # Distance histogram
    # ------------------------------------------------------------------

    def interactive_distance_histogram(
        self,
        cluster_id,
        bins=50,
        xlabel="Distance to cluster ray",
        ylabel="Counts",
        title=None,
        xlims=None,
        ylims=None,
        histnorm=None,
        save_pdf=False,
        save_html=False,
        width=700,
        height=300,
        output="histogram",
        show_fig=True,
    ):
        """
        Histogram of signed angular distances for a single cluster.
        Identical interface to LinearClusterer.interactive_distance_histogram.
        """
        cluster = self.clusters_[cluster_id]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=cluster["point_distance"],
                nbinsx=bins,
                histnorm=histnorm,
                marker=dict(color="royalblue", line=dict(color="black", width=0.8)),
                opacity=0.85,
            )
        )

        if xlims is not None:
            fig.update_xaxes(range=list(xlims))
        if ylims is not None:
            fig.update_yaxes(range=list(ylims))

        fig.update_layout(
            width=width,
            height=height,
            paper_bgcolor="white",
            plot_bgcolor="white",
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
            fig.write_image(
                f"histograms/{output}_cluster{cluster_id}.pdf", format="pdf", scale=3
            )
        if save_html:
            fig.write_html(
                f"histograms/{output}_cluster{cluster_id}.html", include_plotlyjs="cdn"
            )
        if show_fig:
            fig.show()

        return fig

    # ------------------------------------------------------------------
    # Global histogram
    # ------------------------------------------------------------------

    def global_hist(
        self,
        nbins=80,
        save_html=False,
        save_pdf=False,
        output="global_histogram",
    ):
        """
        Histogram of angular distances across all clusters (skipping first two).
        Identical interface to LinearClusterer.global_hist.
        """
        all_distances = []
        for c in self.clusters_[2:]:
            all_distances.extend(np.asarray(c["point_distance"]).ravel())

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=all_distances,
                nbinsx=nbins,
                histnorm=None,
                marker=dict(color="royalblue", line=dict(color="black", width=1)),
                opacity=0.85,
            )
        )

        if save_pdf:
            fig.write_image(f"histograms/{output}.pdf", format="pdf", scale=3)
        if save_html:
            fig.write_html(f"histograms/{output}.html", include_plotlyjs="cdn")

        fig.show()

    # ------------------------------------------------------------------
    # Write DataFrame output
    # ------------------------------------------------------------------

    def write_df_output(
        self,
        df_input: pl.DataFrame,
        cols_to_fit,
        create_file=False,
        selected_cols=None,
        sort_by_arctan=False,
        model_name=None,
        model_path=None,
    ):
        """
        Join cluster assignments back onto the input Polars DataFrame.
        Identical interface and return value to LinearClusterer.write_df_output.
        """
        # 1. Optional arctan sorting
        if sort_by_arctan:
            sorted_clusters = sorted(self.clusters_, key=lambda c: float(c["arctan"]))
        else:
            sorted_clusters = sorted(self.clusters_, key=lambda c: float(c["id"]))
        sorted_ids = [c["id"] for c in sorted_clusters]

        # 2. Build mapping dict (deduplicated — last write wins)
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
            mapping_dict[key] = {
                "cluster": -1,
                "arctan2": np.arctan2(y_val_un, x_val_un),
            }

        rows = [
            {"x_key": k[0], "y_key": k[1], "cluster": v["cluster"], "arctan2": v["arctan2"]}
            for k, v in mapping_dict.items()
        ]
        df_mapping = pl.DataFrame(rows)

        # 3. Prepare input keys
        df_input = df_input.with_columns(
            [
                pl.col(cols_to_fit[0]).round(8).alias("x_key"),
                pl.col(cols_to_fit[1]).round(8).alias("y_key"),
            ]
        )

        # 4. Join
        df_output = df_input.join(df_mapping, on=["x_key", "y_key"], how="left")

        # 5. Format columns
        df_output = df_output.with_columns(
            [
                pl.col("x_key").cast(pl.Utf8),
                pl.col("y_key").cast(pl.Utf8),
                pl.when(pl.col("arctan2").is_not_null())
                .then(pl.col("arctan2").round(8).cast(pl.Utf8))
                .otherwise(pl.lit(None))
                .alias("arctan2"),
            ]
        )

        # 6. Optional file export
        if (
            create_file
            and selected_cols is not None
            and model_name is not None
            and model_path is not None
        ):
            (
                df_output.select(selected_cols)
                .filter(pl.col("cluster").is_not_null())
                .sort("freq")
                .write_csv(model_path, float_precision=8)
            )

        return df_output