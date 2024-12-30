
from concurrent.futures import ProcessPoolExecutor
import random
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from joblib import Parallel, delayed

class ParallelCHoutsideConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
    """
    PcaConvexHullAnomalyDetector:
    - Applies PCA to reduce data before computing Convex Hull.
    - Iteratively removes points that minimize or maintain hull volume (single best point per iteration).
    - Stops when no improvement in one iteration or max_iter reached.
    - Uses parallelization to speed up candidate checks.
    - Uses precomputed hull equations to speed up predict.

    Changes from original:
    - Added PCA for dimensionality reduction.
    - Parallelized candidate removal checks with joblib.
    - Removed tolerance checks and rely on no improvement + max_iter stopping.
    - Stores hull equations to avoid recomputing hull in predict.
    """

    def __init__(self, lam=1.0, tol=1e-3, max_iter=100, n_components=2, n_jobs=-1):
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.Sp = None
        self.pca = None
        self.hull_equations_ = None

    def fit(self, X):
        # Reduce dimensions with PCA
        self.pca = PCA(n_components=self.n_components)
        X_reduced = self.pca.fit_transform(X)
        S = set(map(tuple, X_reduced))
        Sp = S.copy()
        r = set()

        # If not enough points for initial hull
        if len(Sp) < self.n_components + 1:
            self.Sp = Sp
            self.hull_equations_ = None
            return self

        Sp_array = np.array(list(Sp))
        try:
            Sh = ConvexHull(Sp_array)
            vol_m = Sh.volume
        except Exception:
            self.Sp = Sp
            self.hull_equations_ = None
            return self

        def compute_new_volume(Sp, p_tuple):
            Sp_new = Sp - {p_tuple}
            if len(Sp_new) < self.n_components + 1:
                return p_tuple, float('inf')
            try:
                Sh_new = ConvexHull(np.array(list(Sp_new)))
                vol_n = Sh_new.volume
            except Exception:
                vol_n = float('inf')
            return p_tuple, vol_n,Sh_new

        for _ in range(self.max_iter):
            candidates = Sh.points[Sh.vertices]
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_new_volume)(Sp, tuple(p)) for p in candidates
            )

            best_vol = vol_m
            best_point = None
            best_ch=None
            for p_tuple, vol_n,ch_new in results:
                if vol_n <= best_vol:
                    best_vol = vol_n
                    best_point = p_tuple
                    best_ch=ch_new

            if best_point is not None and best_vol <= vol_m:
                # Remove the best point found this iteration
                Sp = Sp - {best_point}
                r.add(best_point)
                vol_m = best_vol

                if len(Sp) < self.n_components + 1:
                    break
                try:
                    Sh = best_ch #ConvexHull(np.array(list(Sp)))
                except Exception:
                    break
            else:
                # No improvement found, stop
                break

        self.Sp = Sp
        Sp_array = np.array(list(self.Sp))
        if len(Sp_array) >= self.n_components + 1:
            try:
                final_hull = ConvexHull(Sp_array)
                self.hull_equations_ = final_hull.equations
            except Exception:
                self.hull_equations_ = None
        else:
            self.hull_equations_ = None

        return self

    def predict(self, X):
        X_reduced = self.pca.transform(X)
        if self.hull_equations_ is None:
            # No hull, consider all points inside
            return np.ones(X_reduced.shape[0], dtype=int)

        A = self.hull_equations_[:, :-1]
        b = self.hull_equations_[:, -1]

        predictions = []
        for x in X_reduced:
            if np.all(A.dot(x) + b <= 1e-12):
                predictions.append(1)  # Inside
            else:
                predictions.append(-1) # Outside
        return np.array(predictions)



class ParallelConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, lam=1.0, tol=1e-3, max_iter=100, n_components=2, n_jobs=-1):
        """
        Parameters:
        - lam: Regularization parameter for balancing size vs. volume.
        - tol: Tolerance for stopping criteria.
        - max_iter: Maximum iterations.
        - n_components: Number of PCA components for visualization.
        - n_jobs: Number of jobs for parallel processing (-1 for all cores).
        """
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.n_components = n_components
        self.Sp = None  # Optimal subset
        self.pca = None  # PCA object
        self.n_jobs = n_jobs


    def fit(self, X):
        """
        Fit the model by reducing dimensions with PCA and finding the convex hull.
        """
        # Step 1: Reduce dimensions with PCA
        self.pca = PCA(n_components=self.n_components)
        X_reduced = self.pca.fit_transform(X)
        S = set(map(tuple, X_reduced))
        Sp = S.copy()
        r = set()  # Removed points

        for _ in range(self.max_iter):
            Sp_array = np.array(list(Sp))
            if len(Sp_array) < self.n_components + 1:
                # Not enough points to form a hull
                break

            # Compute the current hull
            try:
                Sh = ConvexHull(Sp_array)
                vol_c = Sh.volume
            except Exception:
                # If hull fails to compute, stop
                break

            vol_m = vol_c
            S_new_h = Sp
            r_n = set()

            # Parallel computation function
            def compute_new_volume(Sp, p_tuple):
                """Remove p_tuple from Sp, compute new hull volume."""
                Sp_new = Sp - {p_tuple}
                # Need at least (n_components+1) points to form a hull
                if len(Sp_new) < self.n_components + 1:
                    return p_tuple, np.inf
                try:
                    Sh_new = ConvexHull(np.array(list(Sp_new)))
                    vol_n = Sh_new.volume
                except Exception:
                    vol_n = np.inf
                return p_tuple, vol_n

            # Sub-loop: Remove one point at a time in parallel
            candidates = Sh.points[Sh.vertices]
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_new_volume)(Sp, tuple(p)) for p in candidates
            )

            # Decide which points to remove after all parallel computations
            for p_tuple, vol_n in results:
                if vol_n < vol_m - self.tol:
                    # Update if this candidate leads to a smaller hull volume
                    if vol_n < vol_m:
                        vol_m = vol_n
                        S_new_h = Sp - {p_tuple}
                    r_n.add(p_tuple)

            # Update CH and removed points
            Sp = S_new_h
            r.update(r_n)

            # Stopping criteria
            if len(r_n) == 0:
                break

        self.Sp = Sp
        Sp_array = np.array(list(self.Sp))

        # After determining the final Sp, try to compute the final hull equations.
        if len(Sp_array) >= self.n_components + 1:
            # Compute hull on the final set of points
            final_hull = ConvexHull(Sp_array)
            self.hull_equations_ = final_hull.equations
        else:
            # Not enough points to form a hull
            self.hull_equations_ = None

        return self

    def predict(self, X):
        """
        Predict whether points are inside the fitted convex hull or not.

        Instead of adding each point and attempting to build a hull again,
        we leverage the final hull equations. A point is inside if it satisfies
        all half-space inequalities defined by the hull facets.
        """
        X_reduced = self.pca.transform(X)
        predictions = []

        # If we don't have a valid hull, treat all points as inside
        # or as anomalies depending on preference.
        if self.hull_equations_ is None:
            # If no hull could be formed, perhaps all points are considered normal
            # or consider them all anomalies. Here we choose all normal:
            return np.ones(X_reduced.shape[0], dtype=int)

        A = self.hull_equations_[:, :-1]
        b = self.hull_equations_[:, -1]

        # Check if each point satisfies A*x + b <= 0 for all facets
        # We'll add a small numerical tolerance to account for floating-point errors
        tol = 1e-12

        for x in X_reduced:
            if np.all(A.dot(x) + b <= tol):
                predictions.append(1)  # Inside (normal)
            else:
                predictions.append(-1)  # Outside (anomalous)

        return np.array(predictions)

    def plot_pca_with_hull(self, X, y=None):
        """
        Visualize the PCA-reduced data with the convex hull and anomalies.

        Parameters:
        - X: Original input data.
        - y: Optional ground-truth labels (1 for normal, -1 for anomalies).
        """
        X_reduced = self.pca.transform(X)
        Sp_array = np.array(list(self.Sp))

        if len(Sp_array) < 3:
            # If we don't have enough points for a hull, just plot points
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='gray', label='All Points')
            plt.title("PCA-Reduced Data")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            if y is not None:
                plt.scatter(X_reduced[y == -1][:, 0], X_reduced[y == -1][:, 1],
                            c='red', label='Ground-Truth Anomalies', edgecolor='k')
            plt.legend()
            plt.show()
            return

        # Compute Convex Hull
        hull = ConvexHull(Sp_array)

        plt.figure(figsize=(10, 7))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='gray', label='All Points')
        plt.scatter(Sp_array[:, 0], Sp_array[:, 1], c='green', label='Inside Hull')

        # Highlight hull edges
        for simplex in hull.simplices:
            plt.plot(Sp_array[simplex, 0], Sp_array[simplex, 1], 'r-')

        if y is not None:
            plt.scatter(X_reduced[y == -1][:, 0], X_reduced[y == -1][:, 1],
                        c='red', label='Ground-Truth Anomalies', edgecolor='k')

        plt.title("PCA-Reduced Data with Convex Hull")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.show()


# def fit(self, X):
    #     """
    #     Fit the model by reducing dimensions with PCA and finding the convex hull.
    #     """
    #     # Step 1: Reduce dimensions with PCA
    #     self.pca = PCA(n_components=self.n_components)
    #     X_reduced = self.pca.fit_transform(X)
    #     S = set(map(tuple, X_reduced))
    #     Sp = S.copy()
    #     r = set()  # Removed points
    #
    #     for _ in range(self.max_iter):
    #         Sp_array = np.array(list(Sp))
    #         if len(Sp_array) < self.n_components + 1:
    #             # Not enough points to form a hull
    #             break
    #
    #         # Compute the current hull
    #         try:
    #             Sh = ConvexHull(Sp_array)
    #             vol_c = Sh.volume
    #         except Exception:
    #             # If hull fails to compute, stop
    #             break
    #
    #         vol_m = vol_c
    #         S_new_h = Sp
    #         r_n = set()
    #
    #         # Parallel computation function
    #         def compute_new_volume(Sp, p_tuple):
    #             """Remove p_tuple from Sp, compute new hull volume."""
    #             Sp_new = Sp - {p_tuple}
    #             # Need at least (n_components+1) points to form a hull
    #             if len(Sp_new) < self.n_components + 1:
    #                 return p_tuple, np.inf
    #             try:
    #                 Sh_new = ConvexHull(np.array(list(Sp_new)))
    #                 vol_n = Sh_new.volume
    #             except Exception:
    #                 vol_n = np.inf
    #             return p_tuple, vol_n
    #
    #         # Sub-loop: Remove one point at a time in parallel
    #         candidates = Sh.points[Sh.vertices]
    #         results = Parallel(n_jobs=self.n_jobs)(
    #             delayed(compute_new_volume)(Sp, tuple(p)) for p in candidates
    #         )
    #
    #         # Decide which points to remove after all parallel computations
    #         for p_tuple, vol_n in results:
    #             if vol_n < vol_m - self.tol:
    #                 # Update if this candidate leads to a smaller hull volume
    #                 if vol_n < vol_m:
    #                     vol_m = vol_n
    #                     S_new_h = Sp - {p_tuple}
    #                 r_n.add(p_tuple)
    #
    #         # Update CH and removed points
    #         Sp = S_new_h
    #         r.update(r_n)
    #
    #         # Stopping criteria
    #         if len(r_n) == 0:
    #             break
    #
    #     self.Sp = Sp
    #     return self




class PcaConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, lam=1.0, tol=1e-3, max_iter=100, n_components=2):
        """
        Parameters:
        - lam: Regularization parameter for balancing size vs. volume.
        - tol: Tolerance for stopping criteria.
        - max_iter: Maximum iterations.
        - n_components: Number of PCA components for visualization.
        """
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.n_components = n_components
        self.Sp = None  # Optimal subset
        self.pca = None  # PCA object

    def fit(self, X):
        """
        Fit the model by reducing dimensions with PCA and finding the convex hull.
        """
        # Step 1: Reduce dimensions with PCA
        self.pca = PCA(n_components=self.n_components)
        X_reduced = self.pca.fit_transform(X)
        S = set(map(tuple, X_reduced))
        Sp = S.copy()
        r = set()  # Removed points

        # Step 2: Main loop - Compute CH and remove anomalies
        for _ in range(self.max_iter):
            Sp_array = np.array(list(Sp))
            try:
                Sh = ConvexHull(Sp_array)
                vol_c = Sh.volume
            except Exception:
                break  # Stop if ConvexHull computation fails

            vol_m = vol_c
            S_new_h = Sp
            r_n = set()

            # Sub-loop: Remove one point at a time
            for p in Sh.points[Sh.vertices]:  # Only hull vertices
                p_tuple = tuple(p)
                Sp_new = Sp - {p_tuple}
                try:
                    Sh_new = ConvexHull(np.array(list(Sp_new)))
                    vol_n = Sh_new.volume
                except Exception:
                    vol_n = np.inf

                if vol_n < vol_m - self.tol:
                    S_new_h = Sp_new
                    vol_m = vol_n
                    r_n.add(p_tuple)

            # Update CH and removed points
            Sp = S_new_h
            r.update(r_n)

            # Stopping criteria
            if len(r_n) == 0:
                break

        self.Sp = Sp
        return self

    def predict(self, X):
        """
        Predict whether points are inside the fitted convex hull or not.
        """
        X_reduced = self.pca.transform(X)
        Sp_array = np.array(list(self.Sp))

        predictions = []
        for x in X_reduced:
            try:
                ConvexHull(np.vstack([Sp_array, x]))
                predictions.append(1)  # Inside (normal)
            except Exception:
                predictions.append(-1)  # Outside (anomalous)
        return np.array(predictions)

    def plot_pca_with_hull(self, X, y=None):
        """
        Visualize the PCA-reduced data with the convex hull and anomalies.

        Parameters:
        - X: Original input data.
        - y: Optional ground-truth labels (1 for normal, -1 for anomalies).
        """
        X_reduced = self.pca.transform(X)
        Sp_array = np.array(list(self.Sp))

        # Compute Convex Hull
        hull = ConvexHull(Sp_array)

        plt.figure(figsize=(10, 7))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='gray', label='All Points')
        plt.scatter(Sp_array[:, 0], Sp_array[:, 1], c='green', label='Inside Hull')

        # Highlight points on the hull
        for simplex in hull.simplices:
            plt.plot(Sp_array[simplex, 0], Sp_array[simplex, 1], 'r-')

        if y is not None:
            plt.scatter(X_reduced[y == -1][:, 0], X_reduced[y == -1][:, 1],
                        c='red', label='Ground-Truth Anomalies', edgecolor='k')

        plt.title("PCA-Reduced Data with Convex Hull")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.show()

    from sklearn.metrics import accuracy_score, classification_report

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    def compute_fit_accuracy(self, X, y):
        """
        Compute accuracy of the fit using ground-truth labels and analyze anomalies.

        Parameters:
        - X: Original input data.
        - y: Ground-truth labels (0 for normal, 1 for anomalies).

        Returns:
        - Accuracy score.
        """
        y_pred = self.predict(X)

        # Map predicted labels to match ground-truth format: 0 (normal), 1 (anomalies)
        y_pred = np.where(y_pred == -1, 1, 0)  # -1 (anomaly) -> 1, 1 (normal) -> 0

        # Compute accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f"Fit Accuracy: {accuracy:.4f}")

        # Detailed classification report
        print("Classification Report:")
        print(classification_report(y, y_pred, target_names=["Normal (0)", "Anomaly (1)"]))

        # Analyze anomalies
        total_anomalies = np.sum(y == 1)
        detected_anomalies = np.sum(y_pred == 1)
        true_positives = np.sum((y == 1) & (y_pred == 1))  # Correctly identified anomalies

        print(f"\nAnomaly Analysis:")
        print(f"Total anomalies in ground-truth: {total_anomalies}")
        print(f"Anomalies detected by the model: {detected_anomalies}")
        print(f"True positives (correctly identified anomalies): {true_positives}")
        print(f"Missed anomalies: {total_anomalies - true_positives}")
        print(f"False positives (normal misclassified as anomalies): {detected_anomalies - true_positives}")

        return accuracy


class ModifiedConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, threshold=1e-3, tol=1e-3, max_iter=100):
        """
        Parameters:
        - threshold: float, threshold for volume reduction.
        - tol: float, tolerance for stopping criteria.
        - max_iter: int, maximum number of iterations.
        """
        self.threshold = threshold
        self.tol = tol
        self.max_iter = max_iter
        self.Sp = None  # Remaining subset of points after anomaly removal

    def fit(self, X):
        """
        Fit the model by iteratively removing points that reduce the convex hull volume.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), input data.

        Returns:
        - self
        """
        S = set(map(tuple, X))  # Initialize Sp with all points
        Sp = S.copy()
        r = set()  # Removed points

        for _ in range(self.max_iter):
            Sh = ConvexHull(np.array(list(Sp)), qhull_options="QJ")  # Compute convex hull
            vol_c = Sh.volume  # Current volume
            vol_m = vol_c  # Minimum volume initialized to current volume
            S_new_h = Sh  # Initialize best hull
            r_n = set()  # Set of points removed in this iteration

            # Iterate over all points in current convex hull
            for p in Sh.points:
                p_tuple = tuple(p)
                Sp_new = Sp - {p_tuple}  # Remove point p temporarily

                try:
                    Sh_new = ConvexHull(np.array(list(Sp_new)))
                    vol_n = Sh_new.volume
                except Exception:
                    vol_n = np.inf  # If CH fails, set a high volume

                # Check if removing point reduces volume significantly
                if vol_n < vol_m - self.threshold:
                    S_new_h = Sh_new
                    vol_m = vol_n
                    r_n.add(p_tuple)

            # Update hull and removed points
            Sp -= r_n
            r.update(r_n)

            # Check stopping criteria
            if len(r) == len(S) or len(r_n) == 0:
                break

        self.Sp = Sp  # Remaining points
        return self

    def predict(self, X):
        """
        Predict anomalies for new data points.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), input data.

        Returns:
        - ndarray of shape (n_samples,), 1 for normal points, -1 for anomalies.
        """
        if self.Sp is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before 'predict'.")

        Sp_array = np.array(list(self.Sp))
        predictions = []

        for x in X:
            try:
                # Test if point is inside the current convex hull
                test_hull = ConvexHull(np.vstack([Sp_array, x]))
                predictions.append(1)  # Point is inside or on the convex hull
            except Exception:
                predictions.append(-1)  # Point cannot be added, considered an anomaly

        return np.array(predictions)



class OldVersionConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, lam=1.0, tol=1e-3, max_iter=100):
        """
        Parameters:
        - lam: float, regularization parameter to balance size vs. volume of the convex hull.
        - tol: float, tolerance for stopping criteria.
        - max_iter: int, maximum number of iterations for the outer loop.
        """
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.Sp = None  # The anomaly-free subset



    def is_full_dimensional(self,points):
        """
        Check if the given points span the full dimensional space.

        Parameters:
        - points: ndarray of shape (n_samples, n_features)

        Returns:
        - bool: True if the points span the full dimensionality of the space,
          otherwise False.
        """
        if len(points) == 0:
            return False
        return matrix_rank(points) == points.shape[1]

    from concurrent.futures import ProcessPoolExecutor

    def evaluate_point(args):
        """
        Evaluate the effect of adding/removing point `s` on the subset `Sp`.

        Parameters:
        - args: tuple (s, Sp, lam)
            s: tuple, Candidate point.
            Sp: set, Current subset of points.
            lam: float, Regularization parameter.

        Returns:
        - tuple: (candidate, candidate_f)
            The modified subset and its corresponding objective function value.
        """
        s, Sp, lam = args

        if s not in Sp:
            candidate = Sp | {s}
        else:
            candidate = Sp - {s}

        candidate_array = np.array(sorted(candidate))

        # Check if points are full-dimensional
        if len(candidate_array) > 0 and matrix_rank(candidate_array) < candidate_array.shape[1]:
            volume = 0
        else:
            try:
                hull = ConvexHull(candidate_array, qhull_options="QJ")
                volume = hull.volume
            except Exception:
                volume = 0

        candidate_f = len(candidate) - lam * volume
        return candidate, candidate_f

    def fit(self, X, y=None):
        """
        Fit the model by finding the optimal anomaly-free subset (Sp).
        """
        S = set(map(tuple, X))  # Convert data to a set of tuples
        Sp = set()
        S_unchecked = set(S)  # Points that have not yet been fully evaluated
        self.hull_cache = {}  # Cache for convex hull volumes

        current_f = -float('inf')

        for _ in range(self.max_iter):
            best_candidate = Sp
            best_f = current_f

            # Dynamically adjust the subset size
            subset_size = min(len(S_unchecked), 100)
            if subset_size == 0:
                break

            sampled_S = random.sample(list(S_unchecked), subset_size)

            # Prepare arguments for parallel processing
            args = [(s, Sp, self.lam) for s in sampled_S]

            # Parallelize inner loop
            with ProcessPoolExecutor() as executor:
                results = executor.map(self.evaluate_point, args)

            for candidate, candidate_f in results:
                if candidate_f > best_f:
                    best_candidate = candidate
                    best_f = candidate_f

            # Update unchecked points
            S_unchecked -= (best_candidate - Sp)

            # Check stopping criteria
            if abs(best_f - current_f) < self.tol:
                break

            Sp = best_candidate
            current_f = best_f

        self.Sp = Sp
        return self

    def predict(self, X):
        """
        Predict whether each point in X is part of the fitted Sp (anomaly-free subset).
        Returns:
        - 1 for points in Sp (normal/inliers).
        - -1 for points not in Sp (anomalies).
        """
        if self.Sp is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before 'predict'.")

        predictions = []
        for x in X:
            # Check if the point would improve the subset
            candidate = self.Sp | {tuple(x)}
            try:
                hull = ConvexHull(np.array(list(candidate)))
                volume = hull.volume
            except Exception:
                volume = 0

            candidate_f = len(candidate) - self.lam * volume
            fitted_f = len(self.Sp) - self.lam * (ConvexHull(np.array(list(self.Sp))).volume if len(self.Sp) > 2 else 0)

            if candidate_f > fitted_f:
                predictions.append(1)  # Normal (would belong to Sp)
            else:
                predictions.append(-1)  # Anomalous (wouldn't belong to Sp)

        return np.array(predictions)



class BaseConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, lam=1.0, tol=1e-3, max_iter=100):
        """
        Parameters:
        - lam: float, regularization parameter to balance size vs. volume of the convex hull.
        - tol: float, tolerance for stopping criteria.
        - max_iter: int, maximum number of iterations for the outer loop.
        """
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.Sp = None  # The anomaly-free subset

    def fit(self, X, y=None):
        """
        Fit the model by finding the optimal anomaly-free subset (Sp).
        """
        S = set(map(tuple, X))  # Convert data to a set of tuples
        Sp = set()

        current_f = -float('inf')

        # Outer loop: Repeat until stopping criteria is met
        for _ in range(self.max_iter):
            best_candidate = Sp
            best_f = current_f

            # Inner loop: Iterate over all points in S
            for s in S:
                if s not in Sp:
                    # Try adding the point
                    candidate = Sp | {s}
                else:
                    # Try removing the point
                    candidate = Sp - {s}

                # Compute volume of the convex hull
                try:
                    hull = ConvexHull(np.array(list(candidate)))
                    volume = hull.volume
                except Exception:
                    volume = 0  # If not enough points to form a hull, volume is 0

                # Compute objective function
                candidate_f = len(candidate) - self.lam * volume
                if candidate_f > best_f:
                    best_candidate = candidate
                    best_f = candidate_f

            # Check stopping criteria
            if abs(best_f - current_f) < self.tol:
                break

            # Update state
            Sp = best_candidate
            current_f = best_f

        self.Sp = Sp
        return self

    def predict(self, X):
        """
        Predict whether each point in X is part of the fitted Sp (anomaly-free subset).
        Returns:
        - 1 for points in Sp (normal/inliers).
        - -1 for points not in Sp (anomalies).
        """
        if self.Sp is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before 'predict'.")

        predictions = []
        for x in X:
            # Check if the point would improve the subset
            candidate = self.Sp | {tuple(x)}
            try:
                hull = ConvexHull(np.array(list(candidate)))
                volume = hull.volume
            except Exception:
                volume = 0

            candidate_f = len(candidate) - self.lam * volume
            fitted_f = len(self.Sp) - self.lam * (ConvexHull(np.array(list(self.Sp))).volume if len(self.Sp) > 2 else 0)

            if candidate_f > fitted_f:
                predictions.append(1)  # Normal (would belong to Sp)
            else:
                predictions.append(-1)  # Anomalous (wouldn't belong to Sp)

        return np.array(predictions)