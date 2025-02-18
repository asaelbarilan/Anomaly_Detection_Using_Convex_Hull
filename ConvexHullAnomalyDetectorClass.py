
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
import tarfile
import os
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
from sklearn.metrics import f1_score  # For calculating F1-score

import numpy as np
import logging
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from scipy.spatial import ConvexHull
from sklearn.utils.validation import check_is_fitted


import numpy as np
import logging
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from scipy.spatial import ConvexHull
from sklearn.utils.validation import check_is_fitted


class ConfigurableConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(
            self,
            method="pca",
            n_components=2,
            stopping_criteria="naive",  # "naive", "elbow", or "optimal"
            tol=1e-3,
            max_iter=100,
            lam=1.0,
            n_jobs=-1
    ):
        """
        Parameters:
        -----------
        method : str
            Dimensionality reduction ('pca' or 'tsne').
        n_components : int
            Number of dimensions to reduce to.
        stopping_criteria : {"naive", "elbow", "optimal"}
            - naive: remove points if it improves the lam-based score; stop if no improvement.
            - elbow: stop if volume (or lam-based score) no longer changes enough.
            - optimal: if 'y' is provided, remove points while F1 improves.
        tol : float
            Tolerance for "elbow" stopping criteria.
        max_iter : int
            Maximum removal iterations.
        lam : float
            Lambda factor for f(S) = |S| - lam * volume(CH(S)).
        n_jobs : int
            Parallel jobs (not used in this snippet).
        """
        self.method = method
        self.n_components = n_components
        self.stopping_criteria = stopping_criteria
        self.tol = tol
        self.max_iter = max_iter
        self.lam = lam
        self.n_jobs = n_jobs

        # Fitted attributes:
        self.reducer = None
        self.Sp = None
        self.hull_equations_ = None
        self.X_reduced_ = None

        # For "optimal" stopping
        self._best_f1_score = -1
        self._best_Sp = None

    # ----------------------------------------------------------------------
    #  Public Methods
    # ----------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        Fit the Convex Hull anomaly detector with the specified stopping criteria.

        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) (optional)
            Used if stopping_criteria == "optimal" to compute F1.
        """
        # 1) Dimensionality reduction
        X_reduced = self._dimensionality_reduction(X)

        # 2) Initialize sets, volumes, etc.
        Sp, current_score = self._initialize_sets(X_reduced)

        # If "optimal", track best F1 on the full set
        if self.stopping_criteria == "optimal" and y is not None:
            self._best_f1_score = self._compute_f1_for_set(Sp, X, y)
            self._best_Sp = Sp.copy()

        # If "elbow", we track volume changes
        if self.stopping_criteria == "elbow":
            initial_volume = self._try_convex_hull_volume(Sp)
            prev_volume = initial_volume

        # 3) Iteratively remove points
        Sp = self._iteratively_remove_points(
            Sp=Sp,
            X=X,
            y=y,
            current_score=current_score,
            prev_volume=prev_volume if self.stopping_criteria == "elbow" else None
        )

        # 4) Finalize (store hull, etc.)
        self._finalize_hull(Sp)
        return self

    def predict(self, X):
        """
        Predict whether points are inside (+1) or outside (-1) of the final hull.
        """
        check_is_fitted(self, ["hull_equations_", "reducer", "X_reduced_", "Sp"])
        if self.method == "tsne":
            # TSNE can't transform new data. So only valid if X matches training set size.
            if X.shape[0] != self.X_reduced_.shape[0]:
                raise ValueError("TSNE does not support transform for new data.")
            X_reduced = self.X_reduced_
        else:
            X_reduced = self.reducer.transform(X)

        if self.hull_equations_ is None:
            # If no hull, label everything as inlier
            return np.ones(X_reduced.shape[0], dtype=int)

        A = self.hull_equations_[:, :-1]
        b = self.hull_equations_[:, -1]
        predictions = []
        for x in X_reduced:
            # inside if all facets satisfied
            if np.all(A.dot(x) + b <= 1e-12):
                predictions.append(1)
            else:
                predictions.append(-1)
        return np.array(predictions)


    # ----------------------------------------------------------------------
    #  Internal Steps (Helper Methods)
    # ----------------------------------------------------------------------

    def _dimensionality_reduction(self, X):
        """Perform PCA or TSNE reduction on X, store in self.X_reduced_."""
        self.reducer = self._initialize_reducer()
        if self.method == "tsne":
            X_reduced = self.reducer.fit_transform(X)
        else:
            X_reduced = self.reducer.fit_transform(X)
        self.X_reduced_ = X_reduced.copy()
        return X_reduced

    def _initialize_sets(self, X_reduced):
        """
        Convert X_reduced to a set Sp and compute initial score.
        Returns (Sp, current_score).
        """
        Sp = set(map(tuple, X_reduced))
        current_volume = self._try_convex_hull_volume(Sp)
        current_score = self._score(len(Sp), current_volume)
        return Sp, current_score

    def _iteratively_remove_points(self, Sp, X, y, current_score, prev_volume=None):
        """
        Main loop for removing points. For each iteration:
          - Build hull
          - Evaluate candidate removals
          - Remove best point if it improves the lam-based score
          - Check stopping_criteria
        Returns the final Sp.
        """
        count=0
        while True:
            Sp_array = np.array(list(Sp))
            if len(Sp_array) < self.n_components + 1:
                break  # not enough points for hull

            hull = self._try_build_hull(Sp_array)
            if hull is None:
                # can't build hull
                break

            hull_volume = hull.volume
            improved = False

            # We track the best new score after removing one point
            best_point_to_remove = None
            best_new_score = current_score

            candidate_points = hull.points[hull.vertices]
            for p in candidate_points:
                p_tuple = tuple(p)
                Sp_new = Sp - {p_tuple}
                new_volume = self._try_convex_hull_volume(Sp_new)
                new_score = self._score(len(Sp_new), new_volume)

                if new_score > best_new_score:
                    best_new_score = new_score
                    best_point_to_remove = p_tuple

            # If we found a removal that improves the score
            if best_point_to_remove is not None:
                Sp.remove(best_point_to_remove)
                current_score = best_new_score
                improved = True

                # If "optimal", check if F1 improved
                if self.stopping_criteria == "optimal" and y is not None:
                    current_f1 = self._compute_f1_for_set(Sp, X, y)
                    if current_f1 > self._best_f1_score:
                        self._best_f1_score = current_f1
                        self._best_Sp = Sp.copy()

            # Apply stopping criteria
            stop = self._apply_stopping_criteria(
                Sp=Sp,
                improved=improved,
                current_score=current_score,
                prev_volume=prev_volume,
                count=count
            )
            if stop:
                break
            count+=1
            # If "elbow", update prev_volume
            if self.stopping_criteria == "elbow" and prev_volume is not None:
                new_vol = self._try_convex_hull_volume(Sp)
                prev_volume = new_vol

        # End of iteration
        # If "optimal" and we have a best_Sp, use it
        if self.stopping_criteria == "optimal" and y is not None and self._best_Sp is not None:
            return self._best_Sp
        else:
            return Sp

    def _apply_stopping_criteria(self, Sp, improved, current_score, prev_volume,count):
        """
        Check if we should stop based on self.stopping_criteria.
        Return True if we should stop, False otherwise.
        """
        if self.stopping_criteria == "naive":
            # If no improvement => stop
            if count==self.max_iter:
                return True

        elif self.stopping_criteria == "elbow":
            # We look at volume changes or score changes.
            if prev_volume is not None:
                new_volume = self._try_convex_hull_volume(Sp)
                if abs(prev_volume - new_volume) < self.tol:
                    return True

        elif self.stopping_criteria == "optimal":
            # If not improved, we can stop
            if not improved:
                return True

        else:
            # fallback to naive
            if not improved:
                return True

        return False

    def _finalize_hull(self, Sp):
        """
        After finishing removal, build the final hull and store hull_equations_.
        """
        Sp_array_final = np.array(list(Sp))
        if len(Sp_array_final) >= self.n_components + 1:
            final_hull = self._try_build_hull(Sp_array_final)
            if final_hull:
                self.hull_equations_ = final_hull.equations
            else:
                self.hull_equations_ = None
        else:
            self.hull_equations_ = None

        self.Sp = Sp  # keep track of final set

    # ----------------------------------------------------------------------
    #  Utility / Score / Hull Methods
    # ----------------------------------------------------------------------
    def _score(self, num_points, volume):
        """
        Score function: f(S) = |S| - lam * volume.
        Higher is better.
        """
        return num_points - self.lam * volume

    def _try_build_hull(self, points_array):
        """
        Safely build a ConvexHull from points_array.
        Return hull or None if it fails.
        """
        try:
            return ConvexHull(points_array)
        except:
            return None

    def _try_convex_hull_volume(self, Sp):
        """
        Return the volume of the hull formed by Sp.
        If hull building fails or not enough points, return inf.
        """
        arr = np.array(list(Sp))
        if len(arr) < self.n_components + 1:
            return float('inf')
        hull = self._try_build_hull(arr)
        return hull.volume if hull else float('inf')

    def _compute_f1_for_set(self, Sp, X, y):
        """
        Rebuild a hull from Sp, then predict on X, compute F1.
        Inliers => 0, outliers => 1 for F1 scoring.
        """
        arr = np.array(list(Sp))
        hull = self._try_build_hull(arr)
        if hull is None:
            # If we can't build a hull, label everything inlier
            y_pred_binary = np.zeros_like(y)
        else:
            A = hull.equations[:, :-1]
            b = hull.equations[:, -1]
            # transform X if PCA, or use self.X_reduced_ if TSNE
            if self.method == "tsne":
                # assume X is training data
                X_reduced = self.X_reduced_
            else:
                X_reduced = self.reducer.transform(X)

            predictions = []
            for x_ in X_reduced:
                if np.all(A.dot(x_) + b <= 1e-12):
                    predictions.append(0)  # inlier
                else:
                    predictions.append(1)  # outlier
            y_pred_binary = np.array(predictions)

        return f1_score(y, y_pred_binary)

    def _initialize_reducer(self):
        """
        Create PCA or TSNE object based on self.method.
        """
        if self.method == "pca":
            return PCA(n_components=self.n_components)
        elif self.method == "tsne":
            return TSNE(n_components=self.n_components)
        else:
            raise ValueError("method must be 'pca' or 'tsne'.")

class ConvexHullAnomalyDetectorLambda(BaseEstimator, OutlierMixin):
    def __init__(
            self,
            method="pca",
            n_components=2,
            stopping_criteria="naive",  # "naive", "elbow", or "optimal"
            tol=1e-3,
            max_iter=100,
            lam=1.0,
            n_jobs=-1
    ):
        """
        Parameters:
        -----------
        method : str
            Dimensionality reduction ('pca' or 'tsne').
        n_components : int
            Number of dimensions to reduce to.
        stopping_criteria : {"naive", "elbow", "optimal"}
            - naive: remove points if it improves the score, stop if no improvement.
            - elbow: stop if volume (or score) no longer changes enough.
            - optimal: use y labels (if given) to track F1 improvements.
        tol : float
            Tolerance for elbow-based stopping criteria.
        max_iter : int
            Maximum number of removal iterations.
        lam : float
            Lambda factor for weighting volume vs set size in the scoring function f(S).
        n_jobs : int
            Parallel jobs (not used in this snippet).
        """
        self.method = method
        self.n_components = n_components
        self.stopping_criteria = stopping_criteria
        self.tol = tol
        self.max_iter = max_iter
        self.lam = lam
        self.n_jobs = n_jobs

        # Will be filled after fit
        self.reducer = None
        self.Sp = None
        self.hull_equations_ = None
        self.X_reduced_ = None

        # For "optimal" stopping
        self._best_f1_score = -1
        self._best_Sp = None

    def _initialize_reducer(self):
        if self.method == "pca":
            return PCA(n_components=self.n_components)
        elif self.method == "tsne":
            return TSNE(n_components=self.n_components)
        else:
            raise ValueError("Invalid method. Must be 'pca' or 'tsne'.")

    def _score(self, num_points, volume):
        """
        Score function: f(S) = |S| - lam * volume
        The higher, the better.
        """
        return num_points - self.lam * volume

    def fit(self, X, y=None):
        """
        Fit the Convex Hull anomaly detector with the specified stopping criteria.

        X : array-like of shape (n_samples, n_features)
        y : optional, used if stopping_criteria == "optimal" to compute F1.
        """
        # 1) Dimensionality reduction
        self.reducer = self._initialize_reducer()
        if self.method == "tsne":
            # TSNE only has fit_transform
            X_reduced = self.reducer.fit_transform(X)
        else:
            X_reduced = self.reducer.fit_transform(X)
        self.X_reduced_ = X_reduced.copy()

        # Convert data to a set of tuples to easily remove points
        Sp = set(map(tuple, X_reduced))

        # Build initial hull to get initial volume
        current_volume = self._try_convex_hull_volume(Sp)
        current_score = self._score(len(Sp), current_volume)

        # If "optimal", we track F1 with the current set
        if self.stopping_criteria == "optimal" and y is not None:
            # Evaluate F1 with the entire set
            # (No points removed => build hull => predict => measure F1)
            self._best_f1_score = self._compute_f1_for_set(Sp, X, y)
            self._best_Sp = Sp.copy()

        # For "elbow", keep track of the last volume or score to see if improvement < tol
        if self.stopping_criteria == "elbow":
            prev_volume = current_volume  # or prev_score = current_score

        for iteration in range(self.max_iter):
            Sp_array = np.array(list(Sp))
            # If not enough points to build hull, break
            if len(Sp_array) < self.n_components + 1:
                break

            hull = self._try_build_hull(Sp_array)
            if hull is None:
                # Hull building failed => break
                break

            hull_volume = hull.volume
            improved = False

            # For each iteration, see if there's a single point that improves the criterion
            best_point_to_remove = None
            best_new_score = current_score

            # Usually only hull vertices matter for volume changes
            candidate_points = hull.points[hull.vertices]

            for p in candidate_points:
                p_tuple = tuple(p)
                # Remove p temporarily
                Sp_new = Sp - {p_tuple}
                # Compute new volume
                new_volume = self._try_convex_hull_volume(Sp_new)
                new_score = self._score(len(Sp_new), new_volume)

                # Check improvement in "naive" sense (score-based)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_point_to_remove = p_tuple

            # If we found a point that improved the score, remove it
            if best_point_to_remove is not None:
                Sp.remove(best_point_to_remove)
                current_score = best_new_score
                improved = True

                # If "optimal": check if that removal improved F1
                if self.stopping_criteria == "optimal" and y is not None:
                    current_f1 = self._compute_f1_for_set(Sp, X, y)
                    if current_f1 > self._best_f1_score:
                        self._best_f1_score = current_f1
                        self._best_Sp = Sp.copy()

            # ===========================
            #   STOPPING CRITERIA
            # ===========================
            if self.stopping_criteria == "naive":
                # If no improvement => break
                if not improved:
                    break

            elif self.stopping_criteria == "elbow":
                # Check difference in volume (or score) from previous iteration
                new_volume = self._try_convex_hull_volume(Sp)
                if abs(prev_volume - new_volume) < self.tol:
                    # Not enough improvement => stop
                    break
                prev_volume = new_volume

            elif self.stopping_criteria == "optimal" and y is not None:
                # We keep removing points so long as we see an F1 improvement.
                # If no improvement => break
                if not improved:
                    break

            else:
                # If stopping_criteria == "optimal" but no y provided,
                # we basically fallback to naive approach
                if not improved:
                    break

        # After finishing iterations:
        if self.stopping_criteria == "optimal" and y is not None and self._best_Sp is not None:
            # Use the best set found for F1
            self.Sp = self._best_Sp
        else:
            self.Sp = Sp

        # Build final hull
        Sp_array_final = np.array(list(self.Sp))
        if len(Sp_array_final) >= self.n_components + 1:
            final_hull = self._try_build_hull(Sp_array_final)
            if final_hull:
                self.hull_equations_ = final_hull.equations
            else:
                self.hull_equations_ = None
        else:
            self.hull_equations_ = None

        return self

    def predict(self, X):
        """
        Predict whether points are inside (+1=inside) or outside (-1=outside)
        of the final hull.
        """
        check_is_fitted(self, ["reducer", "X_reduced_", "Sp", "hull_equations_"])

        # Handle TSNE's lack of .transform for new data
        if self.method == "tsne":
            # If X is the same data used to train, we can reuse self.X_reduced_
            if X.shape[0] != self.X_reduced_.shape[0]:
                raise ValueError("TSNE does not support transform for new data. "
                                 "Predicting on training set only.")
            X_reduced = self.X_reduced_
        else:
            X_reduced = self.reducer.transform(X)

        # If no hull, label everything as inlier
        if self.hull_equations_ is None:
            return np.ones(X_reduced.shape[0], dtype=int)

        A = self.hull_equations_[:, :-1]
        b = self.hull_equations_[:, -1]
        predictions = []
        for x in X_reduced:
            # Inside if for all facets: A_i * x + b_i <= 1e-12
            if np.all(A.dot(x) + b <= 1e-12):
                predictions.append(1)
            else:
                predictions.append(-1)
        return np.array(predictions)

    # ----------------------------------------------------------------
    #   Helper methods
    # ----------------------------------------------------------------
    def _try_build_hull(self, points_array):
        """Safely build a ConvexHull, return None if fails."""
        try:
            return ConvexHull(points_array)
        except:
            return None

    def _try_convex_hull_volume(self, points_set):
        """Build a hull from points_set, return volume or float('inf') if fails."""
        arr = np.array(list(points_set))
        if len(arr) < self.n_components + 1:
            return float('inf')  # can't form a hull
        hull = self._try_build_hull(arr)
        return hull.volume if hull else float('inf')

    def _compute_f1_for_set(self, Sp, X, y):
        """
        Compute F1 score after building a hull from Sp and predicting on X.
        We interpret +1 => inlier => 0 label; -1 => outlier => 1 label.
        """
        arr = np.array(list(Sp))
        hull = self._try_build_hull(arr)
        if hull is None:
            # If we can't build a hull, label everything as inlier
            y_pred_binary = np.zeros_like(y)
        else:
            # Build eqns
            A = hull.equations[:, :-1]
            b = hull.equations[:, -1]

            # If X is TSNE, we presumably just use self.X_reduced_ but
            # for simplicity let's assume X is the training set
            if self.method == "tsne":
                # Only valid if X is the same as self.X_reduced_
                X_reduced = self.X_reduced_
            else:
                X_reduced = self.reducer.transform(X)

            # Predict
            predictions = []
            for x_ in X_reduced:
                if np.all(A.dot(x_) + b <= 1e-12):
                    predictions.append(0)  # inlier
                else:
                    predictions.append(1)  # outlier

            y_pred_binary = np.array(predictions)

        return f1_score(y, y_pred_binary)



class ConfigurableConvexHullAnomalyDetectorNonCOmpact(BaseEstimator, OutlierMixin):
    def __init__(
            self,
            method="pca",
            n_components=2,
            stopping_criteria="naive",
            tol=1e-3,
            max_iter=100,
            lam=1.0,
            n_jobs=-1
    ):
        """
        Parameters:
        - method: str, dimensionality reduction method ('pca' or 'tsne').
        - n_components: int, number of dimensions to reduce to.
        - stopping_criteria: str, stopping method ('naive', 'elbow', 'optimal').
        - tol: float, tolerance for elbow-based stopping criteria.
        - max_iter: int, maximum iterations to remove points that reduce hull volume.
        - lam: float, (not currently used in the snippet) could be used for weighting volume vs size.
        - n_jobs: int, number of parallel jobs (not used in this snippet).
        """
        self.method = method
        self.n_components = n_components
        self.stopping_criteria = stopping_criteria
        self.tol = tol
        self.max_iter = max_iter
        self.lam = lam
        self.n_jobs = n_jobs

        self.reducer = None  # Will hold PCA or TSNE object
        self.Sp = None  # Final set of points in reduced space
        self.best_score = None
        self.hull_equations_ = None  # Store final hull equations
        self.X_reduced_ = None  # Store dimension-reduced points (especially for TSNE)

    def _initialize_reducer(self):
        if self.method == "pca":
            return PCA(n_components=self.n_components)
        elif self.method == "tsne":
            # Remember: TSNE in sklearn does not implement .transform()
            # We’ll have to store X_reduced_ at fit time if we want to do predictions.
            return TSNE(n_components=self.n_components)
        else:
            raise ValueError("Invalid dimensionality reduction method. Choose 'pca' or 'tsne'.")

    def fit(self, X, y=None):
        """
        Fit the Convex Hull anomaly detector with the specified stopping criteria.
        X: array-like of shape (n_samples, n_features)
        y: optional, used for 'optimal' stopping with an F1-based approach
        """
        # 1) Dimensionality reduction
        self.reducer = self._initialize_reducer()

        if self.method == "tsne":
            # TSNE has only fit_transform
            X_reduced = self.reducer.fit_transform(X)
        else:
            X_reduced = self.reducer.fit_transform(X)

        self.X_reduced_ = X_reduced.copy()  # store for potential re-use

        # 2) Convert to set of tuples to handle iterative removals
        Sp = set(map(tuple, X_reduced))

        best_score = -float('inf')  # track best F1 if stopping_criteria="optimal"
        best_hull_points = None

        prev_volume = float('inf')  # track volume if stopping_criteria="elbow"

        # 3) Iteratively remove points
        for _ in range(self.max_iter):
            Sp_array = np.array(list(Sp))
            # If not enough points to form a hull
            if len(Sp_array) < self.n_components + 1:
                break

            # Attempt to build hull
            try:
                Sh = ConvexHull(Sp_array)
                current_volume = Sh.volume
            except Exception:
                # If the hull fails to build, stop
                break

            # --- Checking stopping criteria ---
            if self.stopping_criteria == "elbow":
                # Stop if volume hasn’t changed enough
                if prev_volume - current_volume < self.tol:
                    break

            elif self.stopping_criteria == "optimal" and y is not None:
                # Evaluate current F1
                y_pred = self.predict(X)
                # By convention, anomaly detectors often use -1 for outliers and 1 for inliers.
                # If your y array is 1 for "anomaly" and 0 for "normal", you may need to adjust it.
                # For example, if y == 1 is an outlier, we convert y_pred:
                y_pred_binary = np.where(y_pred == -1, 1, 0)
                f1 = f1_score(y, y_pred_binary)
                if f1 > best_score:
                    best_score = f1
                    best_hull_points = Sp.copy()

            # Update for next iteration if using elbow
            prev_volume = current_volume

            # 4) Find the single point whose removal yields the greatest volume decrease
            best_point_to_remove = None
            best_new_volume = current_volume

            # The hull’s vertices are the only relevant points for reducing volume
            candidate_points = Sh.points[Sh.vertices]

            for p in candidate_points:
                p_tuple = tuple(p)
                # Remove p_tuple from Sp temporarily
                Sp_new = Sp - {p_tuple}
                if len(Sp_new) < self.n_components + 1:
                    # Not enough points to form a hull
                    new_volume = float('inf')
                else:
                    try:
                        Sh_new = ConvexHull(np.array(list(Sp_new)))
                        new_volume = Sh_new.volume
                    except Exception:
                        new_volume = float('inf')

                # Update best if the volume is decreased
                if new_volume < best_new_volume:
                    best_new_volume = new_volume
                    best_point_to_remove = p_tuple

            # If no point removal improved the volume, we stop
            if best_point_to_remove is None or best_new_volume >= current_volume:
                break

            # Otherwise, remove that point
            Sp.remove(best_point_to_remove)

        # 5) Once done with iteration:
        #    if "optimal", use best hull points. Otherwise use Sp from last iteration
        if self.stopping_criteria == "optimal" and (best_hull_points is not None):
            self.Sp = best_hull_points
        else:
            self.Sp = Sp

        # 6) Build final hull and store equations
        Sp_array_final = np.array(list(self.Sp))
        if len(Sp_array_final) >= self.n_components + 1:
            try:
                final_hull = ConvexHull(Sp_array_final)
                self.hull_equations_ = final_hull.equations  # shape = (n_facets, n_dim+1)
            except Exception as e:
                logging.warning(f"Failed to build final hull: {e}")
                self.hull_equations_ = None
        else:
            self.hull_equations_ = None

        return self

    def predict(self, X):
        """
        Predict whether points are inside (-1 for outside, +1 for inside).
        If using TSNE, we assume X is the same data used in fit (or handle it accordingly).
        """
        check_is_fitted(self, ["reducer", "X_reduced_", "Sp", "hull_equations_"])

        # If TSNE, no built-in transform:
        if self.method == "tsne":
            # If you only do anomaly detection on the training set:
            # re-use self.X_reduced_. Otherwise, you need a workaround for new data.
            X_reduced = self.X_reduced_
            # But if X is a different array (shape or values) than the original,
            # this won't be correct. You might want to raise an error:
            if X.shape[0] != self.X_reduced_.shape[0]:
                raise ValueError("TSNE does not support transform for new data. "
                                 "Predicting on training set only.")
        else:
            # PCA has a .transform()
            X_reduced = self.reducer.transform(X)

        # If we have no hull (too few points or error building), consider them inliers
        if self.hull_equations_ is None:
            return np.ones(X_reduced.shape[0], dtype=int)

        # The hull equations are of shape (num_facets, n_dim+1).
        # Typically, equation is A_i * x + b_i <= 0 means inside
        # But sometimes the sign is reversed, so you have to confirm your sign usage.
        A = self.hull_equations_[:, :-1]  # first n_components columns
        b = self.hull_equations_[:, -1]  # last column

        predictions = []
        for x in X_reduced:
            # If all planes are satisfied, x is inside
            # The typical eqn from `ConvexHull.equations` is of form: normal * x + offset == 0
            # But half-spaces might be normal*x + offset <= 0 for inside
            # You often need a small tolerance for floating point checks.
            # Here, let's interpret inside if normal*x + offset <= 1e-12 for all facets.
            if np.all(A.dot(x) + b <= 1e-12):
                predictions.append(1)  # inside
            else:
                predictions.append(-1)  # outside

        return np.array(predictions)


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