import random

from sklearn.base import BaseEstimator, OutlierMixin
import numpy as np
from scipy.spatial import ConvexHull
from concurrent.futures import ProcessPoolExecutor
from numpy.linalg import matrix_rank
from scipy.spatial import ConvexHull
import numpy as np
import random
from numpy.linalg import matrix_rank

class ConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
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