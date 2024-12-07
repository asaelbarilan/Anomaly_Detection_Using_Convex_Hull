from sklearn.base import BaseEstimator, OutlierMixin
import numpy as np
from scipy.spatial import ConvexHull

class ConvexHullAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, lam=1.0, tol=1e-3, max_iter=100):
        """
        Parameters:
        - lam: float, regularization parameter to balance size vs. volume of the convex hull.
        - tol: float, tolerance for stopping criteria.
        - max_iter: int, maximum number of iterations.
        """
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.Sp = None  # The anomaly-free subset



    def fit(self, X, y=None):
        S = {tuple(x): None for x in X}  # Use dict for efficient operations
        Sp = {}

        current_f = -float('inf')

        for _ in range(self.max_iter):
            best_candidate = Sp
            best_f = current_f

            for s in S.keys():
                if s not in Sp:
                    candidate = {**Sp, s: None}
                else:
                    candidate = {k: v for k, v in Sp.items() if k != s}

                try:
                    hull = ConvexHull(np.array(list(candidate.keys())))
                    volume = hull.volume
                except Exception:
                    volume = 0

                candidate_f = len(candidate) - self.lam * volume
                if candidate_f > best_f:
                    best_candidate = candidate
                    best_f = candidate_f

            if abs(best_f - current_f) < self.tol:
                break

            Sp = best_candidate
            current_f = best_f

        self.Sp = Sp
        return self

    def predict(self, X):
        if self.Sp is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before 'predict'.")

        predictions = []
        for x in X:
            candidate = {**self.Sp, tuple(x): None}
            try:
                hull = ConvexHull(np.array(list(candidate.keys())))
                volume = hull.volume
            except Exception:
                volume = 0

            candidate_f = len(candidate) - self.lam * volume
            fitted_f = len(self.Sp) - self.lam * (
                ConvexHull(np.array(list(self.Sp.keys()))).volume if len(self.Sp) > 2 else 0)

            if candidate_f > fitted_f:
                predictions.append(1)
            else:
                predictions.append(-1)

        return np.array(predictions)