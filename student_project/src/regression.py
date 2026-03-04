"""
Linear regression from scratch (no sklearn).
We use plain gradient descent on MSE.
"""

from typing import Tuple, List
import numpy as np


class LinearRegressionGD:
    def __init__(self, lr: float = 0.1, iters: int = 1000):
        self.lr = lr
        self.iters = iters
        self.w: np.ndarray | None = None
        self.b: float | None = None
        self.costs: List[float] = []

    def _init_params(self, d: int) -> None:
        self.w = np.zeros(d)
        self.b = 0.0

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b  # (n, d) @ (d,) + scalar → (n,)

    def _mse_cost(self, y: np.ndarray, yhat: np.ndarray) -> float:
        n = y.shape[0]
        return float((1 / (2 * n)) * np.sum((yhat - y) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        n, d = X.shape
        self._init_params(d)

        if verbose:
            print("\n====== Training Linear Regression (from scratch) ======")
            print(f"lr={self.lr}, iters={self.iters}, n={n}, d={d}")

        for t in range(self.iters):
            yhat = self._predict_raw(X)
            err = yhat - y
            # gradients
            dw = (X.T @ err) / n
            db = float(np.sum(err) / n)
            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db
            # track cost
            cost = self._mse_cost(y, self._predict_raw(X))
            self.costs.append(cost)

            if verbose and (t % 100 == 0 or t == self.iters - 1):
                print(f"iter {t:4d} | cost {cost:.4f}")

        if verbose:
            print("Done training.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self._predict_raw(X)

    def parameters(self) -> Tuple[np.ndarray, float]:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained yet.")
        return self.w, self.b