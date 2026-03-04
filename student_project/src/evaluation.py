"""
Evaluation metrics using NumPy only.
"""

import numpy as np


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((y - yhat) ** 2))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(mse(y, yhat)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - yhat) ** 2))
    return 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)


def evaluate_all(y: np.ndarray, yhat: np.ndarray, name: str = "set") -> dict:
    metrics = {"MAE": mae(y, yhat), "MSE": mse(y, yhat), "RMSE": rmse(y, yhat), "R2": r2(y, yhat)}
    print(f"\n====== Evaluation ({name}) ======")
    for k, v in metrics.items():
        print(f"{k:>5}: {v:.4f}")
    return metrics