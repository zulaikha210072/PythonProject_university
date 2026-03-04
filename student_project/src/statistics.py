"""
Lightweight NumPy-only stats helpers for quick exploration.
"""

from typing import Dict, List
import numpy as np


def compute_mean(X: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(X, axis=axis)


def compute_std(X: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.std(X, axis=axis)


def compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    cov = compute_covariance_matrix(X)
    s = np.sqrt(np.diag(cov))
    s[s == 0] = 1.0
    return cov / np.outer(s, s)


def summary_vector(y: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(y)),
        "median": float(np.median(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "q25": float(np.percentile(y, 25)),
        "q75": float(np.percentile(y, 75)),
    }


def print_statistics(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
    print("\n====== Basic Statistics ======")
    tgt = summary_vector(y)
    print("Target (math score):")
    for k, v in tgt.items():
        print(f"  {k:>6}: {v:.2f}")

    print("\nFeatures:")
    print(f"  samples: {X.shape[0]}")
    print(f"  dims   : {X.shape[1]}")
    means = compute_mean(X, axis=0)
    for name, m in zip(feature_names, means):
        print(f"  mean({name}): {m:.4f}")