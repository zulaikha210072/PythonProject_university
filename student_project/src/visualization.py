"""
Matplotlib plots for quick checks: hist, scatter, heatmap, and training cost curve.
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt


def plot_hist(y: np.ndarray, title: str = "Math score distribution", bins: int = 20):
    plt.figure(figsize=(9, 5))
    plt.hist(y, bins=bins, color="#6aa9ff", edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_pred_vs_actual(y: np.ndarray, yhat: np.ndarray):
    plt.figure(figsize=(6, 6))
    plt.scatter(y, yhat, s=35, alpha=0.7, edgecolor="k")
    lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="perfect")
    plt.title("Predicted vs Actual (math)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_heatmap(corr: np.ndarray, names: List[str]):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, label="corr")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    # annotate small grid with values
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.show()


def plot_cost(costs: list):
    plt.figure(figsize=(9, 5))
    plt.plot(costs, color="#9b59b6", lw=2)
    plt.title("Training Cost (MSE) over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()