"""
Preprocessing utilities for the student performance project.
Loads the CSV, cleans it, encodes categories, builds X/y, normalizes, and splits.
"""

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from CSV.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV loaded but it's empty.")
        print(f"[preprocessing] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find the file at: {path}")
    except Exception as e:
        raise ValueError(f"Something went wrong while reading CSV: {e}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple missing value handling:
    - numeric: fill with column mean
    - categorical: fill with mode
    """
    total_missing = int(df.isna().sum().sum())
    if total_missing == 0:
        print("[preprocessing] No missing values. Nice!")
        return df.copy()

    fixed = df.copy()
    num_cols = fixed.select_dtypes(include=[np.number]).columns
    cat_cols = fixed.select_dtypes(include=["object"]).columns

    # numeric → mean
    for col in num_cols:
        fixed[col] = fixed[col].fillna(fixed[col].mean())

    # categorical → mode
    for col in cat_cols:
        mode_val = fixed[col].mode()
        if len(mode_val) > 0:
            fixed[col] = fixed[col].fillna(mode_val.iloc[0])
        else:
            fixed[col] = fixed[col].fillna("Unknown")

    print(f"[preprocessing] Filled {total_missing} missing values.")
    return fixed


def encode_categorical_variables(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Super simple label encoding for all object columns.
    Returns the encoded DataFrame + a dict of mappings (so we know what 0/1/2 means).
    """
    encoded = df.copy()
    mappings: Dict[str, Dict] = {}
    cat_cols = encoded.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        uniques = encoded[col].astype(str).unique().tolist()
        mapping = {val: i for i, val in enumerate(uniques)}
        encoded[col] = encoded[col].astype(str).map(mapping)
        mappings[col] = mapping

    print(f"[preprocessing] Encoded {len(cat_cols)} categorical columns.")
    return encoded, mappings


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = "math score",
    drop_cols: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X (features) and y (target).
    By default, we drop reading/writing scores so we predict math from demographics + prep.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if drop_cols is None:
        drop_cols = ["reading score", "writing score"]

    keep_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    feature_cols = [c for c in keep_df.columns if c != target_col]

    X = keep_df[feature_cols].to_numpy()
    y = keep_df[target_col].to_numpy()

    print(f"[preprocessing] X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max normalize each feature to [0,1]. Handles constant columns safely.
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    rng = X_max - X_min
    rng[rng == 0] = 1.0
    Xn = (X - X_min) / rng
    print("[preprocessing] Normalized features to [0,1].")
    return Xn, X_min, X_max


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle and split into train/test (simple and reproducible).
    """
    np.random.seed(seed)
    n = X.shape[0]
    idx = np.random.permutation(n)
    n_test = int(n * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    print(f"[preprocessing] Train: {Xtr.shape[0]} | Test: {Xte.shape[0]}")
    return Xtr, Xte, ytr, yte