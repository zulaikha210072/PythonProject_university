"""
Runs the whole pipeline in a clean, beginner-friendly way.
"""

from src.preprocessing import (
    load_data,
    handle_missing_values,
    encode_categorical_variables,
    prepare_features_target,
    normalize_features,
    train_test_split,
)
from src.statistics import compute_correlation_matrix, print_statistics
from src.regression import LinearRegressionGD
from src.evaluation import evaluate_all
from src.visualization import plot_hist, plot_heatmap, plot_pred_vs_actual, plot_cost


def main():
    print("\n=== Predictive Modeling: Student Performance (NumPy + Matplotlib) ===")

    # 1) Load
    df = load_data("data/StudentsPerformance.csv")

    # 2) Clean
    df = handle_missing_values(df)

    # 3) Encode categories
    df_enc, enc_maps = encode_categorical_variables(df)

    # 4) Build X/y (predict math score from demographics + test prep)
    X, y, feature_names = prepare_features_target(
        df_enc,
        target_col="math score",
        drop_cols=["reading score", "writing score"],
    )

    # quick stats (before normalization so means are in raw label-encoded units)
    print_statistics(X, y, feature_names)

    # 5) Correlations (on current X)
    corr = compute_correlation_matrix(X)
    plot_heatmap(corr, feature_names)

    # 6) Normalize features (helps gradient descent a lot)
    Xn, Xmin, Xmax = normalize_features(X)

    # 7) Split
    Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.2, seed=42)

    # 8) Train model (from scratch)
    model = LinearRegressionGD(lr=0.1, iters=1000)
    model.fit(Xtr, ytr, verbose=True)

    # 9) Cost curve
    plot_cost(model.costs)

    # 10) Predictions
    ytr_hat = model.predict(Xtr)
    yte_hat = model.predict(Xte)

    # 11) Evaluation
    evaluate_all(ytr, ytr_hat, name="train")
    evaluate_all(yte, yte_hat, name="test")

    # 12) Visual check of predictions
    plot_hist(y, title="Distribution of Math Scores", bins=20)
    plot_pred_vs_actual(yte, yte_hat)

    # 13) Print parameters (weights map to feature_names in order)
    w, b = model.parameters()
    print("\n=== Parameters ===")
    print(f"bias: {b:.4f}")
    for name, val in zip(feature_names, w):
        print(f"{name:30s}: {val:.4f}")

    print("\n[Note] I used PyCharm's debugger to step through gradient descent,")
    print("check 'err', 'dw/db', and watch 'cost' go down. Very helpful!")


if __name__ == "__main__":
    main()