"""Classical machine learning baseline for Air Quality Index (AQI) prediction.

This script trains three models on pollutant data and compares them:
  1. Linear Regression      (simple reference point)
  2. Random Forest          (strong tree-based model)
  3. Gradient Boosting      (strong tree-based model)

The results are the classical baseline that a hybrid quantum-classical
model can later be compared against.

Example:
    python aqi_baseline.py --data data/city_day.csv --city Delhi
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path("results")


def parse_args():
    p = argparse.ArgumentParser(description="AQI prediction baseline")
    p.add_argument("--data", required=True, help="Path to the CSV file")
    p.add_argument("--target", default="AQI", help="Name of the AQI column")
    p.add_argument("--date-col", default="Date", help="Name of the date column")
    p.add_argument("--city-col", default="City", help="Name of the city column")
    p.add_argument("--city", default=None, help="Only use rows for this city")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of the most recent rows used for testing")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(args):
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")

    if args.city is not None:
        if args.city_col not in df.columns:
            raise SystemExit(f"Column '{args.city_col}' not found in the data")
        df = df[df[args.city_col] == args.city]
        print(f"Kept {len(df)} rows for city '{args.city}'")

    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. Columns are: {list(df.columns)}"
        )

    # Sort by time so the test set is always *later* than the training set.
    # A random split would leak future information into training.
    if args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
        df = df.sort_values(args.date_col)

    df = df.dropna(subset=[args.target])
    return df


def get_features(df, args):
    """Use every numeric column except the target as an input feature."""
    numeric = df.select_dtypes(include="number")
    feature_cols = [c for c in numeric.columns if c != args.target]
    # Drop columns that are completely empty
    feature_cols = [c for c in feature_cols if numeric[c].notna().any()]
    return df[feature_cols], df[args.target], feature_cols


def build_models(seed):
    # Missing values are filled with the training median inside each pipeline.
    imputer = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "Linear Regression": make_pipeline(imputer(), StandardScaler(), LinearRegression()),
        "Random Forest": make_pipeline(
            imputer(), RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        ),
        "Gradient Boosting": make_pipeline(
            imputer(), GradientBoostingRegressor(random_state=seed)
        ),
    }


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def plot_predictions(y_test, preds, model_name):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, preds, s=10, alpha=0.5)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title(f"{model_name}: predicted vs actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_importance(model, feature_cols):
    forest = model.steps[-1][1]
    order = np.argsort(forest.feature_importances_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(feature_cols)[order], forest.feature_importances_[order])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_data(args)
    X, y, feature_cols = get_features(df, args)
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Time-ordered split: oldest rows train, newest rows test
    split = int(len(df) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    models = build_models(args.seed)
    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
        rows.append({"Model": name, **evaluate(y_test, preds)})

    results = pd.DataFrame(rows).set_index("Model").round(3)
    print("\nResults on the test set:")
    print(results)
    results.to_csv(RESULTS_DIR / "metrics.csv")

    best = results["RMSE"].idxmin()
    print(f"\nBest model by RMSE: {best}")
    plot_predictions(y_test, predictions[best], best)
    plot_importance(models["Random Forest"], feature_cols)
    print(f"Saved metrics and plots to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
