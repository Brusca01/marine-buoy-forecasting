from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED = Path(__file__).parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).parents[1] / "models"
TARGETS = ["WVHT", "WTMP"]


def metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "MAE":  round(float(mean_absolute_error(yt, yp)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "R2":   round(float(r2_score(yt, yp)), 4),
    }


def run(station="42002", time_budget=60):
    from flaml import AutoML
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for target in TARGETS:
        print(f"\n=== AutoML {target} (budget={time_budget}s) ===")
        splits = {}
        for s in ("train", "val", "test"):
            splits[s] = pd.read_csv(
                PROCESSED / f"{s}_{station}_{target}.csv", parse_dates=["timestamp"])

        feat_cols = [c for c in splits["train"].columns if c not in ("timestamp", "y")]
        Xtr = splits["train"][feat_cols].values.astype(float)
        ytr = splits["train"]["y"].values.astype(float)
        Xva = splits["val"][feat_cols].values.astype(float)
        yva = splits["val"]["y"].values.astype(float)
        Xte = splits["test"][feat_cols].values.astype(float)
        yte = splits["test"]["y"].values.astype(float)

        automl = AutoML()
        automl.fit(
            X_train=Xtr, y_train=ytr,
            task="regression",
            metric="rmse",
            time_budget=time_budget,
            eval_method="cv",
            n_splits=5,
            split_type="time",
            seed=42,
            verbose=0,
        )

        m = {
            "val":  metrics(yva, automl.predict(Xva)),
            "test": metrics(yte, automl.predict(Xte)),
        }
        print(f"  best: {automl.best_estimator} | val RMSE={m['val']['RMSE']} R2={m['val']['R2']}")

        joblib.dump({
            "model": automl,
            "features": feat_cols,
            "target": target,
            "name": f"flaml:{automl.best_estimator}",
        }, MODELS_DIR / f"automl_{target}.joblib")

        results[target] = {
            "best_estimator": automl.best_estimator,
            "metrics": m,
        }

    out = MODELS_DIR / "metrics_automl.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[automl] saved -> {MODELS_DIR}")
    return results


if __name__ == "__main__":
    import sys
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run(time_budget=budget)
