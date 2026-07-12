from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
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


def _xy(df, feat_cols):
    X = df[feat_cols].values.astype(float)
    y = df["y"].values.astype(float)
    return X, y


def train(station="42002"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for target in TARGETS:
        print(f"\n=== {target} ===")
        splits = {}
        for s in ("train", "val", "test"):
            splits[s] = pd.read_csv(
                PROCESSED / f"{s}_{station}_{target}.csv", parse_dates=["timestamp"])

        feat_cols = [c for c in splits["train"].columns if c not in ("timestamp", "y")]
        Xtr, ytr = _xy(splits["train"], feat_cols)
        Xva, yva = _xy(splits["val"],   feat_cols)
        Xte, yte = _xy(splits["test"],  feat_cols)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)

        models = {
            "persistence": None,
            "ridge":            Ridge(alpha=1.0),
            "random_forest":    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting":GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

        t0_col = f"{target}_t0"
        t0_idx = feat_cols.index(t0_col)

        target_metrics = {}
        best_name, best_rmse = None, float("inf")

        for name, model in models.items():
            if name == "persistence":
                yp_va = Xva[:, t0_idx]
                yp_te = Xte[:, t0_idx]
            else:
                X_tr_in = Xtr_s if name == "ridge" else Xtr
                X_va_in = Xva_s if name == "ridge" else Xva
                X_te_in = Xte_s if name == "ridge" else Xte
                model.fit(X_tr_in, ytr)
                yp_va = model.predict(X_va_in)
                yp_te = model.predict(X_te_in)

            m = {"val": metrics(yva, yp_va), "test": metrics(yte, yp_te)}
            target_metrics[name] = m
            print(f"  {name:20s} val RMSE={m['val']['RMSE']:.4f} R2={m['val']['R2']:.4f}")

            if name != "persistence" and m["val"]["RMSE"] < best_rmse:
                best_rmse = m["val"]["RMSE"]
                best_name = name

        print(f"  -> best: {best_name}")
        best_model = models[best_name]
        best_scaler = scaler if best_name == "ridge" else None
        joblib.dump({
            "model": best_model,
            "scaler": best_scaler,
            "features": feat_cols,
            "target": target,
            "name": best_name,
        }, MODELS_DIR / f"best_{target}.joblib")

        for name, model in models.items():
            if model is not None:
                joblib.dump({
                    "model": model,
                    "scaler": scaler if name == "ridge" else None,
                    "features": feat_cols,
                    "target": target,
                    "name": name,
                }, MODELS_DIR / f"{name}_{target}.joblib")

        all_metrics[target] = {"best": best_name, "metrics": target_metrics}

    out = MODELS_DIR / "metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2))
    print(f"\n[models] saved -> {MODELS_DIR}")
    return all_metrics


if __name__ == "__main__":
    train()
