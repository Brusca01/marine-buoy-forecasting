from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

PROCESSED = Path(__file__).parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).parents[1] / "models"
CLUSTER_DIR = PROCESSED / "clustering"
TARGETS = ["WVHT", "WTMP"]
UNITS = {"WVHT": "m", "WTMP": "°C"}
STATION = "42002"


def _load_bundle(target):
    path = MODELS_DIR / f"best_{target}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run models.py first.")
    return joblib.load(path)


def _predict_one(bundle, row_df):
    feat = bundle["features"]
    X = row_df[feat].values.astype(float)
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
    return float(bundle["model"].predict(X)[0])


def predict(station=STATION, n_history=168):
    results = {}
    for target in TARGETS:
        clean_f = PROCESSED / f"buoy_{station}_clean.csv"
        if not clean_f.exists():
            raise FileNotFoundError(f"{clean_f} not found. Run clean.py first.")
        df = pd.read_csv(clean_f, parse_dates=["timestamp"])

        test_f = PROCESSED / f"test_{station}_{target}.csv"
        if not test_f.exists():
            raise FileNotFoundError(f"{test_f} not found. Run features.py first.")
        test = pd.read_csv(test_f, parse_dates=["timestamp"])
        feat_cols = [c for c in test.columns if c not in ("timestamp", "y")]
        last_row = test[feat_cols].dropna().iloc[[-1]]

        bundle = _load_bundle(target)
        pred_val = _predict_one(bundle, last_row)
        persistence = float(last_row[f"{target}_t0"].iloc[0])

        last_ts = pd.to_datetime(test["timestamp"].iloc[-1], utc=True)
        pred_ts = last_ts + pd.Timedelta(hours=1)

        hist = df[["timestamp", target]].dropna().tail(n_history)
        history = [{"timestamp": str(t), "value": round(float(v), 3)}
                   for t, v in zip(hist["timestamp"], hist[target])]

        results[target] = {
            "station": station, "target": target, "unit": UNITS.get(target, ""),
            "model_name": bundle["name"],
            "last_timestamp": str(last_ts),
            "predicted_timestamp": str(pred_ts),
            "predicted_value": round(pred_val, 3),
            "persistence_value": round(persistence, 3),
            "history": history,
        }
    return results


def get_metrics():
    results = {}
    m_path = MODELS_DIR / "metrics.json"
    if m_path.exists():
        results["sklearn"] = json.loads(m_path.read_text())
    a_path = MODELS_DIR / "metrics_automl.json"
    if a_path.exists():
        results["automl"] = json.loads(a_path.read_text())
    if not results:
        raise FileNotFoundError("No metrics found. Run models.py and automl.py first.")
    return results


def get_static_clusters():
    f = CLUSTER_DIR / "static.json"
    t = CLUSTER_DIR / "static.csv"
    if not f.exists():
        raise FileNotFoundError("Static clustering not found. Run clustering.py first.")
    info = json.loads(f.read_text())
    tab = pd.read_csv(t)
    info["points"] = [
        {"year_month": str(r["year_month"]), "cluster": int(r["cluster"]),
         "pca_x": round(float(r["pca_x"]), 3), "pca_y": round(float(r["pca_y"]), 3)}
        for _, r in tab.iterrows()
    ]
    return info


def get_dynamic_clusters():
    f = CLUSTER_DIR / "dynamic.json"
    t = CLUSTER_DIR / "dynamic.csv"
    if not f.exists():
        raise FileNotFoundError("Dynamic clustering not found. Run clustering.py first.")
    info = json.loads(f.read_text())
    tab = pd.read_csv(t)
    info["points"] = [
        {"year": int(r["year"]), "cluster": int(r["cluster"]),
         "pca_x": round(float(r["pca_x"]), 3), "pca_y": round(float(r["pca_y"]), 3)}
        for _, r in tab.iterrows()
    ]
    return info
