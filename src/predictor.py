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

STATIONS = {
    "42002": "Buoy 42002 — Gulf of Mexico (primary)",
    "42001": "Buoy 42001 — Gulf of Mexico",
    "46042": "Buoy 46042 — Pacific (Monterey)",
}


def available_stations():
    return [{"id": k, "label": v} for k, v in STATIONS.items()]


def _load_bundle(station, target):
    for candidate in [
        MODELS_DIR / f"best_{station}_{target}.joblib",
        MODELS_DIR / f"best_42002_{target}.joblib",
        MODELS_DIR / f"best_{target}.joblib",
    ]:
        if candidate.exists():
            return joblib.load(candidate)
    raise FileNotFoundError(f"No model found for {station} {target}.")


def _predict_one(bundle, row_df):
    feat = bundle["features"]
    X = row_df[feat].values.astype(float)
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
    return float(bundle["model"].predict(X)[0])


def predict(station="42002", n_history=168):
    results = {}
    src = PROCESSED / "buoys_clean.csv"
    if not src.exists():
        raise FileNotFoundError("Clean data not found. Run clean.py first.")
    df_all = pd.read_csv(src, parse_dates=["timestamp"])

    for target in TARGETS:
        df = df_all[df_all["station_id"].astype(str) == str(station)].copy()
        if df.empty:
            raise ValueError(f"No data for station {station}")

        test_f = PROCESSED / f"test_{station}_{target}.csv"
        if not test_f.exists():
            test_f = PROCESSED / f"test_42002_{target}.csv"
        if not test_f.exists():
            raise FileNotFoundError(f"No test split found. Run features.py first.")

        test = pd.read_csv(test_f, parse_dates=["timestamp"])
        feat_cols = [c for c in test.columns if c not in ("timestamp", "y")]
        last_row = test[feat_cols].dropna().iloc[[-1]]

        bundle = _load_bundle(station, target)
        pred_val = _predict_one(bundle, last_row)
        persistence = float(last_row[f"{target}_t0"].iloc[0])

        last_ts = pd.to_datetime(test["timestamp"].iloc[-1], utc=True)
        pred_ts = last_ts + pd.Timedelta(hours=1)

        hist = df[["timestamp", target]].dropna().tail(n_history)
        history = [{"timestamp": str(t), "value": round(float(v), 3)}
                   for t, v in zip(hist["timestamp"], hist[target])]

        results[target] = {
            "station": station,
            "target": target,
            "unit": UNITS.get(target, ""),
            "model_name": bundle["name"],
            "last_timestamp": str(last_ts),
            "predicted_timestamp": str(pred_ts),
            "predicted_value": round(pred_val, 3),
            "persistence_value": round(persistence, 3),
            "history": history,
        }
    return results


def get_all_metrics():
    """Returns sklearn and automl metrics for ALL stations."""
    result = {"sklearn": {}, "automl": {}}
    for station in STATIONS:
        # sklearn
        for name in [f"metrics_{station}.json", "metrics.json"]:
            p = MODELS_DIR / name
            if p.exists():
                result["sklearn"][station] = json.loads(p.read_text())
                break
        # automl
        for name in [f"metrics_automl_{station}.json", "metrics_automl.json"]:
            p = MODELS_DIR / name
            if p.exists():
                result["automl"][station] = json.loads(p.read_text())
                break
    if not result["sklearn"]:
        raise FileNotFoundError("No metrics found. Run models.py first.")
    return result


def get_static_clusters():
    f = CLUSTER_DIR / "static.json"
    t = CLUSTER_DIR / "static.csv"
    if not f.exists():
        raise FileNotFoundError("Static clustering not found. Run clustering.py first.")
    info = json.loads(f.read_text())
    tab = pd.read_csv(t)
    info["points"] = [
        {"station_id": str(r["station_id"]), "cluster": int(r["cluster"]),
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
        {"station_id": str(r["station_id"]), "year": int(r["year"]),
         "cluster": int(r["cluster"]),
         "pca_x": round(float(r["pca_x"]), 3), "pca_y": round(float(r["pca_y"]), 3)}
        for _, r in tab.iterrows()
    ]
    return info
