from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

PROCESSED = Path(__file__).parents[1] / "data" / "processed"
CLUSTER_DIR = PROCESSED / "clustering"
PROFILE_VARS = ["WVHT", "WTMP", "WSPD", "PRES", "ATMP"]


def _scale(X):
    return StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))


def _best_k(Xs, kmin=2, kmax=8):
    kmax = min(kmax, len(Xs) - 1)
    silhouettes, inertias = {}, {}
    for k in range(kmin, kmax + 1):
        km = KMeans(k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        inertias[k] = round(float(km.inertia_), 2)
        if len(set(labels)) > 1:
            silhouettes[k] = round(float(silhouette_score(Xs, labels)), 3)
    best_k = max(silhouettes, key=silhouettes.get) if silhouettes else kmin
    return best_k, silhouettes, inertias


def _profile(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for v in PROFILE_VARS:
            if v in g.columns:
                row[f"{v}_mean"] = g[v].mean()
                row[f"{v}_std"]  = g[v].std()
        rows.append(row)
    return pd.DataFrame(rows).dropna()


def run_static(df):
    prof = _profile(df, ["station_id"])
    feat = [c for c in prof.columns if c != "station_id"]
    Xs = _scale(prof[feat].values)
    k, silhouettes, inertias = _best_k(Xs)
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(Xs)
    sil = float(silhouette_score(Xs, labels)) if len(set(labels)) > 1 else 0.0
    pca = PCA(2, random_state=42).fit_transform(Xs)
    prof["cluster"] = labels
    prof["pca_x"] = pca[:, 0]
    prof["pca_y"] = pca[:, 1]
    info = {
        "k": int(k),
        "silhouette": round(sil, 3),
        "silhouette_by_k": silhouettes,
        "inertia_by_k": inertias,
        "n_stations": len(prof),
        "clusters": {str(c): prof.loc[prof["cluster"]==c, "station_id"].tolist()
                     for c in sorted(set(labels))},
    }
    return prof, info


def run_dynamic(df):
    df = df.copy()
    df["year"] = pd.to_datetime(df["timestamp"], utc=True).dt.year
    prof = _profile(df, ["station_id", "year"])
    feat = [c for c in prof.columns if c not in ["station_id", "year"]]
    if len(prof) < 3:
        return prof, {"k": 0, "silhouette": 0, "sequences": {}, "n_points": 0}
    Xs = _scale(prof[feat].values)
    k, silhouettes, inertias = _best_k(Xs)
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(Xs)
    sil = float(silhouette_score(Xs, labels)) if len(set(labels)) > 1 else 0.0
    pca = PCA(2, random_state=42).fit_transform(Xs)
    prof["cluster"] = labels
    prof["pca_x"] = pca[:, 0]
    prof["pca_y"] = pca[:, 1]
    sequences = {}
    for st, g in prof.groupby("station_id"):
        g = g.sort_values("year")
        sequences[str(st)] = {
            "years": g["year"].tolist(),
            "clusters": g["cluster"].tolist(),
            "changes": int((np.diff(g["cluster"].values) != 0).sum()),
        }
    info = {
        "k": int(k),
        "silhouette": round(sil, 3),
        "silhouette_by_k": silhouettes,
        "inertia_by_k": inertias,
        "n_points": len(prof),
        "sequences": sequences,
    }
    return prof, info


def cluster():
    src = PROCESSED / "buoys_clean.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run clean.py first.")
    df = pd.read_csv(src, parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    print(f"[clustering] {df['station_id'].nunique()} stations")

    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

    static_tab, static_info = run_static(df)
    static_tab.to_csv(CLUSTER_DIR / "static.csv", index=False)
    (CLUSTER_DIR / "static.json").write_text(json.dumps(static_info, indent=2))
    print(f"[clustering] static k={static_info['k']} sil={static_info['silhouette']}")
    for c, stations in static_info["clusters"].items():
        print(f"   cluster {c}: {stations}")

    dyn_tab, dyn_info = run_dynamic(df)
    dyn_tab.to_csv(CLUSTER_DIR / "dynamic.csv", index=False)
    (CLUSTER_DIR / "dynamic.json").write_text(json.dumps(dyn_info, indent=2))
    print(f"[clustering] dynamic k={dyn_info['k']} sil={dyn_info['silhouette']}")

    return static_info, dyn_info


if __name__ == "__main__":
    cluster()
