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
STATION = "42002"


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


def _profile_monthly(df):
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["timestamp"], utc=True).dt.to_period("M").astype(str)
    rows = []
    for ym, g in df.groupby("year_month"):
        row = {"year_month": ym}
        for v in PROFILE_VARS:
            if v in g.columns:
                row[f"{v}_mean"] = g[v].mean()
                row[f"{v}_std"]  = g[v].std()
        rows.append(row)
    return pd.DataFrame(rows).dropna()


def run_static(df):
    prof = _profile_monthly(df)
    feat = [c for c in prof.columns if c != "year_month"]
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
        "clusters": {str(c): prof.loc[prof["cluster"]==c, "year_month"].tolist()
                     for c in sorted(set(labels))},
        "n_points": len(prof),
    }
    return prof, info


def run_dynamic(df):
    df = df.copy()
    df["year"] = pd.to_datetime(df["timestamp"], utc=True).dt.year
    rows = []
    for yr, g in df.groupby("year"):
        row = {"year": int(yr)}
        for v in PROFILE_VARS:
            if v in g.columns:
                row[f"{v}_mean"] = g[v].mean()
                row[f"{v}_std"]  = g[v].std()
        rows.append(row)
    prof = pd.DataFrame(rows).dropna()
    if len(prof) < 3:
        return prof, {"k": 0, "silhouette": 0, "sequences": {}, "n_points": 0}
    feat = [c for c in prof.columns if c != "year"]
    Xs = _scale(prof[feat].values)
    k, silhouettes, inertias = _best_k(Xs)
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(Xs)
    sil = float(silhouette_score(Xs, labels)) if len(set(labels)) > 1 else 0.0
    pca = PCA(2, random_state=42).fit_transform(Xs)
    prof["cluster"] = labels
    prof["pca_x"] = pca[:, 0]
    prof["pca_y"] = pca[:, 1]
    prof = prof.sort_values("year")
    sequences = {
        STATION: {
            "years": prof["year"].tolist(),
            "clusters": prof["cluster"].tolist(),
            "changes": int((np.diff(prof["cluster"].values) != 0).sum()),
        }
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
    src = PROCESSED / f"buoy_{STATION}_clean.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run clean.py first.")
    df = pd.read_csv(src, parse_dates=["timestamp"])
    print(f"[clustering] buoy {STATION}: {len(df):,} rows")

    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

    static_tab, static_info = run_static(df)
    static_tab.to_csv(CLUSTER_DIR / "static.csv", index=False)
    (CLUSTER_DIR / "static.json").write_text(json.dumps(static_info, indent=2))
    print(f"[clustering] static k={static_info['k']} sil={static_info['silhouette']} points={static_info['n_points']}")

    dyn_tab, dyn_info = run_dynamic(df)
    dyn_tab.to_csv(CLUSTER_DIR / "dynamic.csv", index=False)
    (CLUSTER_DIR / "dynamic.json").write_text(json.dumps(dyn_info, indent=2))
    print(f"[clustering] dynamic k={dyn_info['k']} sil={dyn_info['silhouette']} points={dyn_info['n_points']}")

    return static_info, dyn_info


if __name__ == "__main__":
    cluster()
