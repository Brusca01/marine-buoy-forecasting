from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED = Path(__file__).parents[1] / "data" / "processed"
TARGETS = ["WVHT", "WTMP"]
LAGS = [1, 2, 3, 6, 12, 24]
ROLLS = [3, 6, 24]
HORIZON = 1
STATIONS = ["42002", "42001", "42039", "46042"]


def _build(df, target):
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    out = pd.DataFrame({"timestamp": ts})
    y = pd.to_numeric(df[target], errors="coerce")
    out[f"{target}_t0"] = y.values
    for k in LAGS:
        out[f"{target}_lag{k}"] = y.shift(k).values
    for w in ROLLS:
        out[f"{target}_roll{w}"] = y.rolling(w, min_periods=2).mean().values
    exog = [c for c in df.columns
            if c not in ("station_id", "timestamp", target)
            and pd.api.types.is_numeric_dtype(df[c])]
    for c in exog:
        s = pd.to_numeric(df[c], errors="coerce")
        out[f"{c}_t0"] = s.values
        out[f"{c}_lag1"] = s.shift(1).values
    rad = 2 * np.pi * ts.dt.hour / 24
    out["hour_sin"] = np.sin(rad).values
    out["hour_cos"] = np.cos(rad).values
    rad2 = 2 * np.pi * ts.dt.dayofyear / 365
    out["doy_sin"] = np.sin(rad2).values
    out["doy_cos"] = np.cos(rad2).values
    out["y"] = y.shift(-HORIZON).values
    return out.dropna().reset_index(drop=True)


def build_features(station):
    src = PROCESSED / "buoys_clean.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run clean.py first.")
    df_all = pd.read_csv(src, parse_dates=["timestamp"])
    df = df_all[df_all["station_id"].astype(str) == str(station)].copy()
    if df.empty:
        raise ValueError(f"No data for station {station}")

    results = {}
    for target in TARGETS:
        sup = _build(df, target)
        n = len(sup)
        i_tr = int(n * 0.70)
        i_va = int(n * 0.85)
        splits = {
            "train": sup.iloc[:i_tr],
            "val":   sup.iloc[i_tr:i_va],
            "test":  sup.iloc[i_va:],
        }
        for name, part in splits.items():
            part.to_csv(PROCESSED / f"{name}_{station}_{target}.csv", index=False)
        feat_cols = [c for c in sup.columns if c not in ("timestamp", "y")]
        results[target] = {"splits": splits, "features": feat_cols}
        print(f"[features] {station} {target}: {n} rows, {len(feat_cols)} features | "
              f"train={i_tr} val={i_va-i_tr} test={n-i_va}")
    return results


if __name__ == "__main__":
    import sys
    stations = sys.argv[1:] if len(sys.argv) > 1 else STATIONS
    for st in stations:
        try:
            build_features(st)
        except Exception as e:
            print(f"[features] {st}: {e}")
