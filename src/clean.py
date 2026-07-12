from pathlib import Path
import pandas as pd

RAW = Path(__file__).parents[1] / "data" / "raw"
PROCESSED = Path(__file__).parents[1] / "data" / "processed"

BOUNDS = {
    "WDIR": (0, 360), "WSPD": (0, 120), "GST": (0, 150),
    "WVHT": (0, 30),  "DPD":  (0, 40),  "APD": (0, 40),
    "MWD":  (0, 360), "PRES": (800, 1100), "ATMP": (-40, 55), "WTMP": (-5, 40),
}
TARGETS = ["WVHT", "WTMP"]
STATION = "42002"


def clean(freq="1h"):
    src = RAW / f"buoy_{STATION}.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run load.py first.")

    df = pd.read_csv(src, parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    print(f"[clean] loaded {len(df):,} rows")

    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = float("nan")

    num_cols = [c for c in BOUNDS if c in df.columns]
    g = df.set_index("timestamp").sort_index()
    res = g[num_cols].resample(freq).mean()
    res["station_id"] = STATION
    df = res.reset_index()
    print(f"[clean] after resample {freq}: {len(df):,} rows")

    miss = df[TARGETS].isnull().mean().round(3)
    print("[clean] missing fraction:", miss.to_dict())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / f"buoy_{STATION}_clean.csv", index=False)
    print(f"[clean] saved -> {PROCESSED}")
    return df


if __name__ == "__main__":
    clean()
