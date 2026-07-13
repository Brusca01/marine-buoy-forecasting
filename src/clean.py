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
PRIMARY = "42002"


def clean(freq="1h"):
    src = RAW / "buoys_all.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run load.py first.")

    df = pd.read_csv(src, parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    print(f"[clean] loaded {len(df):,} rows, {df['station_id'].nunique()} stations")

    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = float("nan")

    num_cols = [c for c in BOUNDS if c in df.columns]
    frames = []
    for st, g in df.groupby("station_id"):
        g = g.set_index("timestamp").sort_index()
        res = g[num_cols].resample(freq).mean()
        res["station_id"] = st
        frames.append(res.reset_index())
    df = pd.concat(frames, ignore_index=True)
    print(f"[clean] after resample {freq}: {len(df):,} rows")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "buoys_clean.csv", index=False)
    primary = df[df["station_id"] == PRIMARY].copy()
    primary.to_csv(PROCESSED / f"buoy_{PRIMARY}_clean.csv", index=False)
    print(f"[clean] saved -> {PROCESSED}")
    return df


if __name__ == "__main__":
    clean()
