import sys
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "Qdrant/NOAA-Buoy"
PRIMARY = "42002"
RAW = Path(__file__).parents[1] / "data" / "raw"

COLS = ["station_id", "timestamp", "WDIR", "WSPD", "GST", "WVHT",
        "DPD", "APD", "MWD", "PRES", "ATMP", "WTMP"]


def _read(path):
    if str(path).endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_parquet(path)


def _normalize(df, station):
    df = df.copy()
    df.columns = [str(c).strip().lstrip("#").strip() for c in df.columns]
    ts = next((c for c in df.columns if c.upper() in ("TSTMP", "TIMESTAMP")), None)
    if ts:
        df = df.rename(columns={ts: "timestamp"})
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)
    df["station_id"] = str(station)
    for c in [x for x in COLS if x not in ("station_id", "timestamp")]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df[[c for c in COLS if c in df.columns]]
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["station_id", "timestamp"])
    return df.reset_index(drop=True)


def load_all():
    files = list(HfApi().list_repo_files(HF_REPO, repo_type="dataset"))
    wanted = [f for f in files if ("full_years" in f or "full_2023" in f)
              and "zscore" not in f and "trimmed" not in f
              and not f.startswith("orig_downloads/")]
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in sorted(wanted):
        local = hf_hub_download(HF_REPO, f, repo_type="dataset")
        frames.append(_normalize(_read(local), PRIMARY))
        print(f"  {f}: {len(frames[-1])} rows")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["station_id", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out = RAW / f"buoy_{PRIMARY}.csv"
    df.to_csv(out, index=False)
    print(f"[load] {PRIMARY}: {len(df):,} rows -> {out}")
    return df


if __name__ == "__main__":
    load_all()
