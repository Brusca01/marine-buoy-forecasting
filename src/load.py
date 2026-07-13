import sys
import io
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "Qdrant/NOAA-Buoy"
PRIMARY = "42002"
EXTRA_STATIONS = ["42001", "42039", "46042"]
EXTRA_YEARS = range(2015, 2024)
RAW = Path(__file__).parents[1] / "data" / "raw"

COLS = ["station_id", "timestamp", "WDIR", "WSPD", "GST", "WVHT",
        "DPD", "APD", "MWD", "PRES", "ATMP", "WTMP"]
NDBC_MISSING = ["MM", "99.0", "99.00", "999", "999.0", "9999.0", "99999.0"]


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


def load_primary():
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


def load_ndbc(station, years):
    import requests
    frames = []
    for yr in years:
        url = (f"https://www.ndbc.noaa.gov/view_text_file.php?"
               f"filename={station}h{yr}.txt.gz&dir=data/historical/stdmet/")
        try:
            r = requests.get(url, timeout=60)
        except Exception as e:
            print(f"  {station} {yr}: {e}"); continue
        if r.status_code != 200 or "Unable" in r.text[:200]:
            print(f"  {station} {yr}: not available"); continue
        lines = [l for l in r.text.splitlines() if l.strip()]
        if not lines: continue
        header = lines[0].lstrip("#").split()
        start = 2 if len(lines) > 1 and lines[1].lstrip().startswith("#") else 1
        raw = pd.read_csv(io.StringIO("\n".join(lines[start:])),
                          sep=r"\s+", names=header,
                          na_values=NDBC_MISSING, engine="python")
        cols = {c.upper(): c for c in raw.columns}
        yr_col = cols.get("YY", cols.get("#YY"))
        if yr_col is None: continue
        y = pd.to_numeric(raw[yr_col], errors="coerce")
        y = y.where(y > 100, y + 1900)
        raw["timestamp"] = pd.to_datetime(dict(
            year=y,
            month=pd.to_numeric(raw[cols["MM"]], errors="coerce"),
            day=pd.to_numeric(raw[cols["DD"]], errors="coerce"),
            hour=pd.to_numeric(raw.get(cols.get("HH"), 0), errors="coerce"),
        ), errors="coerce")
        frames.append(_normalize(raw, station))
        print(f"  {station} {yr}: {len(frames[-1])} rows")
    if not frames:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RAW / f"buoy_{station}.csv", index=False)
    return df


def load_all():
    frames = [load_primary()]
    print(f"\n== Loading extra stations: {EXTRA_STATIONS} ==")
    for st in EXTRA_STATIONS:
        df = load_ndbc(st, EXTRA_YEARS)
        if not df.empty:
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    out = RAW / "buoys_all.csv"
    all_df.to_csv(out, index=False)
    print(f"\n[load] total {len(all_df):,} rows, {all_df['station_id'].nunique()} stations -> {out}")
    return all_df


if __name__ == "__main__":
    load_all()
