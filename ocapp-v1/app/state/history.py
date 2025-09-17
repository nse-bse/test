from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import pandas as pd
import streamlit as st
from config import IST, now_ist

MAX_SNAPSHOTS = 120

def _hist_key(symbol: str, expiry: str) -> str:
    return f"{symbol}:{expiry}"

def ensure_live_buffers():
    if "live_buffers" not in st.session_state:
        st.session_state["live_buffers"] = {}
    return st.session_state["live_buffers"]

def push_live_ring_snapshot(symbol: str, expiry: str, ts: str, df_ring: pd.DataFrame, digest: Optional[str]) -> bool:
    bufs = ensure_live_buffers()
    key = _hist_key(symbol, expiry)
    q = bufs.get(key)
    if q is None:
        q = deque(maxlen=MAX_SNAPSHOTS)
        bufs[key] = q

    last_digest = q[-1]["digest"] if len(q) else None
    if digest is not None and digest == last_digest:
        return False

    snap = df_ring.copy()
    try:
        ts_dt = pd.to_datetime(ts)
        if ts_dt.tzinfo is None and IST:
            ts_dt = ts_dt.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        ts_dt = now_ist()
    snap["timestamp"] = ts_dt
    q.append({"digest": digest, "frame": snap})
    return True

def live_frames(symbol: str, expiry: str, n: int = 40) -> List[pd.DataFrame]:
    q = st.session_state.get("live_buffers", {}).get(_hist_key(symbol, expiry))
    if not q: return []
    return [x["frame"] for x in list(q)[-n:]]

def ensure_history_state():
    if "ring_history" not in st.session_state:
        st.session_state["ring_history"] = {}
    return st.session_state["ring_history"]

def record_ring_snapshot(symbol: str, expiry: str, ts: str, df_ring: pd.DataFrame):
    hist = ensure_history_state()
    key = _hist_key(symbol, expiry)
    wanted = [
        "strike",
        "CE_oi_change","PE_oi_change","net_oi_change",
        "CE_iv","PE_iv","CE_vega","PE_vega","CE_delta","PE_delta",
        "CE_oi","PE_oi","CE_ltp","PE_ltp"
    ]
    cols = [c for c in wanted if c in df_ring.columns]
    if "strike" not in cols:
        return
    snap = df_ring.loc[:, cols].copy()
    try:
        ts_dt = pd.to_datetime(ts)
        if ts_dt.tzinfo is None and IST:
            ts_dt = ts_dt.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        ts_dt = now_ist()
    snap["timestamp"] = ts_dt

    snap["strike"] = pd.to_numeric(snap["strike"], errors="coerce")
    for c in snap.columns:
        if c not in ("timestamp", "strike"):
            snap[c] = pd.to_numeric(snap[c], errors="coerce")

    lst = hist.get(key, [])
    lst.append(snap)
    if len(lst) > MAX_SNAPSHOTS:
        lst = lst[-MAX_SNAPSHOTS:]
    hist[key] = lst

def get_last_frames(symbol: str, expiry: str, n: int = 2) -> List[pd.DataFrame]:
    lf = live_frames(symbol, expiry, n)
    if lf:
        return lf[-n:]
    hist = ensure_history_state()
    key = _hist_key(symbol, expiry)
    lst = hist.get(key, [])
    return lst[-n:] if lst else []

def make_long_history(symbol: str, expiry: str, last_n: int = 40) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = get_last_frames(symbol, expiry, n=last_n)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    all_strikes = sorted(set(np.concatenate([f["strike"].dropna().unique() for f in frames])))
    aligned = []
    for f in frames:
        g = f.set_index("strike").reindex(all_strikes)
        g["timestamp"] = f["timestamp"].iloc[0]
        aligned.append(g.reset_index())
    long_df = pd.concat(aligned, ignore_index=True)
    vel_df = pd.DataFrame()
    if len(frames) >= 2 and "net_oi_change" in long_df.columns:
        latest_ts = frames[-1]["timestamp"].iloc[0]; prev_ts = frames[-2]["timestamp"].iloc[0]
        latest = long_df[long_df["timestamp"] == latest_ts][["strike","net_oi_change"]].set_index("strike")
        prev = long_df[long_df["timestamp"] == prev_ts][["strike","net_oi_change"]].set_index("strike")
        vel = (latest["net_oi_change"] - prev["net_oi_change"]).rename("velocity").to_frame().reset_index()
        vel_df = vel.dropna()
    return long_df, vel_df

def get_history_between(symbol: str, expiry: str, start_ts, end_ts):
    hist = ensure_history_state()
    key = _hist_key(symbol, expiry)
    lst = hist.get(key, [])
    if not lst: return []
    frames = []
    for f in lst:
        ts = f["timestamp"].iloc[0]
        if isinstance(ts, str): ts = pd.to_datetime(ts)
        if start_ts <= ts <= end_ts: frames.append(f)
    return frames

def velocity_spike_table(vel_df: pd.DataFrame, k: int = 5, threshold: Optional[float] = None) -> pd.DataFrame:
    if vel_df.empty: return pd.DataFrame()
    df = vel_df.copy()
    if threshold is not None:
        df = df[(df["velocity"].abs() >= threshold)]
        if df.empty: return df
    top_support = df.sort_values("velocity", ascending=False).head(k).copy()
    top_resist = df.sort_values("velocity", ascending=True).head(k).copy()
    out = pd.concat([top_support, top_resist], ignore_index=True)
    out["Label"] = out["velocity"].apply(lambda v: "🟢 Support surge" if v > 0 else "🔴 Resistance surge")
    out["velocity"] = out["velocity"].round(0).astype("Int64")
    out["strike"] = out["strike"].round(0).astype("Int64")
    return out[["strike","velocity","Label"]]
# -------------------- Disk persistence (added) --------------------
import os, re, glob
from datetime import datetime

def _dh_safe_dir(base: str, sym: str, exp: str, ts_iso: str) -> str:
    """
    Builds: data/live/<SYM>/<EXP>/<YYYY-MM-DD>/
    """
    sym = re.sub(r"[^A-Z0-9_]+", "_", str(sym).upper())
    exp = re.sub(r"[^A-Z0-9_]+", "_", str(exp).upper())
    try:
        d = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00")).date().isoformat()
    except Exception:
        d = datetime.now().date().isoformat()
    return os.path.join("data", "live", sym, exp, d)

def persist_live_ring_snapshot(sym: str, exp: str, ts_iso: str, df_ring: pd.DataFrame) -> str | None:
    """
    Save the ring-level DataFrame for a live snapshot to disk as Parquet.
    Returns the filepath or None on failure.
    """
    try:
        outdir = _dh_safe_dir("data/live", sym, exp, ts_iso)
        os.makedirs(outdir, exist_ok=True)
        try:
            t = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00")).strftime("%H%M%S")
        except Exception:
            t = datetime.now().strftime("%H%M%S")
        fp = os.path.join(outdir, f"{t}.parquet")
        df_ring.to_parquet(fp, index=False)
        return fp
    except Exception:
        return None

def load_live_ring_history(sym: str, exp: str, date_iso: str | None = None, limit: int | None = None) -> list[pd.DataFrame]:
    """
    Load today's (or a given date's) saved ring snapshots from disk (Parquet files).
    Returns a list of DataFrames ordered by time.
    """
    if date_iso is None:
        date_iso = datetime.now().date().isoformat()
    base = os.path.join("data", "live", re.sub(r"[^A-Z0-9_]+","_", sym.upper()), re.sub(r"[^A-Z0-9_]+","_", exp.upper()), date_iso)
    files = sorted(glob.glob(os.path.join(base, "*.parquet")))
    if limit is not None and limit > 0:
        files = files[-limit:]
    out = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # best-effort timestamp from filename
            ts = os.path.splitext(os.path.basename(f))[0]  # HHMMSS
            df = df.copy()
            df["timestamp"] = ts
            out.append(df)
        except Exception:
            continue
    return out

def hydrate_inmemory_from_disk(sym: str, exp: str, limit: int = 60):
    """
    Optional: re-populate the in-memory history using the files saved to disk,
    so existing analytics that read the in-memory buffers continue to work.
    """
    frames = load_live_ring_history(sym, exp, limit=limit)
    # Reuse existing helper if available:
    try:
        for df in frames:
            ts = df["timestamp"].iloc[0] if "timestamp" in df.columns and len(df) else ""
            # record_ring_snapshot stores history for non-live; push_live_ring_snapshot is what live mode uses.
            # Use whichever your app expects for live buffers:
            try:
                push_live_ring_snapshot(sym, exp, ts, df, digest=None)  # noqa: F821 (exists in this module)
            except Exception:
                try:
                    record_ring_snapshot(sym, exp, ts, df)  # noqa: F821
                except Exception:
                    pass
    except Exception:
        pass
