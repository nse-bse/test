from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from .chain import find_atm_index
from state.history import get_last_frames

def _nearest_by_delta(df: pd.DataFrame, side: str, target: float) -> Optional[pd.Series]:
    col = f"{side}_delta"
    if col not in df.columns or df.empty: return None
    g = df[[col, "strike"]].copy()
    g["err"] = (g[col] - target).abs()
    g = g.sort_values("err")
    if g.empty: return None
    k = g.iloc[0]["strike"]
    try: return df.loc[df["strike"] == k].iloc[0]
    except Exception: return None

def compute_atm_iv(df: pd.DataFrame, spot: Optional[float]) -> Optional[float]:
    ce = _nearest_by_delta(df, "CE", 0.50)
    pe = _nearest_by_delta(df, "PE", -0.50)
    ivs = []
    if ce is not None and "CE_iv" in ce and pd.notna(ce["CE_iv"]): ivs.append(float(ce["CE_iv"]))
    if pe is not None and "PE_iv" in pe and pd.notna(pe["PE_iv"]): ivs.append(float(pe["PE_iv"]))
    if ivs: return float(np.nanmean(ivs))
    if spot is None or "strike" not in df.columns or df.empty: return None
    i = find_atm_index(df, spot)
    ivs = []
    if "CE_iv" in df.columns and pd.notna(df.loc[i, "CE_iv"]): ivs.append(float(df.loc[i, "CE_iv"]))
    if "PE_iv" in df.columns and pd.notna(df.loc[i, "PE_iv"]): ivs.append(float(df.loc[i, "PE_iv"]))
    return float(np.nanmean(ivs)) if ivs else None

def compute_rr_bf(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    put25 = _nearest_by_delta(df, "PE", -0.25)
    call25 = _nearest_by_delta(df, "CE", 0.25)
    atm_iv = compute_atm_iv(df, None)
    if put25 is None or call25 is None or atm_iv is None:
        return None, None
    s_put = float(put25.get("PE_iv", np.nan))
    s_call = float(call25.get("CE_iv", np.nan))
    if not np.isfinite(s_put) or not np.isfinite(s_call):
        return None, None
    rr = s_put - s_call
    bf = 0.5*(s_put + s_call) - float(atm_iv)
    return float(rr), float(bf)

def compute_vega_weighted_iv_flow(df_now: pd.DataFrame, df_prev: Optional[pd.DataFrame]):
    if df_now.empty or "strike" not in df_now.columns or df_prev is None:
        return pd.DataFrame(), 0.0

    def side_flow(side: str) -> pd.DataFrame:
        iv_col, vega_col, oi_col = f"{side}_iv", f"{side}_vega", f"{side}_oi"
        need = {"strike", iv_col, vega_col, oi_col}
        if not need.issubset(df_now.columns):
            return pd.DataFrame(columns=["strike","flow","side"])
        cur = df_now.loc[:, list(need)].copy().rename(columns={iv_col:"iv", vega_col:"vega", oi_col:"oi"})
        prev = df_prev.loc[:, list(need)].copy().rename(columns={iv_col:"iv", vega_col:"vega", oi_col:"oi"}).set_index("strike")
        cur["iv_prev"] = cur["strike"].map(prev["iv"])
        cur["d_iv"] = pd.to_numeric(cur["iv"] - cur["iv_prev"], errors="coerce").fillna(0.0)
        cur["flow"] = pd.to_numeric(cur["vega"], errors="coerce").fillna(0.0) * \
                      pd.to_numeric(cur["oi"], errors="coerce").fillna(0.0) * cur["d_iv"]
        out = cur[["strike","flow"]].copy()
        out["side"] = side
        return out

    ce = side_flow("CE"); pe = side_flow("PE")
    flows = pd.concat([ce, pe], ignore_index=True)
    flows = flows.dropna(subset=["strike"]).copy()
    flows["strike"] = flows["strike"].round(0).astype("Int64")
    net = float(flows["flow"].sum()) if not flows.empty else 0.0
    return flows, net

def make_iv_skew_history(symbol: str, expiry: str, last_n: int = 60) -> pd.DataFrame:
    frames = get_last_frames(symbol, expiry, n=last_n)
    if not frames: return pd.DataFrame()
    rows = []
    for f in frames:
        ts = f["timestamp"].iloc[0]
        atm_iv = compute_atm_iv(f, None)
        from .iv_skew import compute_rr_bf as _rrbf  # local to avoid circular
        rr, bf = _rrbf(f)
        rows.append({"timestamp": ts, "atm_iv": atm_iv, "rr": rr, "bf": bf})
    return pd.DataFrame(rows).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

def zscore_last(series: pd.Series, window: int = 20) -> Optional[float]:
    if series is None or len(series) < 3: return None
    s = series.dropna()
    if len(s) < 3: return None
    tail = s.iloc[-window:] if len(s) >= window else s
    mu, sd = tail.mean(), tail.std(ddof=0)
    if not np.isfinite(sd) or sd == 0: return 0.0
    return float((s.iloc[-1] - mu) / sd)
