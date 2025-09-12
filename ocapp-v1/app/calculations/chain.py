from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config import NUM_FIELDS
from utils.common import to_float

def group_chain(snapshot: Dict) -> Tuple[pd.DataFrame, Optional[float], List[str]]:
    options = snapshot.get("options", []) or []
    spot_raw = snapshot.get("spot", None)
    spot = to_float(spot_raw) if spot_raw is not None else None
    rows: Dict[float, Dict] = {}
    all_fields = set()
    for o in options:
        s = to_float(o.get("strike"))
        if pd.isna(s): continue
        if s not in rows: rows[s] = {"strike": s}
        side = (o.get("type") or "").upper()
        if side not in ("CE", "PE"): continue
        for k, v in o.items():
            if k in ("strike", "type"): continue
            colname = f"{side}_{k}"
            rows[s][colname] = to_float(v) if k in NUM_FIELDS else v
            all_fields.add(k)
    df = pd.DataFrame(list(rows.values())).sort_values("strike").reset_index(drop=True)
    for c in df.columns:
        if c == "strike": continue
        if c.startswith(("CE_", "PE_")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "CE_oi_change" in df.columns and "PE_oi_change" in df.columns:
        df["net_oi_change"] = df["PE_oi_change"].fillna(0) - df["CE_oi_change"].fillna(0)
        all_fields.add("net_oi_change")
    return df, spot, sorted(list(all_fields))

def find_atm_index(df: pd.DataFrame, spot: Optional[float]) -> int:
    if spot is None or df.empty: return len(df) // 2
    idx = (df["strike"] - spot).abs().idxmin()
    return int(idx)

def compute_pcr(df: pd.DataFrame) -> float:
    ce = df.get("CE_oi")
    pe = df.get("PE_oi")
    ce_sum = float(ce.fillna(0).sum()) if ce is not None else 0.0
    pe_sum = float(pe.fillna(0).sum()) if pe is not None else 0.0
    return pe_sum / ce_sum if ce_sum > 0 else float("inf")

def compute_max_pain(df: pd.DataFrame) -> Optional[float]:
    if df.empty: return None
    strikes = df["strike"].tolist()
    ce_oi = (df["CE_oi"] if "CE_oi" in df else pd.Series([0]*len(strikes))).fillna(0).tolist()
    pe_oi = (df["PE_oi"] if "PE_oi" in df else pd.Series([0]*len(strikes))).fillna(0).tolist()
    best_s, best_cost = strikes[0], float("inf")
    for s in strikes:
        cost = 0.0
        for i, k in enumerate(strikes):
            ce_intr = max(0.0, k - s)
            pe_intr = max(0.0, s - k)
            cost += ce_oi[i] * ce_intr + pe_oi[i] * pe_intr
        if cost < best_cost: best_cost, best_s = cost, s
    return best_s

def estimate_cash_spot_parity_consistent(df: pd.DataFrame, fut_spot: Optional[float], look_radius: int = 5) -> Optional[float]:
    if df.empty or fut_spot is None: return fut_spot
    if "CE_ltp" not in df.columns or "PE_ltp" not in df.columns: return fut_spot
    atm = find_atm_index(df, fut_spot)
    s = max(0, atm - look_radius); e = min(len(df), atm + look_radius + 1)
    ring = df.iloc[s:e].copy().dropna(subset=["CE_ltp","PE_ltp"])
    if ring.empty: return fut_spot
    ring["dist"] = (ring["strike"] - fut_spot).abs()
    row = ring.sort_values("dist").iloc[0]
    K, C, P = float(row["strike"]), float(row["CE_ltp"]), float(row["PE_ltp"])
    S = C - P + K
    return S if np.isfinite(S) else fut_spot

def get_writing_commentary(dfw: pd.DataFrame) -> Tuple[str, str]:
    if "net_oi_change" not in dfw.columns or dfw.empty:
        return "neutral", "No OI change data available for sentiment analysis."
    df2 = dfw[["strike", "net_oi_change"]].dropna().copy()
    if df2.empty: return "neutral", "No usable OI change points."
    pos_sum = float(df2.loc[df2["net_oi_change"] > 0, "net_oi_change"].sum())
    neg_sum = float(df2.loc[df2["net_oi_change"] < 0, "net_oi_change"].sum())
    top_support = df2.sort_values("net_oi_change", ascending=False).iloc[0]
    top_resist  = df2.sort_values("net_oi_change", ascending=True).iloc[0]
    if pos_sum > abs(neg_sum) * 1.3:
        return ("bullish", f"Put writing dominates (ΣPE−CE ≈ {pos_sum:,.0f} vs {neg_sum:,.0f}); strongest support at {top_support['strike']:.0f}.")
    elif abs(neg_sum) > pos_sum * 1.3:
        return ("bearish", f"Call writing dominates (ΣPE−CE ≈ {neg_sum:,.0f} vs {pos_sum:,.0f}); strongest resistance at {top_resist['strike']:.0f}.")
    else:
        return ("mixed", f"Mixed writing (Σ+ ≈ {pos_sum:,.0f}, Σ− ≈ {neg_sum:,.0f}); local support {top_support['strike']:.0f}, resistance {top_resist['strike']:.0f}.")

def top_table(df: pd.DataFrame, col: str, n: int = 5) -> pd.DataFrame:
    if col not in df.columns: return pd.DataFrame()
    t = df[["strike", col]].copy()
    t[col] = pd.to_numeric(t[col], errors="coerce")
    t = t.dropna(subset=[col]).sort_values(col, ascending=False).head(n)
    if t.empty: return t
    t["strike"] = pd.to_numeric(t["strike"], errors="coerce").round(0).astype("Int64")
    t[col] = t[col].round(0).astype("Int64")
    return t.reset_index(drop=True)

def build_movers_long(dfw: pd.DataFrame, n: int = 5):
    ce_ch = top_table(dfw, "CE_oi_change", n)
    pe_ch = top_table(dfw, "PE_oi_change", n)
    ce_oi = top_table(dfw, "CE_oi", n)
    pe_oi = top_table(dfw, "PE_oi", n)
    ch_long = []
    for _, r in ce_ch.iterrows():
        val = pd.to_numeric(r["CE_oi_change"], errors="coerce")
        if pd.notna(val): ch_long.append({"strike": int(r["strike"]), "value": int(val), "side": "CE", "metric": "OI Change"})
    for _, r in pe_ch.iterrows():
        val = pd.to_numeric(r["PE_oi_change"], errors="coerce")
        if pd.notna(val): ch_long.append({"strike": int(r["strike"]), "value": int(val), "side": "PE", "metric": "OI Change"})
    oi_long = []
    for _, r in ce_oi.iterrows():
        val = pd.to_numeric(r["CE_oi"], errors="coerce")
        if pd.notna(val): oi_long.append({"strike": int(r["strike"]), "value": int(val), "side": "CE", "metric": "OI"})
    for _, r in pe_oi.iterrows():
        val = pd.to_numeric(r["PE_oi"], errors="coerce")
        if pd.notna(val): oi_long.append({"strike": int(r["strike"]), "value": int(val), "side": "PE", "metric": "OI"})
    ch_df = pd.DataFrame(ch_long)
    oi_df = pd.DataFrame(oi_long)
    net = []
    if {"CE_oi_change","PE_oi_change"}.issubset(dfw.columns):
        for k, grp in dfw.groupby("strike"):
            ce = pd.to_numeric(grp.get("CE_oi_change"), errors="coerce").sum()
            pe = pd.to_numeric(grp.get("PE_oi_change"), errors="coerce").sum()
            if pd.notna(ce) and pd.notna(pe): net.append({"strike": int(k), "net_oi_change": int(pe - ce)})
    net_df = pd.DataFrame(net).sort_values("net_oi_change", ascending=False).head(n) if net else pd.DataFrame()
    return ce_ch, pe_ch, ce_oi, pe_oi, ch_df, oi_df, net_df
