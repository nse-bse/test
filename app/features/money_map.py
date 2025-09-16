from typing import Optional, Tuple
import numpy as np
import pandas as pd

def compute_money_map(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    g = df[["strike"]].copy()

    def S(col, absval=False):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
        else:
            s = pd.Series(0.0, index=g.index, dtype="float64")
        return s.abs() if absval else s

    ce_ltp, pe_ltp = S("CE_ltp"), S("PE_ltp")
    ce_oi,  pe_oi  = S("CE_oi"),  S("PE_oi")
    ce_ch,  pe_ch  = S("CE_oi_change"), S("PE_oi_change")
    ce_g,   pe_g   = S("CE_gamma", absval=True), S("PE_gamma", absval=True)
    ce_vg,  pe_vg  = S("CE_vega",  absval=True), S("PE_vega",  absval=True)
    ce_vol, pe_vol = S("CE_volume"), S("PE_volume")

    g["tot_oi"]    = ce_oi.fillna(0) + pe_oi.fillna(0)
    g["tot_oi_ch"] = ce_ch.fillna(0) + pe_ch.fillna(0)
    g["prem_stock"] = ce_ltp.fillna(0)*ce_oi.fillna(0) + pe_ltp.fillna(0)*pe_oi.fillna(0)
    g["prem_new"]   = ce_ltp.fillna(0)*ce_ch.fillna(0) + pe_ltp.fillna(0)*pe_ch.fillna(0)
    g["gamma_dlr"]  = ce_g.fillna(0)*ce_oi.fillna(0) + pe_g.fillna(0)*pe_oi.fillna(0)
    g["vega_dlr"]   = ce_vg.fillna(0)*ce_oi.fillna(0) + pe_vg.fillna(0)*pe_oi.fillna(0)
    vol_tot = ce_vol.fillna(0) + pe_vol.fillna(0)
    g["turnover_ratio"] = vol_tot / g["tot_oi"].replace(0, np.nan)

    g = g.sort_values("strike").reset_index(drop=True)
    g["oi_roll3"] = g["tot_oi"].rolling(3, center=True, min_periods=1).sum()
    sd = g["oi_roll3"].std(ddof=0)
    g["wall_z"] = (g["oi_roll3"] - g["oi_roll3"].mean())/sd if sd and np.isfinite(sd) and sd>0 else 0.0
    return g

def money_center_of_mass(dfm: pd.DataFrame, use_premium: bool=True) -> Optional[float]:
    if dfm.empty: return None
    w = dfm["prem_stock"] if use_premium and "prem_stock" in dfm else dfm["tot_oi"]
    if w.fillna(0).sum() <= 0: return None
    return float((dfm["strike"]*w).sum() / w.sum())

def find_walls(
    dfm: pd.DataFrame,
    z_thresh: float = 1.5,
    k: int = 4,
    spot: Optional[float] = None
) -> pd.DataFrame:
    """
    Detect OI/premium clusters ("walls") and classify as support / resistance / magnet.

    Expects dfm to have at least: strike, wall_z, tot_oi, prem_stock.
    Optionally uses any of: CE_oi, PE_oi, CE_ltp, PE_ltp, CE_prem_stock, PE_prem_stock.
    """
    if dfm.empty or "wall_z" not in dfm.columns:
        return pd.DataFrame()

    base_cols = ["strike", "wall_z", "tot_oi", "prem_stock"]
    have = [c for c in ["CE_oi","PE_oi","CE_ltp","PE_ltp","CE_prem_stock","PE_prem_stock"] if c in dfm.columns]
    cand = dfm[dfm["wall_z"] >= z_thresh][base_cols + have].copy()
    if cand.empty:
        return pd.DataFrame(columns=base_cols + ["CE_prem","PE_prem","type","type_reason","type_confidence"])

    # --- CE/PE premium components with fallbacks ---
    def _num(s): return pd.to_numeric(s, errors="coerce").fillna(0.0)

    if {"CE_prem_stock","PE_prem_stock"}.issubset(cand.columns):
        ce_prem = _num(cand["CE_prem_stock"])
        pe_prem = _num(cand["PE_prem_stock"])
    elif {"CE_oi","CE_ltp","PE_oi","PE_ltp"}.issubset(cand.columns):
        ce_prem = _num(cand["CE_oi"]) * _num(cand["CE_ltp"])
        pe_prem = _num(cand["PE_oi"]) * _num(cand["PE_ltp"])
    elif {"CE_oi","PE_oi"}.issubset(cand.columns):
        # Proxy using OI if LTP unavailable
        ce_prem = _num(cand["CE_oi"])
        pe_prem = _num(cand["PE_oi"])
    else:
        # Last resort: split prem_stock evenly → gives mixed → low confidence
        half = _num(cand["prem_stock"]) * 0.5
        ce_prem = half
        pe_prem = half

    cand["CE_prem"] = ce_prem
    cand["PE_prem"] = pe_prem

    # --- take top-k by strength ---
    cand = cand.sort_values(["wall_z","prem_stock","tot_oi"], ascending=False).head(k).copy()

    # --- strike step for "near spot" ---
    try:
        step = float(np.median(np.diff(np.sort(dfm["strike"].dropna().unique()))))
    except Exception:
        step = 100.0

    # --- classification ---
    types, reasons, confs = [], [], []
    for _, r in cand.iterrows():
        k_strike = float(r["strike"])
        ce, pe = float(r["CE_prem"]), float(r["PE_prem"])
        tot = ce + pe
        dom = (abs(ce - pe) / tot) if tot > 0 else 0.0  # dominance 0..1
        conf = float(np.clip(dom, 0.0, 1.0))

        if spot is None or not np.isfinite(spot):
            # No spot → only composition available
            if pe > ce:
                t, reason = "support", "PE-dominated (no spot)"
            elif ce > pe:
                t, reason = "resistance", "CE-dominated (no spot)"
            else:
                t, reason, conf = "neutral", "mixed (no spot)", 0.25
        else:
            near = abs(k_strike - spot) <= 0.5 * step
            if near:
                t, reason, conf = "magnet", "near spot", max(conf, 0.4)
            elif (pe >= ce) and (k_strike < spot):
                t, reason, conf = "support", "PE-dominated below spot", min(1.0, conf + 0.3)
            elif (ce >= pe) and (k_strike > spot):
                t, reason, conf = "resistance", "CE-dominated above spot", min(1.0, conf + 0.3)
            elif pe > ce and k_strike > spot:
                t, reason, conf = "support", "PE-dominated but above spot", max(0.2, conf * 0.6)
            elif ce > pe and k_strike < spot:
                t, reason, conf = "resistance", "CE-dominated but below spot", max(0.2, conf * 0.6)
            else:
                t, reason, conf = "neutral", "mixed", 0.25

        types.append(t); reasons.append(reason); confs.append(conf)

    cand["type"] = types
    cand["type_reason"] = reasons
    cand["type_confidence"] = confs

    # Clean display names (optional)
    cand = cand.rename(columns={"prem_stock": "premium", "wall_z": "wall_z (σ)"})

    return cand.reset_index(drop=True)
def make_money_map_commentary(dfm: pd.DataFrame, spot: Optional[float]) -> str:
    if dfm.empty: return "Money Map not available."
    parts = []
    cm_p = money_center_of_mass(dfm, use_premium=True)
    cm_o = money_center_of_mass(dfm, use_premium=False)
    if cm_p is not None and cm_o is not None:
        parts.append(f"Magnet vs positioning: Premium Center ≈ {cm_p:.0f}, OI Center ≈ {cm_o:.0f}.")
        if spot is not None:
            if abs(spot - cm_p) <= 40:
                parts.append(f"Spot {spot:.0f} → **pin/chop near magnet**.")
            elif spot > cm_p:
                parts.append(f"Spot {spot:.0f} → **gravity slightly down** toward magnet unless momentum flips.")
            else:
                parts.append(f"Spot {spot:.0f} → **gravity slightly up** toward magnet unless momentum flips.")
    walls_tbl = find_walls(dfm, z_thresh=1.5, k=4)
    if not walls_tbl.empty:
        walls_list = walls_tbl["strike"].astype(int).astype(str).tolist()
        parts.append("Sticky walls: " + " / ".join(walls_list) + ".")
    if "prem_new" in dfm.columns and spot is not None:
        above = dfm.loc[dfm["strike"] >= spot, "prem_new"].fillna(0).sum()
        below = dfm.loc[dfm["strike"] <  spot, "prem_new"].fillna(0).sum()
        tot = above + below
        if tot > 0:
            share_above = above / tot
            if share_above >= 0.60: parts.append("New premium **concentrated overhead** → ceiling bias unless those calls unwind.")
            elif share_above <= 0.40: parts.append("New premium **concentrated below** → support bias unless those puts unwind.")
            else: parts.append("New premium **balanced** across strikes.")
    hot = []
    if "gamma_dlr" in dfm.columns and dfm["gamma_dlr"].notna().any():
        hot.append(int(dfm.loc[dfm["gamma_dlr"].idxmax(), "strike"]))
    if "vega_dlr" in dfm.columns and dfm["vega_dlr"].notna().any():
        vk = int(dfm.loc[dfm["vega_dlr"].idxmax(), "strike"])
        if vk not in hot: hot.append(vk)
    if hot:
        parts.append("Dealer hedging hotspots near " + " / ".join(map(str, hot)) + ".")
    if spot is not None:
        up_w = None; dn_w = None
        if not walls_tbl.empty:
            ups = [int(k) for k in walls_tbl["strike"] if k >= spot]
            dns = [int(k) for k in walls_tbl["strike"] if k <= spot]
            up_w = ups[0] if ups else None
            dn_w = dns[-1] if dns else None
        if up_w and dn_w:
            parts.append(f"Base case: **range/pin {dn_w}–{up_w}**; fade edges while overhead writing persists.")
    return " ".join(parts) if parts else "No clear Money Map read yet."
