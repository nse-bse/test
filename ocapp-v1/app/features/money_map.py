# money_map.py
# ---------------------------------------------
# Strike-level "Money Map" utilities
#
# Exposed functions (compatible with your code):
#   - compute_money_map(df)
#   - money_center_of_mass(dfm, use_premium=True)
#   - find_walls(dfm, z_thresh=1.5, k=4, spot=None, k_each=None)
#   - make_money_map_commentary(dfm, spot)
#
# Notes
# -----
# • prem_stock  = CE_mid*CE_oi + PE_mid*PE_oi      → mass (not fresh cash)
# • prem_new    = CE_mid*ΔCE_oi + PE_mid*ΔPE_oi     → signed flow proxy (+in, −out)
# • CE/PE legs are propagated so we *never* need 50/50 splits unless data is missing.
# • "gamma_dlr"/"vega_dlr" remain magnitude indices (not rupees).
# • find_walls() uses price-consistent rules:
#     resistance ⇒ CE-dominated & strike > spot
#     support    ⇒ PE-dominated & strike < spot
#     near spot  ⇒ magnet
#     CE-dom < spot ⇒ neutral/spent overhang (optionally demoted if covering)
# • Optional k_each lets you take top-k above and below spot separately
#   without breaking existing calls (defaults to global top-k).
# ---------------------------------------------

from typing import Optional
import numpy as np
import pandas as pd


# --------------------------
# Core builders
# --------------------------

def compute_money_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build strike-level features used by Money Map.

    Inputs (wide chain; missing cols tolerated):
      strike, CE/PE: ltp, bid, ask, oi, oi_change, volume, gamma, vega

    Returns (no rounding):
      strike, CE_mid, PE_mid, CE_ltp, PE_ltp, CE_oi, PE_oi,
      CE_oi_change, PE_oi_change,
      CE_prem_stock, PE_prem_stock, prem_stock, prem_new,
      tot_oi, tot_oi_ch, gamma_dlr, vega_dlr, turnover_ratio,
      oi_roll3, wall_z
    """
    if df.empty:
        return pd.DataFrame()

    g = df[["strike"]].copy()

    def S(col: str, absval: bool = False) -> pd.Series:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
        else:
            s = pd.Series(0.0, index=g.index, dtype="float64")
        return s.abs() if absval else s

    # Base legs
    ce_ltp, pe_ltp = S("CE_ltp"), S("PE_ltp")
    ce_bid, pe_bid = S("CE_bid"), S("PE_bid")
    ce_ask, pe_ask = S("CE_ask"), S("PE_ask")
    ce_oi,  pe_oi  = S("CE_oi"),  S("PE_oi")
    ce_ch,  pe_ch  = S("CE_oi_change"), S("PE_oi_change")
    ce_g,   pe_g   = S("CE_gamma", absval=True), S("PE_gamma", absval=True)
    ce_vg,  pe_vg  = S("CE_vega",  absval=True), S("PE_vega",  absval=True)
    ce_vol, pe_vol = S("CE_volume"), S("PE_volume")

    # Prefer MID (bid/ask) when both sides are present and >0; else fall back to LTP
    def mid_series(b: pd.Series, a: pd.Series, ltp: pd.Series) -> pd.Series:
        m = (b + a) / 2.0
        use_mid = (b > 0) & (a > 0)
        out = pd.Series(np.where(use_mid, m, ltp), index=g.index, dtype="float64")
        return out.fillna(0.0)

    CE_mid = mid_series(ce_bid, ce_ask, ce_ltp)
    PE_mid = mid_series(pe_bid, pe_ask, pe_ltp)

    # Expose legs downstream (prevents even-split later)
    g["CE_ltp"], g["PE_ltp"] = ce_ltp.fillna(0.0), pe_ltp.fillna(0.0)
    g["CE_mid"], g["PE_mid"] = CE_mid, PE_mid
    g["CE_oi"],  g["PE_oi"]  = ce_oi.fillna(0.0),  pe_oi.fillna(0.0)
    g["CE_oi_change"], g["PE_oi_change"] = ce_ch.fillna(0.0), pe_ch.fillna(0.0)

    # Totals
    g["tot_oi"]    = g["CE_oi"] + g["PE_oi"]
    g["tot_oi_ch"] = g["CE_oi_change"] + g["PE_oi_change"]

    # Leg-specific premium stock and aggregate (mass)
    g["CE_prem_stock"] = g["CE_mid"] * g["CE_oi"]
    g["PE_prem_stock"] = g["PE_mid"] * g["PE_oi"]
    g["prem_stock"]    = g["CE_prem_stock"] + g["PE_prem_stock"]

    # "New" premium proxy = MID × ΔOI (signed; +inflow / −outflow)
    g["prem_new"] = g["CE_mid"] * g["CE_oi_change"] + g["PE_mid"] * g["PE_oi_change"]

    # Magnitude indices (kept for compatibility)
    g["gamma_dlr"] = ce_g.fillna(0.0) * g["CE_oi"] + pe_g.fillna(0.0) * g["PE_oi"]
    g["vega_dlr"]  = ce_vg.fillna(0.0) * g["CE_oi"] + pe_vg.fillna(0.0) * g["PE_oi"]

    # Turnover / OI (guard divide-by-zero)
    vol_tot = ce_vol.fillna(0.0) + pe_vol.fillna(0.0)
    g["turnover_ratio"] = vol_tot / g["tot_oi"].replace(0, np.nan)

    # Rolling cluster metric and z-score
    g = g.sort_values("strike").reset_index(drop=True)
    g["oi_roll3"] = g["tot_oi"].rolling(3, center=True, min_periods=1).sum()
    sd = float(g["oi_roll3"].std(ddof=0))
    g["wall_z"] = (g["oi_roll3"] - g["oi_roll3"].mean()) / sd if (sd and np.isfinite(sd) and sd > 0) else 0.0

    return g


def money_center_of_mass(dfm: pd.DataFrame, use_premium: bool = True) -> Optional[float]:
    """
    Center of mass by premium (prem_stock) or by OI (tot_oi).
    """
    if dfm.empty:
        return None
    w = dfm["prem_stock"] if (use_premium and "prem_stock" in dfm) else dfm["tot_oi"]
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    if w.sum() <= 0:
        return None
    strikes = pd.to_numeric(dfm["strike"], errors="coerce")
    return float((strikes * w).sum() / w.sum())


# --------------------------
# Walls / classification
# --------------------------

def _median_strike_step(strikes: pd.Series) -> float:
    try:
        arr = np.sort(pd.to_numeric(strikes, errors="coerce").dropna().unique())
        diffs = np.diff(arr)
        step = float(np.median(diffs)) if len(diffs) else 0.0
        return step if (np.isfinite(step) and step > 0) else 100.0
    except Exception:
        return 100.0


def find_walls(
    dfm: pd.DataFrame,
    z_thresh: float = 1.5,
    k: int = 4,
    spot: Optional[float] = None,
    k_each: Optional[int] = None,  # optional: top-k above and below spot
) -> pd.DataFrame:
    """
    Detect OI/premium clusters ("walls") and classify as support / resistance / magnet.

    Requires: strike, wall_z, tot_oi, prem_stock.
    Optionally: CE_oi, PE_oi, CE_ltp, PE_ltp, CE_mid, PE_mid, CE_prem_stock, PE_prem_stock.
    """
    if dfm.empty or "wall_z" not in dfm.columns:
        return pd.DataFrame()

    base_cols = ["strike", "wall_z", "tot_oi", "prem_stock"]
    have = [c for c in ["CE_oi","PE_oi","CE_ltp","PE_ltp","CE_mid","PE_mid",
                        "CE_prem_stock","PE_prem_stock",
                        "CE_oi_change","PE_oi_change"]
            if c in dfm.columns]
    cand = dfm[dfm["wall_z"] >= z_thresh][base_cols + have].copy()
    if cand.empty:
        return pd.DataFrame(columns=base_cols + ["CE_prem","PE_prem","prem_provenance",
                                                 "type","type_reason","type_confidence"])

    def _num(s): return pd.to_numeric(s, errors="coerce").fillna(0.0)

    # Leg premium split with provenance
    provenance = pd.Series("even_split", index=cand.index, dtype="object")

    if {"CE_prem_stock","PE_prem_stock"}.issubset(cand.columns):
        ce_prem = _num(cand["CE_prem_stock"])
        pe_prem = _num(cand["PE_prem_stock"])
        provenance[:] = "prem_stock_direct"
    else:
        prem_total = _num(cand.get("prem_stock", 0.0))
        have_oi  = {"CE_oi","PE_oi"}.issubset(cand.columns)
        have_mid = {"CE_mid","PE_mid"}.issubset(cand.columns)
        have_ltp = {"CE_ltp","PE_ltp"}.issubset(cand.columns)

        ce_w = pe_w = None
        if have_oi and have_mid:
            ce_w = _num(cand["CE_oi"]) * _num(cand["CE_mid"])
            pe_w = _num(cand["PE_oi"]) * _num(cand["PE_mid"])
            provenance[:] = "oi_x_mid"
        elif have_oi and have_ltp:
            ce_w = _num(cand["CE_oi"]) * _num(cand["CE_ltp"])
            pe_w = _num(cand["PE_oi"]) * _num(cand["PE_ltp"])
            provenance[:] = "oi_x_ltp"
        elif have_oi:
            ce_w = _num(cand["CE_oi"]); pe_w = _num(cand["PE_oi"])
            provenance[:] = "oi_only"
        elif have_ltp:
            ce_w = _num(cand["CE_ltp"]); pe_w = _num(cand["PE_ltp"])
            provenance[:] = "ltp_only"

        if ce_w is not None and pe_w is not None:
            w_sum = (ce_w + pe_w).replace(0, np.nan)
            ce_share = (ce_w / w_sum).fillna(0.5)
            pe_share = 1.0 - ce_share
            ce_prem = prem_total * ce_share
            pe_prem = prem_total * pe_share
        else:
            half = prem_total * 0.5
            ce_prem = half
            pe_prem = half
            provenance[:] = "even_split"

    cand["CE_prem"] = ce_prem
    cand["PE_prem"] = pe_prem
    cand["prem_provenance"] = provenance

    # Order by strength
    sorter = cand.sort_values(["wall_z","prem_stock","tot_oi"], ascending=False)

    # Optional: take top-k above and below spot for balanced context
    if (k_each is not None) and (spot is not None) and np.isfinite(spot):
        above = sorter[sorter["strike"] >= spot].head(k_each)
        below = sorter[sorter["strike"] <  spot].head(k_each)
        cand = pd.concat([above, below], ignore_index=True)
    else:
        cand = sorter.head(k).copy()

    # Strike step and optional flow
    step = _median_strike_step(dfm["strike"])
    has_flow = {"CE_mid","PE_mid","CE_oi_change","PE_oi_change"}.issubset(cand.columns)
    ce_flow = (_num(cand["CE_mid"]) * _num(cand["CE_oi_change"])) if has_flow else pd.Series(0.0, index=cand.index)

    # Classification (price-consistent)
    CE_RES_THR = 0.55
    PE_SUP_THR = 0.55

    types, reasons, confs = [], [], []
    for i, r in cand.iterrows():
        k_strike = float(r["strike"])
        ce, pe = float(r["CE_prem"]), float(r["PE_prem"])
        tot = ce + pe
        if tot <= 0:
            types.append("neutral"); reasons.append("no premium mass"); confs.append(0.2); continue

        ce_share = ce / tot
        pe_share = 1.0 - ce_share

        # provenance-weighted base confidence
        prov = r.get("prem_provenance", "even_split")
        if prov in ("prem_stock_direct","oi_x_mid","oi_x_ltp"):
            base_conf = 0.6
        elif prov in ("oi_only","ltp_only"):
            base_conf = 0.45
        else:
            base_conf = 0.25

        dom = abs(ce - pe) / tot
        conf = float(np.clip(base_conf + 0.6 * dom, 0.0, 1.0))

        # Near-spot magnet
        is_near = (spot is not None and np.isfinite(spot) and abs(k_strike - spot) <= 0.5 * step)
        if is_near:
            types.append("magnet"); reasons.append("near spot"); confs.append(max(conf, 0.4)); continue

        # No spot → composition-only labels
        if (spot is None) or (not np.isfinite(spot)):
            if pe_share >= PE_SUP_THR:
                types.append("support"); reasons.append("PE-dominated (no spot)"); confs.append(conf); continue
            if ce_share >= CE_RES_THR:
                types.append("resistance"); reasons.append("CE-dominated (no spot)"); confs.append(conf); continue
            types.append("neutral"); reasons.append("mixed (no spot)"); confs.append(max(0.2, 0.8*base_conf)); continue

        # Spot known → price-consistent labeling
        if k_strike > spot:
            if ce_share >= CE_RES_THR:
                types.append("resistance"); reasons.append("CE-dominated above spot"); confs.append(conf); continue
            if pe_share >= PE_SUP_THR:
                types.append("neutral"); reasons.append("PE mass above spot"); confs.append(min(conf, 0.4)); continue
            types.append("neutral"); reasons.append("mixed above spot"); confs.append(min(conf, 0.4)); continue

        if k_strike < spot:
            if pe_share >= PE_SUP_THR:
                types.append("support"); reasons.append("PE-dominated below spot"); confs.append(conf); continue
            if ce_share >= CE_RES_THR:
                # CE overhang behind price → not resistance. Demote; note covering if flows negative.
                spent = has_flow and (ce_flow.loc[i] < 0) and ((spot - k_strike) >= 0.5*step)
                if spent:
                    types.append("neutral"); reasons.append("spent call wall below spot (covering)"); confs.append(min(conf, 0.35)); continue
                types.append("neutral"); reasons.append("CE overhang below spot"); confs.append(min(conf, 0.4)); continue
            types.append("neutral"); reasons.append("mixed below spot"); confs.append(min(conf, 0.4)); continue

        # Exactly at spot
        types.append("magnet"); reasons.append("at-the-money"); confs.append(max(conf, 0.4))

    cand["type"] = types
    cand["type_reason"] = reasons
    cand["type_confidence"] = confs

    # Presentation aliases for UI
    cand = cand.rename(columns={"prem_stock": "premium", "wall_z": "wall_z (σ)"})
    return cand.reset_index(drop=True)


# --------------------------
# Commentary
# --------------------------

def make_money_map_commentary(dfm: pd.DataFrame, spot: Optional[float]) -> str:
    """
    Narrative summary:
      • Premium vs OI centers (magnet/positioning)
      • Sticky walls (top sides)
      • New premium concentration ABOVE vs BELOW spot (inflows)
      • Hedging hotspots (max gamma/vega indices)
      • Base-case range between nearest walls around spot
    """
    if dfm.empty:
        return "Money Map not available."

    parts = []

    # Centers
    cm_p = money_center_of_mass(dfm, use_premium=True)
    cm_o = money_center_of_mass(dfm, use_premium=False)
    if (cm_p is not None) and (cm_o is not None):
        parts.append(f"Magnet vs positioning: Premium Center ≈ {cm_p:.0f}, OI Center ≈ {cm_o:.0f}.")
        if spot is not None and np.isfinite(spot):
            if abs(spot - cm_p) <= 40:
                parts.append(f"Spot {spot:.0f} → pin/chop bias near magnet.")
            elif spot > cm_p:
                parts.append(f"Spot {spot:.0f} → gravity slightly down toward magnet unless momentum flips.")
            else:
                parts.append(f"Spot {spot:.0f} → gravity slightly up toward magnet unless momentum flips.")

    # Show a balanced set of walls around spot if possible
    walls_tbl = find_walls(dfm, z_thresh=1.5, k=4, spot=spot, k_each=2)
    if not walls_tbl.empty:
        walls_list = walls_tbl["strike"].astype(int).astype(str).tolist()
        parts.append("Sticky walls: " + " / ".join(walls_list) + ".")

    # Fresh premium concentration (consider inflows only)
    if "prem_new" in dfm.columns and spot is not None and np.isfinite(spot):
        strikes_num = pd.to_numeric(dfm["strike"], errors="coerce")
        above_in = dfm.loc[strikes_num >= spot, "prem_new"].clip(lower=0).sum()
        below_in = dfm.loc[strikes_num <  spot, "prem_new"].clip(lower=0).sum()
        tot_in = above_in + below_in
        if tot_in > 0:
            share_above = above_in / tot_in
            if share_above >= 0.60:
                parts.append("New premium concentrated overhead → ceiling bias unless calls unwind.")
            elif share_above <= 0.40:
                parts.append("New premium concentrated below → support bias unless puts unwind.")
            else:
                parts.append("New premium balanced across strikes.")

    # Hedging hotspots (magnitude indices)
    hot = []
    if "gamma_dlr" in dfm.columns and dfm["gamma_dlr"].notna().any():
        hot.append(int(dfm.loc[dfm["gamma_dlr"].idxmax(), "strike"]))
    if "vega_dlr" in dfm.columns and dfm["vega_dlr"].notna().any():
        vk = int(dfm.loc[dfm["vega_dlr"].idxmax(), "strike"])
        if vk not in hot:
            hot.append(vk)
    if hot:
        parts.append("Dealer hedging hotspots near " + " / ".join(map(str, hot)) + ".")

    # Base-case corridor between nearest walls around spot
    if spot is not None and np.isfinite(spot) and not walls_tbl.empty:
        ups = [int(k) for k in walls_tbl["strike"] if k >= spot]
        dns = [int(k) for k in walls_tbl["strike"] if k <= spot]
        up_w = ups[0] if ups else None
        dn_w = dns[-1] if dns else None
        if up_w and dn_w:
            parts.append(f"Base case: range/pin {dn_w}–{up_w}; fade edges while overhead writing persists.")

    return " ".join(parts) if parts else "No clear Money Map read yet."
