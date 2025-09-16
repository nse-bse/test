from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from typing import Optional

def _mid(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = (a + b) / 2.0
    if isinstance(m, pd.Series):
        return m.where(m > 0)
    else:
        return m if m > 0 else np.nan

def compute_spread_stats(df: pd.DataFrame, spot: Optional[float], band: int = 3):
    if df.empty or spot is None: return None
    from calculations.chain import find_atm_index
    atm = find_atm_index(df, spot)
    lo = max(0, atm - band); hi = min(len(df), atm + band + 1)
    ring = df.iloc[lo:hi].copy()
    for side in ("CE", "PE"):
        if f"{side}_ask" in ring.columns and f"{side}_bid" in ring.columns:
            ring[f"{side}_mid"] = _mid(ring[f"{side}_ask"], ring[f"{side}_bid"])
            ring[f"{side}_spr"] = pd.to_numeric(ring[f"{side}_ask"], errors="coerce") - pd.to_numeric(ring[f"{side}_bid"], errors="coerce")
            ring[f"{side}_spr_bps"] = 1e4 * (ring[f"{side}_spr"] / ring[f"{side}_mid"])
    spr_bps_med = float(pd.concat([ring.get("CE_spr_bps"), ring.get("PE_spr_bps")]).replace([np.inf, -np.inf], np.nan).dropna().median()) \
                  if "CE_spr_bps" in ring.columns and not ring.empty else np.nan
    spr_med = float(pd.concat([ring.get("CE_spr"), ring.get("PE_spr")]).replace([np.inf, -np.inf], np.nan).dropna().median()) \
              if "CE_spr" in ring.columns and not ring.empty else np.nan
    if len(df) <= atm: return None
    row = df.iloc[atm]
    ce_mid = _mid(row.get("CE_ask"), row.get("CE_bid"))
    pe_mid = _mid(row.get("PE_ask"), row.get("PE_bid"))
    ce_spr = float(pd.to_numeric(row.get("CE_ask"), errors="coerce") - pd.to_numeric(row.get("CE_bid"), errors="coerce")) if "CE_ask" in row else np.nan
    pe_spr = float(pd.to_numeric(row.get("PE_ask"), errors="coerce") - pd.to_numeric(row.get("PE_bid"), errors="coerce")) if "PE_ask" in row else np.nan
    ce_bps = float(1e4 * ce_spr / ce_mid) if pd.notna(ce_mid) and ce_mid > 0 else np.nan
    pe_bps = float(1e4 * pe_spr / pe_mid) if pd.notna(pe_mid) and pe_mid > 0 else np.nan
    return {
        "median_spread_bps": spr_bps_med,
        "median_spread_rs": spr_med,
        "atm": {
            "strike": float(row["strike"]),
            "CE_spread_rs": ce_spr, "CE_spread_bps": ce_bps,
            "PE_spread_rs": pe_spr, "PE_spread_bps": pe_bps
        }
    }

def impact_cost_proxy(df: pd.DataFrame, spot: Optional[float], qty: int = 50):
    if df.empty or spot is None: return None
    from calculations.chain import find_atm_index
    i = find_atm_index(df, spot)
    if len(df) <= i: return None
    row = df.iloc[i]
    def _slip(side):
        bid = pd.to_numeric(row.get(f"{side}_bid"), errors="coerce")
        ask = pd.to_numeric(row.get(f"{side}_ask"), errors="coerce")
        mid = (bid + ask) / 2 if pd.notna(bid) and pd.notna(ask) else np.nan
        if pd.isna(mid) or mid <= 0 or pd.isna(ask) or pd.isna(bid): return np.nan, np.nan
        buy_ic  = float((ask - mid) / mid) if mid else np.nan
        sell_ic = float((mid - bid) / mid) if mid else np.nan
        return buy_ic, sell_ic
    ce_buy_ic, ce_sell_ic = _slip("CE")
    pe_buy_ic, pe_sell_ic = _slip("PE")
    ic_vals = []
    for t in (ce_buy_ic, ce_sell_ic, pe_buy_ic, pe_sell_ic):
        if np.isfinite(t): ic_vals.append(t)
    ic = float(np.nanmean(ic_vals)) if ic_vals else np.nan
    return {"ic_mean": ic, "ic_bps": (ic * 1e4) if np.isfinite(ic) else np.nan}

def liquidity_stress(med_spr_bps: float, ic_bps: float, ring: pd.DataFrame) -> Optional[float]:
    if ring is None or np.isnan(med_spr_bps) or np.isnan(ic_bps): return None
    tr = ring.get("turnover_ratio") if "turnover_ratio" in ring.columns else None
    tr_inv = float(np.nanmedian(1.0 / tr.replace(0, np.nan))) if isinstance(tr, pd.Series) and tr.notna().any() else 1.0
    z_spr = np.clip(med_spr_bps / 50.0, 0.0, 1.0)
    z_ic  = np.clip(ic_bps / 30.0, 0.0, 1.0)
    z_tr  = np.clip(tr_inv / 10.0, 0.0, 1.0)
    return float(0.5*z_spr + 0.3*z_ic + 0.2*z_tr)
# --- Robust intraday normalization for liquidity stress ---

def _robust_z(x: pd.Series, value: float) -> float | None:
    """
    Median/MAD z-score (robust to outliers). Returns None if insufficient data.
    """
    if x is None or len(x) < 6 or not np.isfinite(value):
        return None
    s = pd.to_numeric(pd.Series(x, dtype="float64"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 6:
        return None
    med = float(s.median())
    mad = float((s - med).abs().median())
    if mad == 0:
        return None
    return float((value - med) / (1.4826 * mad))

def _percentile(x: pd.Series, value: float) -> float | None:
    """
    Empirical percentile of 'value' within 'x' (0..100).
    """
    if x is None or len(x) < 6 or not np.isfinite(value):
        return None
    s = pd.to_numeric(pd.Series(x, dtype="float64"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 6:
        return None
    # rank of value among historical values
    return float((s <= value).mean() * 100.0)

def liquidity_stress_with_stats(
    med_spr_bps: float,
    ic_bps: float,
    ring: pd.DataFrame,
    hist_values: Optional[List[float]] = None
) -> dict:
    """
    Wraps liquidity_stress() and adds robust z-score + percentile
    vs the intraday history (pass a list/series of previous stresses).
    """
    stress = liquidity_stress(med_spr_bps, ic_bps, ring)
    z = _robust_z(hist_values, stress) if (hist_values is not None and stress is not None) else None
    pct = _percentile(hist_values, stress) if (hist_values is not None and stress is not None) else None
    return {
        "stress": stress,            # 0..1
        "z_robust": z,               # None or ~[-3..+3]
        "percentile_0_100": pct      # None or 0..100  (higher = worse vs today)
    }

def gex_curve(df: pd.DataFrame, spot: Optional[float]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    out = pd.DataFrame({"strike": df["strike"]})
    def S(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    out["gex"] = S("CE_gamma").fillna(0)*S("CE_oi").fillna(0) + S("PE_gamma").fillna(0)*S("PE_oi").fillna(0)
    return out.dropna().sort_values("strike").reset_index(drop=True)

def realized_vol_annualized(spots: List, minutes_per_year: int = 252*390) -> Optional[float]:
    """
    Accepts a list of floats (spot levels) or a list of dicts with key 'spot'.
    Returns annualized intraday RV in %.
    """
    if not spots or len(spots) < 5:
        return None
    # normalize to floats
    if isinstance(spots[0], dict):
        seq = [pd.to_numeric(x.get("spot"), errors="coerce") for x in spots]
    else:
        seq = [pd.to_numeric(x, errors="coerce") for x in spots]
    s = pd.Series(seq, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        return None
    r = np.log(s).diff().dropna()
    rv = float(np.sqrt(minutes_per_year) * r.std(ddof=0)) * 100.0
    return rv

def pcr_vol(df: pd.DataFrame) -> Optional[float]:
    if df.empty: return None
    ce = pd.to_numeric(df.get("CE_volume"), errors="coerce").fillna(0).sum() if "CE_volume" in df else 0
    pe = pd.to_numeric(df.get("PE_volume"), errors="coerce").fillna(0).sum() if "PE_volume" in df else 0
    return float(pe/ce) if ce > 0 else np.inf

def _num(x):
    try:
        v = float(x); return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def _liquidity_tier(median_bps: float, impact_bps: float, stress: float) -> tuple[str, str]:
    m = _num(median_bps); i = _num(impact_bps); s = _num(stress)
    hi = sum([int(m >= 800) if np.isfinite(m) else 0, int(i >= 800) if np.isfinite(i) else 0, int(s >= 70) if np.isfinite(s) else 0])
    if hi >= 2: return "Tight", "🔴"
    med = sum([int(400 <= m < 800) if np.isfinite(m) else 0, int(400 <= i < 800) if np.isfinite(i) else 0, int(40 <= s < 70) if np.isfinite(s) else 0])
    if med >= 1: return "Caution", "🟡"
    return "Normal", "🟢"

def make_quick_execution_commentary(
    liq: dict,
    iv_z: float | None,
    gamma_regime: str | None,
    pcr: float | None,
    rails: dict | None,
    sup_zones: list[int] | None,
    res_zones: list[int] | None,
) -> tuple[str, str]:
    median_bps = liq.get("median_spread_bps") if liq else np.nan
    impact_bps = liq.get("ic_bps") if liq else np.nan
    stress     = liq.get("liquidity_stress") if liq else np.nan
    tier, dot = _liquidity_tier(median_bps, impact_bps, stress)
    parts = []
    liq_bits = []
    if np.isfinite(_num(median_bps)): liq_bits.append(f"spread ~{_num(median_bps):.0f} bps")
    if np.isfinite(_num(impact_bps)): liq_bits.append(f"impact ~{_num(impact_bps):.0f} bps")
    if np.isfinite(_num(stress)):     liq_bits.append(f"stress {int(_num(stress))}/100")
    if tier == "Tight":
        parts.append(f"💧 Liquidity: {dot} **{tier}**" + (f" ({' • '.join(liq_bits)})" if liq_bits else "") + ". Use patient limits; prefer ATM±1 or futures; avoid market orders.")
    elif tier == "Caution":
        parts.append(f"💧 Liquidity: {dot} **{tier}**" + (f" ({' • '.join(liq_bits)})" if liq_bits else "") + ". Prefer limits; split size.")
    else:
        parts.append(f"💧 Liquidity: {dot} **{tier}**" + (f" ({' • '.join(liq_bits)})" if liq_bits else "") + ".")
    if iv_z is not None:
        if iv_z > 1.5:   parts.append("⚡ **Vol expanding** — respect breakouts.")
        elif iv_z < -1.5: parts.append("🧘 **Vol compressing** — mean-revert favoured.")
        else:             parts.append("〽️ **Vol steady.**")
    gr = (gamma_regime or "")
    if "Short" in gr: parts.append("📟 **Short γ** — breakout risk.")
    elif "Long" in gr: parts.append("📟 **Long γ** — fade edges ok.")
    if pcr is not None and np.isfinite(_num(pcr)):
        pv = _num(pcr)
        if pv > 1.2:   parts.append(f"🟢 **PCR {pv:.2f}** bullish tilt.")
        elif pv < 0.8: parts.append(f"🔴 **PCR {pv:.2f}** bearish tilt.")
        else:          parts.append(f"⚖️ **PCR {pv:.2f}** balanced.")
    r50 = rails.get(0.5) if rails else None
    if r50 and all(np.isfinite([_num(r50[0]), _num(r50[1])])): parts.append(f"🎯 **Cone {int(r50[0])}–{int(r50[1])}**.")
    if sup_zones: parts.append("🟢 **Sup** " + ", ".join(map(str, sup_zones)) + ".")
    if res_zones: parts.append("🔴 **Res** " + ", ".join(map(str, res_zones)) + ".")
    text = " ".join(parts)
    severity = "error" if tier == "Tight" else ("warning" if tier == "Caution" else "info")
    return severity, text
# --- 1% Execution Suite (edge ÷ pressure, INR slip, sizing) ---

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _mid_series(ask: pd.Series, bid: pd.Series) -> pd.Series:
    a, b = _to_num(ask), _to_num(bid)
    m = (a + b) / 2.0
    return m.where(m > 0)

def _spread_pct_series(ask: pd.Series, bid: pd.Series) -> pd.Series:
    m = _mid_series(ask, bid)
    spr = _to_num(ask) - _to_num(bid)
    spr_pct = spr / m
    return spr_pct.replace([np.inf, -np.inf], np.nan)

def liquidity_stress_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strike-level pressure proxy that’s robust to missing fields.
    Columns created:
      - ce_mid, pe_mid
      - ce_spread_pct, pe_spread_pct, spread_pct (best side)
      - impact_proxy  (half-spread + thinness kicker)
      - liq_pressure_raw, liquidity_pressure_v2 (0..1)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # CE/PE mids + spread pct
    if {"CE_ask", "CE_bid"}.issubset(out.columns):
        out["ce_mid"] = _mid_series(out["CE_ask"], out["CE_bid"])
        out["ce_spread_pct"] = _spread_pct_series(out["CE_ask"], out["CE_bid"])
    else:
        out["ce_mid"] = np.nan
        out["ce_spread_pct"] = np.nan

    if {"PE_ask", "PE_bid"}.issubset(out.columns):
        out["pe_mid"] = _mid_series(out["PE_ask"], out["PE_bid"])
        out["pe_spread_pct"] = _spread_pct_series(out["PE_ask"], out["PE_bid"])
    else:
        out["pe_mid"] = np.nan
        out["pe_spread_pct"] = np.nan

    # Choose the better (lower) spread% side at each strike
    out["spread_pct"] = np.nanmin(
        np.vstack([out["ce_spread_pct"].to_numpy(), out["pe_spread_pct"].to_numpy()]),
        axis=0,
    )
    out.loc[~np.isfinite(out["spread_pct"]), "spread_pct"] = np.nan

    # Thin-liquidity kicker via inverse turnover if present
    # IMPORTANT CHANGE: .fillna(0.0) so missing turnover contributes 0 (not NaN)
    tr = _to_num(out.get("turnover_ratio", pd.Series(np.nan, index=out.index))).replace(0, np.nan)
    tr_inv_norm = (1.0 / tr).clip(lower=0, upper=10).fillna(0.0)  # ← key fix

    # Impact proxy ~ half-spread + small kicker for thin books
    out["impact_proxy"] = (0.5 * out["spread_pct"]) + (0.02 * tr_inv_norm)

    # Raw “bps-like” pressure
    out["liq_pressure_raw"] = out["impact_proxy"] * 1e4

    # Robust 0..1 normalization by 99th percentile (guard if all NaN/0)
    q99 = out["liq_pressure_raw"].quantile(0.99)
    if pd.notna(q99) and q99 > 0:
        out["liquidity_pressure_v2"] = (out["liq_pressure_raw"] / q99).clip(0, 1.0)
    else:
        out["liquidity_pressure_v2"] = np.nan

    return out


def estimate_rupee_impact(df: pd.DataFrame, *, lot_size: int = 25) -> pd.DataFrame:
    """
    Adds: rupee_impact_per_lot (expected slip to cross one lot, INR).
    Uses best_mid * spread_pct * lot_size.
    """
    if df is None or df.empty: 
        return df
    out = df.copy()
    best_mid = out[["ce_mid","pe_mid"]].min(axis=1)
    out["rupee_impact_per_lot"] = (best_mid * out["spread_pct"] * lot_size).replace([np.inf, -np.inf], np.nan)
    return out

def plan_order_sizes(df: pd.DataFrame, *, max_slippage_bps: float = 10.0, lot_size: int = 25) -> pd.DataFrame:
    """
    Adds: max_lots_under_slippage (conservative rule of thumb).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    cur_bps = out["spread_pct"] * 1e4
    out["max_lots_under_slippage"] = np.where(
        (pd.notna(cur_bps)) & (cur_bps <= float(max_slippage_bps)),
        1,
        0
    ).astype(int)
    if "rupee_impact_per_lot" in out.columns:
        out.loc[(out["max_lots_under_slippage"] == 1) & (out["rupee_impact_per_lot"] <= 5), "max_lots_under_slippage"] = 2
    return out

def _edge_score(df: pd.DataFrame) -> pd.Series:
    """
    Edge proxy: participation (|net_oi_change| or |ΔOI|) + proximity to ATM (via delta).
    Returns 0..1 score.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if "net_oi_change" in df.columns:
        mag = pd.to_numeric(df["net_oi_change"], errors="coerce").abs()
    else:
        ce = pd.to_numeric(df.get("CE_oi_change"), errors="coerce")
        pe = pd.to_numeric(df.get("PE_oi_change"), errors="coerce")
        mag = (ce.fillna(0).abs() + pe.fillna(0).abs())

    prox = pd.Series(1.0, index=df.index)
    if "delta" in df.columns:
        d = pd.to_numeric(df["delta"], errors="coerce").abs()
        prox = np.exp(-((d - 0.5) ** 2) / (2 * (0.2 ** 2)))
    elif {"strike"}.issubset(df.columns):
        s = pd.to_numeric(df["strike"], errors="coerce")
        prox = 1.0 / (1.0 + (s - s.median()).abs() / (s.median() + 1e-9))

    def _rank01(x: pd.Series) -> pd.Series:
        if x.isna().all():
            return pd.Series(np.nan, index=x.index)
        return x.rank(pct=True, method="average")

    mag01 = _rank01(mag.fillna(0))
    prox01 = _rank01(prox.fillna(0))
    edge = 0.7 * mag01 + 0.3 * prox01
    return edge.clip(0, 1.0)

def rank_strikes_by_edge_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: edge_score (0..1), edge_per_pressure, rank_epp (1 = best).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "liquidity_pressure_v2" not in out.columns:
        out = liquidity_stress_v2(out)
    out["edge_score"] = _edge_score(out)
    denom = out["liquidity_pressure_v2"].replace(0, np.nan)
    out["edge_per_pressure"] = out["edge_score"] / denom
    out["rank_epp"] = out["edge_per_pressure"].rank(ascending=False, method="min")
    return out.sort_values(["rank_epp", "edge_per_pressure"], ascending=[True, False]).reset_index(drop=True)

def find_execution_windows(df_ranked: pd.DataFrame, *, pressure_pctile: float = 0.25, min_edge_per_pressure: float = 0.0) -> pd.DataFrame:
    """
    Filter for low-pressure pocket + positive edge/pressure.
    """
    if df_ranked is None or df_ranked.empty:
        return pd.DataFrame()
    q = df_ranked["liquidity_pressure_v2"].quantile(pressure_pctile)
    sel = df_ranked[
        (df_ranked["liquidity_pressure_v2"] <= q) &
        (df_ranked["edge_per_pressure"] >= float(min_edge_per_pressure))
    ].copy()
    keep = ["strike","edge_per_pressure","liquidity_pressure_v2","rupee_impact_per_lot","max_lots_under_slippage"]
    for c in keep:
        if c not in sel.columns:
            sel[c] = np.nan
    return sel.sort_values("edge_per_pressure", ascending=False).reset_index(drop=True)
