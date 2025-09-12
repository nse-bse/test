from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

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

def gex_curve(df: pd.DataFrame, spot: Optional[float]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    out = pd.DataFrame({"strike": df["strike"]})
    def S(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    out["gex"] = S("CE_gamma").fillna(0)*S("CE_oi").fillna(0) + S("PE_gamma").fillna(0)*S("PE_oi").fillna(0)
    return out.dropna().sort_values("strike").reset_index(drop=True)

def realized_vol_annualized(spots: List[Dict], minutes_per_year: int = 252*390) -> Optional[float]:
    if not spots or len(spots) < 5: return None
    s = pd.Series([x["spot"] for x in spots], dtype="float64").replace([np.inf,-np.inf], np.nan).dropna()
    if len(s) < 5: return None
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
