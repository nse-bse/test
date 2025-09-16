from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from calculations.chain import find_atm_index
from calculations.iv_skew import compute_atm_iv, make_iv_skew_history, zscore_last, compute_rr_bf
from state.history import get_last_frames

def compute_gamma_regime(df: pd.DataFrame, spot: Optional[float], ring_size: int = 10) -> Tuple[str, float]:
    if df.empty or spot is None: return "Unknown", 0.0
    atm_idx = find_atm_index(df, spot)
    lo = max(0, atm_idx - ring_size); hi = min(len(df), atm_idx + ring_size + 1)
    ring = df.iloc[lo:hi].copy()
    gex = 0.0
    if {"CE_gamma","CE_oi"}.issubset(ring.columns):
        gex += (ring["CE_gamma"].fillna(0) * ring["CE_oi"].fillna(0)).sum()
    if {"PE_gamma","PE_oi"}.issubset(ring.columns):
        gex += (ring["PE_gamma"].fillna(0) * ring["PE_oi"].fillna(0)).sum()
    if not np.isfinite(gex): return "Unknown", 0.0
    regime = "Long γ (fade edges)" if gex > 0 else "Short γ (breakouts riskier)"
    return regime, float(gex)

def compute_iv_rails(spot: float, atm_iv: float, minutes_left: float, probs=(0.3,0.5,0.7)) -> Dict[float, Tuple[float,float]]:
    if spot is None or atm_iv is None or minutes_left <= 0: return {}
    sigma = atm_iv / 100.0
    tau = minutes_left / (60*24*365)
    rails = {}
    for p in probs:
        z = {0.3: 0.524, 0.5: 0.674, 0.7: 1.036}.get(p, 0.674)
        band = z * sigma * np.sqrt(tau)
        rails[p] = (spot * (1 - band), spot * (1 + band))
    return rails

def detect_stop_hunt(symbol: str, expiry: str) -> Optional[Dict]:
    frames = get_last_frames(symbol, expiry, n=2)
    if len(frames) < 2:  return None
    f_prev, f_now = frames
    atm_iv_prev = compute_atm_iv(f_prev, None)
    atm_iv_now = compute_atm_iv(f_now, None)
    if atm_iv_prev is None or atm_iv_now is None: return None
    d_iv = atm_iv_now - atm_iv_prev
    iv_hist = make_iv_skew_history(symbol, expiry, last_n=30)
    iv_z = zscore_last(iv_hist["atm_iv"], window=20) if not iv_hist.empty else None
    if "net_oi_change" not in f_now.columns or "net_oi_change" not in f_prev.columns: return None
    vel = (f_now.set_index("strike")["net_oi_change"] - f_prev.set_index("strike")["net_oi_change"]).fillna(0)
    hot_strike = vel.abs().idxmax()
    hot_val = vel.loc[hot_strike]
    if iv_z is None: return None
    if abs(d_iv) > 0.5 and iv_z > 1.5:
        tag = "True Break ✅" if np.sign(hot_val) > 0 else "Stop Hunt ⚠️"
        return {"strike": int(hot_strike), "iv_z": iv_z, "d_iv": d_iv, "tag": tag}
    return None

def score_zones(df_ring: pd.DataFrame, atm_iv: Optional[float], rr: Optional[float], rails: Dict) -> pd.DataFrame:
    if df_ring.empty: return pd.DataFrame()
    half = rails.get(0.5) if rails else None
    out = []
    for _, r in df_ring.iterrows():
        k = float(r["strike"])
        ce_oi = float(r.get("CE_oi", 0) or 0); pe_oi = float(r.get("PE_oi", 0) or 0)
        net = float(r.get("net_oi_change", 0) or 0)
        ztype = "Support" if pe_oi >= ce_oi else "Resistance"
        emoji = "🟢" if ztype == "Support" else "🔴"
        score = (np.log1p(ce_oi + pe_oi) / 10.0) + (net / 10000.0)
        if rr is not None: score += (rr / 10.0) * (1 if ztype == "Support" else -1)
        if half and (half[0] <= k <= half[1]): score += 2.0
        out.append({"emoji": emoji, "type": ztype, "strike": int(round(k)), "score": float(score)})
    df_out = pd.DataFrame(out).sort_values("score", ascending=False).head(4).reset_index(drop=True)
    df_out["score"] = df_out["score"].round(2)
    return df_out

def _format_zone_list(strikes: List[int], kind: str) -> str:
    emj = "🟢" if kind == "support" else "🔴"
    return " / ".join([f"{emj} {int(k)}" for k in strikes]) if strikes else "–"

def summarize_zones(df: pd.DataFrame, top_n: int=2):
    if "net_oi_change" not in df.columns or df.empty: return [], []
    df_sum = df[["strike", "net_oi_change"]].dropna().copy()
    if df_sum.empty: return [], []
    df_sum["strike"] = df_sum["strike"].round(0).astype(int)
    top_support = df_sum[df_sum["net_oi_change"] > 0].sort_values("net_oi_change", ascending=False).head(top_n)["strike"].tolist()
    top_resistance = df_sum[df_sum["net_oi_change"] < 0].sort_values("net_oi_change", ascending=True).head(top_n)["strike"].tolist()
    return top_support, top_resistance

def summarize_trade_bias(pcr: float, sentiment: str, atm_iv: Optional[float], iv_z: Optional[float]) -> str:
    msg = []
    pcr_msg = "PCR is balanced"
    if pcr > 1.2: pcr_msg = "PCR is bullish (puts > calls)"
    elif pcr < 0.8: pcr_msg = "PCR is bearish (calls > puts)"
    msg.append(f"**PCR:** {pcr_msg} ({pcr:.2f})")
    msg.append(f"Writing sentiment is {sentiment}.")
    iv_msg = ""
    if iv_z is not None and atm_iv is not None:
        if iv_z > 1.5: iv_msg = f"IV is spiking ({atm_iv:.2f}%)"
        elif iv_z < -1.5: iv_msg = f"IV is compressing ({atm_iv:.2f}%)"
        else: iv_msg = f"IV is stable ({atm_iv:.2f}%)"
    if iv_msg: msg.append(f"**Vol:** {iv_msg}")
    return " ".join(msg)

def build_trade_commentary(
    gamma_regime: str,
    rails: Dict,
    pcr: float,
    sentiment: str,
    commentary: str,
    atm_iv: Optional[float],
    iv_z: Optional[float],
    sup_zones: List[int],
    res_zones: List[int],
    vel_df: pd.DataFrame,
    stop_event: Optional[Dict]
) -> str:
    parts = []
    if "Long" in gamma_regime:
        parts.append("Market in **long gamma** → chop/mean reversion likely. Fade edges, book quicker.")
    elif "Short" in gamma_regime:
        parts.append("Market in **short gamma** → breakouts riskier but can run. Don’t fade blindly.")
    if pcr > 1.2: parts.append(f"PCR {pcr:.2f} → bullish lean. {commentary}")
    elif pcr < 0.8: parts.append(f"PCR {pcr:.2f} → bearish lean. {commentary}")
    else: parts.append(f"PCR {pcr:.2f} → balanced. {commentary}")
    if atm_iv is not None and iv_z is not None:
        if iv_z > 1.5: parts.append(f"IV spiking ({atm_iv:.2f}%), respect breakouts.")
        elif iv_z < -1.5: parts.append(f"IV compressing ({atm_iv:.2f}%), favor fades.")
        else: parts.append(f"IV stable ({atm_iv:.2f}%).")
    if sup_zones: parts.append(f"Supports 🟢 {', '.join(map(str, sup_zones))}")
    if res_zones: parts.append(f"Resistances 🔴 {', '.join(map(str, res_zones))}")
    r50 = rails.get(0.5) if rails else None
    if r50: parts.append(f"Intraday move likely contained in {int(r50[0])}–{int(r50[1])}.")
    if not vel_df.empty:
        up = vel_df.sort_values("velocity", ascending=False).iloc[0]
        dn = vel_df.sort_values("velocity", ascending=True).iloc[0]
        parts.append(f"Momentum: 🟢 build {int(up['strike'])} (+{int(up['velocity'])}) | 🔴 build {int(dn['strike'])} ({int(dn['velocity'])})")
    if stop_event:
        parts.append(f"{stop_event['tag']} seen at {stop_event['strike']} (ΔIV {stop_event['d_iv']:+.2f}, z={stop_event['iv_z']:+.2f}).")
    return " ".join(parts)
