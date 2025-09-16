from typing import Optional, Tuple
import numpy as np
import pandas as pd
from state.history import get_last_frames

def _p95_pos(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna(); s = s[s > 0]
    return float(np.percentile(s, 95)) if len(s) else 1.0

def _p95_neg_mag(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna(); s = -s[s < 0]
    return float(np.percentile(s, 95)) if len(s) else 1.0

def _norm_pos(v: float, p95: float) -> float:
    return max(0.0, float(v)) / (p95 if p95 > 0 else 1.0)

def _norm_neg_to_pos(v: float, p95: float) -> float:
    return max(0.0, -float(v)) / (p95 if p95 > 0 else 1.0)

def _persistence_for_strike(frames, strike: float, side: str, cond: str="write_pos") -> float:
    if not frames: return np.nan
    col = f"{side}_oi_change"; vals = []
    for f in frames:
        row = f.loc[f["strike"] == strike]
        if row.empty or col not in row.columns: continue
        v = float(row.iloc[0][col]); ok = (v > 0) if cond == "write_pos" else (v < 0)
        vals.append(1.0 if ok else 0.0)
    return float(np.mean(vals)) if vals else np.nan

def build_gate_runway_tables(
    df_ring: pd.DataFrame,
    spot: float,
    atm_idx_in_ring: int,
    symbol: str,
    expiry: str,
    up_n: int = 6,
    down_n: int = 6,
    lookback_frames: int = 6,
    pass_threshold: float = 0.85,
    w_write: float = 0.6,
    w_stock: float = 0.3,
    w_unwind: float = 0.1,
):
    need_cols = {"PE_oi_change","PE_oi","CE_oi_change","CE_oi","strike"}
    if df_ring.empty or not need_cols.issubset(df_ring.columns) or spot is None:
        return pd.DataFrame(), pd.DataFrame(), None, None
    pe_ch_p95  = _p95_pos(df_ring["PE_oi_change"])
    pe_oi_p95  = _p95_pos(df_ring["PE_oi"])
    ce_ch_p95  = _p95_pos(df_ring["CE_oi_change"])
    ce_oi_p95  = _p95_pos(df_ring["CE_oi"])
    ce_ch_neg95 = _p95_neg_mag(df_ring["CE_oi_change"])
    pe_ch_neg95 = _p95_neg_mag(df_ring["PE_oi_change"])
    atm_strike = float(df_ring.iloc[atm_idx_in_ring]["strike"])
    up_band = df_ring[df_ring["strike"] >= atm_strike].copy().head(up_n)
    down_band = df_ring[df_ring["strike"] <= atm_strike].copy().tail(down_n).iloc[::-1].copy()
    frames = get_last_frames(symbol, expiry, n=lookback_frames)

    def _score_up(r):
        s_write  = _norm_pos(r.get("PE_oi_change", 0), pe_ch_p95)
        s_stock  = _norm_pos(r.get("PE_oi", 0),      pe_oi_p95)
        s_unwind = _norm_neg_to_pos(r.get("CE_oi_change", 0), ce_ch_neg95)
        return float(w_write*s_write + w_stock*s_stock + w_unwind*s_unwind)

    def _score_down(r):
        s_write  = _norm_pos(r.get("CE_oi_change", 0), ce_ch_p95)
        s_stock  = _norm_pos(r.get("CE_oi", 0),      ce_oi_p95)
        s_unwind = _norm_neg_to_pos(r.get("PE_oi_change", 0), pe_ch_neg95)
        return float(w_write*s_write + w_stock*s_stock + w_unwind*s_unwind)

    up_rows, down_rows = [], []
    for _, r in up_band.iterrows():
        k = float(r["strike"]); sc = _score_up(r)
        write_ok = float(r.get("PE_oi_change", 0)) > 0
        opp_press = _norm_pos(r.get("CE_oi_change", 0), ce_ch_p95)
        persist = _persistence_for_strike(frames, k, "PE", "write_pos")
        gate_ok = bool(write_ok and sc >= pass_threshold)
        up_rows.append({
            "strike": int(round(k)),
            "PE_oi_change": _round_int(r.get("PE_oi_change")),
            "PE_oi": _round_int(r.get("PE_oi")),
            "CE_oi_change": _round_int(r.get("CE_oi_change")),
            "CE_oi": _round_int(r.get("CE_oi")),
            "score": round(min(sc, 1.6), 2),
            "persistence": None if np.isnan(persist) else round(persist, 2),
            "opp_pressure": round(opp_press, 2),
            "gate_ok": gate_ok,
        })
    for _, r in down_band.iterrows():
        k = float(r["strike"]); sc = _score_down(r)
        write_ok = float(r.get("CE_oi_change", 0)) > 0
        opp_press = _norm_pos(r.get("PE_oi_change", 0), pe_ch_p95)
        persist = _persistence_for_strike(frames, k, "CE", "write_pos")
        gate_ok = bool(write_ok and sc >= pass_threshold)
        down_rows.append({
            "strike": int(round(k)),
            "CE_oi_change": _round_int(r.get("CE_oi_change")),
            "CE_oi": _round_int(r.get("CE_oi")),
            "PE_oi_change": _round_int(r.get("PE_oi_change")),
            "PE_oi": _round_int(r.get("PE_oi")),
            "score": round(min(sc, 1.6), 2),
            "persistence": None if np.isnan(persist) else round(persist, 2),
            "opp_pressure": round(opp_press, 2),
            "gate_ok": gate_ok,
        })
    up_tbl = pd.DataFrame(up_rows); down_tbl = pd.DataFrame(down_rows)
    up_clear  = _compute_clear(up_tbl)
    down_clear = _compute_clear(down_tbl)
    return up_tbl, down_tbl, up_clear, down_clear

def _round_int(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return int(round(v))
        return np.nan
    except Exception:
        return np.nan

def majority_pass(sym: str, exp: str, strike: float, side: str, thresh: float, frames_n: int = 3) -> Optional[bool]:
    frames = get_last_frames(sym, exp, n=frames_n)
    if len(frames) < frames_n: return None
    ok = 0
    for f in frames:
        row = f.loc[f["strike"] == strike]
        if row.empty: continue
        if side == "UP":
            pe_ch = float(row["PE_oi_change"].iloc[0] or 0) > 0
            ce_un = float(row["CE_oi_change"].iloc[0] or 0) < 0
            stock = np.log1p(max(0.0, float(row["PE_oi"].iloc[0] or 0)))/10.0
            ok += int(pe_ch and ce_un and stock >= thresh)
        else:
            ce_ch = float(row["CE_oi_change"].iloc[0] or 0) > 0
            pe_un = float(row["PE_oi_change"].iloc[0] or 0) < 0
            stock = np.log1p(max(0.0, float(row["CE_oi"].iloc[0] or 0)))/10.0
            ok += int(ce_ch and pe_un and stock >= thresh)
    return ok >= 2

def _delta_w(side: str, s: pd.Series) -> float:
    col = f"{side}_delta"
    if col not in s or pd.isna(s[col]):  return 1.0
    d = abs(float(s[col])); return float(np.clip(d/0.50, 0.4, 1.2))

def step_gradient_penalty(tbl: pd.DataFrame, side: str):
    col = f"{side}_oi_change"
    vals = tbl[col].fillna(0).astype(float).values
    if len(vals) < 3: return np.ones_like(vals)
    slope = np.gradient(vals)
    return np.where(slope >= 0, 1.0, 0.8)

def _compute_clear(tbl: pd.DataFrame) -> Optional[int]:
    if tbl.empty: return None
    streak = 0
    for ok in tbl["gate_ok"].tolist():
        if ok: streak += 1
        else: break
    if streak == 0: return None
    return int(tbl.iloc[streak-1]["strike"])

def _compute_clear_with_skip(tbl: pd.DataFrame) -> Optional[int]:
    if tbl.empty: return None
    flags = tbl["gate_ok"].tolist(); strikes = tbl["strike"].tolist()
    i, skipped, last = 0, False, None
    while i < len(flags):
        if flags[i]:
            last = strikes[i]; i += 1; continue
        if not skipped and i+2 < len(flags) and flags[i+1] and flags[i+2]:
            skipped = True; last = strikes[i+2]; i += 3; continue
        break
    return None if last is None else int(last)

def confidence_score(runway_ratio: float, vel_ok: float, iv_z: Optional[float], gamma_regime: str, pcr: float) -> float:
    iv_boost    = np.clip((iv_z or 0)/2.0, -1, 1)
    gamma_boost = (0.5 if "Short" in (gamma_regime or "") else (-0.5 if "Long" in (gamma_regime or "") else 0))
    pcr_align   = (0.25 if pcr>1.2 else (-0.25 if pcr<0.8 else 0))
    return float(np.clip(100*(0.45*runway_ratio + 0.20*vel_ok + 0.20*max(0,iv_boost) + 0.10*max(0,gamma_boost) + 0.05*max(0,pcr_align)), 0, 100))

def apply_runway_enhancements(
    up_tbl: pd.DataFrame,
    down_tbl: pd.DataFrame,
    df_ring: pd.DataFrame,
    strict: float,
    use_delta: bool,
    opp_cap: float,
    liq_pct: int,
    use_majority: bool,
    maj_frames: int,
    allow_skip: bool,
    sym: str, exp: str
):
    pe_floor = np.percentile(df_ring["PE_oi"].dropna(), liq_pct) if "PE_oi" in df_ring else 0
    ce_floor = np.percentile(df_ring["CE_oi"].dropna(), liq_pct) if "CE_oi" in df_ring else 0
    del_cols = [c for c in ["strike","PE_delta","CE_delta"] if c in df_ring.columns]
    dd = df_ring[del_cols].drop_duplicates("strike") if del_cols else pd.DataFrame(columns=["strike"])
    def _enhance(tbl: pd.DataFrame, side: str):
        if tbl.empty:  return tbl
        t = tbl.copy()
        if not dd.empty: t = t.merge(dd, on="strike", how="left")
        if use_delta:
            if side == "UP": t["score"] = t.apply(lambda r: r["score"]*_delta_w("PE", r), axis=1)
            else:            t["score"] = t.apply(lambda r: r["score"]*_delta_w("CE", r), axis=1)
        t["score"] = t["score"] * (step_gradient_penalty(t, "PE" if side=="UP" else "CE"))
        if side == "UP":
            write_ok = t["PE_oi_change"].fillna(0).astype(float) > 0
            liq_ok   = t["PE_oi"].fillna(0).astype(float) >= pe_floor
        else:
            write_ok = t["CE_oi_change"].fillna(0).astype(float) > 0
            liq_ok   = t["CE_oi"].fillna(0).astype(float) >= ce_floor
        score_ok = t["score"] >= strict
        opp_ok   = t["opp_pressure"] <= opp_cap
        t["gate_ok"] = (write_ok & score_ok & opp_ok & liq_ok)
        if use_majority:
            if side == "UP":
                mp = t["strike"].apply(lambda k: majority_pass(sym, exp, float(k), "UP", strict, maj_frames))
            else:
                mp = t["strike"].apply(lambda k: majority_pass(sym, exp, float(k), "DOWN", strict, maj_frames))
            t["gate_ok"] = t["gate_ok"] & mp.fillna(True)
        def reason_up(r):
            if r.get("PE_oi",0) < pe_floor: return "low_liq"
            if r["opp_pressure"] > opp_cap: return "opp_cap"
            if float(r.get("PE_oi_change",0)) <= 0: return "no_write"
            if float(r.get("CE_oi_change",0)) >= 0: return "no_unwind"
            return "pass" if r["gate_ok"] else "weak_score"
        def reason_down(r):
            if r.get("CE_oi",0) < ce_floor: return "low_liq"
            if r["opp_pressure"] > opp_cap: return "opp_cap"
            if float(r.get("CE_oi_change",0)) <= 0: return "no_write"
            if float(r.get("PE_oi_change",0)) >= 0: return "no_unwind"
            return "pass" if r["gate_ok"] else "weak_score"
        t["reason"] = t.apply(reason_up if side=="UP" else reason_down, axis=1)
        return t
    up_adj    = _enhance(up_tbl, "UP")
    down_adj  = _enhance(down_tbl, "DOWN")
    up_clear    = _compute_clear_with_skip(up_adj) if allow_skip else _compute_clear(up_adj)
    down_clear  = _compute_clear_with_skip(down_adj) if allow_skip else _compute_clear(down_adj)
    return up_adj, down_adj, up_clear, down_clear
