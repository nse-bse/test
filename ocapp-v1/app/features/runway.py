from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from state.history import get_last_frames

# -------------------- robust small helpers --------------------

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

def _round_int(x):
    try:
        v = float(x)
        if np.isfinite(v): return int(round(v))
    except Exception:
        pass
    return np.nan

def _persistence_for_strike(frames, strike: float, side: str, cond: str="write_pos") -> float:
    """share of frames where the directional condition holds"""
    if not frames: return np.nan
    col = f"{side}_oi_change"; vals = []
    for f in frames:
        row = f.loc[f["strike"] == strike]
        if row.empty or col not in row.columns: continue
        v = float(row.iloc[0][col])
        ok = (v > 0) if cond == "write_pos" else (v < 0)
        vals.append(1.0 if ok else 0.0)
    return float(np.mean(vals)) if vals else np.nan

def majority_pass(sym: str, exp: str, strike: float, side: str, thresh: float, frames_n: int = 3) -> Optional[bool]:
    """2-of-3 pass heuristic using history frames."""
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
    """delta-based emphasis near 50Δ, bounded and safe if missing."""
    col = f"{side}_delta"
    if col not in s or pd.isna(s[col]):  return 1.0
    d = abs(float(s[col])); return float(np.clip(d/0.50, 0.4, 1.2))

def step_gradient_penalty(tbl: pd.DataFrame, side: str):
    """soft-penalize if writing decays as we step along the path."""
    col = f"{side}_oi_change"
    vals = tbl[col].fillna(0).astype(float).values
    if len(vals) < 3: return np.ones_like(vals)
    slope = np.gradient(vals)
    return np.where(slope >= 0, 1.0, 0.8)

def _compute_clear(tbl: pd.DataFrame) -> Optional[int]:
    """clear level = last strike in the initial pass streak"""
    if tbl.empty: return None
    streak = 0
    for ok in tbl["gate_ok"].tolist():
        if ok: streak += 1
        else: break
    if streak == 0: return None
    return int(tbl.iloc[streak-1]["strike"])

def _compute_clear_with_skip(tbl: pd.DataFrame) -> Optional[int]:
    """allow a single skip if immediately followed by two passes"""
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

# -------------------- core scoring (unchanged API) --------------------

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
    """
    Builds raw UP/DOWN strike tables with gate score & basic pass logic.
    Keep signature for compatibility; enhancements are applied later.
    """
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

# -------------------- confidence score (kept) --------------------

def confidence_score(runway_ratio: float, vel_ok: float, iv_z: Optional[float], gamma_regime: str, pcr: float) -> float:
    iv_boost    = np.clip((iv_z or 0)/2.0, -1, 1)
    gamma_boost = (0.5 if "Short" in (gamma_regime or "") else (-0.5 if "Long" in (gamma_regime or "") else 0))
    pcr_align   = (0.25 if pcr>1.2 else (-0.25 if pcr<0.8 else 0))
    return float(np.clip(100*(0.45*runway_ratio + 0.20*vel_ok + 0.20*max(0,iv_boost) + 0.10*max(0,gamma_boost) + 0.05*max(0,pcr_align)), 0, 100))

# -------------------- adaptive enhancements (extended) --------------------

def _dynamic_settings(
    base_opp_cap: float,
    regime: Optional[str],
    iv_z: Optional[float],
    breakout_up: Optional[float],
    breakout_dn: Optional[float],
    inside_p50: Optional[bool],
    pin_top: Optional[float],
    minutes_since_open: Optional[float],
    minutes_to_close: Optional[float],
) -> Dict[str, object]:
    """Compute adaptive knobs; stays safe if context is None."""
    # Defaults
    first2_cap = max(0.0, base_opp_cap - 0.1)
    later_cap  = base_opp_cap + 0.1
    allow_skip = False
    unwind_boost = 0.0
    min_runway = 2
    blockers = {"time_block": False, "pin_block": False, "regime_block": False}

    # Pin / regime blockers
    if pin_top is not None and pin_top >= 0.35:
        blockers["pin_block"] = True
    if inside_p50 and regime and "Long" in regime:
        blockers["regime_block"] = True

    # Time windows (block early/late unless breakout ≥ 0.60)
    breakout_any = max(breakout_up or 0.0, breakout_dn or 0.0)
    if minutes_since_open is not None and minutes_since_open < 15 and breakout_any < 0.60:
        blockers["time_block"] = True
    if minutes_to_close is not None and minutes_to_close < 30 and breakout_any < 0.60:
        blockers["time_block"] = True

    # Short-gamma + impulse → loosen, skip-1, add unwind weight
    if (regime and "Short" in regime) and ((iv_z is not None and abs(iv_z) >= 2.0) or (breakout_any >= 0.60)):
        allow_skip = True
        unwind_boost = 0.1  # additive on normalized unwind component
        first2_cap = base_opp_cap        # don’t over-tighten early
        later_cap  = base_opp_cap + 0.1
        min_runway = 2

    # Long-gamma or chop → tighten
    if regime and "Long" in regime and not blockers["regime_block"]:
        first2_cap = min(first2_cap, 0.9)
        later_cap  = min(later_cap, 1.0)
        min_runway = 3

    return {
        "first2_cap": float(first2_cap),
        "later_cap": float(later_cap),
        "allow_skip_extra": bool(allow_skip),
        "unwind_boost": float(unwind_boost),
        "min_runway": int(min_runway),
        "blockers": blockers,
    }

def _apply_unwind_boost(tbl: pd.DataFrame, side: str, boost: float) -> pd.DataFrame:
    if boost <= 0 or tbl.empty: return tbl
    t = tbl.copy()
    # Estimate normalized unwind magnitude using p95 of the appropriate side’s change
    if side == "UP":
        p95_neg = _p95_neg_mag(t["CE_oi_change"] if "CE_oi_change" in t else pd.Series([0]))
        norm_unw = t["CE_oi_change"].apply(lambda v: _norm_neg_to_pos(v or 0.0, p95_neg))
    else:
        p95_neg = _p95_neg_mag(t["PE_oi_change"] if "PE_oi_change" in t else pd.Series([0]))
        norm_unw = t["PE_oi_change"].apply(lambda v: _norm_neg_to_pos(v or 0.0, p95_neg))
    t["score"] = (t["score"].astype(float) + boost * norm_unw.astype(float)).clip(upper=1.6)
    return t

def _cumulative_flow_ok(tbl: pd.DataFrame, side: str, take_first_n: int = 3) -> bool:
    if tbl.empty: return False
    take = min(take_first_n, len(tbl))
    if side == "UP":
        # reward PE writing and CE unwind; penalize CE writing
        sub = tbl.head(take)
        pe = pd.to_numeric(sub["PE_oi_change"], errors="coerce").fillna(0)
        ce = pd.to_numeric(sub["CE_oi_change"], errors="coerce").fillna(0)
        net = float((pe - ce).sum())
        return net > 0
    else:
        sub = tbl.head(take)
        ce = pd.to_numeric(sub["CE_oi_change"], errors="coerce").fillna(0)
        pe = pd.to_numeric(sub["PE_oi_change"], errors="coerce").fillna(0)
        net = float((ce - pe).sum())
        return net > 0

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
    sym: str, exp: str,
    # ---- NEW optional context to enable adaptivity (safe defaults) ----
    regime: Optional[str] = None,
    iv_z: Optional[float] = None,
    breakout_up: Optional[float] = None,
    breakout_dn: Optional[float] = None,
    inside_p50: Optional[bool] = None,
    pin_top: Optional[float] = None,
    minutes_since_open: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    min_runway_override: Optional[int] = None,
):
    """
    Finalize gate_ok with delta weighting, liquidity floor, dynamic opp-caps,
    optional history majority, optional skip-1, unwind boost in short-gamma impulse,
    and min-runway enforcement. Signature stays backward compatible.
    """
    # Liquidity floors
    pe_floor = np.percentile(df_ring["PE_oi"].dropna(), liq_pct) if "PE_oi" in df_ring else 0
    ce_floor = np.percentile(df_ring["CE_oi"].dropna(), liq_pct) if "CE_oi" in df_ring else 0

    # Dynamic settings from context
    dyn = _dynamic_settings(
        base_opp_cap=float(opp_cap),
        regime=regime, iv_z=iv_z,
        breakout_up=breakout_up, breakout_dn=breakout_dn,
        inside_p50=inside_p50, pin_top=pin_top,
        minutes_since_open=minutes_since_open, minutes_to_close=minutes_to_close,
    )
    first2_cap = dyn["first2_cap"]
    later_cap  = dyn["later_cap"]
    allow_skip_final = bool(allow_skip or dyn["allow_skip_extra"])
    unwind_boost = dyn["unwind_boost"]
    min_runway_needed = int(min_runway_override or dyn["min_runway"])
    blockers = dyn["blockers"]

    # Prepare delta cols for weighting (safe if missing)
    del_cols = [c for c in ["strike","PE_delta","CE_delta"] if c in df_ring.columns]
    dd = df_ring[del_cols].drop_duplicates("strike") if del_cols else pd.DataFrame(columns=["strike"])

    def _enhance(tbl: pd.DataFrame, side: str):
        if tbl is None or tbl.empty:  return tbl

        # Merge deltas
        t = tbl.copy()
        if not dd.empty:
            t = t.merge(dd, on="strike", how="left")

        # Delta weighting
        if use_delta:
            if side == "UP": t["score"] = t.apply(lambda r: r["score"]*_delta_w("PE", r), axis=1)
            else:            t["score"] = t.apply(lambda r: r["score"]*_delta_w("CE", r), axis=1)

        # Step gradient penalty
        t["score"] = t["score"] * (step_gradient_penalty(t, "PE" if side=="UP" else "CE"))

        # Short-gamma impulse → boost unwind component
        t = _apply_unwind_boost(t, side, unwind_boost)

        # Base booleans
        if side == "UP":
            write_ok = t["PE_oi_change"].fillna(0).astype(float) > 0
            liq_ok   = t["PE_oi"].fillna(0).astype(float) >= pe_floor
            opp = t["opp_pressure"].astype(float)
        else:
            write_ok = t["CE_oi_change"].fillna(0).astype(float) > 0
            liq_ok   = t["CE_oi"].fillna(0).astype(float) >= ce_floor
            opp = t["opp_pressure"].astype(float)

        score_ok = t["score"].astype(float) >= float(strict)

        # Distance-aware opposition caps
        idx = np.arange(len(t))
        caps = np.where(idx < 2, first2_cap, later_cap)  # stricter first two
        opp_ok = opp <= caps

        # Majority confirmation (optional)
        maj_ok = pd.Series([True]*len(t))
        if use_majority:
            if side == "UP":
                maj_ok = t["strike"].apply(lambda k: majority_pass(sym, exp, float(k), "UP", strict, maj_frames))
            else:
                maj_ok = t["strike"].apply(lambda k: majority_pass(sym, exp, float(k), "DOWN", strict, maj_frames))
            maj_ok = maj_ok.fillna(True)

        # Apply blockers (pin, regime-in-rails, early/late)
        block_mask = False
        if any(blockers.values()):
            block_mask = True

        # Final gate_ok
        t["gate_ok"] = (write_ok & score_ok & opp_ok & liq_ok & maj_ok)
        if block_mask:
            t.loc[:, "gate_ok"] = False

        # Reasons (first-fail reason per row)
        def reason_up(r):
            if block_mask: return ("pin_block" if blockers.get("pin_block") else
                                   "regime_block" if blockers.get("regime_block") else
                                   "time_block")
            if r.get("PE_oi",0) < pe_floor: return "low_liq"
            if r["opp_pressure"] > (first2_cap if r.name < 2 else later_cap): return "opp_cap"
            if float(r.get("PE_oi_change",0)) <= 0: return "no_write"
            if float(r.get("CE_oi_change",0)) >= 0: return "no_unwind"
            return "pass" if r["gate_ok"] else "weak_score"

        def reason_down(r):
            if block_mask: return ("pin_block" if blockers.get("pin_block") else
                                   "regime_block" if blockers.get("regime_block") else
                                   "time_block")
            if r.get("CE_oi",0) < ce_floor: return "low_liq"
            if r["opp_pressure"] > (first2_cap if r.name < 2 else later_cap): return "opp_cap"
            if float(r.get("CE_oi_change",0)) <= 0: return "no_write"
            if float(r.get("PE_oi_change",0)) >= 0: return "no_unwind"
            return "pass" if r["gate_ok"] else "weak_score"

        t["reason"] = t.apply(reason_up if side=="UP" else reason_down, axis=1)
        return t

    up_adj   = _enhance(up_tbl, "UP")
    down_adj = _enhance(down_tbl, "DOWN")

    # Clear levels (with optional skip-1)
    up_clear   = _compute_clear_with_skip(up_adj) if allow_skip_final else _compute_clear(up_adj)
    down_clear = _compute_clear_with_skip(down_adj) if allow_skip_final else _compute_clear(down_adj)

    # Enforce min runway & cumulative flow guard on the final clear result
    def _passes_streak_len(tbl) -> int:
        if tbl is None or tbl.empty: return 0
        c = 0
        for ok in tbl["gate_ok"].tolist():
            if ok: c += 1
            else: break
        return c

    if _passes_streak_len(up_adj) < min_runway_needed or not _cumulative_flow_ok(up_adj, "UP", take_first_n=max(2, min_runway_needed)) or any(d for d in blockers.values()):
        up_clear = None
    if _passes_streak_len(down_adj) < min_runway_needed or not _cumulative_flow_ok(down_adj, "DOWN", take_first_n=max(2, min_runway_needed)) or any(d for d in blockers.values()):
        down_clear = None

    return up_adj, down_adj, up_clear, down_clear
