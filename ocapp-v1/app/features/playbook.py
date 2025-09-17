# features/playbook.py

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------- small helpers -------------------------

def _isnum(x) -> bool:
    return isinstance(x, (int, float, np.floating)) and np.isfinite(x)

def _fmt_pct01(x: Optional[float]) -> str:
    return f"{x*100:.0f}%" if _isnum(x) else "–"

def _fmt_signed2(x: Optional[float]) -> str:
    return f"{x:+.2f}" if _isnum(x) else "–"

def _nearest_strike(strikes: List[float], target: float) -> Optional[float]:
    if target is None or not strikes:
        return None
    return float(min(strikes, key=lambda k: abs(k - target)))

def _pick_above(strikes: List[float], k: float) -> Optional[float]:
    ups = [s for s in strikes if s >= k]
    return float(min(ups)) if ups else None

def _pick_below(strikes: List[float], k: float) -> Optional[float]:
    dns = [s for s in strikes if s <= k]
    return float(max(dns)) if dns else None

def _rail_inside(rails: Optional[dict], spot: Optional[float]) -> Optional[bool]:
    try:
        r50 = rails.get(0.5) if isinstance(rails, dict) else None
        if r50 and _isnum(spot):
            lo, hi = float(r50[0]), float(r50[1])
            return (lo <= spot <= hi)
    except Exception:
        pass
    return None

def _vel_tilt(vel_df: Optional[pd.DataFrame]) -> float:
    if vel_df is None or vel_df.empty or "velocity" not in vel_df.columns:
        return 0.0
    up = float(vel_df.loc[vel_df["velocity"] > 0, "velocity"].sum())
    dn = float(-vel_df.loc[vel_df["velocity"] < 0, "velocity"].sum())
    tot = up + dn
    return (up - dn) / tot if tot > 0 else 0.0

def _suggest_size_from_liquidity(stress: Optional[float]) -> int:
    """Ultra-simple position size suggestion from liquidity stress [0..1]."""
    if not _isnum(stress):
        return 1
    if stress >= 0.85:
        return 0
    if stress >= 0.70:
        return 1
    if stress >= 0.50:
        return 2
    return 3

# ------------------------- leg builders -------------------------

def _leg(side: str, action: str, strike: float, qty: int = 1) -> Dict:
    return {"side": side, "action": action, "strike": float(strike), "qty": int(qty)}

def _iron_fly(center_k: float, wing_lo: float, wing_hi: float, lots: int = 1) -> List[Dict]:
    # Short straddle + wings (1:1)
    return [
        _leg("CE", "SELL", center_k, lots),
        _leg("PE", "SELL", center_k, lots),
        _leg("CE", "BUY",  wing_hi, lots),
        _leg("PE", "BUY",  wing_lo, lots),
    ]

def _credit_condor(lo_short: float, lo_long: float, hi_short: float, hi_long: float, lots: int = 1) -> List[Dict]:
    # PE spread + CE spread, all short strikes inside long strikes
    return [
        _leg("PE", "SELL", lo_short, lots),
        _leg("PE", "BUY",  lo_long,  lots),
        _leg("CE", "SELL", hi_short, lots),
        _leg("CE", "BUY",  hi_long,  lots),
    ]

def _debit_call_spread(k_buy: float, k_sell: float, lots: int = 1) -> List[Dict]:
    return [_leg("CE", "BUY", k_buy, lots), _leg("CE", "SELL", k_sell, lots)]

def _debit_put_spread(k_buy: float, k_sell: float, lots: int = 1) -> List[Dict]:
    return [_leg("PE", "BUY", k_buy, lots), _leg("PE", "SELL", k_sell, lots)]

# ------------------------- main API -------------------------

def build_playbook(
    *,
    df_ring: pd.DataFrame,
    working_spot: Optional[float],
    rails: Optional[dict],
    em: Optional[dict],
    pcr: Optional[float],
    iv_z: Optional[float],
    regime: Optional[str],
    vel_df: Optional[pd.DataFrame],
    pin_k: Optional[float],
    pin_top: Optional[float],
    walls_tbl: Optional[pd.DataFrame],
    liq_stress: Optional[float],
    direction_score: Optional[int],
) -> List[Dict]:
    """
    Produce a set of ready-to-trade templates (plays) with strikes, rationale and suggested size.
    Returns a list of dicts with keys: name, bias, rationale, legs, levels, size_lots.
    """
    plays: List[Dict] = []

    if df_ring is None or df_ring.empty or not _isnum(working_spot):
        return plays

    strikes = [float(s) for s in df_ring["strike"].dropna().tolist()]
    if not strikes:
        return plays

    atm = _nearest_strike(strikes, working_spot) or float(strikes[len(strikes)//2])
    inside_p50 = _rail_inside(rails, working_spot)
    r50 = rails.get(0.5) if isinstance(rails, dict) else None
    sigma_pts = (em or {}).get("sigma_pts", None)

    # pin logic (safe)
    pin_ok = (_isnum(pin_top) and pin_top >= 0.35)
    center_label = f"Pin {int(pin_k)}" if (pin_ok and _isnum(pin_k)) else "ATM"
    center_k = _nearest_strike(strikes, float(pin_k)) if (pin_ok and _isnum(pin_k)) else atm

    # velocity tilt & bias
    tilt = _vel_tilt(vel_df)
    dscore = int(direction_score) if isinstance(direction_score, (int, float)) else 50
    bias = (
        "Trend" if dscore >= 60 else
        "Mean-Revert" if dscore <= 40 else
        "Neutral"
    )

    # wing selection helpers
    # Prefer rails(50%) if available; else use ~0.7σ for wings; fallback fixed width ~1 step
    if r50 and all(map(_isnum, r50)):
        lo_w, hi_w = float(r50[0]), float(r50[1])
    elif _isnum(sigma_pts):
        lo_w, hi_w = working_spot - 0.7 * float(sigma_pts), working_spot + 0.7 * float(sigma_pts)
    else:
        # ~one step away each side
        lo_w, hi_w = working_spot - (strikes[1] - strikes[0] if len(strikes) > 1 else 100.0), \
                     working_spot + (strikes[1] - strikes[0] if len(strikes) > 1 else 100.0)

    wing_lo = _pick_below(strikes, lo_w) or strikes[0]
    wing_hi = _pick_above(strikes, hi_w) or strikes[-1]

    # size suggestion (lots) from liquidity stress
    size_lots = _suggest_size_from_liquidity(liq_stress)

    # Common rationale prefix (None-safe IV-z)
    rationale_prefix = (
        f"{center_label} center; "
        f"rails inside={bool(inside_p50) if inside_p50 is not None else '–'}; "
        f"IV-z={_fmt_signed2(iv_z)} if shown"
    )

    # ------------------ Play 1: Iron Fly (pin/range) ------------------
    # Good when inside rails, pin risk, or long-gamma
    if (inside_p50 is True) or pin_ok or (isinstance(regime, str) and "Long" in regime):
        legs_if = _iron_fly(center_k, wing_lo, wing_hi, max(1, size_lots or 1))
        plays.append({
            "name": "Iron Fly (pin/range)",
            "bias": "Range / Mean-Revert",
            "rationale": f"{rationale_prefix}; favor credit around magnet.",
            "legs": legs_if,
            "levels": {
                "entry": float(center_k),
                "targets": [float(wing_lo), float(wing_hi)],
                "invalidates": list(map(float, r50)) if r50 else None
            },
            "size_lots": max(1, size_lots or 1)
        })

    # ------------------ Play 2: Credit Condor (inside rails) ------------------
    if r50 and inside_p50 is not None:
        mid = float(center_k)
        inner_lo = _pick_below(strikes, (mid + r50[0]) / 2) or wing_lo
        inner_hi = _pick_above(strikes, (mid + r50[1]) / 2) or wing_hi
        legs_ic = _credit_condor(inner_lo, wing_lo, inner_hi, wing_hi, max(1, size_lots or 1))
        plays.append({
            "name": "Credit Condor (rails)",
            "bias": "Range / Fade",
            "rationale": f"{rationale_prefix}; collect theta within p50 rails.",
            "legs": legs_ic,
            "levels": {
                "entry": float(mid),
                "invalidates": list(map(float, r50)) if r50 else None
            },
            "size_lots": max(1, size_lots or 1)
        })

    # ------------------ Play 3: Debit Call Spread (break ↑) ------------------
    if (dscore >= 60 and tilt >= 0.05) or (isinstance(regime, str) and "Short" in regime and tilt >= 0):
        k_buy = _pick_above(strikes, working_spot)
        k_sell = _pick_above(strikes, (working_spot + (sigma_pts or 0))) if _isnum(sigma_pts) else _pick_above(strikes, working_spot * 1.01)
        if _isnum(k_buy) and _isnum(k_sell) and k_sell > k_buy:
            plays.append({
                "name": "Debit Call Spread (breakout ↑)",
                "bias": "Trend Up",
                "rationale": f"{rationale_prefix}; directional bias ON (score={dscore}), tilt={tilt:+.2f}.",
                "legs": _debit_call_spread(k_buy, k_sell, max(1, size_lots or 1)),
                "levels": {
                    "entry": float(k_buy),
                    "targets": [float(k_sell)],
                    "invalidates": [float(r50[0])] if r50 else None
                },
                "size_lots": max(1, size_lots or 1)
            })

    # ------------------ Play 4: Debit Put Spread (break ↓) ------------------
    if (dscore >= 60 and tilt <= -0.05) or (isinstance(regime, str) and "Short" in regime and tilt <= 0):
        k_sell = _pick_below(strikes, (working_spot - (sigma_pts or 0))) if _isnum(sigma_pts) else _pick_below(strikes, working_spot * 0.99)
        k_buy  = _pick_below(strikes, working_spot)
        if _isnum(k_buy) and _isnum(k_sell) and k_buy > k_sell:
            plays.append({
                "name": "Debit Put Spread (breakout ↓)",
                "bias": "Trend Down",
                "rationale": f"{rationale_prefix}; directional bias ON (score={dscore}), tilt={tilt:+.2f}.",
                "legs": _debit_put_spread(k_buy, k_sell, max(1, size_lots or 1)),
                "levels": {
                    "entry": float(k_buy),
                    "targets": [float(k_sell)],
                    "invalidates": [float(r50[1])] if r50 else None
                },
                "size_lots": max(1, size_lots or 1)
            })

    # Fallback: if nothing built (very edge case), at least emit a neutral iron fly at ATM
    if not plays:
        legs_if = _iron_fly(atm, wing_lo, wing_hi, 1)
        plays.append({
            "name": "Iron Fly (fallback)",
            "bias": "Range",
            "rationale": f"{rationale_prefix}; fallback when signals mixed/insufficient.",
            "legs": legs_if,
            "levels": {"entry": float(atm), "targets": [float(wing_lo), float(wing_hi)]},
            "size_lots": 1
        })

    return plays


# ------------------------- UI renderer -------------------------

def render_playbook_card(play: Dict, lot_size_default: int = 25) -> None:
    """
    Streamlit card renderer for a single play dict returned by build_playbook.
    Expected keys: name, bias, rationale, legs[list], levels{...}, size_lots
    """
    name = play.get("name", "Play")
    bias = play.get("bias", "—")
    rationale = play.get("rationale", "")
    legs = play.get("legs", []) or []
    levels = play.get("levels", {}) or {}
    size_lots = int(play.get("size_lots", 1))

    with st.container(border=True):
        topL, topR = st.columns([3, 1])
        with topL:
            st.markdown(f"### {name}")
            st.caption(bias)
            if rationale:
                st.write(rationale)
        with topR:
            st.metric("Suggested size (lots)", max(0, size_lots))
            st.caption(f"Lot size ref: {lot_size_default}")

        # Legs table
        if legs:
            df = pd.DataFrame(legs)
            df["strike"] = df["strike"].map(lambda x: f"{x:.0f}")
            df["qty"] = df["qty"].astype(int)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Levels (optional)
        if levels:
            c1, c2, c3 = st.columns(3)
            if levels.get("entry") is not None:
                c1.metric("Entry ref", f"{levels['entry']:.0f}")
            if levels.get("targets"):
                t = ", ".join([f"{float(x):.0f}" for x in levels["targets"]])
                c2.metric("Targets", t)
            if levels.get("invalidates"):
                inv = ", ".join([f"{float(x):.0f}" for x in levels["invalidates"]])
                c3.metric("Invalidation", inv)

        with st.expander("Play JSON", expanded=False):
            st.code(pd.Series(play).to_json(force_ascii=False, indent=2), language="json")
