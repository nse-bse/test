# features/predict.py
"""
Predictive intraday utilities:
- expected_move_nowcast: 1σ move to close + probabilistic cones (25/50/75% central).
- pin_probability: finish-probability per strike using GEX × Gaussian cone.
- breakout_probabilities: odds of breaching IV rails (up/down) by close.
- wall_shift_velocity: are OI walls moving toward or away from spot.
- session_phase: quick regime/phase classifier for intraday context.
"""

from __future__ import annotations
from math import erf, sqrt
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------
# Basic normal math (no SciPy)
# ------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _inv_norm_cdf(p: float) -> float:
    """
    Acklam's rational approximation of Φ^{-1}(p).
    Domain: 0 < p < 1. Max error ~ 4.5e-4 which is fine for cones.
    """
    if not (0.0 < p < 1.0):
        if p <= 0.0:
            return -np.inf
        if p >= 1.0:
            return np.inf

    # Coefficients
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [  7.784695709041462e-03,  3.224671290700398e-01,
           2.445134137142996e+00,  3.754408661907416e+00 ]

    plow  = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


# ------------------------
# 1) Expected move nowcast
# ------------------------

def expected_move_nowcast(
    spot: Optional[float],
    atm_iv_pct: Optional[float],
    minutes_left: Optional[float],
    sessions_per_year: float = 252.0,
    minutes_per_session: float = 390.0,
) -> Optional[Dict]:
    """
    Compute 1σ move to close (in points) and central probability cones.
    Returns: {"sigma_pts": float, "cones": {0.25:(lo,hi), 0.50:(lo,hi), 0.75:(lo,hi)}}
    """
    if spot is None or atm_iv_pct is None or minutes_left is None:
        return None
    if not np.isfinite(spot) or not np.isfinite(atm_iv_pct) or minutes_left <= 0:
        return None

    iv = float(atm_iv_pct) / 100.0
    t_years = float(minutes_left) / (sessions_per_year * minutes_per_session)
    sigma_pts = max(1.0, float(spot) * iv * np.sqrt(t_years))  # floor to avoid zeros

    # Central probabilities → two-sided quantiles:
    # central p => tail prob = (1-p)/2, z = Φ^{-1}((1+p)/2)
    cones = {}
    for p_central in (0.25, 0.50, 0.75):
        z = _inv_norm_cdf(0.5 * (1.0 + p_central))  # e.g., 0.50 -> z ≈ 0.674
        lo = float(spot) - z * sigma_pts
        hi = float(spot) + z * sigma_pts
        cones[p_central] = (lo, hi)

    return {"sigma_pts": sigma_pts, "cones": cones}


# ------------------------
# 2) Pin probability (GEX × cone)
# ------------------------

def pin_probability(
    gex_df: pd.DataFrame,
    spot: Optional[float],
    sigma_pts: Optional[float],
    beta: float = 4.0,
):
    """
    Combine GEX magnitude with Gaussian likelihood to produce finish probs per strike.
    Returns (probs_df, best_strike). probs_df columns: ['strike','pin_prob'] sorted desc.
    """
    if gex_df is None or gex_df.empty or spot is None or sigma_pts is None or sigma_pts <= 0:
        return None, None

    df = gex_df.dropna(subset=["strike", "gex"]).copy()
    if df.empty:
        return None, None

    strikes = df["strike"].astype(float).to_numpy()
    gex_mag = np.abs(df["gex"].astype(float).to_numpy())

    # Softmax over GEX magnitude (dealer magnetization)
    g_min, g_max = float(gex_mag.min()), float(gex_mag.max())
    if g_max - g_min < 1e-12:
        w_gex = np.ones_like(gex_mag) / max(1, len(gex_mag))
    else:
        g_norm = (gex_mag - g_min) / (g_max - g_min)
        w_gex = np.exp(beta * g_norm)
        w_gex = w_gex / w_gex.sum()

    # Gaussian likelihood of finishing at strike given sigma
    z = (strikes - float(spot)) / float(sigma_pts)
    w_gauss = np.exp(-0.5 * z * z)
    if w_gauss.sum() <= 0:
        return None, None
    w_gauss = w_gauss / w_gauss.sum()

    probs = w_gex * w_gauss
    total = probs.sum()
    if total <= 0:
        return None, None
    probs = probs / total

    out = pd.DataFrame({"strike": df["strike"].values, "pin_prob": probs})
    out = out.sort_values("pin_prob", ascending=False).reset_index(drop=True)
    best = int(out.iloc[0]["strike"])
    return out, best


# ------------------------
# 3) Breakout odds vs rails
# ------------------------

def breakout_probabilities(
    spot: Optional[float],
    rails_50: Optional[Tuple[float, float]],
    sigma_pts: Optional[float],
) -> Optional[Dict[str, float]]:
    """
    Approximate probability of close finishing beyond the 50% rail bounds.
    Returns {"prob_up_break": float, "prob_dn_break": float} in [0,1].
    """
    if spot is None or rails_50 is None or sigma_pts is None or sigma_pts <= 0:
        return None
    lo, hi = float(rails_50[0]), float(rails_50[1])

    # P(Close > hi) and P(Close < lo) under normal with σ = sigma_pts
    z_up = (hi - float(spot)) / float(sigma_pts)
    z_dn = (lo - float(spot)) / float(sigma_pts)
    prob_up = max(0.0, min(1.0, 1.0 - _norm_cdf(z_up)))
    prob_dn = max(0.0, min(1.0, _norm_cdf(z_dn)))
    return {"prob_up_break": prob_up, "prob_dn_break": prob_dn}


# ------------------------
# 4) Wall-shift velocity
# ------------------------

def wall_shift_velocity(
    curr_walls_df: Optional[pd.DataFrame],
    prev_walls_df: Optional[pd.DataFrame],
    spot: Optional[float],
    k: int = 3,
) -> Optional[float]:
    """
    Average signed shift (pts) of the k nearest walls toward (+) or away (−) from spot.
    Positive => compression/pinning risk. Negative => release/room to run.
    """
    if spot is None or curr_walls_df is None or prev_walls_df is None:
        return None
    if curr_walls_df.empty or prev_walls_df.empty:
        return None

    def _nearest(df: pd.DataFrame) -> np.ndarray:
        dd = df.copy()
        if "strike" not in dd.columns:
            return np.array([])
        dd["dist"] = (dd["strike"].astype(float) - float(spot)).abs()
        return dd.sort_values("dist").head(k)["strike"].astype(float).to_numpy()

    c = _nearest(curr_walls_df)
    p = _nearest(prev_walls_df)
    n = min(len(c), len(p))
    if n == 0:
        return None

    # Positive when current walls are closer to spot than previous walls
    # sign with respect to spot location to avoid left/right bias
    shift = np.sign(float(spot) - p[:n]) * (p[:n] - c[:n])
    return float(np.mean(shift))


# ------------------------
# 5) Session phase classifier
# ------------------------

def session_phase(
    minutes_since_open: Optional[float],
    iv_z: Optional[float],
    regime_text: Optional[str],
    pin_prob_top: Optional[float] = None,
) -> str:
    """
    Very light classifier for intraday context.
    Returns one of: 'Open Drive', 'Mean-Revert', 'Trend', 'Late Pin', 'Unknown'
    """
    if minutes_since_open is None or not np.isfinite(minutes_since_open):
        return "Unknown"

    if minutes_since_open <= 30:
        return "Open Drive"

    reg = regime_text or ""
    if ("Short Gamma" in reg) and (iv_z is not None and np.isfinite(iv_z) and iv_z > 1.5):
        return "Trend"

    if (minutes_since_open >= 180) and (pin_prob_top is not None) and np.isfinite(pin_prob_top) and pin_prob_top >= 0.35:
        return "Late Pin"

    return "Mean-Revert"
