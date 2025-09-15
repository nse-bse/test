# app/ui/app_main.py
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import json
import requests

# ------------------ local modules ------------------
from config import (
    now_ist, IST,
    NIFTY_HISTORICAL_EXPIRIES, BANKNIFTY_HISTORICAL_EXPIRIES,
    FNO_STOCKS, INDEX_SYMBOLS,
)
from utils.time_utils import market_bounds, next_refresh_in_seconds, on_market_tick
from data.api import (
    load_expiries, load_snapshot_from_api, load_snapshot_from_hist_api,
    load_cash_spot_from_api, snapshot_digest,
)
from calculations.chain import (
    group_chain, find_atm_index, compute_pcr, compute_max_pain,
    estimate_cash_spot_parity_consistent, get_writing_commentary, build_movers_long,
)
from calculations.iv_skew import (
    compute_atm_iv, compute_rr_bf, make_iv_skew_history, zscore_last, compute_vega_weighted_iv_flow,
)
from state.history import (
    push_live_ring_snapshot, record_ring_snapshot, make_long_history,
    get_last_frames, velocity_spike_table,
)
from features.gamma_and_rails import (
    compute_gamma_regime, compute_iv_rails, detect_stop_hunt, score_zones,
    summarize_zones, summarize_trade_bias, build_trade_commentary, _format_zone_list,
)
from features.money_map import (
    compute_money_map, money_center_of_mass, find_walls, make_money_map_commentary
)
from features.liquidity import (
    compute_spread_stats, impact_cost_proxy, liquidity_stress, gex_curve,
    realized_vol_annualized, pcr_vol, make_quick_execution_commentary,
    liquidity_stress_v2, estimate_rupee_impact, plan_order_sizes, rank_strikes_by_edge_pressure,
    find_execution_windows, 
)
from features.liquidity import liquidity_stress_with_stats

from features.runway import (
    build_gate_runway_tables, apply_runway_enhancements, confidence_score,
)
from ui.styling import style_chain, movers_chart, net_oi_change_chart, style_mover_table

# Predictive layer
from features.predict import (
    expected_move_nowcast, pin_probability, breakout_probabilities,
    wall_shift_velocity, session_phase,
)

# Soft dependency: heartbeat
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# ---------------------------- helpers ----------------------------
def _isnum(x):
    return isinstance(x, (int, float)) and np.isfinite(x)

@st.cache_data(show_spinner=False, ttl=300)
def _load_expiries_for_session(source: str, symbol: str, api_base: str):
    """Unifies live + historical lists when in Historical mode. Stocks fallback → BANKNIFTY monthly."""
    if source == "Live API":
        return load_expiries(api_base, symbol)
    live_expiries, _ = load_expiries(api_base, symbol)
    historical = NIFTY_HISTORICAL_EXPIRIES if symbol == "NIFTY" else BANKNIFTY_HISTORICAL_EXPIRIES
    try:
        all_exps = sorted(
            set(live_expiries) | set(historical),
            key=lambda x: datetime.strptime(x, "%d%b%y"),
            reverse=True,
        )
    except Exception:
        all_exps = sorted(set(live_expiries) | set(historical), reverse=True)
    return all_exps, None


def _direction_score(
    regime: str,
    iv_z: Optional[float],
    vel_df: pd.DataFrame,
    up_tbl: Optional[pd.DataFrame],
    down_tbl: Optional[pd.DataFrame],
    net_iv_flow: Optional[float],
    pcr_window: float
) -> int:
    """
    Lightweight 0–100 score: >60 trend/break bias; <40 mean-revert bias.
    Components: gamma regime, IV impulse, velocity tilt, gates edge, IV-flow, PCR tilt.
    """
    score = 50.0

    # Gamma regime
    if regime:
        if "Short" in regime: score += 12
        elif "Long" in regime: score -= 12

    # IV impulse: more |z| pushes away from mean-revert; sign adds direction only via velocity/gates
    if iv_z is not None and np.isfinite(iv_z):
        score += np.clip(abs(iv_z), 0, 3.0) * 6.0  # up to +18 when vol is impulsive

    # Velocity tilt
    if vel_df is not None and not vel_df.empty and "velocity" in vel_df.columns:
        up_v = float(vel_df.loc[vel_df["velocity"] > 0, "velocity"].sum())
        dn_v = float(-vel_df.loc[vel_df["velocity"] < 0, "velocity"].sum())
        total = up_v + dn_v
        if total > 0:
            tilt = (up_v - dn_v) / total  # -1..+1
            score += float(tilt) * 10.0

    # Gates edge
    try:
        up_edge = float(up_tbl["score"].mean()) if (up_tbl is not None and not up_tbl.empty) else 0.0
        dn_edge = float(down_tbl["score"].mean()) if (down_tbl is not None and not down_tbl.empty) else 0.0
        score += np.tanh((up_edge - dn_edge) / 5.0) * 10.0
    except Exception:
        pass

    # IV flow: net vol buying/selling
    if net_iv_flow is not None and np.isfinite(net_iv_flow):
        score += np.tanh(net_iv_flow / 5e6) * 6.0

    # PCR tilt: very low/high moves score from neutrality
    if np.isfinite(pcr_window):
        score += np.tanh((pcr_window - 1.0) * 1.5) * 6.0

    return int(np.clip(round(score), 0, 100))


# ------------------------------ app ------------------------------

def render_app():
    # ---- session state ----
    ss = st.session_state
    ss.setdefault("last_fetch", 0.0)
    ss.setdefault("snapshot", None)
    ss.setdefault("last_live_digest", None)
    ss.setdefault("last_hist_digest", None)
    ss.setdefault("spot_hist", {})  # {f"{sym}:{exp}": [spot,...]}
    ss.setdefault("api_hits", 0)

    # Historical manual control state
    ss.setdefault("hist_date_pending", datetime.today().date())
    ss.setdefault("hist_time_pending", "15:30")
    ss.setdefault("hist_sel_key", None)
    ss.setdefault("current_source", None)
    ss.setdefault("liq_hist", {})   # { "SYM:EXP": [stress0..n] }  (floats 0..1)

    # ---- sidebar: source & symbol ----
    st.sidebar.header("Data Source & Symbol")
    source = st.sidebar.selectbox("Source", ["Live API", "Historical API"], index=0)
    api_base = st.sidebar.text_input("API Base", value="http://127.0.0.1:8000")
    inst = st.sidebar.radio("Instrument", ["Index", "Stock"], index=0, horizontal=True)

    if inst == "Index":
        symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
    else:
        default_idx = FNO_STOCKS.index("RELIANCE") if "RELIANCE" in FNO_STOCKS else 0
        symbol = st.sidebar.selectbox("Symbol", FNO_STOCKS, index=default_idx)
    symbol = symbol.strip().upper()

    expiry_list, warn_msg = _load_expiries_for_session(source, symbol, api_base)
    if warn_msg:
        st.sidebar.warning(f"Could not fetch expiries from API. Using static list. Reason: {warn_msg}")
    expiry = st.sidebar.selectbox("Expiry", expiry_list, index=0)

    st.sidebar.header("Live cadence")
    cadence_mode = st.sidebar.selectbox(
        "Cadence", ["Market schedule (09:15 → 09:17 → every 3 min)", "Fixed seconds"], index=0
    )
    refresh_sec = st.sidebar.number_input(
        "Refresh interval (seconds for Fixed)", min_value=10, max_value=600, value=180, step=10
    )

    # Default: Cash (API) for indices; Futures for stocks
    _default_spot_idx = 1 if symbol in INDEX_SYMBOLS else 0
    spot_source = st.sidebar.selectbox(
        "Use spot as",
        ["Futures (from OC snapshot)", "Cash (API)", "Cash (estimate via parity)"],
        index=_default_spot_idx,
    )

    # ---- view controls ----
    st.sidebar.header("View")
    strike_window = st.sidebar.slider("± Strikes around ATM", min_value=5, max_value=40, value=15, step=1)
    with st.sidebar.expander("Analytics Ring Settings", expanded=False):
        fixed_skew_ring = st.number_input("Ring size (± strikes)", min_value=5, max_value=25, value=10, step=1)
    with st.sidebar.expander("Heatmap Settings", expanded=True):
        enable_heatmap = st.checkbox("Enable heatmaps", value=True)
        available_metric_choices = ["oi_change", "oi", "volume", "ltp", "net_oi_change"]
        heatmap_metrics = st.multiselect(
            "Heatmap metrics",
            available_metric_choices,
            default=["oi_change", "oi", "net_oi_change"],
        )

    # Heartbeat (rerun)
    if st_autorefresh:
        hb = st_autorefresh(interval=60000, key="heartbeat")
        st.sidebar.caption(f"🫀 heartbeat #{hb}")
    else:
        st.sidebar.warning("`streamlit-autorefresh` not installed → no timed re-runs.")

    # ---- handle source transitions & stale views ----
    prev_source = ss.get("current_source")
    ss["current_source"] = source
    if source == "Historical API":
        sel_key = (api_base, symbol, expiry)
        if prev_source != "Historical API" or ss.get("hist_sel_key") != sel_key:
            ss["hist_sel_key"] = sel_key
            ss["snapshot"] = None
            ss["last_hist_digest"] = None

    # ---- fetch logic ----
    snapshot = ss["snapshot"]
    now = now_ist()
    seconds_to_next_tick = None
    did_fetch = False

    if source == "Live API" and ss.get("last_fetch") is not None:
        auto = st.sidebar.checkbox("Auto refresh", value=True)
        fetch_live_now = st.sidebar.button("Fetch Live Snapshot")
        if auto:
            if cadence_mode.startswith("Market"):
                seconds_to_next_tick = next_refresh_in_seconds(now)
                should_fetch = on_market_tick(now, ss["last_fetch"])
            else:
                elapsed = time.time() - ss["last_fetch"]
                seconds_to_next_tick = int(max(0, refresh_sec - elapsed))
                should_fetch = elapsed >= refresh_sec
        else:
            should_fetch = False
        if fetch_live_now: should_fetch = True
        if should_fetch:
            ss["snapshot"] = load_snapshot_from_api(api_base, symbol, expiry)
            ss["last_fetch"] = time.time()
            ss["last_live_digest"] = snapshot_digest(ss["snapshot"]) if ss["snapshot"] else None
            ss["api_hits"] += 1
            did_fetch = True
            now = now_ist()
            seconds_to_next_tick = next_refresh_in_seconds(now) if cadence_mode.startswith("Market") else int(refresh_sec)

    elif source == "Historical API":
        with st.sidebar.expander("Historical Settings", expanded=True):
            _ = st.date_input("Date", key="hist_date_pending", value=ss["hist_date_pending"])
            def _slots(start="09:15", end="15:30", step=3):
                slots, cur, end_t = [], datetime.strptime(start, "%H:%M"), datetime.strptime(end, "%H:%M")
                while cur <= end_t:
                    slots.append(cur.strftime("%H:%M")); cur += timedelta(minutes=step)
                return slots
            all_times = _slots()
            try:
                default_idx = all_times.index(ss.get("hist_time_pending", "15:30"))
            except ValueError:
                default_idx = len(all_times) - 1
            sel_time = st.selectbox("Time (HH:MM)", all_times, index=default_idx, key="hist_time_select")
            ss["hist_time_pending"] = sel_time
            fetch_hist_now = st.button("Fetch Historical Snapshot", use_container_width=True)
            st.caption("Changing Date/Time does not update the view. Click **Fetch** to load that snapshot.")
        if fetch_hist_now:
            pending_date = ss.get("hist_date_pending", datetime.today().date())
            pending_time = ss.get("hist_time_pending", "15:30")
            ss["snapshot"] = load_snapshot_from_hist_api(
                api_base, symbol, expiry, pending_date.strftime("%Y-%m-%d"), pending_time
            )
            ss["last_hist_digest"] = snapshot_digest(ss["snapshot"]) if ss["snapshot"] else None
            ss["api_hits"] += 1
            did_fetch = True

    snapshot = ss["snapshot"]
    if snapshot is None:
        st.title("Option Chain Dashboard")
        left, right = st.columns([3,1])
        with left:
            st.caption("No snapshot yet." if source == "Historical API" else "Waiting for first live fetch…")
            if seconds_to_next_tick is not None:
                st.caption(
                    f"Next auto-fetch in ~{seconds_to_next_tick}s → "
                    f"at **{(now + timedelta(seconds=seconds_to_next_tick)).strftime('%H:%M:%S')} IST**"
                )
        with right:
            st.metric("API hits (session)", ss.get("api_hits", 0))
        return

    # ---- processing ----
    sym = snapshot.get("symbol", symbol)
    exp = snapshot.get("expiry", expiry)
    ts = snapshot.get("timestamp", "?")

    df, fut_spot, all_fields = group_chain(snapshot)

    # Spot routing
    cash_spot = None
    if symbol in FNO_STOCKS:  # STOCKS → always Cash(API)
        cash_spot = load_cash_spot_from_api(api_base, sym)
        working_spot = cash_spot if cash_spot is not None else fut_spot
        if cash_spot is None:
            st.warning("Cash spot unavailable for stock; using Futures spot.")
    else:  # INDICES → honor dropdown
        working_spot = fut_spot
        if spot_source == "Cash (API)":
            cash_spot = load_cash_spot_from_api(api_base, sym)
            working_spot = cash_spot if cash_spot is not None else fut_spot
            if cash_spot is None:
                st.warning("Cash spot unavailable; using Futures spot.")
        elif spot_source == "Cash (estimate via parity)":
            cash_spot = estimate_cash_spot_parity_consistent(df, fut_spot)
            working_spot = cash_spot if cash_spot is not None else fut_spot
            if cash_spot is None:
                st.warning("Parity estimate unavailable; using Futures spot.")

    # Keep a short spot history (floats) for RV & diagnostics
    if did_fetch and (working_spot is not None):
        key = f"{sym}:{exp}"
        buf = ss["spot_hist"].setdefault(key, [])
        buf.append(float(working_spot))
        if len(buf) > 360:
            del buf[:len(buf) - 360]

    if df.empty:
        st.warning("No option rows returned by API for this selection.")
        return

    atm_idx = find_atm_index(df, working_spot)
    start = max(0, atm_idx - strike_window)
    end = min(len(df), atm_idx + strike_window + 1)
    dfw = df.iloc[start:end].reset_index(drop=True)
    atm_idx_local = atm_idx - start

    df_ana = df.iloc[max(0, atm_idx - fixed_skew_ring):min(len(df), atm_idx + fixed_skew_ring + 1)].reset_index(drop=True)
    atm_idx_ana = atm_idx - max(0, atm_idx - fixed_skew_ring)

    # record frames
    if did_fetch:
        if source == "Live API":
            push_live_ring_snapshot(sym, exp, ts, df_ana, ss.get("last_live_digest"))
        else:
            record_ring_snapshot(sym, exp, ts, df_ana)

    if "net_oi_change" in df.columns and "net_oi_change" not in all_fields:
        all_fields.append("net_oi_change")
    default_fields = [f for f in ["ltp","oi","oi_change","iv","delta","net_oi_change"] if f in all_fields]
    selected_fields = st.sidebar.multiselect("Fields to display", all_fields, default=default_fields)

    heat_cols: List[str] = []
    if enable_heatmap:
        for base in heatmap_metrics:
            if base == "net_oi_change":
                if "net_oi_change" in dfw.columns:
                    heat_cols.append("net_oi_change")
            else:
                ce_c = f"CE_{base}"; pe_c = f"PE_{base}"
                if ce_c in dfw.columns: heat_cols.append(ce_c)
                if pe_c in dfw.columns: heat_cols.append(pe_c)

    pcr_window = compute_pcr(df_ana)
    pcr_full = compute_pcr(df)
    show_both_pcr = (
        np.isfinite(pcr_window) and np.isfinite(pcr_full) and
        (abs(pcr_window - pcr_full) / max(1e-9, abs(pcr_full)) > 0.05)
    )

    max_pain_window = compute_max_pain(df_ana)
    max_pain_full = compute_max_pain(df)

    # --------------------------- Title + header ---------------------------
    st.title("Option Chain Dashboard")
    topL, topR = st.columns([3,1])
    with topL:
        st.caption(f"Last fetched: {ts}")
        if source == "Live API" and seconds_to_next_tick is not None:
            if cadence_mode.startswith("Market"):
                st.caption(
                    f"Next market tick in ~{seconds_to_next_tick}s → "
                    f"at **{(now + timedelta(seconds=seconds_to_next_tick)).strftime('%H:%M:%S')} IST** "
                    "(09:15 → 09:17 → every 3 mins)"
                )
            else:
                st.caption(
                    f"Auto-fetch in ~{seconds_to_next_tick}s → "
                    f"at **{(now + timedelta(seconds=seconds_to_next_tick)).strftime('%H:%M:%S')} IST**"
                )
    with topR:
        st.metric("API hits (session)", ss.get("api_hits", 0))

    # ---- Tabs (added 'Commentary') ----
    tab_overview, tab_runway, tab_money, tab_chain, tab_writing, tab_velocity, tab_movers, tab_iv, tab_commentary, tab_diag = st.tabs(
        ["Overview", "Gates & Runway", "Money Map", "Chain", "Writing", "Velocity", "Movers", "IV & Skew", "Commentary", "Diagnostics"]
    )

    # --------------------------- Commentary (One-Glance + Narrative + AI JSON) ---------------------------
    with tab_commentary:
        st.subheader("One-Glance Commentary")
    
        # ---------- helpers ----------
        def _isnum(x):
            return isinstance(x, (int, float, np.floating)) and np.isfinite(x)
    
        def _fmt(x, n=2, suffix=""):
            return f"{x:.{n}f}{suffix}" if _isnum(x) else "–"
    
        def _pct(x):
            return f"{x*100:.0f}%" if _isnum(x) else "–"
    
        def _safe_mean(vs):
            vs = [v for v in vs if _isnum(v)]
            return float(np.mean(vs)) if vs else None
    
        # Direction score: 0 (mean revert) .. 100 (directional)
        def _direction_score(regime, iv_z, vel_df, up_tbl, down_tbl, net_flow, pcr_ring):
            score = 50.0
    
            # gamma regime
            if isinstance(regime, str):
                if "Short" in regime: score += 15
                elif "Long" in regime: score -= 15
    
            # IV impulse
            if _isnum(iv_z):
                score += np.clip(abs(iv_z), 0, 4) * 6  # up to +24 on big impulse
    
            # runway pass ratios
            def _pass_ratio(tbl):
                if tbl is None or isinstance(tbl, (list, tuple)) or len(getattr(tbl, "columns", [])) == 0: 
                    return None
                return float(tbl["gate_ok"].mean()) if "gate_ok" in tbl.columns and len(tbl) else None
            r_up, r_dn = _pass_ratio(up_tbl), _pass_ratio(down_tbl)
            if _isnum(r_up) or _isnum(r_dn):
                # favor the stronger side
                d = (r_up or 0) - (r_dn or 0)
                score += d * 20  # +/-20 tilt
    
            # velocity (net)
            if vel_df is not None and not vel_df.empty and "velocity" in vel_df.columns:
                v = float(vel_df["velocity"].sum())
                score += np.tanh(v / 5_000) * 10  # soft clip
    
            # vega-weighted IV net flow (signed)
            if _isnum(net_flow):
                score += np.tanh(net_flow / 1e6) * 10
    
            # PCR tilt (extremes reduce trend score)
            if _isnum(pcr_ring):
                if pcr_ring > 1.4: score -= 6   # heavy put writing bias → reversion risk
                if pcr_ring < 0.6: score -= 6   # heavy call writing bias → reversion risk
    
            return int(np.clip(score, 0, 100))
    
        # ---------- inputs & light recompute ----------
        now2 = now_ist()
        df_for_calc = df_ana if not df_ana.empty else df
    
        atm_iv_now = compute_atm_iv(df_for_calc, working_spot)
        iv_hist = make_iv_skew_history(sym, exp, last_n=40)
        iv_z = zscore_last(iv_hist["atm_iv"], window=20) if (iv_hist is not None and not iv_hist.empty and "atm_iv" in iv_hist) else None
    
        regime, gex_val = compute_gamma_regime(df_for_calc, working_spot, ring_size=fixed_skew_ring if fixed_skew_ring else 10)
    
        mstart, mend = market_bounds(now2)
        minutes_left = max(1, (mend - now2).total_seconds() / 60.0) if mend else 60.0
    
        rails = compute_iv_rails(working_spot, atm_iv_now, minutes_left)
        em = expected_move_nowcast(working_spot, atm_iv_now, minutes_left)
    
        r50 = rails.get(0.5) if isinstance(rails, dict) else None
        breakout = breakout_probabilities(working_spot, r50, em["sigma_pts"]) if (r50 and em) else None
    
        gc = gex_curve(df_for_calc, working_spot)
        probs, pin_k = (None, None)
        if em and not gc.empty:
            probs, pin_k = pin_probability(gc, working_spot, em["sigma_pts"])
    
        # runway quick calc (non-strict; small lookback)
        _, vel_df = make_long_history(sym, exp, last_n=10)
        up_tbl = down_tbl = None
        if not df_ana.empty and working_spot is not None:
            up_tbl, down_tbl, *_ = build_gate_runway_tables(
                df_ring=df_ana, spot=working_spot, atm_idx_in_ring=atm_idx_ana, symbol=sym, expiry=exp,
                up_n=6, down_n=6, lookback_frames=6, pass_threshold=0.85,
            )
    
        frames = get_last_frames(sym, exp, n=2)
        df_prev_for_flow = frames[-2] if len(frames) == 2 else None
        flows, net_flow = compute_vega_weighted_iv_flow(df_for_calc, df_prev_for_flow)
    
        # money map + walls + wall-shift velocity
        dfm = compute_money_map(df_for_calc)
        wall_tbl = find_walls(dfm, z_thresh=1.5, k=4) if not dfm.empty else pd.DataFrame()
        prev_walls = None
        if len(frames) == 2:
            prev_dfm = compute_money_map(frames[-2])
            prev_walls = find_walls(prev_dfm, z_thresh=1.5, k=4)
        wsv = wall_shift_velocity(wall_tbl, prev_walls, working_spot) if not wall_tbl.empty else None
    
        # liquidity pack
        liq_stats = compute_spread_stats(df_for_calc, working_spot, band=3)
        icp = impact_cost_proxy(df_for_calc, working_spot, qty=50)
        liq_data = {}
        if liq_stats and icp:
            liq_data["median_spread_bps"] = liq_stats.get("median_spread_bps")
            liq_data["ic_bps"] = icp.get("ic_bps")
            ring_df = compute_money_map(df_ana if not df_ana.empty else df)  # used as 'ring' for turnover if present
            stats = liquidity_stress_with_stats(
                liq_data["median_spread_bps"], liq_data["ic_bps"], ring_df,
                ss["liq_hist"].get(f"{sym}:{exp}")
            )
            liq_data["liquidity_stress"] = stats["stress"]

            # push to intraday history buffer
            key = f"{sym}:{exp}"
            buf = ss["liq_hist"].setdefault(key, [])
            if stats["stress"] is not None:
                buf.append(float(stats["stress"]))
                if len(buf) > 480:    # ~ a full day @ 1 per tick
                    del buf[:len(buf) - 480]
    
        # sentiment / PCR
        pcr_ring = compute_pcr(df_ana)
        sentiment, _writing_text = get_writing_commentary(df_ana)
    
        # phase & direction score
        minutes_since_open = (now2 - mstart).total_seconds() / 60.0 if mstart else None
        pin_top = float(probs["pin_prob"].iloc[0]) if (probs is not None and not probs.empty and "pin_prob" in probs.columns) else None
        phase = session_phase(minutes_since_open, iv_z, regime, pin_prob_top=pin_top)
        dir_score = _direction_score(regime, iv_z, vel_df, up_tbl, down_tbl, net_flow, pcr_ring)
    
        # ---------- headline narrative ----------
        gamma_label = "Long-gamma" if isinstance(regime, str) and "Long" in regime else ("Short-gamma" if isinstance(regime, str) and "Short" in regime else "Mixed")
        if gamma_label == "Long-gamma":
            st.markdown("**🟢 Long-gamma: mean-revert bias; prefer fade/credit structures.**")
        elif gamma_label == "Short-gamma":
            st.markdown("**🔴 Short-gamma: trend/breakout risk; prefer momentum/debit structures.**")
        else:
            st.markdown("**⚖️ Mixed gamma: stay tactical; let rails & IV guide.**")
    
        if _isnum(pin_top) and pin_k is not None:
            st.markdown(f"🎯 **Pin risk** near **{int(pin_k)}** (top pin {_pct(pin_top)}).")
    
        # Mean-revert vs breakout label
        if dir_score <= 40:
            st.info("**Mean-revert bias ON** (score ≤40).")
        elif dir_score >= 60:
            st.warning("**Directional bias ON** (score ≥60).")
        else:
            st.caption("Neutral/transition — wait for alignment (rails + IV + velocity).")
    
        # ---------- One-glance metric rows ----------
        a1, a2, a3 = st.columns([1.2, 1, 1])
        with a1:
            st.metric("Session phase", phase if phase else "–")
            if em: st.metric("Expected move (1σ)", f"±{em['sigma_pts']:.0f} pts")
            if r50: st.caption(f"Rails 50%: {int(r50[0])} ↔ {int(r50[1])}")
        with a2:
            st.metric("Gamma Regime", regime if isinstance(regime, str) else "–")
            if breakout and "prob_up_break" in breakout:
                st.metric("Breakout↑ (vs 50% rail)", _pct(breakout["prob_up_break"]))
        with a3:
            st.metric("Direction score", f"{dir_score}/100")
            if probs is not None and pin_k is not None:
                st.metric("Pin target", f"{int(pin_k)}")
    
        b1, b2, b3 = st.columns(3)
        with b1:
            if not dfm.empty:
                cm_p = money_center_of_mass(dfm, use_premium=True)
                cm_o = money_center_of_mass(dfm, use_premium=False)
                st.metric("Premium Center", f"{cm_p:.0f}" if _isnum(cm_p) else "–")
                st.metric("OI Center", f"{cm_o:.0f}" if _isnum(cm_o) else "–")
        with b2:
            if _isnum(wsv):
                st.metric("Wall-shift velocity", f"{wsv:+.0f} pts")
            if _isnum(net_flow):
                st.metric("Net IV Flow", f"{net_flow:,.0f}")
        with b3:
            if liq_data and _isnum(liq_data.get("median_spread_bps")):
                st.metric("Median spread (bps)", f"{liq_data['median_spread_bps']:.0f}")
            if liq_data and _isnum(liq_data.get("liquidity_stress")):
                st.metric("Liquidity stress", f"{liq_data['liquidity_stress']*100:.0f} / 100")
    
        # ---------- Actionable bullets ----------
        bullets = []
        if "Short" in (regime or ""):
            bullets.append("Short-gamma: **breakouts expand**; prefer directional/debit structures.")
        elif "Long" in (regime or ""):
            bullets.append("Long-gamma: **mean-revert**; prefer fade/credit structures.")
        if _isnum(iv_z) and abs(iv_z) >= 2.0:
            bullets.append(f"IV impulse z={iv_z:+.2f}: expect **larger range** than average.")
        if _isnum(pin_top) and pin_top >= 0.35 and pin_k is not None:
            bullets.append(f"High pin risk near **{int(pin_k)}** (**{_pct(pin_top)}**).")
        if breakout and _isnum(breakout.get("prob_up_break")) and breakout["prob_up_break"] >= 0.55:
            bullets.append("Upside breakout odds **>55%** vs 50% rail; look for **CE-unwind + PE-writing** on Runway.")
        if _isnum(wsv) and wsv > 0:
            bullets.append("Walls moving **toward** spot → compression/pin risk; **scalp edges**.")
        if liq_data and _isnum(liq_data.get("median_spread_bps")) and liq_data["median_spread_bps"] > 120:
            bullets.append("Wide spreads: **cut size** or wait for **liquidity pockets**.")
        if dir_score >= 60: bullets.append("**Directional bias ON** (score ≥60).")
        if dir_score <= 40: bullets.append("**Mean-revert bias ON** (score ≤40).")
    
        st.markdown("#### Actionable")
        if bullets:
            st.markdown("\n".join([f"- {b}" for b in bullets]))
        else:
            st.caption("Signals are mixed—be selective and wait for alignment (Runway + Velocity).")
    
        st.markdown("#### Invalidation / Flip")
        st.markdown(
            "- **Flip to trend** if 50% rail breaks **and** IV-z ≥ 2 **and** OI flow one-sided.\n"
            "- **Flip back to fade** if pushes fail and IV cools while GEX pulls price back toward magnet.\n"
        )
    
        # ---------- AI JSON payload (machine-readable) ----------
        ai_payload = {
            "meta": {
                "symbol": sym, "expiry": exp, "timestamp": ts,
                "instrument_kind": "stock" if sym in FNO_STOCKS else "index",
                "source": "live" if isinstance(ts, str) and ":" in ts else "historical"
            },
            "prices": {
                "working_spot": working_spot,
                "futures_spot_from_oc": fut_spot,
                "cash_spot_api": cash_spot
            },
            "iv": {
                "atm_iv_percent": atm_iv_now,
                "iv_zscore": iv_z,
                "net_vega_weighted_iv_flow": net_flow
            },
            "gamma": {
                "regime": regime,
                "gex_total_relative": gex_val
            },
            "rails": {
                "p50": list(map(float, r50)) if r50 else None,
                "expected_move_sigma_points": em["sigma_pts"] if em else None,
                "breakout_prob_up": breakout.get("prob_up_break") if breakout else None,
                "breakout_prob_down": breakout.get("prob_dn_break") if breakout else None
            },
            "pinning": {
                "pin_target_strike": int(pin_k) if pin_k is not None else None,
                "pin_top_probability": float(pin_top) if _isnum(pin_top) else None
            },
            "oi_walls": {
                "wall_shift_velocity_pts": wsv,
                "premium_center": float(money_center_of_mass(dfm, use_premium=True)) if not dfm.empty else None,
                "oi_center": float(money_center_of_mass(dfm, use_premium=False)) if not dfm.empty else None
            },
            "liquidity": {
                "median_spread_bps": liq_data.get("median_spread_bps") if liq_data else None,
                "impact_cost_bps": liq_data.get("ic_bps") if liq_data else None,
                "liquidity_stress_0to1": liq_data.get("liquidity_stress") if liq_data else None
            },
            "sentiment": {
                "pcr_ring": pcr_ring,
                "writing_sentiment": sentiment
            },
            "scores": {
                "direction_score_0to100": dir_score
            }
        }
        with st.expander("Machine-readable Commentary (AI JSON)", expanded=False):
            st.json(ai_payload)

    # --------------------------- Overview ---------------------------
    with tab_overview:
        st.subheader("Key Metrics")
        c1, c2, c3, _, _ = st.columns(5)
        c1.metric("Spot Used", f"{working_spot:,.2f}" if working_spot is not None else "–")
        if fut_spot is not None:
            c2.metric("Futures (OC spot)", f"{fut_spot:,.2f}")
        if cash_spot is not None and fut_spot is not None:
            basis = (fut_spot - cash_spot)
            c3.metric("Basis (Fut−Cash)", f"{basis:,.2f}", help="Cash via parity S≈C−P+K (ignores carry/dividends)")

        cards2 = st.columns(3)
        if show_both_pcr:
            cards2[0].metric("PCR (ring)", f"{pcr_window:.2f}", help=f"Computed on fixed ±{fixed_skew_ring} strikes")
        else:
            cards2[0].metric("PCR", f"{pcr_full:.2f}" if np.isfinite(pcr_full) else "∞")
        if max_pain_window != max_pain_full:
            cards2[2].metric("Max Pain (ring)", f"{max_pain_window:.0f}" if max_pain_window is not None else "-")
        else:
            cards2[2].metric("Max Pain", f"{max_pain_full:.0f}" if max_pain_full is not None else "-")

        c6, c7 = st.columns(2)
        c6.metric("ATM Strike", f"{df.iloc[atm_idx]['strike']:.0f}" if len(df) else "-")
        c7.metric("Expiry", exp)

        st.caption(f"Analytics ring uses ±{fixed_skew_ring} strikes around ATM. Window view shows ±{strike_window} around ATM.")
        if not df_ana.empty:
            st.caption(f"Analytics ring strikes: {df_ana['strike'].min():.0f} → {df_ana['strike'].max():.0f}")

        st.markdown("---")
        st.subheader("Trading Overview (intraday lens)")

        # Liquidity pack (with intraday stats)
        liq_stats = compute_spread_stats(df_ana if not df_ana.empty else df, working_spot, band=3)
        icp = impact_cost_proxy(df_ana if not df_ana.empty else df, working_spot, qty=50)
        liq_data, stats = {}, None
        if liq_stats and icp:
            liq_data["median_spread_bps"] = liq_stats.get("median_spread_bps")
            liq_data["ic_bps"] = icp.get("ic_bps")
            ring_df = compute_money_map(df_ana if not df_ana.empty else df)  # turnover if present
            stats = liquidity_stress_with_stats(
                liq_data["median_spread_bps"], liq_data["ic_bps"], ring_df,
                ss["liq_hist"].get(f"{sym}:{exp}")
            )
            liq_data["liquidity_stress"] = stats["stress"]

            # push to intraday history buffer (so the percentile stabilizes intraday)
            key = f"{sym}:{exp}"
            buf = ss["liq_hist"].setdefault(key, [])
            if stats["stress"] is not None:
                buf.append(float(stats["stress"]))
                if len(buf) > 480:  # keep ~1 trading day
                    del buf[:len(buf) - 480]

        if liq_data:
            cL1, cL2, cL3 = st.columns(3)
            med_bps = liq_data.get("median_spread_bps")
            ic_bps = liq_data.get("ic_bps")
            stress = liq_data.get("liquidity_stress")

            cL1.metric("Median spread (bps)", f"{med_bps:.0f}" if _isnum(med_bps) else "–")
            cL2.metric("Impact proxy (bps)", f"{ic_bps:.0f}" if _isnum(ic_bps) else "–")
            cL3.metric(
                "Liquidity stress",
                f"{(stress * 100):.0f} / 100" if _isnum(stress) else "–",
                help="Higher = harder fills"
            )
                # Context vs today's intraday history
            if stats is not None:
                pct = stats.get("percentile_0_100")
                zrb = stats.get("z_robust")
                if (pct is not None) or (zrb is not None):
                    st.caption(
                        f"Liquidity stress vs today: "
                        f"{(f'{pct:.0f}th pct' if pct is not None else '—')}"
                        f"{' · ' if (pct is not None and zrb is not None) else ''}"
                        f"{(f'z≈{zrb:+.2f}' if zrb is not None else '')}"
                    )

    # ---- 1% Execution Suite (additive; optional panel) ----
    with st.expander("1% Execution Suite (edge ÷ pressure, size planner, INR slip)", expanded=False):
        lot = st.number_input("Lot size", min_value=1, value=25, step=1, key="onepct_lot")
        bps = st.number_input("Max slippage (bps)", min_value=1.0, value=10.0, step=1.0, key="onepct_bps")

        # Use the same analytics ring you already derived
        df_for_calc = df_ana if not df_ana.empty else df

        dfp = liquidity_stress_v2(df_for_calc)                         # dials + liquidity_pressure_v2
        dfp = estimate_rupee_impact(dfp, lot_size=int(lot))            # rupee_impact_per_lot
        dfp = plan_order_sizes(dfp, max_slippage_bps=float(bps), lot_size=int(lot))  # max_lots_under_slippage
        ranked = rank_strikes_by_edge_pressure(dfp)                    # edge_per_pressure, rank_epp

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top strikes by Edge ÷ Pressure**")
            st.dataframe(
                ranked[["strike","edge_score","liquidity_pressure_v2","edge_per_pressure",
                        "max_lots_under_slippage","rupee_impact_per_lot"]],
                use_container_width=True, hide_index=True
            )
        with c2:
            st.markdown("**Order Planner (₹ impact)**")
            st.dataframe(
                dfp[["strike","spread_pct","impact_proxy","rupee_impact_per_lot","max_lots_under_slippage"]],
                use_container_width=True, hide_index=True
            )

        # Optional: alert-style filter for easy entries
        try:
            wins = find_execution_windows(ranked, pressure_pctile=0.25, min_edge_per_pressure=0.0)
            if wins is not None and not wins.empty:
                st.markdown("**Execution Windows (low pressure & positive edge)**")
                st.dataframe(
                    wins[["strike","edge_per_pressure","liquidity_pressure_v2","rupee_impact_per_lot","max_lots_under_slippage"]],
                    use_container_width=True, hide_index=True
                )
        except Exception:
            pass

        with st.expander("Raw 1% debug (no rounding)", expanded=False):
            st.dataframe(dfp, use_container_width=True, hide_index=True)
            st.download_button("Download raw 1% table (CSV)", dfp.to_csv(index=False).encode("utf-8"),
                               file_name="liquidity_1pct_raw.csv", mime="text/csv")


        st.markdown("---")

        # Compute metrics for commentary
        atm_iv_now = compute_atm_iv(df_ana, working_spot)
        iv_hist = make_iv_skew_history(sym, exp, last_n=40)
        iv_z = zscore_last(iv_hist["atm_iv"], window=20) if not iv_hist.empty else None
        regime, gex_val = compute_gamma_regime(df_ana, working_spot, ring_size=fixed_skew_ring)

        mstart, mend = market_bounds(now)
        minutes_left = max(1, (mend - now).total_seconds() / 60.0)
        rails = compute_iv_rails(working_spot, atm_iv_now, minutes_left)

        # Expected move (to close)
        em = expected_move_nowcast(working_spot, atm_iv_now, minutes_left)
        if em:
            st.metric("Expected move (1σ)", f"±{em['sigma_pts']:.0f} pts")
            c = em["cones"]
            st.caption(
                f"Cones → 25%: {c[0.25][0]:.0f} ↔ {c[0.25][1]:.0f} | "
                f"50%: {c[0.50][0]:.0f} ↔ {c[0.50][1]:.0f} | "
                f"75%: {c[0.75][0]:.0f} ↔ {c[0.75][1]:.0f}"
            )

        sup_zones, res_zones = summarize_zones(df_ana, top_n=2)
        sev, qtext = make_quick_execution_commentary(
            liq=liq_data, iv_z=iv_z, gamma_regime=regime, pcr=pcr_window, rails=rails,
            sup_zones=sup_zones, res_zones=res_zones,
        )
        if sev == "error": st.error(qtext)
        elif sev == "warning": st.warning(qtext)
        else: st.info(qtext)

        colz1, colz2 = st.columns(2)
        colz1.metric("Supports", _format_zone_list(sup_zones, "support"))
        colz2.metric("Resistances", _format_zone_list(res_zones, "resistance"))

        sentiment, commentary = get_writing_commentary(df_ana)
        bias_note = summarize_trade_bias(pcr_window, sentiment, atm_iv_now, iv_z)

        st.markdown("#### Bias / Sentiment")
        if sentiment == "bullish": st.success(f"🟢 {bias_note}")
        elif sentiment == "bearish": st.error(f"🔴 {bias_note}")
        else: st.info(f"⚖️ {bias_note}")

        pcrV = pcr_vol(df_ana)
        if pcrV is not None and np.isfinite(pcrV):
            st.caption(f"PCR-Vol (ring): **{pcrV:.2f}** → use with momentum checks")

        long_df, vel_df = make_long_history(sym, exp, last_n=10)
        if not vel_df.empty:
            up = vel_df.sort_values("velocity", ascending=False).iloc[0]
            dn = vel_df.sort_values("velocity", ascending=True).iloc[0]
            st.markdown(
                f"**Momentum:** 🟢 Support surge {int(up['strike'])} (+{int(up['velocity'])}) | "
                f"🔴 Resistance surge {int(dn['strike'])} ({int(dn['velocity'])})"
            )
        else:
            st.caption("Momentum: waiting for velocity data (needs ≥2 snapshots).")

        # Gamma regime metric + GEX curve + pins
        regime_emoji = "🟢" if "Long" in regime else ("🔴" if "Short" in regime else "❓")
        st.metric("Gamma Regime", f"{regime_emoji} {regime}", help=f"GEX={gex_val:,.0f}")
        gc = gex_curve(df_ana if not df_ana.empty else df, working_spot)
        if not gc.empty:
            st.markdown("**Gamma exposure by strike (relative)**")
            gex_ch = alt.Chart(gc).mark_bar().encode(
                x=alt.X("strike:O", title="Strike"),
                y=alt.Y("gex:Q", title="GEX (γ×OI, rel.)"),
                tooltip=["strike","gex"]
            ).properties(height=220)
            st.altair_chart(gex_ch, use_container_width=True)
            k_mag = gc.iloc[gc["gex"].rank(pct=True).idxmax()]["strike"]
            st.caption(f"Likely pin/magnet region near **{int(k_mag)}** (largest GEX).")

        # Rails + breakout odds
        if rails and rails.get(0.5):
            r50 = rails[0.5]
            st.metric("IV Rails (50%)", f"{r50[0]:.0f} ↔ {r50[1]:.0f}")
            all_zones = (sup_zones or []) + (res_zones or [])
            if all_zones and working_spot is not None:
                nz = min(all_zones, key=lambda k: abs(k - working_spot))
                inside = (r50[0] <= nz <= r50[1])
                st.caption(f"{'🎯' if inside else '🚧'} Nearest zone {int(nz)} is {'inside' if inside else 'outside'} the 50% cone.")

            if em:
                bp = breakout_probabilities(working_spot, r50, em["sigma_pts"])
                if bp:
                    cA, cB = st.columns(2)
                    cA.metric("Breakout↑ prob (vs 50% rail)", f"{bp['prob_up_break']*100:.0f}%")
                    cB.metric("Breakout↓ prob (vs 50% rail)", f"{bp['prob_dn_break']*100:.0f}%")

        event = detect_stop_hunt(sym, exp)
        if event:
            st.info(f"{event['tag']} seen at {event['strike']} — ΔIV {event['d_iv']:+.2f}, z={event['iv_z']:+.2f}")

        rr_val, _bf_ = compute_rr_bf(df_ana)
        zones_df = score_zones(df_ana, atm_iv_now, rr_val, rails)
        if not zones_df.empty:
            st.markdown("#### Top Zone Scores")
            st.dataframe(zones_df[["emoji","type","strike","score"]], use_container_width=True, hide_index=True)

        st.markdown("#### Live Commentary")
        try:
            commentary_text = build_trade_commentary(
                gamma_regime=regime, rails=rails, pcr=pcr_window, sentiment=sentiment, commentary=commentary,
                atm_iv=atm_iv_now, iv_z=iv_z, sup_zones=sup_zones, res_zones=res_zones, vel_df=vel_df, stop_event=event,
            )
            st.info(commentary_text if commentary_text else "No commentary yet.")
        except Exception as e:
            st.warning(f"Commentary unavailable: {e}")

    # ---------------------- Gates & Runway ----------------------
    with tab_runway:
        st.subheader("Directional Gates & Runway (near LTP)")
        if df_ana.empty or not {"PE_oi_change","PE_oi","CE_oi_change","CE_oi"}.issubset(df_ana.columns) or working_spot is None:
            st.info("Need PE/CE OI & ΔOI around ATM to compute gates.")
        else:
            colA, colB, colC = st.columns([1,1,1])
            with colA:
                up_steps = st.number_input("Upward steps (strikes ↑)", 3, 12, 6, 1)
            with colB:
                down_steps = st.number_input("Downward steps (strikes ↓)", 3, 12, 6, 1)
            with colC:
                strict = st.slider("Strictness (pass threshold)", 0.5, 1.2, 0.85, 0.05)

            with st.expander("Refinements (optional, safe defaults)", expanded=False):
                USE_DELTA      = st.checkbox("Delta-weight gate score", value=False)
                OPP_CAP        = st.slider("Opposition cap (force FAIL above)", 0.8, 1.5, 1.0, 0.05)
                LIQ_PCT        = st.slider("Liquidity floor (OI percentile)", 0, 40, 0, 5)
                USE_MAJORITY   = st.checkbox("Majority confirmation (2 of last 3)", value=False)
                MAJ_FRAMES     = st.number_input("Majority lookback frames", 3, 7, 3, 1)
                ALLOW_ONE_SKIP = st.checkbox("Allow skip-1 bridge", value=False)

            up_tbl, down_tbl, up_clear, down_clear = build_gate_runway_tables(
                df_ring=df_ana, spot=working_spot, atm_idx_in_ring=atm_idx_ana, symbol=sym, expiry=exp,
                up_n=int(up_steps), down_n=int(down_steps), lookback_frames=6, pass_threshold=float(strict),
            )

            up_tbl, down_tbl, up_clear, down_clear = apply_runway_enhancements(
                up_tbl=up_tbl, down_tbl=down_tbl, df_ring=df_ana, strict=float(strict),
                use_delta=USE_DELTA, opp_cap=float(OPP_CAP), liq_pct=int(LIQ_PCT),
                use_majority=USE_MAJORITY, maj_frames=int(MAJ_FRAMES), allow_skip=ALLOW_ONE_SKIP,
                sym=sym, exp=exp,
            )

            atm_iv_now2 = compute_atm_iv(df_ana, working_spot)
            iv_hist2 = make_iv_skew_history(sym, exp, last_n=40)
            iv_z2 = zscore_last(iv_hist2["atm_iv"], window=20) if not iv_hist2.empty else None
            regime2, _gex2 = compute_gamma_regime(df_ana, working_spot, ring_size=fixed_skew_ring)
            _, vel_df_local = make_long_history(sym, exp, last_n=10)

            runway_ratio_up = float(up_tbl["gate_ok"].mean()) if not up_tbl.empty else 0.0
            runway_ratio_down = float(down_tbl["gate_ok"].mean()) if not down_tbl.empty else 0.0
            vel_ok_up = 1.0 if (not vel_df_local.empty and vel_df_local["velocity"].max() > 0) else 0.0
            vel_ok_down = 1.0 if (not vel_df_local.empty and vel_df_local["velocity"].min() < 0) else 0.0

            conf_up = confidence_score(runway_ratio_up, vel_ok_up, iv_z2, regime2, pcr_window)
            conf_down = confidence_score(runway_ratio_down, vel_ok_down, iv_z2, regime2, pcr_window)

            c_conf1, c_conf2 = st.columns(2)
            c_conf1.metric("Confidence ↑", f"{conf_up:.0f}")
            c_conf2.metric("Confidence ↓", f"{conf_down:.0f}")

            def _chart(tbl, title):
                if tbl.empty: return None
                dd = tbl.copy()
                dd["pass"] = dd["gate_ok"].map(lambda x: "Pass" if x else "Fail")
                return (
                    alt.Chart(dd)
                    .mark_bar()
                    .encode(
                        x=alt.X("strike:O", title="Strike"),
                        y=alt.Y("score:Q", title="Gate score"),
                        color=alt.Color("pass:N", scale=alt.Scale(domain=["Pass","Fail"], range=["#2ecc71","#e74c3c"])),
                        tooltip=[alt.Tooltip("strike:O"), alt.Tooltip("score:Q"), alt.Tooltip("persistence:Q"), alt.Tooltip("opp_pressure:Q")],
                    )
                    .properties(height=220, title=title)
                )

            colU, colD = st.columns(2)
            with colU:
                ch_up = _chart(up_tbl, "Up path (needs PE writing + CE unwind)")
                if ch_up: st.altair_chart(ch_up, use_container_width=True)
            with colD:
                ch_dn = _chart(down_tbl, "Down path (needs CE writing + PE unwind)")
                if ch_dn: st.altair_chart(ch_dn, use_container_width=True)

            st.markdown("#### Details (latest tick)")
            cL, cR = st.columns(2)
            with cL:
                if not up_tbl.empty:
                    st.markdown("**Up path table**")
                    st.dataframe(
                        up_tbl[["strike","PE_oi_change","PE_oi","CE_oi_change","CE_oi","score","persistence","opp_pressure","gate_ok","reason"]],
                        use_container_width=True, hide_index=True
                    )
            with cR:
                if not down_tbl.empty:
                    st.markdown("**Down path table**")
                    st.dataframe(
                        down_tbl[["strike","CE_oi_change","CE_oi","PE_oi_change","PE_oi","score","persistence","opp_pressure","gate_ok","reason"]],
                        use_container_width=True, hide_index=True
                    )

            st.caption(
                "Rule of thumb: an advance needs **fresh put writing stepping up** (ΔPE_OI>0) and ideally some **call unwind** (ΔCE_OI<0) at each next strike. "
                "The first strike that fails is your **gate**. Same logic mirrors downside with calls."
            )

    # --------------------------- Money Map ---------------------------
    with tab_money:
        st.subheader("Money Map (where capital sits & flows)")
        dfm = compute_money_map(df_ana if not df_ana.empty else df)
        if dfm.empty:
            st.info("Need OI / ΔOI (and ideally LTP, gamma, vega) to compute money map.")
        else:
            st.markdown("#### Money Map commentary")
            try:
                mm_comment = make_money_map_commentary(dfm, working_spot)
                st.info(mm_comment)
            except Exception as e:
                st.warning(f"Commentary error: {e}")

            cm_p = money_center_of_mass(dfm, use_premium=True)
            cm_o = money_center_of_mass(dfm, use_premium=False)
            c1,c2,c3 = st.columns(3)
            c1.metric("Premium Center", f"{cm_p:.0f}" if cm_p else "–")
            c2.metric("OI Center", f"{cm_o:.0f}" if cm_o else "–")
            wall_tbl = find_walls(dfm, z_thresh=1.5, k=4)
            c3.metric("Walls flagged", f"{len(wall_tbl)}")

            # Wall-shift velocity vs previous ring snapshot
            prev_walls = None
            frames_mm = get_last_frames(sym, exp, n=2)
            if len(frames_mm) == 2:
                prev_dfm = compute_money_map(frames_mm[-2])
                prev_walls = find_walls(prev_dfm, z_thresh=1.5, k=4)
            wsv = wall_shift_velocity(wall_tbl, prev_walls, working_spot) if not wall_tbl.empty else None
            if wsv is not None:
                st.metric(
                    "Wall-shift velocity",
                    f"{wsv:+.0f} pts",
                    help=">0 = walls moving toward spot (compression/pin risk); <0 = release/room to run."
                )

            left, right = st.columns(2)
            with left:
                if "prem_stock" in dfm.columns:
                    st.markdown("**Premium-at-risk by strike**")
                    st.altair_chart(
                        alt.Chart(dfm).mark_bar().encode(
                            x=alt.X("strike:O", title="Strike"),
                            y=alt.Y("prem_stock:Q", title="Premium stock"),
                            tooltip=["strike","prem_stock","tot_oi"]
                        ).properties(height=240),
                        use_container_width=True
                    )
            with right:
                if "prem_new" in dfm.columns:
                    st.markdown("**New premium today (flow)**")
                    st.altair_chart(
                        alt.Chart(dfm).mark_bar().encode(
                            x=alt.X("strike:O", title="Strike"),
                            y=alt.Y("prem_new:Q", title="New premium (ΔOI×LTP)"),
                            tooltip=["strike","prem_new","tot_oi_ch"]
                        ).properties(height=240),
                        use_container_width=True
                    )

            if {"gamma_dlr","vega_dlr"}.issubset(dfm.columns):
                st.markdown("**Dealer hedging concentrations (relative)**")
                hv = dfm.melt(id_vars=["strike"], value_vars=["gamma_dlr","vega_dlr"], var_name="metric", value_name="value")
                st.altair_chart(
                    alt.Chart(hv).mark_bar().encode(
                        x=alt.X("strike:O"), y=alt.Y("value:Q"), color="metric:N",
                        tooltip=["metric","strike","value"]
                    ).properties(height=240),
                    use_container_width=True
                )

            if not wall_tbl.empty:
                st.markdown("**Walls / clusters (rolling OI z-score)**")
                wall_tbl = wall_tbl.rename(columns={"wall_z":"wall_z (σ)", "prem_stock":"premium"})
                st.dataframe(wall_tbl, use_container_width=True, hide_index=True)

            st.markdown("**Top concentrations**")
            t1 = dfm.sort_values("prem_stock", ascending=False).head(5)[["strike","prem_stock","tot_oi"]]
            t2 = dfm.sort_values("prem_new", ascending=False).head(5)[["strike","prem_new","tot_oi_ch"]]
            colA, colB = st.columns(2)
            with colA:  st.dataframe(t1, use_container_width=True, hide_index=True)
            with colB:  st.dataframe(t2, use_container_width=True, hide_index=True)

    # ----------------------------- Chain -----------------------------
    with tab_chain:
        st.subheader("Option Chain Data")
        dfv, styler = style_chain(dfw, atm_idx_local, selected_fields, working_spot, heat_cols=heat_cols)
        st.markdown(styler.to_html(), unsafe_allow_html=True)
        csv = dfv.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download displayed data as CSV", data=csv, file_name=f'option_chain_{sym}_{exp}_{ts}.csv', mime='text/csv')

    # ---------------------------- Writing ----------------------------
    with tab_writing:
        st.subheader("Option Writing Analysis")
        sentiment, commentary = get_writing_commentary(df_ana)
        if sentiment == "bullish": st.success(commentary)
        elif sentiment == "bearish": st.error(commentary)
        else: st.info(commentary)

        st.markdown("#### Net OI Change (PE - CE) Fortress Chart (ring)")
        net_chart = net_oi_change_chart(df_ana)
        if net_chart: st.altair_chart(net_chart, use_container_width=True)
        else: st.warning("Net OI Change data not available for charting.")

        st.markdown("#### Key Support & Resistance Levels (Based on OI Change, ring)")
        if "net_oi_change" in df_ana.columns and not df_ana.empty:
            df_support = df_ana.sort_values("net_oi_change", ascending=False).head(3).copy()
            df_resist  = df_ana.sort_values("net_oi_change", ascending=True).head(3).copy()
            df_summary = pd.concat([df_support, df_resist])
            df_summary = df_summary[["strike", "net_oi_change"]].dropna()
            if not df_summary.empty:
                df_summary["Sentiment"] = df_summary["net_oi_change"].apply(lambda x: "🟢 Strong Support (put writing)" if x > 0 else "🔴 Strong Resistance (call writing)")
                df_summary = df_summary.rename(columns={"net_oi_change": "Net OI Change (PE - CE)"})
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
            else:
                st.info("No significant OI change data to display.")

    # ---------------------------- Velocity ----------------------------
    with tab_velocity:
        st.subheader("Net OI Change Velocity (Intraday)")
        colv1, colv2 = st.columns([2,1])
        with colv2:
            last_n = st.number_input("Last N snapshots", 5, 240, 40, 5)
            spike_thresh = st.number_input("Spike threshold (abs)", 0, 500000, 25000, 5000)

        long_df, vel_df = make_long_history(sym, exp, last_n=int(last_n))

        st.markdown("#### Heatmap: Net OI Change over time (ring)")
        if not long_df.empty and "net_oi_change" in long_df.columns:
            hdf = long_df.dropna(subset=["strike","timestamp","net_oi_change"]).copy()
            ts_order = sorted(hdf["timestamp"].unique())
            strike_order = sorted(hdf["strike"].unique())
            heat = (
                alt.Chart(hdf).mark_rect().encode(
                    x=alt.X("timestamp:N", sort=ts_order, title="Time"),
                    y=alt.Y("strike:O", sort=strike_order, title="Strike"),
                    color=alt.Color("net_oi_change:Q", scale=alt.Scale(scheme="redyellowgreen"), title="PE−CE"),
                    tooltip=[
                        alt.Tooltip("timestamp:N", title="Time"),
                        alt.Tooltip("strike:O", title="Strike"),
                        alt.Tooltip("net_oi_change:Q", title="Net (PE−CE)", format=","),
                    ],
                ).properties(height=360)
            )
            st.altair_chart(heat, use_container_width=True)
        else:
            st.info("Not enough history yet. Wait for a couple of auto-fetch ticks.")

        st.markdown("#### Latest Velocity by Strike (ring)")
        if not vel_df.empty:
            vmax = float(vel_df["velocity"].abs().max()) + 1
            vbar = (
                alt.Chart(vel_df).mark_bar().encode(
                    y=alt.Y("strike:O", sort="-x", title="Strike"),
                    x=alt.X("velocity:Q", title="Velocity (Δ net OI change)", scale=alt.Scale(domain=[-vmax, vmax])),
                    color=alt.condition(alt.datum.velocity > 0, alt.value("#6ab04c"), alt.value("#E74C3C")),
                    tooltip=[alt.Tooltip("strike:O"), alt.Tooltip("velocity:Q", format=",")],
                ).properties(height=300)
            )
            st.altair_chart(vbar, use_container_width=True)
            st.markdown("#### Surges (Top moves since last tick)")
            surge_tbl = velocity_spike_table(vel_df, k=5, threshold=spike_thresh if spike_thresh > 0 else None)
            if not surge_tbl.empty: st.dataframe(surge_tbl, use_container_width=True, hide_index=True)
            else: st.info("No velocity surges above threshold.")
        else:
            st.info("Velocity will appear after two or more snapshots are captured.")

    # ----------------------------- Movers (FIXED) -----------------------------
    with tab_movers:
        st.subheader("Top Movers & Analytics")
        # FIX: use the **full chain** so we always have enough rows/metrics
        base_df = df if not df.empty else dfw
        ce_ch, pe_ch, ce_oi, pe_oi, ch_df, oi_df, net_df = build_movers_long(base_df, n=5)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**OI Change (Top 5)**")
            ch = movers_chart(ch_df, "OI Change")
            if ch is not None: st.altair_chart(ch, use_container_width=True)
            else: st.info("No OI change data.")
        with colB:
            st.markdown("**Open Interest (Top 5)**")
            ch2 = movers_chart(oi_df, "Open Interest")
            if ch2 is not None: st.altair_chart(ch2, use_container_width=True)
            else: st.info("No OI data.")

        cL, cR = st.columns(2)
        with cL:
            st.markdown("#### CE Movers")
            if ce_ch is not None: st.dataframe(style_mover_table(ce_ch, "CE_oi_change"), use_container_width=True)
            if ce_oi is not None: st.dataframe(style_mover_table(ce_oi, "CE_oi"), use_container_width=True)
        with cR:
            st.markdown("#### PE Movers")
            if pe_ch is not None: st.dataframe(style_mover_table(pe_ch, "PE_oi_change"), use_container_width=True)
            if pe_oi is not None: st.dataframe(style_mover_table(pe_oi, "PE_oi"), use_container_width=True)

        if net_df is not None and not net_df.empty:
            st.markdown("#### Net OI Change (PE−CE): Top 5 by magnitude")
            st.dataframe(net_df, use_container_width=True, hide_index=True)

    # ---------------------------- IV & Skew ----------------------------
    with tab_iv:
        st.subheader("IV & Skew (intraday)")
        iv_hist = make_iv_skew_history(sym, exp, last_n=60)
        df_for_calc = df_ana if not df_ana.empty else df
        atm_iv_now = compute_atm_iv(df_for_calc, working_spot)
        atm_iv_prev = None
        iv_z = None
        if not iv_hist.empty and "atm_iv" in iv_hist.columns:
            if len(iv_hist) >= 2:
                atm_iv_prev = iv_hist["atm_iv"].iloc[-2]
            iv_z = zscore_last(iv_hist["atm_iv"], window=20)
        c1, c2, c3 = st.columns(3)
        c1.metric("ATM IV", f"{atm_iv_now:.2f}%" if atm_iv_now is not None else "–",
                  help="Mean of CE/PE IV at ~50Δ (fallback: ATM strike).")
        if atm_iv_prev is not None and atm_iv_now is not None:
            c2.metric("ΔIV (last tick)", f"{(atm_iv_now - atm_iv_prev):+.2f} pts")
        else:
            c2.metric("ΔIV (last tick)", "–")
        c3.metric("IV impulse (z)", f"{iv_z:+.2f}" if iv_z is not None else "–",
                  help="Z-score of ATM IV over last ~20 snapshots; |z|≥2 suggests a vol impulse.")

        buf = st.session_state.get("spot_hist", {}).get(f"{sym}:{exp}")
        rv = realized_vol_annualized(list(buf)) if buf else None
        if rv is not None and atm_iv_now is not None:
            c4, c5 = st.columns(2)
            c4.metric("Realized Vol (intraday, ann.)", f"{rv:.2f}%")
            c5.metric("IV − RV", f"{(atm_iv_now - rv):+.2f} pts",
                      help=">0 = options rich vs current realized; <0 = cheap")

        if not iv_hist.empty:
            base = alt.Chart(iv_hist).encode(x=alt.X("timestamp:T", title="Time"))
            iv_line = base.mark_line().encode(y=alt.Y("atm_iv:Q", title="ATM IV"))
            st.altair_chart(iv_line.properties(height=220), use_container_width=True)
        else:
            st.info("No IV history yet. It will build as snapshots accumulate.")

        st.markdown("---")
        rr, bf = compute_rr_bf(df_for_calc)
        cc1, cc2 = st.columns(2)
        cc1.metric("Risk Reversal (25Δ)", f"{rr:+.2f} pts" if rr is not None else "–",
                    help="σ(25Δ put) − σ(25Δ call). Positive = puts richer.")
        cc2.metric("Butterfly (25Δ)", f"{bf:+.2f} pts" if bf is not None else "–",
                    help="Smile curvature: 0.5*(σ25P+σ25C) − σATM")
        if not iv_hist.empty and {"rr","bf"}.issubset(iv_hist.columns):
            rr_df = iv_hist.dropna(subset=["rr"])
            bf_df = iv_hist.dropna(subset=["bf"])
            charts = []
            if not rr_df.empty:
                charts.append(alt.Chart(rr_df).mark_line().encode(
                    x=alt.X("timestamp:T", title="Time"), y=alt.Y("rr:Q", title="RR (25Δ)")
                ).properties(height=180))
            if not bf_df.empty:
                charts.append(alt.Chart(bf_df).mark_line().encode(
                    x=alt.X("timestamp:T", title="Time"), y=alt.Y("bf:Q", title="BF (25Δ)")
                ).properties(height=180))
            if charts:
                st.altair_chart(alt.vconcat(*charts), use_container_width=True)

        st.markdown("---")
        frames = get_last_frames(sym, exp, n=2)
        df_prev_for_flow = frames[-2] if len(frames) == 2 else None
        flows, net_flow = compute_vega_weighted_iv_flow(df_for_calc, df_prev_for_flow)
        cL, cR = st.columns([3,1])
        with cR:
            st.metric("Net IV Flow", f"{net_flow:,.0f}", help="Σ(vega × OI × ΔIV) over ring. Positive = net vol buying.")
        if flows.empty:
            st.info("Need IV, Vega, OI in two consecutive snapshots to compute IV flows.")
        else:
            flows_plot = flows.copy()
            flows_plot["abs_flow"] = flows_plot["flow"].abs()
            topN = st.slider("Top strikes by |IV flow|", 5, 20, 8, 1)
            keep = flows_plot.sort_values("abs_flow", ascending=False).head(topN)["strike"].tolist()
            show = flows_plot[flows_plot["strike"].isin(keep)]
            chart = (
                alt.Chart(show)
                .mark_bar()
                .encode(
                    y=alt.Y("strike:O", sort="-x", title="Strike"),
                    x=alt.X("flow:Q", title="Vega-weighted IV Flow"),
                    color=alt.Color("side:N", scale=alt.Scale(domain=["CE","PE"], range=["#2E86DE","#E74C3C"])),
                    tooltip=[alt.Tooltip("side:N"), alt.Tooltip("strike:O"), alt.Tooltip("flow:Q", format=",")],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
        st.caption("Flow proxy: vega × OI × ΔIV per side & strike; aggregated over your analytics ring.")
# --------------------------- Commentary (AI payload + emoji) ---------------------------
    with tab_commentary:
        st.subheader("Commentary (AI payload + quick glance)")
    
        # Cash OHLC (for day range + LTP)
        cash_ohlc = _fetch_cash_ohlc(api_base, sym, inst)  # {"open","high","low","ltp"} or None
        fut_ohlc = None  # optional, set if you later add a futures OHLC endpoint
    
        # Recompute a few things we need (cheap ops)
        df_for_calc = df_ana if not df_ana.empty else df
        atm_iv_now = compute_atm_iv(df_for_calc, working_spot)
        iv_hist = make_iv_skew_history(sym, exp, last_n=40)
        iv_z = zscore_last(iv_hist["atm_iv"], window=20) if not iv_hist.empty else None
        mstart, mend = market_bounds(now)
        minutes_left = max(1, (mend - now).total_seconds() / 60.0) if mend else 60
        rails = compute_iv_rails(working_spot, atm_iv_now, minutes_left)
        regime, _gex = compute_gamma_regime(df_for_calc, working_spot, ring_size=fixed_skew_ring)
    
        # Expected move + breakout
        em = expected_move_nowcast(working_spot, atm_iv_now, minutes_left)
        breakout_probs_dict = breakout_probabilities(
            working_spot, rails.get(0.50) if rails else None, (em or {}).get("sigma_pts")
        ) if rails and em else None
    
        # GEX + pin probs
        gc = gex_curve(df_for_calc, working_spot)
        pin_table_list = []
        if not gc.empty and em:
            probs_df, pin_k = pin_probability(gc, working_spot, em["sigma_pts"])
            if probs_df is not None and not probs_df.empty:
                pin_table_list = [{"strike": float(r.strike), "prob": float(r.pin_prob)} for _, r in probs_df.head(5).iterrows()]
    
        # Supports / resistances summary
        sup_zones, res_zones = summarize_zones(df_for_calc, top_n=2)
        supports = sup_zones or []
        resistances = res_zones or []
    
        # Money map, walls, wall-shift velocity (vs previous ring)
        dfm = compute_money_map(df_for_calc)
        walls_tbl = find_walls(dfm, z_thresh=1.5, k=4) if not dfm.empty else pd.DataFrame()
        walls_list = []
        if not walls_tbl.empty:
            for _, r in walls_tbl.iterrows():
                walls_list.append({"strike": float(r["strike"]), "z": float(r["wall_z"]), "type": "support" if r.get("type","support")=="support" else "resistance"})
        prev_frames = get_last_frames(sym, exp, n=2)
        prev_walls_tbl = pd.DataFrame()
        if len(prev_frames) == 2:
            prev_dfm = compute_money_map(prev_frames[-2])
            prev_walls_tbl = find_walls(prev_dfm, z_thresh=1.5, k=4) if not prev_dfm.empty else pd.DataFrame()
        wsv = wall_shift_velocity(walls_tbl, prev_walls_tbl, working_spot) if not walls_tbl.empty else None
    
        premium_center = money_center_of_mass(dfm, use_premium=True) if not dfm.empty else None
        oi_center = money_center_of_mass(dfm, use_premium=False) if not dfm.empty else None
        top_prem_stock = []
        top_prem_new = []
        if not dfm.empty:
            if "prem_stock" in dfm.columns:
                top_prem_stock = dfm.nlargest(3, "prem_stock")[["strike","prem_stock"]].assign(value=lambda x:x["prem_stock"]).drop(columns=["prem_stock"]).to_dict("records")
            if "prem_new" in dfm.columns:
                top_prem_new = dfm.nlargest(3, "prem_new")[["strike","prem_new"]].assign(value=lambda x:x["prem_new"]).drop(columns=["prem_new"]).to_dict("records")
    
        # Momentum/velocity (last N)
        _, vel_df_local = make_long_history(sym, exp, last_n=10)
        vel_up_sum = float(vel_df_local[vel_df_local["velocity"]>0]["velocity"].sum()) if not vel_df_local.empty else 0.0
        vel_dn_sum = float(-vel_df_local[vel_df_local["velocity"]<0]["velocity"].sum()) if not vel_df_local.empty else 0.0
        vel_tilt = (vel_up_sum - vel_dn_sum) / max(1e-9, (vel_up_sum + vel_dn_sum)) if (vel_up_sum + vel_dn_sum) > 0 else 0.0
        spikes_tbl = velocity_spike_table(vel_df_local, k=5, threshold=None) if not vel_df_local.empty else pd.DataFrame()
        vel_spikes = [{"strike": float(r["strike"]), "abs_value": float(abs(r["velocity"])), "side": "PE" if r["velocity"]>0 else "CE"} for _, r in spikes_tbl.iterrows()] if not spikes_tbl.empty else []
    
        # IV flow
        frames2 = get_last_frames(sym, exp, n=2)
        df_prev_for_flow = frames2[-2] if len(frames2) == 2 else None
        flows, net_flow = compute_vega_weighted_iv_flow(df_for_calc, df_prev_for_flow)
        iv_top_flows = []
        if not flows.empty:
            flows_plot = flows.assign(abs_flow=lambda x: x["flow"].abs()).nlargest(5, "abs_flow")
            iv_top_flows = [{"strike": float(r["strike"]), "side": r["side"], "flow": float(r["flow"])} for _, r in flows_plot.iterrows()]
    
        # Liquidity pack
        liq_stats = compute_spread_stats(df_for_calc, working_spot, band=3)
        icp = impact_cost_proxy(df_for_calc, working_spot, qty=50)
        median_spread_bps = liq_stats.get("median_spread_bps") if liq_stats else None
        impact_bps_50qty = icp.get("ic_bps") if icp else None
        liq_stress = liquidity_stress(median_spread_bps, impact_bps_50qty, dfm) if (median_spread_bps and impact_bps_50qty and not dfm.empty) else None
    
        # Realized vol gap
        buf = st.session_state.get("spot_hist", {}).get(f"{sym}:{exp}")
        rv = realized_vol_annualized(list(buf)) if buf else None
        iv_minus_rv = (atm_iv_now - rv) if (atm_iv_now is not None and rv is not None) else None
    
        # meta & clock
        minutes_since_open = (now - mstart).total_seconds()/60.0 if mstart else None
        minutes_to_close = (mend - now).total_seconds()/60.0 if mend else None
    
        # pin probability top (for phase)
        pin_top = (pin_table_list[0]["prob"] if pin_table_list else None)
        phase = session_phase(minutes_since_open, iv_z, regime, pin_prob_top=pin_top)
    
        meta = {
            "symbol": sym, "instrument": inst.upper(), "expiry": exp,
            "timestamp": ts, "source": source, "dte_days": None, "session_phase": phase
        }
        market_clock = {"minutes_since_open": minutes_since_open, "minutes_to_close": minutes_to_close}
    
        # Build payload (first pass)
        payload = _build_commentary_json_v2(
            meta=meta, market_clock=market_clock,
            working_spot=working_spot, basis=(fut_spot - cash_spot) if (cash_spot is not None and fut_spot is not None) else None,
            cash_ohlc=cash_ohlc, fut_ohlc=fut_ohlc,
            atm_strike=float(df.iloc[atm_idx]["strike"]) if len(df) else None,
            supports=[float(x) for x in (supports or [])],
            resistances=[float(x) for x in (resistances or [])],
            max_pain=float(compute_max_pain(df_for_calc)) if compute_max_pain(df_for_calc) is not None else None,
            gex_pin=None,
            atm_iv=atm_iv_now, iv_z=iv_z, rr25=(compute_rr_bf(df_for_calc)[0]), bf25=(compute_rr_bf(df_for_calc)[1]),
            iv_minus_rv=iv_minus_rv,
            expected_move=em, rails=rails, gamma_regime=regime, breakout_probs=breakout_probs_dict,
            pcr_ring=pcr_window, pcr_full=pcr_full, pin_table=pin_table_list,
            premium_center=premium_center, oi_center=oi_center, walls=walls_list,
            wall_shift_velocity_pts=wsv,
            top_prem_stock=top_prem_stock, top_prem_new=top_prem_new,
            vel_up_sum=vel_up_sum, vel_dn_sum=vel_dn_sum, vel_tilt=vel_tilt, vel_spikes=vel_spikes,
            iv_net_flow=net_flow, iv_top_flows=iv_top_flows,
            median_spread_bps=median_spread_bps, impact_bps_50qty=impact_bps_50qty, liq_stress=liq_stress,
            direction_score=None, bias=None, tags=None,
            decision=None, data_quality={"missing": []}
        )
    
        # Tags + decision + scores
        score, bias, tags, decision = _auto_tags_and_decision(payload)
        payload["derived"]["direction_score_0_100"] = score
        payload["derived"]["bias"] = bias
        payload["derived"]["tags"] = tags
        payload["decision"] = decision
    
        # Human glance (emoji)
        st.markdown("#### Quick glance")
        st.text(_emoji_summary(payload))
    
        # JSON pretty + download
        st.markdown("#### AI payload (JSON v2)")
        pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        st.code(pretty, language="json")
        st.download_button(
            "Download commentary.json",
            data=pretty.encode("utf-8"),
            file_name=f"commentary_{sym}_{exp}_{ts.replace(':','-')}.json",
            mime="application/json",
            use_container_width=True
        )

    # --------------------------- Diagnostics ---------------------------
    with tab_diag:
        st.subheader("Diagnostics")
        with st.expander("Raw API Snapshot (as received)", expanded=False):
            st.json(snapshot)
        st.markdown("#### Consistency Debug (Live vs Historical)")
        buf_len = len(st.session_state.get("live_buffers", {}).get(f"{sym}:{exp}", [])) if source == "Live API" else "—"
        st.write({
            "live_buffer_len": buf_len,
            "last_live_digest": st.session_state.get("last_live_digest"),
            "last_hist_digest": st.session_state.get("last_hist_digest"),
            "using_spot_source": spot_source,
            "futures_spot": fut_spot,
            "cash_spot": cash_spot,
            "working_spot": working_spot,
            "atm_idx": atm_idx,
            "display_window": [start, end],
            "analytics_ring": fixed_skew_ring,
        })
        present_cols = [c for c in ["CE_iv","PE_iv","CE_delta","PE_delta","CE_oi_change","PE_oi_change","CE_oi","PE_oi"] if c in df.columns]
        st.write({"present_columns": present_cols})

# ---------- Commentary helpers (OHLC + payload + tags) ----------

_INDEX_NAME_MAP = {
    "NIFTY": "NIFTY 50",
    "NIFTY 50": "NIFTY 50",
    "NIFTY-I": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTY BANK": "NIFTY BANK",
    "BANKNIFTY-I": "NIFTY BANK",
}

@st.cache_data(show_spinner=False, ttl=10)
def _fetch_cash_ohlc(api_base: str, symbol: str, instrument: str):
    """
    Returns dict with lower-case keys: {"open","high","low","ltp"} or None.
    Uses your local endpoints:
      - Index:  GET {api_base}/nse/index/ohlc?symbols=NIFTY 50
      - Stock:  GET {api_base}/nse/stocks/ohlc?symbols=RELIANCE
    """
    try:
        if instrument.upper() == "INDEX":
            human = _INDEX_NAME_MAP.get(symbol.upper(), symbol)
            url = f"{api_base}/nse/index/ohlc"
            params = {"symbols": human}
        else:
            url = f"{api_base}/nse/stocks/ohlc"
            params = {"symbols": symbol.upper()}
        r = requests.get(url, params=params, timeout=2.5)
        if r.status_code != 200:
            return None
        js = r.json() or {}
        # Normalize to lower-case keys the app expects
        out = {
            "open": float(js.get("Open")) if js.get("Open") is not None else None,
            "high": float(js.get("High")) if js.get("High") is not None else None,
            "low":  float(js.get("Low"))  if js.get("Low")  is not None else None,
            "ltp":  float(js.get("LTP"))  if js.get("LTP")  is not None else None,
        }
        return out
    except Exception:
        return None

def _day_range_position(cash_ohlc: dict, working_spot: float):
    try:
        lo, hi = float(cash_ohlc["low"]), float(cash_ohlc["high"])
        if hi > lo and working_spot is not None:
            return max(0.0, min(1.0, (float(working_spot) - lo) / (hi - lo)))
    except Exception:
        pass
    return None

def _build_commentary_json_v2(
    *,
    meta, market_clock,           # dicts
    working_spot, basis,          # floats/None
    cash_ohlc=None, fut_ohlc=None,
    atm_strike=None, supports=None, resistances=None, max_pain=None, gex_pin=None,
    atm_iv=None, iv_z=None, rr25=None, bf25=None, iv_minus_rv=None,
    expected_move=None, rails=None, gamma_regime="Neutral", breakout_probs=None,
    pcr_ring=None, pcr_full=None, pin_table=None,
    premium_center=None, oi_center=None, walls=None, wall_shift_velocity_pts=None,
    top_prem_stock=None, top_prem_new=None,
    vel_up_sum=None, vel_dn_sum=None, vel_tilt=None, vel_spikes=None,
    iv_net_flow=None, iv_top_flows=None,
    median_spread_bps=None, impact_bps_50qty=None, liq_stress=None,
    direction_score=None, bias=None, tags=None,
    decision=None, data_quality=None
):
    return {
        "meta": meta,
        "market_clock": market_clock,
        "price": {
            "working_spot": working_spot,
            "basis": basis,
            "day_range_position": _day_range_position(cash_ohlc or {}, working_spot),
            "cash_ohlc": cash_ohlc,
            "fut_ohlc": fut_ohlc,
        },
        "strikes": {
            "atm_strike": atm_strike,
            "supports": supports or [],
            "resistances": resistances or [],
            "max_pain": max_pain,
            "gex_pin_strike": gex_pin,
        },
        "volatility": {
            "atm_iv": atm_iv, "iv_z": iv_z, "rr25": rr25, "bf25": bf25, "iv_minus_rv": iv_minus_rv,
            "expected_move_pts_1sigma": (expected_move or {}).get("sigma_pts") if expected_move else None,
            "cones": (expected_move or {}).get("cones") if expected_move else None
        },
        "rails": {
            "gamma_regime": gamma_regime,
            "p25": (rails or {}).get(0.25) if rails else None,
            "p50": (rails or {}).get(0.50) if rails else None,
            "p75": (rails or {}).get(0.75) if rails else None,
            "breakout_prob_vs_p50": breakout_probs or None
        },
        "positioning": {
            "pcr_ring": pcr_ring, "pcr_full": pcr_full,
            "pin_probabilities": pin_table or []
        },
        "money_map": {
            "premium_center": premium_center, "oi_center": oi_center,
            "walls": walls or [], "wall_shift_velocity_pts": wall_shift_velocity_pts,
            "top_premium_stock": top_prem_stock or [], "top_premium_new": top_prem_new or []
        },
        "momentum": {
            "velocity_up_sum": vel_up_sum, "velocity_down_sum": vel_dn_sum,
            "velocity_tilt": vel_tilt, "velocity_spikes": vel_spikes or []
        },
        "iv_flow": { "net_flow": iv_net_flow, "top_flows": iv_top_flows or [] },
        "liquidity": {
            "median_spread_bps": median_spread_bps,
            "impact_bps_50qty": impact_bps_50qty,
            "stress": liq_stress
        },
        "derived": {
            "direction_score_0_100": direction_score,
            "bias": bias,
            "tags": tags or []
        },
        "decision": decision or {},
        "data_quality": data_quality or {"missing": []}
    }

def _auto_tags_and_decision(payload: dict):
    """Lightweight rules → tags, direction score, suggestion."""
    tags = []
    dr = payload["rails"]
    mm = payload["money_map"]
    pos = payload["positioning"]
    vol = payload["volatility"]
    mom = payload["momentum"]
    liq = payload["liquidity"]

    # regime
    g = (dr.get("gamma_regime") or "Neutral").lower().replace(" ", "_")
    if "long" in g: tags.append("long_gamma")
    elif "short" in g: tags.append("short_gamma")
    else: tags.append("gamma_neutral")

    # rails/location
    p50 = dr.get("p50")
    spot = payload["price"]["working_spot"]
    if p50 and spot is not None:
        inside = (p50[0] <= spot <= p50[1])
        tags.append("rails_inside" if inside else "rails_outside_run")

    # pin risk
    pins = pos.get("pin_probabilities") or []
    if pins and pins[0].get("prob", 0) >= 0.3:
        tags.append("pin_risk_high")
    else:
        tags.append("pin_risk_low")

    # breakout probs
    bp = dr.get("breakout_prob_vs_p50") or {}
    up_p, dn_p = bp.get("up", 0.0), bp.get("down", 0.0)
    if max(up_p, dn_p) >= 0.55:
        tags.append("breakout_prob_up_high" if up_p > dn_p else "breakout_prob_down_high")
    else:
        tags.append("breakout_prob_balanced")

    # PCR
    pcr = pos.get("pcr_ring")
    if pcr is not None:
        if pcr <= 0.9: tags.append("pcr_bullish")
        elif pcr >= 1.1: tags.append("pcr_bearish")
        else: tags.append("pcr_neutral")

    # vol impulse / pricing
    iv_z = vol.get("iv_z")
    if iv_z is not None and abs(iv_z) >= 2.0:
        tags.append("iv_impulse_up" if iv_z > 0 else "iv_impulse_down")
    iv_minus_rv = vol.get("iv_minus_rv")
    if iv_minus_rv is not None:
        tags.append("options_rich_vs_rv" if iv_minus_rv > 0 else "options_cheap_vs_rv")

    # momentum tilt
    tilt = (mom.get("velocity_tilt") or 0.0)
    if tilt >= 0.15: tags.append("velocity_solid_up")
    elif tilt <= -0.15: tags.append("velocity_solid_down")
    elif tilt > 0: tags.append("velocity_slight_up")
    elif tilt < 0: tags.append("velocity_slight_down")

    # liquidity
    stress = liq.get("stress")
    if stress is not None:
        tags.append("liquidity_ok" if stress <= 0.5 else "liquidity_stressed")

    # basis
    basis = payload["price"].get("basis")
    if basis is not None:
        tags.append("basis_positive" if basis > 0 else "basis_negative")

    # direction score (0-100): rails & tilt & breakout
    score = 50
    score += 15 * tilt
    score += 20 * (up_p - dn_p)
    if "long_gamma" in tags and "rails_inside" in tags:
        score -= 10
    if "pcr_bullish" in tags: score += 5
    if "pcr_bearish" in tags: score -= 5
    score = max(0, min(100, round(score)))

    # bias & suggestion
    if "long_gamma" in tags and "rails_inside" in tags and abs(tilt) < 0.08:
        bias = "Mean-Revert"
        suggestion = "credit_condor"
    else:
        if score >= 58 and up_p > dn_p:
            bias = "Trend Up"
            suggestion = "debit_call_spread"
        elif score >= 58 and dn_p > up_p:
            bias = "Trend Down"
            suggestion = "debit_put_spread"
        else:
            bias = "Range / Fade"
            suggestion = "iron_fly"

    decision = {
        "suggestion": suggestion,
        "confidence_0_1": round(min(0.95, 0.45 + abs(score - 50) / 100), 2),
        "reasons": [t for t in tags if t.startswith("breakout_") or "gamma" in t or "rails" in t][:3],
        "levels": {}
    }
    return score, bias, tags, decision

def _emoji_summary(payload: dict) -> str:
    # line 1
    ds = payload["derived"].get("direction_score_0_100")
    phase = payload["meta"].get("session_phase")
    L1 = f"🧭 {phase} · {ds}/100"

    # line 2
    em = payload["volatility"].get("expected_move_pts_1sigma")
    p50 = payload["rails"].get("p50")
    if em and p50:
        L2 = f"🎯 ±{em:.0f} | p50: {p50[0]:.0f} ↔ {p50[1]:.0f}"
    elif em:
        L2 = f"🎯 ±{em:.0f}"
    else:
        L2 = "🎯 —"

    # line 3
    pins = payload["positioning"].get("pin_probabilities") or []
    pin_txt = f"{int(pins[0]['strike'])} ({pins[0]['prob']*100:.0f}%)" if pins else "—"
    walls = payload["money_map"].get("walls") or []
    sup = next((int(w["strike"]) for w in walls if w.get("type")=="support"), None)
    res = next((int(w["strike"]) for w in walls if w.get("type")=="resistance"), None)
    wsv = payload["money_map"].get("wall_shift_velocity_pts")
    walls_txt = f"{sup or '—'} / {res or '—'}"
    L3 = f"📌 {pin_txt} | 🧱 {walls_txt} ({wsv:+.0f}→)" if wsv is not None else f"📌 {pin_txt} | 🧱 {walls_txt}"

    # line 4
    nf = payload["iv_flow"].get("net_flow")
    tilt = payload["momentum"].get("velocity_tilt")
    L4 = f"🌊 IV net {nf:,.0f} ; 💨 tilt {tilt:+.02f}" if nf is not None and tilt is not None else "🌊 —"

    # line 5
    sp = payload["liquidity"].get("median_spread_bps")
    st = payload["liquidity"].get("stress")
    L5 = f"💧 {sp:.0f}bps | stress {st:.2f}" if sp is not None and st is not None else "💧 —"

    # line 6
    dec = payload.get("decision") or {}
    L6 = f"🛠️ {dec.get('suggestion','—')} ({(dec.get('confidence_0_1') or 0)*100:.0f}%)"

    return "\n".join([L1, L2, L3, L4, L5, L6])

# Streamlit entrypoint
if __name__ == "__main__":
    render_app()
