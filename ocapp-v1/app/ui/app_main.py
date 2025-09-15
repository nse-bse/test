# app/ui/app_main.py

import time
from datetime import datetime, timedelta
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Soft dependency: heartbeat (re-run without full reload)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ------------------ local modules (per your refactor layout) ------------------

from config import (
    now_ist, IST,
    NIFTY_HISTORICAL_EXPIRIES, BANKNIFTY_HISTORICAL_EXPIRIES,
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
from features.money_map import compute_money_map, money_center_of_mass, find_walls, make_money_map_commentary
from features.liquidity import (
    compute_spread_stats, impact_cost_proxy, liquidity_stress, gex_curve,
    realized_vol_annualized, pcr_vol, make_quick_execution_commentary,
)
from features.runway import (
    build_gate_runway_tables, apply_runway_enhancements, confidence_score,
)
from ui.styling import style_chain, movers_chart, net_oi_change_chart, style_mover_table


# ---------------------------- helpers ----------------------------

@st.cache_data(show_spinner=False, ttl=300)
def _load_expiries_for_session(source: str, symbol: str, api_base: str):
    """Unifies live + historical lists when in Historical mode."""
    if source == "Live API":
        return load_expiries(api_base, symbol)
    else:
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


# ------------------------------ app ------------------------------

def render_app():
    # ---- session state ----
    ss = st.session_state
    ss.setdefault("last_fetch", 0.0)
    ss.setdefault("snapshot", None)
    ss.setdefault("last_live_digest", None)
    ss.setdefault("last_hist_digest", None)
    ss.setdefault("spot_hist", {})  # {f"{sym}:{exp}": deque([...])}
    ss.setdefault("api_hits", 0)

    # Historical manual control state
    ss.setdefault("hist_date_pending", datetime.today().date())
    ss.setdefault("hist_time_pending", "15:30")
    ss.setdefault("hist_sel_key", None)   # (api_base, symbol, expiry) when in Historical
    ss.setdefault("current_source", None) # track source transitions

    # ---- sidebar: source & symbol ----
    st.sidebar.header("Data Source & Symbol")
    source = st.sidebar.selectbox("Source", ["Live API", "Historical API"], index=0)
    api_base = st.sidebar.text_input("API Base", value="http://127.0.0.1:8000")
    symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)

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

    # default to Cash (API) for index symbols; else keep Futures
    _index_syms = {"NIFTY", "BANKNIFTY", "NIFTY 50", "NIFTY BANK", "NIFTY-I", "BANKNIFTY-I"}
    _default_spot_idx = 1 if symbol.upper() in _index_syms else 0
    spot_source = st.sidebar.selectbox(
        "Use spot as",
        ["Futures (from OC snapshot)", "Cash (API)", "Cash (estimate via parity)"],
        index=_default_spot_idx,
    )


    # ---------- Historical controls (manual-only; no auto-fetch) ----------
    if source == "Historical API":
        with st.sidebar.expander("Historical Settings", expanded=True):
            base_date = st.date_input("Date", key="hist_date_pending", value=ss["hist_date_pending"])

            def _slots(start="09:15", end="15:30", step=3):
                slots = []
                cur = datetime.strptime(start, "%H:%M")
                end_t = datetime.strptime(end, "%H:%M")
                while cur <= end_t:
                    slots.append(cur.strftime("%H:%M"))
                    cur += timedelta(minutes=step)
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
    else:
        # Live only
        auto = st.sidebar.checkbox("Auto refresh", value=True)
        fetch_live_now = st.sidebar.button("Fetch Live Snapshot")

    st.sidebar.header("View")
    strike_window = st.sidebar.slider("± Strikes around ATM", 5, 40, 15, 1)
    with st.sidebar.expander("Analytics Ring Settings"):
        fixed_skew_ring = st.number_input("Ring size (± strikes)", 5, 25, 10, 1)
    with st.sidebar.expander("Heatmap Settings", expanded=True):
        enable_heatmap = st.checkbox("Enable heatmaps", value=True)
        available_metric_choices = ["oi_change", "oi", "volume", "ltp", "net_oi_change"]
        heatmap_metrics = st.multiselect(
            "Heatmap metrics",
            available_metric_choices,
            default=["oi_change", "oi", "net_oi_change"],
        )

    # Heartbeat (re-run without full reload)
    if st_autorefresh:
        hb = st_autorefresh(interval=30000, key="heartbeat")  # ~1s
        st.sidebar.caption(f"🫀 heartbeat #{hb}")
    else:
        st.sidebar.warning("`streamlit-autorefresh` not installed → no timed re-runs.")

    # ---- handle source transitions & stale views ----
    prev_source = ss.get("current_source")
    ss["current_source"] = source
    if source == "Historical API":
        sel_key = (api_base, symbol, expiry)
        if prev_source != "Historical API" or ss.get("hist_sel_key") != sel_key:
            # entering Historical or changing (api_base, symbol, expiry) while in Historical
            ss["hist_sel_key"] = sel_key
            ss["snapshot"] = None
            ss["last_hist_digest"] = None

    # ---- fetch logic ----
    snapshot = ss["snapshot"]
    now = now_ist()
    seconds_to_next_tick = None
    did_fetch = False

    if source == "Live API" and ss.get("last_fetch") is not None:
        auto_val = auto if "auto" in locals() else False

        if auto_val:
            if cadence_mode.startswith("Market"):
                seconds_to_next_tick = next_refresh_in_seconds(now)
                should_fetch = on_market_tick(now, ss["last_fetch"])
            else:
                elapsed = time.time() - ss["last_fetch"]
                seconds_to_next_tick = int(max(0, refresh_sec - elapsed))
                should_fetch = elapsed >= refresh_sec
        else:
            should_fetch = False

        if "fetch_live_now" in locals() and fetch_live_now:
            should_fetch = True

        if should_fetch:
            ss["snapshot"] = load_snapshot_from_api(api_base, symbol, expiry)
            ss["last_fetch"] = time.time()
            ss["last_live_digest"] = snapshot_digest(ss["snapshot"]) if ss["snapshot"] else None
            ss["api_hits"] += 1
            did_fetch = True
            # recompute next tick label
            now = now_ist()
            if cadence_mode.startswith("Market"):
                seconds_to_next_tick = next_refresh_in_seconds(now)
            else:
                seconds_to_next_tick = int(refresh_sec)

    elif source == "Historical API":
        # Manual fetch only
        if "fetch_hist_now" in locals() and fetch_hist_now:
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
        if source == "Historical API":
            st.title("Option Chain Dashboard")
            topL, topR = st.columns([3,1])
            with topL:
                st.caption("No historical snapshot loaded. Choose **Date/Time** and click **Fetch Historical Snapshot**.")
            with topR:
                st.metric("API hits (session)", ss.get("api_hits", 0))
            return
        else:
            st.title("Option Chain Dashboard")
            topL, topR = st.columns([3,1])
            with topL:
                st.caption("Waiting for first live fetch…")
                if seconds_to_next_tick is not None:
                    st.caption(
                        f"Next auto-fetch in ~{seconds_to_next_tick}s → "
                        f"at **{(now + timedelta(seconds=seconds_to_next_tick)).strftime('%H:%M:%S')} IST**"
                    )
            with topR:
                st.metric("API hits (session)", ss.get("api_hits", 0))
            return  # stop until we have a snapshot

    # ---- processing ----
    sym = snapshot.get("symbol", symbol)
    exp = snapshot.get("expiry", expiry)
    ts = snapshot.get("timestamp", "?")

    df, fut_spot, all_fields = group_chain(snapshot)
    working_spot = fut_spot
    cash_spot = None
    if spot_source == "Cash (API)":
        cash_spot = load_cash_spot_from_api(api_base, symbol)
        working_spot = cash_spot if cash_spot is not None else fut_spot
        if cash_spot is None:
            st.warning("Could not fetch Cash spot price from API. Falling back to Futures spot.")
    elif spot_source == "Cash (estimate via parity)":
        cash_spot = estimate_cash_spot_parity_consistent(df, fut_spot)
        working_spot = cash_spot if cash_spot is not None else fut_spot
        if cash_spot is None:
            st.warning("Could not estimate Cash spot price. Falling back to Futures spot.")

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
        if source == "Live API" and "auto" in locals() and auto and seconds_to_next_tick is not None:
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
        st.caption(f"Last fetched: {ts}")
    with topR:
        st.metric("API hits (session)", ss.get("api_hits", 0))

    # ---- Tabs ----
    tab_overview, tab_runway, tab_money, tab_chain, tab_writing, tab_velocity, tab_movers, tab_iv, tab_diag = st.tabs(
        ["Overview", "Gates & Runway", "Money Map", "Chain", "Writing", "Velocity", "Movers", "IV & Skew", "Diagnostics"]
    )

    # --------------------------- Overview ---------------------------
    with tab_overview:
        st.subheader("Key Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
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

        # Liquidity pack
        liq_stats = compute_spread_stats(df_ana if not df_ana.empty else df, working_spot, band=3)
        icp = impact_cost_proxy(df_ana if not df_ana.empty else df, working_spot, qty=50)
        liq_data = {}
        if liq_stats and icp:
            liq_data["median_spread_bps"] = liq_stats.get("median_spread_bps")
            liq_data["ic_bps"] = icp.get("ic_bps")
            liq_data["liquidity_stress"] = liquidity_stress(
                liq_data["median_spread_bps"], liq_data["ic_bps"],
                compute_money_map(df_ana if not df_ana.empty else df)
            )
        if liq_data:
            cL1, cL2, cL3 = st.columns(3)
            med_bps = liq_data.get("median_spread_bps")
            ic_bps = liq_data.get("ic_bps")
            stress = liq_data.get("liquidity_stress")
            cL1.metric("Median spread (bps)", f"{med_bps:.0f}" if med_bps is not None and np.isfinite(med_bps) else "–")
            cL2.metric("Impact proxy (bps)", f"{ic_bps:.0f}" if ic_bps is not None and np.isfinite(ic_bps) else "–")
            cL3.metric("Liquidity stress", f"{(stress*100):.0f} / 100" if stress is not None else "–", help="Higher = harder fills")

        st.markdown("---")

        # Compute metrics for commentary
        atm_iv_now = compute_atm_iv(df_ana, working_spot)
        iv_hist = make_iv_skew_history(sym, exp, last_n=40)
        iv_z = zscore_last(iv_hist["atm_iv"], window=20) if not iv_hist.empty else None
        regime, gex_val = compute_gamma_regime(df_ana, working_spot, ring_size=fixed_skew_ring)

        mstart, mend = market_bounds(now)
        minutes_left = max(1, (mend - now).total_seconds() / 60.0)
        rails = compute_iv_rails(working_spot, atm_iv_now, minutes_left)

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

        # Gamma regime metric + GEX curve
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

        # Rails
        if rails:
            r50 = rails.get(0.5)
            if r50:
                st.metric("IV Rails (50%)", f"{r50[0]:.0f} ↔ {r50[1]:.0f}")
                all_zones = (sup_zones or []) + (res_zones or [])
                if all_zones and working_spot is not None:
                    nz = min(all_zones, key=lambda k: abs(k - working_spot))
                    inside = (r50[0] <= nz <= r50[1])
                    st.caption(f"{'🎯' if inside else '🚧'} Nearest zone {int(nz)} is {'inside' if inside else 'outside'} the 50% cone.")

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
        # Research mode removed → only Live (last N)
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

    # ----------------------------- Movers -----------------------------
    with tab_movers:
        st.subheader("Top Movers & Analytics (window)")
        ce_ch, pe_ch, ce_oi, pe_oi, ch_df, oi_df, net_df = build_movers_long(dfw, n=5)
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
            st.dataframe(style_mover_table(ce_ch, "CE_oi_change"), use_container_width=True)
            st.dataframe(style_mover_table(ce_oi, "CE_oi"), use_container_width=True)
        with cR:
            st.markdown("#### PE Movers")
            st.dataframe(style_mover_table(pe_ch, "PE_oi_change"), use_container_width=True)
            st.dataframe(style_mover_table(pe_oi, "PE_oi"), use_container_width=True)

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


# Streamlit entrypoint
if __name__ == "__main__":
    render_app()
