from typing import Dict, List, Optional, Tuple
import hashlib, json as _json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    NIFTY_EXPIRIES_FALLBACK, BANKNIFTY_EXPIRIES_FALLBACK,
    FNO_STOCKS,
)
from utils.common import to_float
from config import IST, now_ist

# =========================
# HTTP session (unchanged)
# =========================
@st.cache_resource(show_spinner=False)
def get_http() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, connect=3, read=3, backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# =========================
# Expiries (unchanged)
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def load_expiries(api_base: str, symbol: str) -> Tuple[List[str], Optional[str]]:
    url = f"{api_base}/expiries?symbol={symbol}"
    try:
        r = get_http().get(url, timeout=6)
        if r.status_code == 200:
            data = r.json()
            exps = data.get("expiries") or data.get("data") or []
            exps = [str(x).strip() for x in exps if str(x).strip()]
            if exps:
                return exps, None
        return (
            NIFTY_EXPIRIES_FALLBACK if symbol == "NIFTY" else BANKNIFTY_EXPIRIES_FALLBACK,
            f"Empty response (HTTP {r.status_code})",
        )
    except Exception as e:
        return (
            NIFTY_EXPIRIES_FALLBACK if symbol == "NIFTY" else BANKNIFTY_EXPIRIES_FALLBACK,
            f"{e}",
        )

# =========================
# OC snapshots (unchanged)
# =========================
def load_snapshot_from_api(api_base: str, symbol: str, expiry: str) -> Optional[Dict]:
    try:
        url = f"{api_base}/oc?symbol={symbol}&expiry={expiry}"
        r = get_http().get(url, timeout=10)
        if r.status_code != 200:
            st.error(f"API Error: Status code {r.status_code} from {url}")
            return None
        return r.json()
    except requests.exceptions.Timeout:
        st.error("API request timed out after 10 seconds.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during API fetch: {e}")
        return None

def load_snapshot_from_hist_api(api_base: str, symbol: str, expiry: str, date: str, time_hhmm: str) -> Optional[Dict]:
    try:
        url = f"{api_base}/oc/hist?symbol={symbol}&expiry={expiry}&date={date}&time={time_hhmm}"
        r = get_http().get(url, timeout=10)
        if r.status_code != 200:
            st.error(f"API Error: Status code {r.status_code} from {url}")
            return None
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Historical API request timed out after 10 seconds.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Historical API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during historical fetch: {e}")
        return None

# ======================================================
# Unified "truth of spot": prefer local OHLC LTPs
#   - Indices  -> /nse/index/ohlc   (LTP)
#   - Stocks   -> /nse/stocks/ohlc  (LTP)
#   - Fallback -> {api_base}/spot   (legacy)
# ======================================================

_INDEX_OHLC_URL = "http://localhost:8000/nse/index/ohlc"
_STOCK_OHLC_URL = "http://localhost:8000/nse/stocks/ohlc"

# Accept the usual variants you use across the app
_INDEX_NAME_MAP = {
    "NIFTY": "NIFTY 50",
    "NIFTY 50": "NIFTY 50",
    "NIFTY-I": "NIFTY 50",
    "NIFTY_50": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTY BANK": "NIFTY BANK",
    "BANKNIFTY-I": "NIFTY BANK",
}

def _symbol_to_index_name(symbol: str) -> Optional[str]:
    if not symbol:
        return None
    k = symbol.strip().upper().replace("_", " ")
    return _INDEX_NAME_MAP.get(k)

# Tiny caches so Streamlit reruns don't spam your local services.
@st.cache_data(show_spinner=False, ttl=1.0)
def _load_index_spot_from_local_api(index_human_name: str) -> Optional[float]:
    try:
        r = get_http().get(_INDEX_OHLC_URL, params={"symbols": index_human_name}, timeout=2.5)
        if r.status_code != 200:
            return None
        data = r.json()  # {"High":..,"LTP":..,"Low":..,"Open":..}
        v = to_float(data.get("LTP"))
        if v is None or not np.isfinite(v) or not (1 <= v < 1e7):
            return None
        return float(v)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=1.0)
def _load_stock_spot_from_local_api(stock_symbol: str) -> Optional[float]:
    try:
        # requests will URL-encode special tickers like "M&M", "MCDOWELL-N", etc.
        r = get_http().get(_STOCK_OHLC_URL, params={"symbols": stock_symbol}, timeout=2.5)
        if r.status_code != 200:
            return None
        data = r.json()  # {"High":..,"LTP":..,"Low":..,"Open":..}
        v = to_float(data.get("LTP"))
        if v is None or not np.isfinite(v) or not (1 <= v < 1e7):
            return None
        return float(v)
    except Exception:
        return None

def _legacy_spot(api_base: str, symbol: str) -> Optional[float]:
    try:
        url = f"{api_base}/spot?symbol={symbol}"
        r = get_http().get(url, timeout=6)
        if r.status_code == 200:
            js = r.json()
            v = to_float(js.get("spot"))
            return v if np.isfinite(v) else None
    except Exception:
        pass
    return None

def load_cash_spot_from_api(api_base: str, symbol: str) -> Optional[float]:
    """
    Unified 'truth of spot':
      - Index symbols -> local Index OHLC LTP
      - Stock symbols -> local Stocks OHLC LTP
      - Fallback      -> legacy {api_base}/spot
    Signature unchanged.
    """
    # Index path
    idx = _symbol_to_index_name(symbol)
    if idx:
        v = _load_index_spot_from_local_api(idx)
        if v is not None:
            return v

    # Stock path
    sym_up = (symbol or "").strip().upper()
    if sym_up in FNO_STOCKS:
        v = _load_stock_spot_from_local_api(sym_up)
        if v is not None:
            return v

    # Fallback to legacy REST spot
    return _legacy_spot(api_base, symbol)

# =========================
# Digest (unchanged)
# =========================
def snapshot_digest(snap: Dict) -> str:
    try:
        opts = snap.get("options", []) or []
        def _k(o):
            t = str(o.get("type"))
            s = to_float(o.get("strike"))
            s = s if np.isfinite(s) else float("inf")
            return (t, s)
        opts_sorted = sorted(opts, key=_k)
        norm = {
            "symbol": snap.get("symbol"),
            "expiry": snap.get("expiry"),
            "timestamp": snap.get("timestamp"),
            "spot": snap.get("spot"),
            "options": opts_sorted,
        }
        raw = _json.dumps(norm, separators=(",", ":"), sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
    except Exception:
        return "digest_error"
