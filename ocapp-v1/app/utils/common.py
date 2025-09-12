import time
import numpy as np
import pandas as pd
import streamlit as st

def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        raise RuntimeError("No rerun method available in this Streamlit version")

def emit_meta_refresh_after(seconds: int, cache_buster: bool = True):
    seconds = max(1, int(seconds))
    if cache_buster:
        st.markdown(
            f"<meta http-equiv='refresh' content='{seconds};url=?_ts={int(time.time())}'>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<meta http-equiv='refresh' content='{seconds}'>", unsafe_allow_html=True)

def to_float(x):
    if isinstance(x, (int, float)): return float(x)
    if x is None: return float("nan")
    s = str(x).strip().replace(",", "")
    if s in {"", "-", "nan", "NaN", "null", "None"}: return float("nan")
    try: return float(s)
    except Exception: return float("nan")
