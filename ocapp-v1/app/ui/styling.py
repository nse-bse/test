from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import altair as alt

def style_chain(
    df: pd.DataFrame,
    atm_idx: int,
    selected_fields: List[str],
    spot: Optional[float],
    heat_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, "pd.io.formats.style.Styler"]:
    NUM_FIELDS = {"ltp","oi","oi_change","iv","delta","gamma","theta","vega","volume","net_oi_change"}
    ce_cols = [f"CE_{f}" for f in selected_fields if f"CE_{f}" in df.columns]
    pe_cols = [f"PE_{f}" for f in selected_fields if f"PE_{f}" in df.columns]
    other_cols = [c for c in selected_fields if c not in NUM_FIELDS]
    if "net_oi_change" in selected_fields and "net_oi_change" in df.columns:
        other_cols.append("net_oi_change")
    cols = ce_cols + ["strike"] + pe_cols + other_cols
    dfv = df.loc[:, [c for c in cols if c in df.columns]].copy()
    dfv.replace({"": np.nan, "-": np.nan, "NaN": np.nan, "nan": np.nan, "null": np.nan, "None": np.nan}, inplace=True)
    num_cols = [c for c in dfv.columns if c != "strike" and pd.api.types.is_numeric_dtype(dfv[c])]

    def row_style(row):
        styles = []
        is_atm = (row.name == atm_idx)
        k = row["strike"]
        call_itm = (spot is not None and pd.notna(k) and k < spot)
        put_itm  = (spot is not None and pd.notna(k) and k > spot)
        for c in dfv.columns:
            bg = ""
            if c == "strike" and is_atm: bg = "background-color:#fff3cd;"
            elif c.startswith("CE_") and call_itm: bg = "background-color:#e7f3ff;"
            elif c.startswith("PE_") and put_itm:  bg = "background-color:#ffe7e7;"
            styles.append(bg)
        return styles

    styler = dfv.style.apply(row_style, axis=1)
    if heat_cols:
        heat_subset = [c for c in heat_cols if c in dfv.columns and c in num_cols]
        if heat_subset:
            standard_heat_cols = [c for c in heat_subset if c != "net_oi_change"]
            if standard_heat_cols:
                styler = styler.background_gradient(subset=standard_heat_cols)
            if "net_oi_change" in heat_subset and "net_oi_change" in dfv.columns:
                styler = styler.background_gradient(subset=["net_oi_change"], cmap="RdYlGn",
                                                    vmin=dfv["net_oi_change"].min(), vmax=dfv["net_oi_change"].max())
    fmt = {"strike": "{:,.0f}"}; fmt.update({c: "{:,.2f}" for c in num_cols})
    styler = styler.format(fmt, na_rep="–")
    return dfv, styler

def movers_chart(df_long: pd.DataFrame, title: str):
    if df_long is None or df_long.empty: return None
    ord = df_long.groupby("strike")["value"].sum().sort_values(ascending=True).index.tolist()
    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            y=alt.Y("strike:N", sort=ord, title="Strike"),
            x=alt.X("value:Q", title=title),
            color=alt.Color("side:N", scale=alt.Scale(domain=["CE","PE"], range=["#2E86DE","#E74C3C"])),
            tooltip=[alt.Tooltip("side:N"), alt.Tooltip("strike:N"), alt.Tooltip("value:Q", title=title, format=",")]
        )
        .properties(height=240)
        .interactive()
    )
    return chart

def net_oi_change_chart(dfw: pd.DataFrame):
    if "net_oi_change" not in dfw.columns or dfw.empty:
        return None
    df_chart = dfw[["strike", "net_oi_change"]].copy()
    df_chart["net_oi_change"] = pd.to_numeric(df_chart["net_oi_change"], errors="coerce")
    df_chart = df_chart.dropna().sort_values("strike")
    if df_chart.empty:
        return None
    max_val = df_chart["net_oi_change"].abs().max() + 1
    chart = alt.Chart(df_chart).mark_bar().encode(
        y=alt.Y("strike:O", sort="-x", title="Strike"),
        x=alt.X("net_oi_change:Q", title="Net OI Change (PE - CE)", scale=alt.Scale(domain=[-max_val, max_val])),
        color=alt.condition(alt.datum.net_oi_change > 0, alt.value("#6ab04c"), alt.value("#E74C3C")),
        tooltip=[alt.Tooltip("strike:O"), alt.Tooltip("net_oi_change:Q", title="Net Change", format=",")]
    ).properties(height=300).interactive()
    return chart

def style_mover_table(df: pd.DataFrame, value_col: str):
    if df.empty or value_col not in df.columns: return df
    sty = df.style.format({"strike": "{:,.0f}", value_col: "{:,}"}).bar(subset=[value_col], color="#6ab04c")
    return sty
