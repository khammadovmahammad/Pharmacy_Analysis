"""
Shared sidebar + formatting helpers for the loan-pilot pages.
Keeping this in one place means a filter added here appears on every page.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.loan_analytics import (CONFIG, SEGMENT_ORDER, available_months,
                                      load_customer_features, apply_filters)


def page_setup(title: str, icon: str = "💊"):
    st.set_page_config(page_title=f"{title} | Pharmacy Analytics",
                       page_icon=icon, layout="wide")
    st.title(title)


@st.cache_data(show_spinner=False)
def _branch_options(_f: pd.DataFrame, key: str) -> list[str]:
    br = _f["primary_branch"].dropna().unique().tolist()
    br.sort(key=lambda v: (len(v), v))
    return ["All Branches"] + br


def sidebar(features_needed: bool = True):
    """
    Renders the shared sidebar and returns (features, filtered_features, state).

    `as_of` re-runs the whole model at a past month, which is how the model
    was validated - it is not just a display filter.
    """
    months = available_months()

    st.sidebar.header("Model window")
    as_of = st.sidebar.selectbox("As-of month (end of observation)", months,
                                 index=len(months) - 1)
    st.sidebar.caption(
        f"Segments and scores use the {CONFIG['WINDOW_MONTHS']} months ending "
        f"{as_of}. Customers with less than {CONFIG['MIN_TENURE_MONTHS']} months "
        f"of history are held out as New/Insufficient."
    )

    f = load_customer_features(as_of)

    st.sidebar.header("Filters")
    branch = st.sidebar.selectbox("Primary branch", _branch_options(f, as_of))
    segments = st.sidebar.multiselect("Customer segment", SEGMENT_ORDER, default=[])
    score_range = st.sidebar.slider("Behavioural score", 0, 100, (0, 100), 5)
    min_tx = st.sidebar.number_input("Min transactions in window", 0, 200, 0, 1)
    min_tenure = st.sidebar.number_input("Min observed months (tenure)", 0, 24, 0, 1)

    with st.sidebar.expander("Value filters"):
        b_max = float(f["avg_basket"].quantile(0.99))
        s_max = float(f["monthly_spend"].quantile(0.99))
        basket_range = st.slider("Average basket", 0.0, round(b_max, 1),
                                 (0.0, round(b_max, 1)))
        spend_range = st.slider("Monthly spend", 0.0, round(s_max, 1),
                                (0.0, round(s_max, 1)))

    ff = apply_filters(f, branch=branch, segments=segments or None,
                       min_score=score_range[0], max_score=score_range[1],
                       min_tx=min_tx, min_tenure=min_tenure,
                       basket_range=(basket_range[0], basket_range[1] * 1.5),
                       spend_range=(spend_range[0], spend_range[1] * 1.5))

    st.sidebar.metric("Customers in view", f"{len(ff):,}",
                      f"{100*len(ff)/max(len(f),1):.1f}% of base")
    return f, ff, {"as_of": as_of, "branch": branch, "segments": segments}


def kpi_row(items: list[tuple[str, str, str | None]]):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def money(v) -> str:
    return f"{v:,.0f}"


def sample(df: pd.DataFrame, n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Scatter plots choke on 900k points; a sample answers the same question."""
    return df.sample(n, random_state=seed) if len(df) > n else df
