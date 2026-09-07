"""Customer Explorer - find a customer, see the profile behind the score."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.dashboard_ui import kpi_row, page_setup, sidebar
from analytics.loan_analytics import CONFIG, explain_score, run_query

page_setup("Customer Explorer")
f, ff, state = sidebar()

st.caption("Sidebar filters narrow the list below. Search by customer key for a "
           "single profile.")

c1, c2, c3 = st.columns([2, 1, 1])
search = c1.text_input("Customer key (exact or prefix)")
sort_by = c2.selectbox("Sort by", ["score", "monthly_spend", "tx_win", "avg_basket",
                                   "tenure_months"])
limit = c3.number_input("Rows", 50, 5000, 500, 50)

view = ff
if search:
    s = search.strip()
    view = view[view["cust"].astype(str).str.startswith(s)]

cols = ["cust", "primary_branch", "segment", "score", "score_band", "coverage",
        "intensity", "tx_win", "monthly_spend", "avg_basket", "median_basket",
        "tenure_months", "recency_months", "n_branches", "primary_branch_share",
        "eligible"]
st.dataframe(view.sort_values(sort_by, ascending=False)[cols].head(int(limit)).style.format({
    "score": "{:.1f}", "intensity": "{:.1f}", "monthly_spend": "{:.2f}",
    "avg_basket": "{:.2f}", "median_basket": "{:.2f}", "primary_branch_share": "{:.2f}"}),
    use_container_width=True, height=420)
st.caption(f"{len(view):,} customers match.")

st.subheader("Customer profile")
default = str(view["cust"].iloc[0]) if len(view) else ""
key = st.text_input("Customer key to profile", value=default)

if key:
    row = f[f["cust"].astype(str) == key.strip()]
    if row.empty:
        st.info("No customer with that key.")
        st.stop()
    r = row.iloc[0]

    kpi_row([
        ("Segment", str(r["segment"]), None),
        ("Score", f"{r['score']:.1f}", str(r["score_band"])),
        ("Eligible", "Yes" if r["eligible"] else "No", None),
        ("Monthly spend", f"{r['monthly_spend']:.2f}", None),
        ("Tenure", f"{int(r['tenure_months'])} m", None),
        ("Last purchase", f"{int(r['recency_months'])} m ago", None),
    ])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Behavioural metrics**")
        prof = pd.DataFrame({
            "metric": ["Primary branch", "Branches used", "Loyalty to primary branch",
                       "Transactions (lifetime)", "Transactions (window)",
                       f"Active months of last {CONFIG['WINDOW_MONTHS']}",
                       "Months with 2+ purchases", "Months with 3+ purchases",
                       "Transactions per active month", "Average basket",
                       "Median basket", "Lifetime spend", "First month", "Last month"],
            "value": [r["primary_branch"], int(r["n_branches"]),
                      f"{100*r['primary_branch_share']:.0f}%", int(r["n_tx"]),
                      int(r["tx_win"]), int(r["coverage"]), int(r["depth_2"]),
                      int(r["depth_3"]), f"{r['intensity']:.2f}",
                      f"{r['avg_basket']:.2f}", f"{r['median_basket']:.2f}",
                      f"{r['total_spend']:,.2f}", r["first_month"], r["last_month"]],
        })
        st.dataframe(prof, use_container_width=True, hide_index=True)
    with c2:
        br = explain_score(r)
        fig = px.bar(br[br["component"] != "TOTAL"], x="points", y="component",
                     orientation="h", title="Score breakdown")
        fig.update_layout(xaxis_title="Points (of 100)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Purchase history**")
    try:
        hist = run_query(f"""
            SELECT CAST(date_trunc('month', tarix_saat) AS DATE) AS month,
                   CAST(aptek_id AS VARCHAR) AS branch,
                   COUNT(*) AS transactions,
                   SUM(CAST(net_amount AS DOUBLE)) AS net_spend
            FROM {CONFIG['TX_TABLE']}
            WHERE {CONFIG['QUALIFYING_FILTER']} AND musteri_acari = ?
            GROUP BY 1, 2 ORDER BY 1
        """, [key.strip()])
        if hist.empty:
            st.info("No qualifying transactions found.")
        else:
            c1, c2 = st.columns(2)
            m = hist.groupby("month", as_index=False).agg(
                transactions=("transactions", "sum"), net_spend=("net_spend", "sum"))
            with c1:
                fig = px.bar(m, x="month", y="transactions", title="Transactions per month")
                fig.update_layout(xaxis_title="", yaxis_title="Transactions")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(m, x="month", y="net_spend", title="Spend per month")
                fig.update_layout(xaxis_title="", yaxis_title="Net spend")
                st.plotly_chart(fig, use_container_width=True)
            byb = (hist.groupby("branch", as_index=False)
                       .agg(transactions=("transactions", "sum"),
                            net_spend=("net_spend", "sum"))
                       .sort_values("transactions", ascending=False))
            st.dataframe(byb, use_container_width=True, hide_index=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"History unavailable: {exc}")
