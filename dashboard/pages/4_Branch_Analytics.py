"""Branch Analytics - every branch on the metrics that matter for lending."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.dashboard_ui import kpi_row, page_setup, sidebar
from analytics.loan_analytics import CONFIG, branch_summary, load_branch_month

page_setup("Branch Analytics")
f, ff, state = sidebar()

b = branch_summary(f, state["as_of"])          # always the full base, not the filter
st.caption("This page always uses the complete customer base so branches stay "
           "comparable; sidebar filters apply to the other pages.")

kpi_row([
    ("Branches", f"{len(b):,}", None),
    ("With full 24-month history", f"{int((b['months_present'] >= 24).sum()):,}", None),
    ("Median primary customers", f"{b['primary_customers'].median():,.0f}", None),
    ("Median eligibility rate", f"{b['eligibility_rate_pct'].median():.1f}%", None),
    ("Total eligible customers", f"{int(b['eligible'].sum()):,}", None),
])

st.subheader("Size is not quality")
c1, c2 = st.columns(2)
with c1:
    d = b[b["primary_customers"] >= 200]
    fig = px.scatter(d, x="primary_customers", y="eligibility_rate_pct",
                     size="eligible", color="elig_median_monthly_spend",
                     hover_name="branch", color_continuous_scale="Viridis",
                     title="Branch size vs eligibility rate",
                     labels={"primary_customers": "Customers whose primary branch this is",
                             "eligibility_rate_pct": "Eligible %",
                             "elig_median_monthly_spend": "Median monthly spend"})
    st.plotly_chart(fig, use_container_width=True)
with c2:
    d = b[b["primary_customers"] >= 200]
    fig = px.scatter(d, x="net_sales", y="eligible", hover_name="branch",
                     color="eligibility_rate_pct", color_continuous_scale="Viridis",
                     title="Net sales vs eligible customers",
                     labels={"net_sales": "Net sales", "eligible": "Eligible customers",
                             "eligibility_rate_pct": "Eligible %"})
    st.plotly_chart(fig, use_container_width=True)
st.caption("The largest branches by sales are not the ones with the highest share of "
           "loan-eligible customers - which is exactly why the pilot ranking is a "
           "weighted score and not a sort on revenue.")

st.subheader("Distribution of branch quality")
c1, c2, c3 = st.columns(3)
with c1:
    fig = px.histogram(b, x="eligibility_rate_pct", nbins=40,
                       title="Eligibility rate across branches")
    fig.update_layout(xaxis_title="Eligible % of primary customers", yaxis_title="Branches")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.histogram(b, x="primary_customer_ratio_pct", nbins=40,
                       title="Branch loyalty (primary customer ratio)")
    fig.update_layout(xaxis_title="% of visitors for whom this is their pharmacy",
                      yaxis_title="Branches")
    st.plotly_chart(fig, use_container_width=True)
with c3:
    fig = px.histogram(b[b["tx_cv_recent"].notna()], x="tx_cv_recent", nbins=40,
                       title="Month-to-month volatility (recent window)")
    fig.update_layout(xaxis_title="Coefficient of variation of monthly transactions",
                      yaxis_title="Branches")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Branch comparison table")
c1, c2, c3 = st.columns(3)
min_cust = c1.number_input("Min primary customers", 0, 10000, 300, 50)
min_months = c2.slider("Min months present", 1, 24, 12)
sort_by = c3.selectbox("Sort by", ["eligible", "eligibility_rate_pct", "net_sales",
                                   "primary_customers", "elig_median_monthly_spend",
                                   "mean_score", "primary_customer_ratio_pct"])

view = b[(b["primary_customers"] >= min_cust) & (b["months_present"] >= min_months)]
cols = ["branch", "unique_customers", "primary_customers", "primary_customer_ratio_pct",
        "tx", "tx_per_primary_customer", "net_sales", "avg_basket",
        "n_frequent", "n_regular", "eligible", "eligibility_rate_pct",
        "mean_score", "elig_median_score", "elig_median_monthly_spend",
        "median_loyalty", "tx_cv_recent", "tx_growth_pct", "months_present",
        "pct_hist_6m", "return_rate_pct", "rx_share_pct"]
cols = [c for c in cols if c in view.columns]
st.dataframe(view.sort_values(sort_by, ascending=False)[cols].style.format({
    "unique_customers": "{:,.0f}", "primary_customers": "{:,.0f}",
    "primary_customer_ratio_pct": "{:.1f}%", "tx": "{:,.0f}",
    "tx_per_primary_customer": "{:.1f}", "net_sales": "{:,.0f}", "avg_basket": "{:.2f}",
    "n_frequent": "{:,.0f}", "n_regular": "{:,.0f}", "eligible": "{:,.0f}",
    "eligibility_rate_pct": "{:.1f}%", "mean_score": "{:.1f}",
    "elig_median_score": "{:.1f}", "elig_median_monthly_spend": "{:.2f}",
    "median_loyalty": "{:.2f}", "tx_cv_recent": "{:.2f}", "tx_growth_pct": "{:.1f}%",
    "pct_hist_6m": "{:.1f}%", "return_rate_pct": "{:.2f}%", "rx_share_pct": "{:.1f}%",
}), use_container_width=True, height=460)
st.download_button("Download branch table (CSV)",
                   view[cols].to_csv(index=False).encode(),
                   file_name=f"branch_metrics_{state['as_of']}.csv")

st.subheader("Single branch detail")
pick = st.selectbox("Branch", view.sort_values("eligible", ascending=False)["branch"].tolist())
row = b[b["branch"] == pick].iloc[0]
kpi_row([
    ("Primary customers", f"{row['primary_customers']:,.0f}", None),
    ("Eligible", f"{row['eligible']:,.0f}", f"{row['eligibility_rate_pct']:.1f}%"),
    ("Frequent", f"{row.get('n_frequent', 0):,.0f}", None),
    ("Regular", f"{row.get('n_regular', 0):,.0f}", None),
    ("Median score", f"{row['median_score']:.1f}", None),
    ("Loyalty", f"{row['primary_customer_ratio_pct']:.1f}%", None),
])
bm = load_branch_month(state["as_of"])
d = bm[bm["branch"] == pick]
c1, c2 = st.columns(2)
with c1:
    fig = px.line(d, x="month", y="tx", markers=True, title=f"Monthly transactions - {pick}")
    fig.update_layout(xaxis_title="", yaxis_title="Transactions")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.line(d, x="month", y="customers", markers=True,
                  title=f"Monthly active customers - {pick}")
    fig.update_layout(xaxis_title="", yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)

seg_cols = [c for c in b.columns if c.startswith("n_") and c != "n_branches"]
mix = (pd.DataFrame({"segment": [c[2:].replace("_", " ").title() for c in seg_cols],
                     "customers": [row[c] for c in seg_cols]}))
fig = px.bar(mix, x="segment", y="customers", title=f"Segment mix - {pick}")
fig.update_layout(xaxis_title="", yaxis_title="Customers")
st.plotly_chart(fig, use_container_width=True)
