"""Customer Segmentation - transparent, rule-based groups and what they are worth."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.dashboard_ui import kpi_row, page_setup, sidebar
from analytics.loan_analytics import (CONFIG, SEGMENT_COLORS, SEGMENT_ORDER,
                                      monthly_sales_by_segment, segment_summary)

page_setup("Customer Segmentation")
f, ff, state = sidebar()

if ff.empty:
    st.warning("No customers match the current filters.")
    st.stop()

L = CONFIG["WINDOW_MONTHS"]
with st.expander("Segmentation rules (all thresholds live in analytics/loan_analytics.py)"):
    st.markdown(f"""
| Segment | Rule on the last {L} months | Why this cut |
|---|---|---|
| **Frequent** | active in ≥{CONFIG['SEG_FREQUENT_COVERAGE']} months **and** ≥3 transactions in ≥{CONFIG['SEG_FREQUENT_DEPTH3']} of them | 87% of these customers stayed active in ≥5 of the following 6 months |
| **Regular** | active in ≥{CONFIG['SEG_REGULAR_COVERAGE']} months | 61% future persistence |
| **Semi-Regular** | active in {CONFIG['SEG_SEMI_COVERAGE']}–{CONFIG['SEG_REGULAR_COVERAGE']-1} months | 27% future persistence |
| **Sporadic** | active in 1–{CONFIG['SEG_SEMI_COVERAGE']-1} months | 7% future persistence |
| **Dormant** | no purchase in the window | <1% future persistence |
| **New / Insufficient** | tenure < {CONFIG['MIN_TENURE_MONTHS']} months | not judgeable, whatever the frequency |

Every rate above was measured on a real holdout: segments built on data to 2025-12,
outcomes observed over 2026-01 to 2026-06.
""")

summ = segment_summary(ff)
kpi_row([
    ("Frequent", f"{int(summ.loc[summ.segment == 'Frequent', 'customers'].sum()):,}", None),
    ("Regular", f"{int(summ.loc[summ.segment == 'Regular', 'customers'].sum()):,}", None),
    ("Semi-Regular", f"{int(summ.loc[summ.segment == 'Semi-Regular', 'customers'].sum()):,}", None),
    ("Sporadic", f"{int(summ.loc[summ.segment == 'Sporadic', 'customers'].sum()):,}", None),
    ("Dormant", f"{int(summ.loc[summ.segment == 'Dormant', 'customers'].sum()):,}", None),
    ("New / Insufficient", f"{int(summ.loc[summ.segment == 'New/Insufficient', 'customers'].sum()):,}", None),
])

c1, c2 = st.columns([1, 1])
with c1:
    fig = px.bar(summ, x="segment", y="customers", color="segment",
                 color_discrete_map=SEGMENT_COLORS, title="Customers by segment")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.pie(summ, names="segment", values="customers", hole=0.45,
                 color="segment", color_discrete_map=SEGMENT_COLORS,
                 title="Share of the customer base")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("What each segment is worth")
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(summ, x="segment", y="median_monthly_spend", color="segment",
                 color_discrete_map=SEGMENT_COLORS,
                 title="Median monthly spend by segment")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Monthly spend")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.bar(summ, x="segment", y="median_basket", color="segment",
                 color_discrete_map=SEGMENT_COLORS,
                 title="Median average basket by segment")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Basket")
    st.plotly_chart(fig, use_container_width=True)
st.caption("Spend separates the segments sharply; basket barely does. Recurrence is "
           "what distinguishes these groups, not ticket size.")

c1, c2 = st.columns(2)
with c1:
    fig = px.box(ff, x="segment", y="score", color="segment",
                 color_discrete_map=SEGMENT_COLORS, points=False,
                 category_orders={"segment": SEGMENT_ORDER},
                 title="Behavioural score distribution by segment")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.bar(summ, x="segment", y="avg_tenure_months", color="segment",
                 color_discrete_map=SEGMENT_COLORS,
                 title="Average tenure by segment (months)")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Months")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Segment comparison table")
show = summ[["segment", "customers", "pct_of_customers", "median_tx_window",
             "median_basket", "median_monthly_spend", "avg_tenure_months",
             "median_recency_months", "median_score", "eligible", "eligibility_rate_pct"]]
st.dataframe(show.style.format({
    "customers": "{:,.0f}", "pct_of_customers": "{:.1f}%", "median_tx_window": "{:.0f}",
    "median_basket": "{:.2f}", "median_monthly_spend": "{:.2f}",
    "avg_tenure_months": "{:.1f}", "median_recency_months": "{:.0f}",
    "median_score": "{:.1f}", "eligible": "{:,.0f}", "eligibility_rate_pct": "{:.1f}%",
}), use_container_width=True)

# ------------------------------------------------------------------ trends
st.subheader("Monthly sales contribution by current segment")
try:
    trend = monthly_sales_by_segment(ff[["cust", "segment"]].assign(
        segment=ff["segment"].astype(str)), state["as_of"],
        key=f"{state['as_of']}|{state['branch']}|{len(ff)}")
    fig = px.area(trend, x="month", y="net_sales", color="segment",
                  color_discrete_map=SEGMENT_COLORS,
                  title="Net sales by segment over time")
    fig.update_layout(xaxis_title="", yaxis_title="Net sales")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Segments are assigned on the current window and then applied backwards, "
               "so this shows where today's segments came from - not a historical re-run.")
except Exception as exc:  # noqa: BLE001
    st.info(f"Monthly segment trend unavailable: {exc}")

# ------------------------------------------------------------------ drilldown
st.subheader("Drill into a segment")
pick = st.selectbox("Segment", SEGMENT_ORDER)
d = ff[ff["segment"] == pick]
if d.empty:
    st.info("No customers in this segment under the current filters.")
else:
    kpi_row([
        ("Customers", f"{len(d):,}", None),
        ("Median score", f"{d['score'].median():.1f}", None),
        ("Median monthly spend", f"{d['monthly_spend'].median():.2f}", None),
        ("Eligible", f"{int(d['eligible'].sum()):,}",
         f"{100*d['eligible'].mean():.1f}% of segment"),
        ("Median tenure", f"{d['tenure_months'].median():.0f} m", None),
    ])
    top = (d.groupby("primary_branch", observed=True)
            .agg(customers=("cust", "size"), eligible=("eligible", "sum"),
                 median_spend=("monthly_spend", "median"))
            .sort_values("customers", ascending=False).head(20).reset_index())
    st.dataframe(top, use_container_width=True)
