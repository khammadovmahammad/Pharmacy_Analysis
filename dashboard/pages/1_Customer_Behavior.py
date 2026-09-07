"""Customer Behavior - how the customer base actually purchases."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.dashboard_ui import kpi_row, page_setup, sample, sidebar
from analytics.loan_analytics import CONFIG, SEGMENT_COLORS

page_setup("Customer Behavior")
f, ff, state = sidebar()

if ff.empty:
    st.warning("No customers match the current filters.")
    st.stop()

kpi_row([
    ("Customers", f"{len(ff):,}", None),
    ("Median transactions (window)", f"{ff['tx_win'].median():.0f}", None),
    ("Median basket", f"{ff['avg_basket'].median():.2f}", None),
    ("Median monthly spend", f"{ff['monthly_spend'].median():.2f}", None),
    ("Median tenure (months)", f"{ff['tenure_months'].median():.0f}", None),
    ("Bought last month", f"{100*(ff['recency_months'] == 0).mean():.1f}%", None),
])

st.caption(
    f"Window = {CONFIG['WINDOW_MONTHS']} months ending {state['as_of']}. Frequency "
    "metrics are measured inside that window; basket and tenure are lifetime."
)

# ---------------------------------------------------------------- frequency
st.subheader("Purchase frequency")
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(ff["tx_win"].clip(upper=40), nbins=40,
                       title="Transactions in the observation window")
    fig.update_layout(xaxis_title="Transactions (capped at 40)",
                      yaxis_title="Customers", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    cov = (ff["coverage"].value_counts().sort_index()
           .rename_axis("active_months").reset_index(name="customers"))
    fig = px.bar(cov, x="active_months", y="customers",
                 title="Active months out of the window")
    fig.update_layout(xaxis_title="Months with at least one purchase",
                      yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)
st.caption("Active-month coverage, not raw transaction count, is the cleanest single "
           "predictor of whether a customer keeps buying (holdout AUC 0.90).")

# ------------------------------------------------------------------- value
st.subheader("Basket and spending")
c1, c2 = st.columns(2)
with c1:
    d = ff[ff["avg_basket"] <= ff["avg_basket"].quantile(0.99)]
    fig = px.histogram(d, x="avg_basket", nbins=60, title="Average basket value")
    fig.update_layout(xaxis_title="Average basket", yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    d = ff[ff["monthly_spend"] <= ff["monthly_spend"].quantile(0.99)]
    fig = px.histogram(d, x="monthly_spend", nbins=60,
                       title="Monthly spend inside the window")
    fig.update_layout(xaxis_title="Monthly spend", yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ tenure/recency
st.subheader("Tenure, recency and inactivity")
c1, c2, c3 = st.columns(3)
with c1:
    t = (ff["tenure_months"].value_counts().sort_index()
         .rename_axis("tenure_months").reset_index(name="customers"))
    fig = px.bar(t, x="tenure_months", y="customers", title="Customer tenure (months)")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    r = (ff["recency_months"].clip(upper=12).value_counts().sort_index()
         .rename_axis("months_since_last").reset_index(name="customers"))
    fig = px.bar(r, x="months_since_last", y="customers",
                 title="Months since last purchase")
    st.plotly_chart(fig, use_container_width=True)
with c3:
    gap = (ff["observed_months"] - ff["active_months_life"]).clip(lower=0)
    fig = px.histogram(gap, nbins=24,
                       title="Inactive months inside the relationship")
    fig.update_layout(xaxis_title="Months without a purchase",
                      yaxis_title="Customers", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------- relationships
st.subheader("Does buying more often mean spending more per visit?")
s = sample(ff)
c1, c2 = st.columns(2)
with c1:
    fig = px.scatter(s, x="tx_win", y="avg_basket", color="segment",
                     color_discrete_map=SEGMENT_COLORS, opacity=0.35,
                     title="Frequency vs average basket",
                     range_x=[0, float(ff["tx_win"].quantile(0.995)) + 1],
                     range_y=[0, float(ff["avg_basket"].quantile(0.99)) + 1])
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.scatter(s, x="tx_win", y="monthly_spend", color="segment",
                     color_discrete_map=SEGMENT_COLORS, opacity=0.35,
                     title="Frequency vs monthly spend",
                     range_x=[0, float(ff["tx_win"].quantile(0.995)) + 1],
                     range_y=[0, float(ff["monthly_spend"].quantile(0.99)) + 1])
    st.plotly_chart(fig, use_container_width=True)
st.caption("Basket size is nearly independent of frequency (Spearman -0.10 across the "
           "whole base), which is why basket drives loan sizing and not the stability score.")

# -------------------------------------------------------- behavioural matrix
st.subheader("Behavioural matrix: consistency vs intensity")
m = ff.copy()
m["intensity_band"] = pd.cut(m["intensity"], [0, 1.001, 2, 3, 4, 6, np.inf],
                             labels=["<=1", "1-2", "2-3", "3-4", "4-6", "6+"])
mat = (m.groupby(["coverage", "intensity_band"], observed=False)
        .size().rename("customers").reset_index())
pivot = mat.pivot(index="intensity_band", columns="coverage", values="customers").fillna(0)
fig = px.imshow(pivot, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
                labels=dict(x="Active months in window",
                            y="Transactions per active month", color="Customers"),
                title="Customer count by consistency and intensity")
st.plotly_chart(fig, use_container_width=True)
st.caption("The pilot population sits in the top-right: customers present in almost "
           "every month who also buy more than once when they come.")
