"""Pilot Branch Selection - the ranked recommendation and the evidence behind it."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.dashboard_ui import kpi_row, page_setup, sidebar
from analytics.loan_analytics import (CONFIG, PILOT_COMPONENT_LABELS, branch_summary,
                                      pilot_ranking, pilot_sensitivity)

page_setup("Pilot Branch Selection")
f, ff, state = sidebar()
b = branch_summary(f, state["as_of"])

st.markdown(
    "A branch qualifies for the pilot only if it has a **full history** and enough "
    "**eligible customers** to produce a readable result. Qualified branches are then "
    "scored on seven components, each converted to a percentile rank inside the "
    "qualified set so no raw unit dominates."
)

with st.sidebar:
    st.header("Pilot gate")
    min_months = st.slider("Min months present", 1, 24, CONFIG["BRANCH_MIN_MONTHS_PRESENT"])
    min_elig = st.number_input("Min eligible customers", 50, 5000,
                               CONFIG["BRANCH_MIN_ELIGIBLE"], 50)
    st.header("Pilot weights")
    w = {}
    for k, label in PILOT_COMPONENT_LABELS.items():
        w[k] = st.slider(label, 0.0, 0.5, float(CONFIG["PILOT_WEIGHTS"][k]), 0.05)
    tot = sum(w.values()) or 1.0
    w = {k: v / tot for k, v in w.items()}

ranked = pilot_ranking(b, weights=w, min_months=min_months, min_eligible=min_elig)
if ranked.empty:
    st.error("No branch passes the gate. Loosen the minimum months or eligible customers.")
    st.stop()

top = ranked.iloc[0]
second = ranked.iloc[1] if len(ranked) > 1 else None

st.success(
    f"**Recommended pilot branch: {top['branch']}**  -  pilot score "
    f"{top['pilot_score']:.1f}/100, {int(top['eligible']):,} eligible customers "
    f"({top['eligibility_rate_pct']:.1f}% of its own customer base), median monthly "
    f"spend {top['elig_median_monthly_spend']:.0f}."
)
if second is not None:
    st.info(f"**Fallback: {second['branch']}** - pilot score {second['pilot_score']:.1f}, "
            f"{int(second['eligible']):,} eligible customers "
            f"({second['eligibility_rate_pct']:.1f}%).")

kpi_row([
    ("Branches qualified", f"{len(ranked):,}", f"of {len(b):,}"),
    ("Eligible in top branch", f"{int(top['eligible']):,}", None),
    ("Eligible in top 3", f"{int(ranked.head(3)['eligible'].sum()):,}", None),
    ("Top-branch loyalty", f"{top['primary_customer_ratio_pct']:.0f}%", None),
    ("Top-branch volatility", f"{top['tx_cv_recent']:.2f}", "lower is steadier"),
])

# --------------------------------------------------------------- ranking
st.subheader("Ranked branches")
cols = ["rank", "branch", "pilot_score", "eligible", "eligibility_rate_pct",
        "primary_customers", "n_frequent", "n_regular", "elig_median_score",
        "elig_median_monthly_spend", "primary_customer_ratio_pct", "tx_cv_recent",
        "tx_growth_pct", "net_sales", "return_rate_pct"]
cols = [c for c in cols if c in ranked.columns]
st.dataframe(ranked[cols].style.format({
    "pilot_score": "{:.1f}", "eligible": "{:,.0f}", "eligibility_rate_pct": "{:.1f}%",
    "primary_customers": "{:,.0f}", "n_frequent": "{:,.0f}", "n_regular": "{:,.0f}",
    "elig_median_score": "{:.1f}", "elig_median_monthly_spend": "{:.2f}",
    "primary_customer_ratio_pct": "{:.1f}%", "tx_cv_recent": "{:.2f}",
    "tx_growth_pct": "{:.1f}%", "net_sales": "{:,.0f}", "return_rate_pct": "{:.2f}%",
}).background_gradient(subset=["pilot_score"], cmap="Greens"),
    use_container_width=True, height=420)

c1, c2 = st.columns([2, 1])
with c1:
    d = ranked.head(15).sort_values("pilot_score")
    fig = px.bar(d, x="pilot_score", y="branch", orientation="h",
                 title="Top 15 branches by pilot suitability", text="eligible")
    fig.update_layout(xaxis_title="Pilot suitability score", yaxis_title="Branch",
                      yaxis_type="category")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    comp_cols = [f"c_{k}" for k in CONFIG["PILOT_WEIGHTS"]]
    r = ranked.iloc[0]
    fig = go.Figure()
    for i in range(min(3, len(ranked))):
        rr = ranked.iloc[i]
        fig.add_trace(go.Scatterpolar(
            r=[rr[c] for c in comp_cols] + [rr[comp_cols[0]]],
            theta=[PILOT_COMPONENT_LABELS[k] for k in CONFIG["PILOT_WEIGHTS"]]
                  + [PILOT_COMPONENT_LABELS[list(CONFIG["PILOT_WEIGHTS"])[0]]],
            fill="toself", name=str(rr["branch"])))
    fig.update_layout(title="Component profile of the top 3",
                      polar=dict(radialaxis=dict(range=[0, 100])))
    st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Why {top['branch']} ranks first")
comp = pd.DataFrame({
    "component": [PILOT_COMPONENT_LABELS[k] for k in CONFIG["PILOT_WEIGHTS"]],
    "percentile": [top[f"c_{k}"] for k in CONFIG["PILOT_WEIGHTS"]],
    "weight_%": [100 * w[k] for k in CONFIG["PILOT_WEIGHTS"]],
})
comp["points"] = comp["percentile"] * comp["weight_%"] / 100
fig = px.bar(comp.sort_values("points"), x="points", y="component", orientation="h",
             color="percentile", color_continuous_scale="Greens",
             title="Points contributed by each component")
fig.update_layout(xaxis_title="Points (of 100)", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

strong = comp.sort_values("percentile", ascending=False).head(2)["component"].tolist()
weak = comp.sort_values("percentile").head(1)["component"].tolist()
st.markdown(f"""
**Advantages** - strongest on {', '.join(strong).lower()}.

**Watch-outs** - weakest on {weak[0].lower()}; the pilot design should compensate
(for example by sizing loans to observed monthly spend rather than to basket value).

**Potential sample** - {int(top['eligible']):,} eligible customers today; at a 20-30%
take-up that is roughly {int(0.2*top['eligible']):,}-{int(0.3*top['eligible']):,} loans,
enough to measure repayment behaviour within two quarters.
""")

# ------------------------------------------------------------- sensitivity
st.subheader("Is this recommendation an artefact of the weights?")
n_draws = st.select_slider("Random weightings to test", [500, 1000, 2000, 5000], 2000)
sens = pilot_sensitivity(ranked, n_draws=n_draws)
c1, c2 = st.columns([1, 1])
with c1:
    d = sens.head(12).sort_values("pct_in_top5")
    fig = px.bar(d, x="pct_in_top5", y="branch", orientation="h",
                 title=f"% of {n_draws} random weightings placing the branch in the top 5")
    fig.update_layout(xaxis_title="% of weightings", yaxis_title="Branch",
                      yaxis_type="category")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.dataframe(sens.head(12).style.format({
        "pct_in_top5": "{:.1f}%", "pct_ranked_first": "{:.1f}%"}),
        use_container_width=True, height=430)
st.caption("Weights are drawn around the configured values. A branch that stays in the "
           "top 5 under most weightings is a robust choice; one that appears only under "
           "a narrow weighting is not.")

# ----------------------------------------------------------- avoid / scale
st.subheader("Branches to avoid for now")
avoid = ranked.tail(8).sort_values("pilot_score")
st.dataframe(avoid[["rank", "branch", "pilot_score", "eligible", "eligibility_rate_pct",
                    "primary_customer_ratio_pct", "tx_cv_recent", "net_sales"]].style.format({
    "pilot_score": "{:.1f}", "eligible": "{:,.0f}", "eligibility_rate_pct": "{:.1f}%",
    "primary_customer_ratio_pct": "{:.1f}%", "tx_cv_recent": "{:.2f}",
    "net_sales": "{:,.0f}"}), use_container_width=True)

big = b.sort_values("net_sales", ascending=False).head(10)[["branch", "net_sales", "eligible",
                                                            "eligibility_rate_pct"]]
big = big.merge(ranked[["branch", "rank", "pilot_score"]], on="branch", how="left")
st.markdown("**Largest branches by revenue and where they actually rank** - "
            "revenue leadership does not carry over to lending suitability.")
st.dataframe(big.style.format({"net_sales": "{:,.0f}", "eligible": "{:,.0f}",
                               "eligibility_rate_pct": "{:.1f}%", "pilot_score": "{:.1f}"}),
             use_container_width=True)
