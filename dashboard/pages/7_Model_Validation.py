"""Model Validation - the checks that make the model auditable."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.dashboard_ui import page_setup, sidebar
from analytics.loan_analytics import CONFIG, validation_checks

page_setup("Model Validation")
f, ff, state = sidebar()

st.subheader("Structural checks (recomputed live)")
checks = validation_checks(f)
st.dataframe(checks.style.map(
    lambda v: "background-color:#e6f4ea" if v == "PASS" else
              ("background-color:#fce8e6" if v == "FAIL" else ""),
    subset=["status"]), use_container_width=True, hide_index=True)
if (checks["status"] == "FAIL").any():
    st.error("At least one structural check failed - fix before using the output.")

st.subheader("Out-of-time evidence")
st.markdown("""
The model was fitted on nothing: thresholds and weights were chosen from observed
distributions, then tested against a period the model never saw. Segments and scores
were built on data up to **2025-12** and compared with actual behaviour over
**2026-01 to 2026-06**. Re-run `validate_model.py` after any data refresh to
reproduce the tables below.
""")

seg_evidence = pd.DataFrame({
    "segment": ["Frequent", "Regular", "Semi-Regular", "Sporadic", "Dormant"],
    "customers_at_cutoff": [32394, 90327, 99597, 146140, 204523],
    "active_next_6m_%": [98.8, 96.9, 86.7, 58.9, 20.3],
    "mean_future_active_months": [5.46, 4.50, 3.00, 1.44, 0.36],
    "persisted_5_of_6_%": [87.2, 60.7, 26.8, 7.0, 0.8],
    "median_spend_next_6m": [443.0, 186.1, 85.8, 23.9, 0.0],
})
st.dataframe(seg_evidence, use_container_width=True, hide_index=True)
fig = px.bar(seg_evidence, x="segment", y="persisted_5_of_6_%",
             title="Future persistence by segment (holdout)")
fig.update_layout(xaxis_title="", yaxis_title="% active in 5+ of the next 6 months")
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Score bands against future behaviour (holdout gains table)**")
gains = pd.DataFrame({
    "score_band": ["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-100"],
    "customers": [222090, 39981, 57103, 57508, 52412, 47237, 41410, 55240],
    "persisted_5_of_6_%": [0.85, 2.79, 5.94, 13.37, 26.40, 44.49, 63.52, 83.69],
    "mean_future_active_months": [0.38, 0.97, 1.46, 2.24, 3.07, 3.90, 4.60, 5.33],
})
c1, c2 = st.columns([1, 1])
with c1:
    st.dataframe(gains, use_container_width=True, hide_index=True)
with c2:
    fig = px.line(gains, x="score_band", y="persisted_5_of_6_%", markers=True,
                  title="Monotonic lift across score bands")
    fig.add_hline(y=21.2, line_dash="dash", annotation_text="base rate 21.2%")
    fig.update_layout(xaxis_title="Score band", yaxis_title="% persisted")
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
**Discrimination.** AUC for predicting "active in ≥5 of the next 6 months":
0.918 for the configured score, 0.9187 for a logistic regression fitted on the same
components — the fixed, explainable weights lose essentially nothing.

**Window choice.** 6 months beat 3, 12 and 18 months (AUC 0.912 / 0.909 / 0.901 / 0.884
on transaction count), and {CONFIG['WINDOW_MONTHS']} months of history are available
for 84.6% of customers.

**Cut-off choice.** At score ≥ {CONFIG['ELIGIBILITY_MIN_SCORE']}, 70.2% of customers
persisted against a 21.2% base rate. Raising the cut to 80 lifts that to 83.7% but
removes more than half the population.

**Known limitations.**
- Purchase persistence is a proxy for repayment behaviour, not a substitute. It measures
  whether a customer keeps a stable relationship, not whether they can repay.
- Primary branch is assigned over the whole history; a customer who recently switched
  pharmacies is still counted at the old branch.
- Pension flag comes from `customer_segments.customer_type` and reflects discount usage,
  not verified income.
- Branches that opened mid-period are excluded from the pilot ranking rather than
  penalised, because their history cannot support a stability estimate.
""")
