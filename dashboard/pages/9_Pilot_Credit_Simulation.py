"""
Pilot Credit Simulation - the analyst view, driven by the live model.

The interface itself lives in analytics/credit_sim_ui.py and is shared with
executive_app.py, which renders the identical page from a pre-built extract.
Only the data source differs: this page recomputes the model from DuckDB at
the selected as-of month.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from analytics import credit_sim_ui as ui
from analytics.dashboard_ui import page_setup, sidebar
from analytics.loan_analytics import branch_summary, pilot_ranking, run_query

SELECTION_SIZE = 100

page_setup("Pilot Credit Simulation")
f, _ff, state = sidebar()


@st.cache_data(show_spinner=False)
def pilot_branches(_features: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """The 100 selected branches, with their reference geography."""
    ranked = pilot_ranking(branch_summary(_features, as_of))
    top = ranked[ranked["rank"] <= SELECTION_SIZE][["rank", "branch", "pilot_score"]]
    try:
        ref = run_query("SELECT branch_key, branch_name, chain, town, geo_category "
                        "FROM branch_reference")
        top = top.merge(ref, left_on="branch", right_on="branch_key", how="left")
    except Exception:
        top["branch_name"] = top["branch"]
        top["chain"] = top["town"] = top["geo_category"] = "Not mapped"
    return top


branches = pilot_branches(f, state["as_of"])
pool = f[f["eligible"] & f["primary_branch"].isin(set(branches["branch"]))]

st.markdown(
    f"A simulation of the credit pilot across the **{len(branches)} selected branches** and "
    f"their **{len(pool):,} eligible customers**. Risk grade and credit limit are derived "
    "from observed purchasing behaviour. Take-up, package choice, default and recovery are "
    "**assumptions** - we have never lent to these customers, so no historical data supports "
    "them. Change them in the sidebar and every figure on this page recalculates."
)
st.caption("Branch figures are always computed on the full customer base. The customer "
           "filters in the sidebar shape the other pages, not this one; use *Filter* below "
           "the branch table to narrow the branch list.")

ui.render(pool[ui.CUSTOMER_COLUMNS], branches, ui.simulation_controls())
