"""
Pilot Credit Simulation - standalone executive app.

One page, one password, no database. Reads a ~1 MB Parquet extract built by
`scripts/06_build_executive_extract.py`, so it starts in about a second and
carries no customer identifiers.

Run locally:   streamlit run executive_app.py
Deploy:        see DEPLOY.md
"""

import hmac
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "dashboard"))

import pandas as pd
import streamlit as st

from analytics import credit_sim_ui as ui

DATA_DIR = ROOT / "dashboard" / "executive_data"

st.set_page_config(page_title="Pilot Credit Simulation", page_icon="💊", layout="wide")


# ===========================================================================
# ACCESS
# ===========================================================================

def check_password() -> bool:
    """Shared-password gate. Returns True once the visitor is through."""
    if st.session_state.get("auth_ok"):
        return True

    expected = None
    try:
        expected = st.secrets["app_password"]
    except Exception:
        pass

    if not expected:
        st.error("No password is configured for this deployment.")
        st.caption("Set `app_password` in the app's secrets (see DEPLOY.md). "
                   "Until then the app stays locked.")
        return False

    st.title("Pilot Credit Simulation")
    st.caption("Enter the access password to open the model.")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Open") and entered:
            if hmac.compare_digest(entered, str(expected)):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("That password is not correct.")
    return False


if not check_password():
    st.stop()


# ===========================================================================
# DATA
# ===========================================================================

@st.cache_data(show_spinner="Loading the pilot data...")
def load_extract() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    customers = pd.read_parquet(DATA_DIR / "customers.parquet")
    branches = pd.read_parquet(DATA_DIR / "branches.parquet")
    meta_path = DATA_DIR / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return customers, branches, meta


if not (DATA_DIR / "customers.parquet").exists():
    st.error("The pilot data extract is missing.")
    st.caption("Build it with `python scripts/06_build_executive_extract.py`, "
               "then commit `dashboard/executive_data/`.")
    st.stop()

pool, branches, meta = load_extract()

st.title("Pilot Credit Simulation")
st.markdown(
    f"A simulation of the credit pilot across the **{len(branches)} selected branches** and "
    f"their **{len(pool):,} eligible customers**. Risk grade and credit limit are derived "
    "from observed purchasing behaviour. Take-up, package choice, default and recovery are "
    "**assumptions** - the pharmacy has never lent to these customers, so no historical data "
    "supports them. Change anything in the sidebar and every figure recalculates."
)
if meta.get("as_of"):
    st.caption(f"Customer behaviour measured to **{meta['as_of']}** · extract built "
               f"{meta.get('built', 'n/a')} · no customer identifiers are held in this app.")

ui.render(pool, branches, ui.simulation_controls())
