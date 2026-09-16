"""
Final 100 Pilot Selection - which branches were chosen, and why.

The selection is the top N of the existing validated behavioural ranking
(`pilot_ranking`, unchanged). Geography and cash/POS come from the
`branch_reference` table and are CONTEXT ONLY - they never enter the score.
Build that table first with `python scripts/05_build_branch_reference.py`.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.dashboard_ui import kpi_row, page_setup, sidebar
from analytics.loan_analytics import branch_summary, pilot_ranking, run_query

SELECTION_SIZE = 100
CATEGORY_ORDER = ["Region", "Baku Peripheral", "Baku Central"]
CATEGORY_COLORS = {"Region": "#1b7f4b", "Baku Peripheral": "#4c9f70", "Baku Central": "#f2b134"}

page_setup("Final 100 Pilot Selection")
f, _ff, state = sidebar()


@st.cache_data(show_spinner=False)
def load_reference() -> pd.DataFrame:
    return run_query("SELECT * FROM branch_reference")


def med(series: pd.Series, fmt: str = "{:,.0f}", empty: str = "n/a") -> str:
    """Median as display text. `empty` covers the case where nothing was left out."""
    if len(series) == 0 or pd.isna(series.median()):
        return empty
    return fmt.format(series.median())


def versus(series: pd.Series, fmt: str = "{:,.0f}") -> str | None:
    """Delta caption against the branches that missed the cut; None when none did."""
    text = med(series, fmt, empty="")
    return f"vs {text} not selected" if text else None


try:
    ref = load_reference()
except Exception:
    st.error("`branch_reference` is missing. Build it with:\n\n"
             "`python scripts/05_build_branch_reference.py`")
    st.stop()

b = branch_summary(f, state["as_of"])
ranked = pilot_ranking(b)                      # default CONFIG weights - unchanged
if ranked.empty:
    st.error("No branch passes the pilot gate at this as-of month.")
    st.stop()

with st.sidebar:
    st.header("Selection")
    n_selected = st.number_input("Selection size", 10, int(len(ranked)), SELECTION_SIZE, 10)
    st.caption(f"The approved decision is {SELECTION_SIZE} branches. Change this only to "
               "test what a different cut would have captured.")

r = ranked.merge(ref.drop(columns=["snapshot_rank", "snapshot_pilot_score"]),
                 left_on="branch", right_on="branch_key", how="left")
r["geo_category"] = r["geo_category"].fillna("Not mapped")
r["town"] = r["town"].fillna("Not mapped")
r["chain"] = r["chain"].fillna("Unknown")
r["cash_status"] = r["cash_status"].fillna("No cash data")

sel = r[r["rank"] <= n_selected].copy()
rest = r[r["rank"] > n_selected].copy()
sel_ok = sel[sel["cash_status"] == "OK"]

st.markdown(
    f"The **{n_selected} branches below are the top {n_selected} of the validated "
    f"behavioural ranking** of {len(r):,} qualified branches - the same model, weights and "
    "gates as before. Geography and cash/POS are shown alongside to explain what kind of "
    "markets the pilot will run in; **they did not influence the selection.**"
)
st.caption("Branch figures are always computed on the full customer base. The customer "
           "filters in the sidebar shape the other pages, not this one; use *Filter the "
           "view* below the table to narrow the branch list.")


# =====================================================================
# 1. EXECUTIVE KPIs
# =====================================================================

total_eligible = r["eligible"].sum()
captured = sel["eligible"].sum()

kpi_row([
    ("Branches selected", f"{len(sel)}", f"of {len(r):,} qualified"),
    ("Eligible customers", f"{int(captured):,}", f"{100*captured/total_eligible:.1f}% of all qualified"),
    ("Median pilot score", f"{sel['pilot_score'].median():.1f}",
     versus(rest["pilot_score"], "{:.1f}")),
    ("Median eligibility rate", f"{sel['eligibility_rate_pct'].median():.1f}%",
     versus(rest["eligibility_rate_pct"], "{:.1f}%")),
    ("Median eligible per branch", f"{sel['eligible'].median():,.0f}",
     versus(rest["eligible"], "{:,.0f}")),
])

counts = sel["geo_category"].value_counts()
kpi_row([
    ("Region", f"{counts.get('Region', 0)}",
     f"{100*counts.get('Region', 0)/len(sel):.0f}% of selection"),
    ("Baku Peripheral", f"{counts.get('Baku Peripheral', 0)}",
     f"{100*counts.get('Baku Peripheral', 0)/len(sel):.0f}% of selection"),
    ("Baku Central", f"{counts.get('Baku Central', 0)}",
     f"{100*counts.get('Baku Central', 0)/len(sel):.0f}% of selection"),
    ("Towns / territories", f"{sel['town'].nunique()} / {sel['territory'].nunique()}",
     f"{sel['chain'].nunique()} chains"),
    ("Primary customer base", f"{int(sel['primary_customers'].sum()):,}", "addressable footfall"),
])

flagged = int((sel["cash_status"] != "OK").sum())
rest_ok = rest[rest["cash_status"] == "OK"]
kpi_row([
    ("Median cash share", f"{sel_ok['cash_share_pct'].median():.1f}%",
     versus(rest_ok["cash_share_pct"], "{:.1f}%")),
    ("Monthly cash turnover", f"{sel_ok['avg_monthly_cash'].sum():,.0f}", "AZN / month"),
    ("Monthly total turnover", f"{sel_ok['avg_monthly_total'].sum():,.0f}", "AZN / month"),
    ("Net sales in selection", f"{sel['net_sales'].sum():,.0f}", "AZN over the window"),
    ("Cash data flagged", f"{flagged}", "shown, never counted as zero"),
])

st.caption("Cash and POS figures are **sales value in AZN, not customer counts**, and exclude "
           "the 2025-01 and 2025-02 reporting gaps. They describe the market a branch sits in; "
           "they are not evidence about repayment.")


# =====================================================================
# 2. GEOGRAPHIC COMPOSITION
# =====================================================================

st.subheader("Where the pilot will run")

c1, c2 = st.columns([1, 1])

with c1:
    comp = (r.assign(status=np.where(r["rank"] <= n_selected, "Selected", "Not selected"))
             .groupby(["geo_category", "status"]).size().reset_index(name="branches"))
    fig = px.bar(comp, x="branches", y="geo_category", color="status", orientation="h",
                 category_orders={"geo_category": CATEGORY_ORDER},
                 color_discrete_map={"Selected": "#1b7f4b", "Not selected": "#d7d9dc"},
                 title="Selected branches by geographic category")
    fig.update_layout(xaxis_title="Branches", yaxis_title="", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    rate = (r.groupby("geo_category")
             .agg(qualified=("branch", "size"),
                  selected=("rank", lambda s: int((s <= n_selected).sum()))))
    rate["selection_rate_%"] = (100 * rate["selected"] / rate["qualified"]).round(1)
    st.dataframe(rate.reindex([c for c in CATEGORY_ORDER if c in rate.index]),
                 use_container_width=True)

with c2:
    cum = r.sort_values("rank").copy()
    cum["cumulative_eligible"] = cum["eligible"].cumsum()
    cum["pct_of_eligible"] = 100 * cum["cumulative_eligible"] / total_eligible
    fig = px.line(cum, x="rank", y="pct_of_eligible",
                  title="Eligible customers captured as the cut deepens")
    fig.add_vline(x=n_selected, line_dash="dash", line_color="#1b7f4b")
    fig.add_hline(y=100 * captured / total_eligible, line_dash="dot", line_color="#1b7f4b")
    fig.update_layout(xaxis_title="Branches included (by pilot rank)",
                      yaxis_title="% of all qualified eligible customers")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"The top {n_selected} branches hold {100*captured/total_eligible:.0f}% of every "
               "eligible customer in the qualified network. The curve flattens because branches "
               "further down the ranking are both smaller and weaker.")

if len(sel_ok):
    fig = px.scatter(sel_ok, x="cash_concentration_pct", y="cash_opportunity_pct",
                     color="geo_category", size="eligible", hover_name="branch",
                     hover_data={"town": True, "pilot_score": ":.1f",
                                 "cash_share_pct": ":.1f", "avg_monthly_cash": ":,.0f"},
                     category_orders={"geo_category": CATEGORY_ORDER},
                     color_discrete_map=CATEGORY_COLORS,
                     title="Cash market profile of the selected branches "
                           "(percentiles within Baku / region)")
    fig.add_vline(x=50, line_dash="dot", line_color="#9aa0a6")
    fig.add_hline(y=50, line_dash="dot", line_color="#9aa0a6")
    fig.update_layout(xaxis_title="Cash concentration - share of sales taken in cash",
                      yaxis_title="Cash opportunity - size of the cash business",
                      legend_title="")
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# 3. THE SELECTED BRANCHES
# =====================================================================

st.subheader(f"The {n_selected} selected branches")

with st.expander("Filter the view", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        pick_cat = st.multiselect("Geographic category",
                                  [c for c in CATEGORY_ORDER if c in set(sel["geo_category"])])
        pick_chain = st.multiselect("Chain", sorted(sel["chain"].unique()))
    with c2:
        pick_town = st.multiselect("Town / settlement", sorted(sel["town"].unique()))
        rank_range = st.slider("Pilot rank", 1, int(n_selected), (1, int(n_selected)))
    with c3:
        rate_range = st.slider("Eligibility rate %", 0.0, 100.0, (0.0, 100.0), 1.0)
        cash_range = st.slider("Cash share %", 0.0, 100.0, (0.0, 100.0), 1.0)
        include_flagged = st.checkbox("Include branches without usable cash data", True)
    show_all = st.checkbox("Show every column", False)

view = sel[sel["rank"].between(*rank_range)
           & sel["eligibility_rate_pct"].between(*rate_range)].copy()
if pick_cat:
    view = view[view["geo_category"].isin(pick_cat)]
if pick_chain:
    view = view[view["chain"].isin(pick_chain)]
if pick_town:
    view = view[view["town"].isin(pick_town)]
if include_flagged:
    view = view[view["cash_share_pct"].between(*cash_range) | view["cash_share_pct"].isna()]
else:
    view = view[view["cash_share_pct"].between(*cash_range)]

CORE_COLUMNS = ["rank", "branch", "branch_name", "chain", "town", "geo_category",
                "pilot_score", "eligible", "eligibility_rate_pct", "net_sales",
                "avg_monthly_cash", "cash_share_pct", "cash_status"]
EXTRA_COLUMNS = ["territory", "primary_customers", "elig_median_score",
                 "elig_median_monthly_spend", "avg_monthly_pos", "avg_monthly_total",
                 "pos_share_pct", "cash_opportunity_pct", "tx_cv_recent"]

columns = CORE_COLUMNS + (EXTRA_COLUMNS if show_all else [])
columns = [c for c in columns if c in view.columns]

st.dataframe(view[columns].style.format({
    "pilot_score": "{:.1f}", "eligible": "{:,.0f}", "eligibility_rate_pct": "{:.1f}%",
    "net_sales": "{:,.0f}", "avg_monthly_cash": "{:,.0f}", "avg_monthly_pos": "{:,.0f}",
    "avg_monthly_total": "{:,.0f}", "cash_share_pct": "{:.1f}%", "pos_share_pct": "{:.1f}%",
    "primary_customers": "{:,.0f}", "elig_median_score": "{:.1f}",
    "elig_median_monthly_spend": "{:.2f}", "cash_opportunity_pct": "{:.0f}",
    "tx_cv_recent": "{:.2f}",
}, na_rep="-").background_gradient(subset=["pilot_score"], cmap="Greens"),
    use_container_width=True, height=520)

st.caption(f"Showing {len(view)} of {len(sel)} selected branches. "
           "**Pilot rank, score, eligible customers and eligibility rate come from the "
           "validated model. Cash columns are descriptive context only.**")

st.download_button("Download the selection as CSV",
                   sel[[c for c in CORE_COLUMNS + EXTRA_COLUMNS if c in sel.columns]]
                   .to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"pilot_selection_top{n_selected}.csv", mime="text/csv")


# =====================================================================
# 4. DECISION RATIONALE
# =====================================================================

st.subheader(f"Why these {n_selected} branches")

cash_by_cat = sel_ok.groupby("geo_category")

clusters = (sel.groupby(["geo_category", "town"])
              .agg(branches=("branch", "size"), best_rank=("rank", "min"),
                   median_score=("pilot_score", "median"), eligible=("eligible", "sum"))
              .reset_index().sort_values("branches", ascending=False))
top_clusters = clusters[clusters["branches"] >= 3].head(6)

cut_score = sel["pilot_score"].min()
next_score = f"{rest['pilot_score'].max():.1f}" if len(rest) else "n/a - every qualified branch is in"

t20 = r[r["rank"] <= 20]["eligible"].sum()
t50 = r[r["rank"] <= 50]["eligible"].sum()

st.markdown(f"""
**1. The ranking is the foundation.** These branches are ranks 1-{n_selected} of the
behavioural branch model, which scores customer quality, eligibility, loyalty, stability
and scale - each converted to a percentile inside the qualified set. Nothing on this page
changed that model. The cut is a deliberate balance between branch quality, addressable
customer volume and having enough branches to run a readable pilot.

**2. What distinguishes them.** The selected branches carry a median pilot score of
**{sel['pilot_score'].median():.1f}** against **{med(rest['pilot_score'], '{:.1f}')}** for the
qualified branches left out, a median eligibility rate of
**{sel['eligibility_rate_pct'].median():.1f}%** against
**{med(rest['eligibility_rate_pct'], '{:.1f}%')}**, and **{sel['eligible'].median():,.0f}**
eligible customers per branch against **{med(rest['eligible'])}**. Median net sales
run **{sel['net_sales'].median():,.0f}** against **{med(rest['net_sales'])} AZN**.
Median cash share, by contrast, is **{sel_ok['cash_share_pct'].median():.1f}%** against
**{med(rest_ok['cash_share_pct'], '{:.1f}%')}** - practically identical,
which is the clearest evidence that cash data played no part in the selection.

**3. Geographic composition.** The selection is
**{counts.get('Region', 0)} Region**, **{counts.get('Baku Peripheral', 0)} Baku Peripheral**
and **{counts.get('Baku Central', 0)} Baku Central** branches, spread over
{sel['town'].nunique()} towns and settlements in {sel['territory'].nunique()} territories.
It is not a Baku-concentrated pilot: regional branches are the largest single block. The
model naturally selects
**{100*counts.get('Baku Peripheral', 0)/max(len(r[r['geo_category']=='Baku Peripheral']),1):.0f}%**
of peripheral Baku branches but only
**{100*counts.get('Baku Central', 0)/max(len(r[r['geo_category']=='Baku Central']),1):.0f}%**
of central Baku ones - the settlement belt simply holds better customers for this product.
That spread also lets the pilot test three genuinely different market types rather than one.
""")

if len(top_clusters):
    st.markdown("**Clusters worth running together** - several strong branches in one place, "
                "which cuts supervision cost and gives a cleaner read than scattered sites:")
    st.dataframe(top_clusters.style.format({"median_score": "{:.1f}", "eligible": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)

cash_lines = []
for cat in CATEGORY_ORDER:
    if cat in cash_by_cat.groups:
        g = cash_by_cat.get_group(cat)
        cash_lines.append(
            f"- **{cat}** - median cash share **{g['cash_share_pct'].median():.1f}%**, "
            f"{g['avg_monthly_cash'].median():,.0f} AZN cash and "
            f"{g['avg_monthly_pos'].median():,.0f} AZN card per month, "
            f"median net sales {g['net_sales'].median():,.0f} AZN.")

st.markdown("**4. Cash versus card profile.** Cash data is a descriptive market signal here, "
            "not a scoring component and not evidence about repayment:\n\n"
            + "\n".join(cash_lines))

low_cash_high_rank = sel_ok[(sel_ok["rank"] <= 25)].nsmallest(3, "cash_share_pct")
high_cash_low_rank = sel_ok[sel_ok["rank"] > n_selected * 0.6].nlargest(3, "cash_share_pct")

st.markdown(f"""
The three categories separate cleanly: regional branches are the most cash-oriented but the
smallest by turnover, central Baku is the least cash-oriented but the largest, and peripheral
Baku sits in between on both. Two kinds of exception are worth naming for the committee.
Strongly ranked branches with a low cash share -
{", ".join(f"**{row.branch}** ({row.town}, rank {int(row['rank'])}, {row.cash_share_pct:.0f}% cash)"
           for _, row in low_cash_high_rank.iterrows())} -
are in the pilot on behavioural merit and should stay there. Lower-ranked branches with a
strong cash market -
{", ".join(f"**{row.branch}** ({row.town}, rank {int(row['rank'])}, {row.cash_share_pct:.0f}% cash)"
           for _, row in high_cash_low_rank.iterrows())} -
are worth watching as a secondary hypothesis, not as a reason to promote them.

**5. Why {n_selected} and not 20 or all {len(r)}.** The top 20 branches hold only
**{100*t20/total_eligible:.0f}%** of eligible customers and the top 50 **{100*t50/total_eligible:.0f}%** -
too few to measure repayment behaviour across different market types within a couple of
quarters. {n_selected} branches capture **{100*captured/total_eligible:.0f}%** of the entire
eligible base while keeping the median score at **{sel['pilot_score'].median():.1f}**. Going
deeper adds customers at a falling quality: the pilot score at rank {n_selected} is
**{cut_score:.1f}** and the next branch scores **{next_score}**. Note that there is no
natural break in the data at {n_selected} - the ranking is smooth there, so this is an
operational judgement about capacity and quality control, not a threshold the model found.
""")

st.markdown(f"""
**6. Key findings for the committee.**
- Regional branches are the largest block of the pilot ({counts.get('Region', 0)} of
  {len(sel)}), so this is a national test, not a Baku test.
- Inside Baku, the settlement belt outperforms the centre on the behavioural model and
  simultaneously runs a higher cash share - the one place where the two signals agree.
- The eligible customer base is concentrated: {n_selected} of {len(r)} qualified branches
  hold {100*captured/total_eligible:.0f}% of all eligible customers.
- **{flagged}** selected branches have no usable cash data. They are in the pilot on
  behavioural merit; their cash profile is unknown, not zero.
- Cash share and pilot rank are close to unrelated across the selection, so cash-heavy
  markets are a segmentation lens for reading pilot results, not a selection criterion.
""")


# =====================================================================
# 5. LIMITATIONS
# =====================================================================

with st.expander("Limitations and data quality - read before acting on this page"):
    missing = sel[sel["cash_status"] != "OK"]
    st.markdown(f"""
- **Cash and POS are sales value, not customers.** A branch taking most of its money in cash
  is not necessarily serving mostly cash-paying customers. Customer-level payment data does
  not exist yet.
- **Cash share is not a validated risk predictor.** It has never been tested against a
  repayment outcome. It is used here only to describe the market a branch sits in.
- **Two months are excluded** from every cash figure: 2025-01, where 177 of 179 branches
  reported exactly zero cash against normal card sales, and 2025-02, which ran at roughly a
  quarter of the surrounding level. Both are reporting gaps.
- **Missing cash data is not zero cash.** {len(missing)} selected branches carry a flag rather
  than a number. They are concentrated in one territory, so their absence is systematic, not
  random - if they are outside the Payonix acquiring network they could be among the most
  cash-oriented branches we have.
- **Territory is not a town.** `territory` is a sales region: the Mingəçevir territory includes
  Ağdaş and Yevlax branches, Şəki includes Oğuz, Qax, Balakən and Zaqatala. The town column is
  parsed from the branch name and is the more accurate geography.
- **The Baku central/peripheral split is our mapping**, derived from branch names because the
  source data labels every Baku branch simply "Bakı". It should be confirmed against the
  official branch register before it drives field operations.
- **The behavioural model measures purchase persistence, not repayment.** It has never observed
  a loan outcome. That is what the pilot is for.
""")
    if len(missing):
        st.dataframe(missing[["rank", "branch", "branch_name", "territory", "town",
                              "eligible", "cash_status"]]
                     .style.format({"eligible": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)
