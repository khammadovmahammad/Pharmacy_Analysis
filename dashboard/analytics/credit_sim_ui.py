"""
Shared interface for the pilot credit simulation.

Used by two entry points that differ only in where the customer data comes
from, so the interface itself exists once:

    dashboard/pages/9_Pilot_Credit_Simulation.py   live model over DuckDB
    executive_app.py                               pre-built Parquet extract

Imports deliberately stop at streamlit / pandas / numpy / plotly / credit_sim.
Nothing here may import loan_analytics or dashboard_ui, because those pull in
duckdb and the executive app ships without a database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import credit_sim as cs

CUSTOMER_COLUMNS = ["primary_branch", "segment", "score", "monthly_spend",
                    "avg_basket", "n_tx", "tenure_months"]
BRANCH_COLUMNS = ["branch", "rank", "branch_name", "chain", "town", "geo_category"]


def kpi_row(items: list[tuple[str, str, str | None]]):
    for col, (label, value, delta) in zip(st.columns(len(items)), items):
        col.metric(label, value, delta)


def integer_shares(weights: dict, total: int) -> dict:
    """Split `total` into whole percents by weight, summing to exactly `total`."""
    raw = {k: w * total for k, w in weights.items()}
    parts = {k: int(np.floor(v)) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: raw[k] - parts[k],
                      reverse=True)[:total - sum(parts.values())]:
        parts[key] += 1
    return parts


# ===========================================================================
# CONTROLS
# ===========================================================================

def simulation_controls() -> dict:
    """
    Render every simulation control in the sidebar and return the config.

    Widget keys carry the scenario name, so switching preset re-initialises
    each control to that scenario's value. Stops the script if a risk grade's
    outcome mix does not total 100%.
    """
    with st.sidebar:
        st.header("Scenario")
        scen = st.radio("Preset", list(cs.SCENARIOS) + ["Custom"], index=1,
                        help="Switching a preset resets the controls below to that scenario.")
        preset = cs.merge_config(cs.SCENARIOS.get(scen, {}))
        k = scen

        st.header("Take-up (assumption)")
        take_up, offer_rate = {}, {}
        for g in cs.RISK_ORDER:
            c1, c2 = st.columns(2)
            take_up[g] = c1.slider(f"{g} take-up %", 0, 100,
                                   int(round(100 * preset["TAKE_UP"][g])), 1,
                                   key=f"{k}_tu_{g}") / 100
            offer_rate[g] = c2.slider(f"{g} offered %", 0, 100,
                                      int(round(100 * preset["OFFER_RATE"][g])), 5,
                                      key=f"{k}_of_{g}") / 100

        st.header("Credit limits (policy)")
        cmin, cmax = st.slider("Overall credit range (AZN)", 50, 500,
                               (int(preset["CREDIT_MIN"]), int(preset["CREDIT_MAX"])), 10,
                               key=f"{k}_range")
        caps = {}
        for g in cs.RISK_ORDER:
            lo_d, hi_d = preset["RISK_CAPS"][g]
            if g == "High":
                v = st.slider("High-risk limit (AZN)", cmin, cmax,
                              int(min(max(hi_d, cmin), cmax)), 10, key=f"{k}_cap_{g}")
                caps[g] = (v, v)
            else:
                caps[g] = st.slider(
                    f"{g}-risk range (AZN)", cmin, cmax,
                    (int(min(max(lo_d, cmin), cmax)), int(min(max(hi_d, cmin), cmax))),
                    10, key=f"{k}_cap_{g}")
        afford = st.slider("Affordability guardrail - max x monthly spend", 1.0, 15.0,
                           float(preset["AFFORDABILITY_MULTIPLE"]), 0.5, key=f"{k}_afford",
                           help="A limit may not exceed this multiple of the customer's "
                                "observed monthly pharmacy spend. Customers whose capped "
                                "limit falls below the minimum are not offered credit.")

        st.header("Package rates (policy)")
        rates = {t: st.slider(f"{t}-month value added %", 0.0, 40.0,
                              100 * preset["PACKAGE_RATES"][t], 0.5, key=f"{k}_rate_{t}") / 100
                 for t in cs.TERMS}

        st.header("Outcome mix (assumption)")
        st.caption("Per 100 customers who take credit: how many choose each term, and how "
                   "many never repay. Each grade must total 100%.")
        term_mix, default_rate, mix_ok = {}, {}, True
        for g in cs.RISK_ORDER:
            with st.expander(f"{g} risk", expanded=(g == "Low")):
                d_def = int(round(100 * preset["DEFAULT_RATE"][g]))
                defaults = integer_shares(preset["TERM_MIX"][g], 100 - d_def)
                shares = {}
                cols = st.columns(3)
                for i, t in enumerate(cs.TERMS):
                    shares[t] = cols[i].number_input(f"{t}m %", 0, 100, defaults[t], 1,
                                                     key=f"{k}_mix_{g}_{t}")
                lost = st.number_input("Lost / never repaid %", 0, 100, d_def, 1,
                                       key=f"{k}_lost_{g}")
                total = sum(shares.values()) + lost
                if total != 100:
                    st.error(f"Totals {total}% - must be 100%")
                    mix_ok = False
                repaying = sum(shares.values())
                term_mix[g] = ({t: shares[t] / repaying for t in cs.TERMS} if repaying
                               else {1: 1.0, 2: 0.0, 3: 0.0})
                default_rate[g] = lost / 100

        st.header("Financial assumptions")
        recovery = st.slider("Recovery on defaulted principal %", 0, 100,
                             int(round(100 * preset["RECOVERY_RATE"])), 5, key=f"{k}_rec") / 100
        repeat = st.slider("Repeat borrowing % of repayers", 0, 100,
                           int(round(100 * preset["REPEAT_RATE"])), 5, key=f"{k}_rep") / 100
        reborrow_gap = st.slider("Months between repaying and borrowing again", 0, 6,
                                 int(preset["REBORROW_GAP_MONTHS"]), 1, key=f"{k}_gap",
                                 help="0 means a customer who repays this month borrows again "
                                      "the same month, so capital never idles. On a 1-month "
                                      "product this is the biggest single lever on how hard "
                                      "the capital works.")
        ramp = st.slider("Months to onboard all borrowers", 1, 12,
                         int(preset["ACQUISITION_MONTHS"]), 1, key=f"{k}_ramp")
        funding = st.slider("Funding cost % per year", 0.0, 40.0,
                            100 * preset["FUNDING_COST_ANNUAL"], 0.5, key=f"{k}_fund") / 100
        opex = st.number_input("Operating cost per loan (AZN)", 0.0, 50.0,
                               float(preset["OPEX_PER_LOAN"]), 0.5, key=f"{k}_opex")
        constrain = st.checkbox("Cap available capital", False, key=f"{k}_cc")
        capital_limit = st.number_input("Capital available (AZN)", 10_000, 20_000_000,
                                        500_000, 10_000, key=f"{k}_cl") if constrain else None

    if not mix_ok:
        st.error("Fix the outcome mix in the sidebar - each risk grade must total 100%.")
        st.stop()

    return dict(CREDIT_MIN=cmin, CREDIT_MAX=cmax, RISK_CAPS=caps,
                AFFORDABILITY_MULTIPLE=afford, PACKAGE_RATES=rates, TAKE_UP=take_up,
                OFFER_RATE=offer_rate, TERM_MIX=term_mix, DEFAULT_RATE=default_rate,
                RECOVERY_RATE=recovery, REPEAT_RATE=repeat,
                REBORROW_GAP_MONTHS=reborrow_gap, ACQUISITION_MONTHS=ramp,
                FUNDING_COST_ANNUAL=funding, OPEX_PER_LOAN=opex,
                CAPITAL_LIMIT=capital_limit)


# ===========================================================================
# RENDER
# ===========================================================================

def render(pool: pd.DataFrame, branches: pd.DataFrame, cfg: dict) -> None:
    """Score the pool, run the 12-month ledger, and draw the whole page."""
    pool = pool.copy()
    pool["risk_grade"] = cs.assign_risk_grade(pool, cfg)
    scored = cs.assign_credit_limit(pool, cfg)
    portfolio, sim, S = cs.run(scored, cfg)
    n_branches = len(branches)

    # ------------------------------------------------------------- KPIs
    counts = scored["risk_grade"].value_counts()
    kpi_row([
        ("Pilot branches", f"{n_branches}", None),
        ("Eligible customers", f"{len(scored):,}",
         f"{int(scored['offerable'].sum()):,} offerable"),
        ("Low risk", f"{int(counts.get('Low', 0)):,}",
         f"{100*counts.get('Low', 0)/len(scored):.0f}% of pool"),
        ("Medium risk", f"{int(counts.get('Medium', 0)):,}",
         f"{100*counts.get('Medium', 0)/len(scored):.0f}% of pool"),
        ("High risk", f"{int(counts.get('High', 0)):,}",
         f"{100*counts.get('High', 0)/len(scored):.0f}% of pool"),
    ])
    kpi_row([
        ("Expected borrowers", f"{S['borrowers']:,.0f}",
         f"{S['loans']:,.0f} loans in 12 months"),
        ("Average loan", f"{S['avg_loan']:,.0f} AZN", None),
        ("Initial capital required", f"{S['month1_capital']:,.0f}",
         "AZN to fund the first month"),
    ])
    kpi_row([
        ("Interest revenue", f"{S['interest']:,.0f}", "AZN over 12 months, value added only"),
        ("Credit losses", f"{S['losses']:,.0f}", f"{100*S['loss_rate']:.1f}% of disbursement"),
        ("Net credit contribution", f"{S['net_contribution']:,.0f}", "interest - losses"),
        ("Net profit", f"{S['net_profit']:,.0f}", "after funding and operating costs"),
    ])
    if S["unfunded_demand"] > 0:
        st.warning(f"Capital cap is binding: **{S['unfunded_demand']:,.0f} AZN** of demand "
                   "could not be funded. Raise the capital cap or lower take-up.")
    st.caption(f"**Initial capital** is the cash needed on day one to fund the first month's "
               f"cohort. Sustaining the pilot needs more as lending ramps up: the deepest "
               f"funding point over the year is **{S['required_capital']:,.0f} AZN**, from "
               f"which {S['disbursed']:,.0f} AZN is lent in total as repaid loans are lent "
               f"again ({S['recycled']:.1f}x recycling). Principal repayment is not revenue "
               "and is excluded from every revenue figure above.")

    # ----------------------------------------------------------- funnel
    st.subheader("From eligible customer to borrower")
    c1, c2 = st.columns([1, 1])
    with c1:
        repaid = sim["borrowers_repaid"].sum()
        steps = pd.DataFrame({
            "stage": ["Eligible in pilot branches", "Passing the affordability floor",
                      "Offered credit", "Take credit (first loan)", "Loans over 12 months",
                      "Loans repaid", "Loans lost"],
            "customers": [len(scored), int(scored["offerable"].sum()),
                          portfolio["offered"].sum(), portfolio["takers"].sum(),
                          S["loans"], repaid, S["loans"] - repaid],
        })
        fig = px.funnel(steps, x="customers", y="stage", title="Customer funnel")
        fig.update_layout(yaxis_title="", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pf = portfolio.reset_index()
        pf["avg_limit"] = pf["avg_limit"].round(0)
        st.dataframe(pf[["risk_grade", "eligible", "offerable", "offered", "takers",
                         "avg_limit", "exposure_if_all_drew"]].style.format({
            "eligible": "{:,.0f}", "offerable": "{:,.0f}", "offered": "{:,.0f}",
            "takers": "{:,.0f}", "avg_limit": "{:,.0f}", "exposure_if_all_drew": "{:,.0f}"}),
            use_container_width=True, hide_index=True)
        st.caption("`exposure_if_all_drew` is the theoretical maximum if every offerable "
                   "customer took their full limit - not a forecast.")

        fig = px.box(scored[scored["offerable"]], x="risk_grade", y="credit_limit",
                     color="risk_grade", category_orders={"risk_grade": cs.RISK_ORDER},
                     color_discrete_map=cs.RISK_COLORS,
                     title="Assigned credit limit by risk grade")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="AZN", height=300)
        st.plotly_chart(fig, use_container_width=True)

    mix = (scored.groupby(["segment", "risk_grade"], observed=True).size()
           .reset_index(name="customers"))
    mix = mix[mix["customers"] > 0]
    fig = px.bar(mix, x="segment", y="customers", color="risk_grade", barmode="stack",
                 category_orders={"risk_grade": cs.RISK_ORDER},
                 color_discrete_map=cs.RISK_COLORS,
                 title="Risk grade within each frequency segment")
    fig.update_layout(xaxis_title="", yaxis_title="Customers", legend_title="", height=340)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------- cash flow
    st.subheader("Twelve-month cash flow and capital")
    labels = {"disbursed": "Credit disbursed", "principal_repaid": "Principal repaid",
              "interest_income": "Interest income", "credit_loss": "Credit loss"}
    flow = sim.melt(id_vars="month", value_vars=list(labels),
                    var_name="flow", value_name="azn")
    flow["flow"] = flow["flow"].map(labels)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(flow, x="month", y="azn", color="flow", barmode="group",
                     color_discrete_map={"Credit disbursed": "#6c8ebf",
                                         "Principal repaid": "#4c9f70",
                                         "Interest income": "#1b7f4b",
                                         "Credit loss": "#e8743b"},
                     title="Monthly flows")
        fig.update_layout(xaxis_title="Month", yaxis_title="AZN", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim["month"], y=sim["outstanding"],
                                 name="Outstanding portfolio", fill="tozeroy",
                                 line=dict(color="#6c8ebf")))
        fig.add_trace(go.Scatter(x=sim["month"], y=-sim["cumulative_cash"],
                                 name="Capital deployed", line=dict(color="#1b7f4b", width=3)))
        fig.update_layout(title="Portfolio and capital", xaxis_title="Month",
                          yaxis_title="AZN", legend=dict(orientation="h", y=-.2))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(sim[["month", "new_loans", "disbursed", "principal_repaid",
                          "interest_income", "credit_loss", "outstanding", "net_cash_flow",
                          "cumulative_net"]].style.format({
            "new_loans": "{:,.0f}", "disbursed": "{:,.0f}", "principal_repaid": "{:,.0f}",
            "interest_income": "{:,.0f}", "credit_loss": "{:,.0f}", "outstanding": "{:,.0f}",
            "net_cash_flow": "{:,.0f}", "cumulative_net": "{:,.0f}"}),
            use_container_width=True, hide_index=True, height=460)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim["month"], y=sim["cumulative_interest"],
                                 name="Cumulative interest", line=dict(color="#1b7f4b")))
        fig.add_trace(go.Scatter(x=sim["month"], y=sim["cumulative_loss"],
                                 name="Cumulative losses", line=dict(color="#e8743b")))
        fig.add_trace(go.Scatter(x=sim["month"], y=sim["cumulative_net"],
                                 name="Net contribution",
                                 line=dict(color="#0f151a", dash="dash")))
        fig.update_layout(title="Cumulative revenue vs. losses", xaxis_title="Month",
                          yaxis_title="AZN", legend=dict(orientation="h", y=-.2))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Capital is recycled **{S['recycled']:.1f}x** over the year: "
                   f"{S['disbursed']:,.0f} AZN lent from {S['required_capital']:,.0f} AZN "
                   "of capital, because repaid loans are lent again.")

    # ---------------------------------------------------------- branches
    st.subheader("Branch-level simulation")
    bb = cs.branch_breakdown(scored, cfg)
    keep = [c for c in ["branch", "rank", "branch_name", "town", "geo_category"]
            if c in branches.columns]
    bb = bb.merge(branches[keep], on="branch", how="left").sort_values("rank")

    with st.expander("Filter", expanded=False):
        c1, c2, c3 = st.columns(3)
        cats = c1.multiselect("Geographic category",
                              sorted(bb["geo_category"].dropna().unique()))
        towns = c2.multiselect("Town / settlement", sorted(bb["town"].dropna().unique()))
        rr = c3.slider("Pilot rank", 1, n_branches, (1, n_branches))
    view = bb[bb["rank"].between(*rr)]
    if cats:
        view = view[view["geo_category"].isin(cats)]
    if towns:
        view = view[view["town"].isin(towns)]

    cols = ["rank", "branch", "branch_name", "town", "geo_category", "eligible", "offerable",
            "low", "medium", "high", "borrowers", "loans", "avg_limit", "disbursed",
            "interest", "losses", "net_contribution", "required_capital"]
    cols = [c for c in cols if c in view.columns]
    st.dataframe(view[cols].style.format({
        "eligible": "{:,.0f}", "offerable": "{:,.0f}", "low": "{:,.0f}", "medium": "{:,.0f}",
        "high": "{:,.0f}", "borrowers": "{:,.0f}", "loans": "{:,.0f}", "avg_limit": "{:,.0f}",
        "disbursed": "{:,.0f}", "interest": "{:,.0f}", "losses": "{:,.0f}",
        "net_contribution": "{:,.0f}", "required_capital": "{:,.0f}"})
        .background_gradient(subset=["net_contribution"], cmap="Greens"),
        use_container_width=True, height=460, hide_index=True)
    st.caption(f"Showing {len(view)} of {len(bb)} branches. Each branch is run through the "
               "same 12-month ledger on its own customer mix, so branch totals reconcile "
               "with the portfolio above.")
    st.download_button("Download the branch simulation as CSV",
                       bb[cols].to_csv(index=False).encode("utf-8-sig"),
                       file_name="pilot_credit_simulation_by_branch.csv", mime="text/csv")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(view.nlargest(15, "net_contribution").sort_values("net_contribution"),
                     x="net_contribution", y="branch", orientation="h", color="geo_category",
                     title="Top 15 branches by expected net contribution")
        fig.update_layout(xaxis_title="AZN", yaxis_title="", yaxis_type="category",
                          legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        by_cat = view.groupby("geo_category").agg(
            branches=("branch", "size"), disbursed=("disbursed", "sum")).reset_index()
        fig = px.bar(by_cat, x="geo_category", y="disbursed", color="geo_category",
                     title="Disbursement by geographic category", text="branches")
        fig.update_layout(xaxis_title="", yaxis_title="AZN", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------- assumptions/limitations
    st.subheader("What is measured, what is assumed")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Observed** - measured from transaction history")
        st.dataframe(pd.DataFrame({
            "variable": ["Frequency segment", "Behavioural score", "Monthly spend",
                         "Average basket", "Transactions", "Tenure", "Primary branch"],
            "in this pool": [
                f"{100*(scored['segment'].astype(str) == 'Frequent').mean():.0f}% Frequent",
                f"median {scored['score'].median():.0f}",
                f"median {scored['monthly_spend'].median():.0f} AZN",
                f"median {scored['avg_basket'].median():.1f} AZN",
                f"median {scored['n_tx'].median():.0f}",
                f"median {scored['tenure_months'].median():.0f} months",
                f"{scored['primary_branch'].nunique()} branches"],
        }), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Assumed** - no historical evidence, set by management")
        gap = cfg["REBORROW_GAP_MONTHS"]
        st.dataframe(pd.DataFrame({
            "assumption": ["Take-up (Low/Med/High)", "Offered (Low/Med/High)",
                           "Loss rate (Low/Med/High)", "Recovery on default",
                           "Repeat borrowing", "Re-borrow gap", "Onboarding ramp",
                           "Package rates"],
            "current value": [
                " / ".join(f"{100*cfg['TAKE_UP'][g]:.0f}%" for g in cs.RISK_ORDER),
                " / ".join(f"{100*cfg['OFFER_RATE'][g]:.0f}%" for g in cs.RISK_ORDER),
                " / ".join(f"{100*cfg['DEFAULT_RATE'][g]:.0f}%" for g in cs.RISK_ORDER),
                f"{100*cfg['RECOVERY_RATE']:.0f}%", f"{100*cfg['REPEAT_RATE']:.0f}%",
                "same month" if gap == 0 else f"{gap} month(s)",
                f"{cfg['ACQUISITION_MONTHS']} months",
                " / ".join(f"{100*cfg['PACKAGE_RATES'][t]:.1f}%" for t in cs.TERMS)],
        }), use_container_width=True, hide_index=True)

    with st.expander("Limitations - read before taking these numbers to a committee"):
        st.markdown(f"""
- **No default rate here is derived from data.** The pharmacy has never lent to these
  customers. We hold no income, no existing debt, no credit-bureau record and no repayment
  history. The loss assumptions are management's judgement, and they could be wrong by a
  wide margin - try moving them and watch what happens to net profit.
- **The behavioural score predicts shopping persistence, not repayment.** It was validated
  out-of-time at AUC 0.918 against whether customers kept buying - a reasonable proxy for a
  stable commercial relationship, and nothing more.
- **Credit limits are large relative to observed spending.** The median eligible customer
  spends {scored['monthly_spend'].median():.0f} AZN a month at the pharmacy. The
  affordability guardrail currently caps limits at {cfg['AFFORDABILITY_MULTIPLE']:g}x that,
  which leaves {int((~scored['offerable']).sum()):,} customers below the
  {cfg['CREDIT_MIN']} AZN floor and therefore unoffered.
- **Everyone who accepts is assumed to draw their full limit.** Real revolving credit is
  drawn partially, so disbursement, capital and interest here are an upper bound for a
  given take-up rate.
- **Repayment is bullet at maturity.** A 3-month loan returns nothing until month 3. If the
  product is sold with monthly instalments instead, capital recycles faster and the capital
  requirement falls.
- **Customers shop at more than one branch.** 76.7% of this pool has transactions at several
  branches (median primary-branch share 0.90), so branch-level figures allocate a customer
  wholly to their primary branch.
- **A new borrower and a repeat borrower are the same person.** Repeat lending recycles
  principal within the existing borrower pool; it does not add customers. The re-borrow gap
  is currently **{'the same month' if gap == 0 else f'{gap} month(s)'}**, producing
  **{S['loans']/max(S['borrowers'], 1):.1f} loans per borrower** over the year.
""")
