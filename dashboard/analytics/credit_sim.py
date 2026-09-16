"""
============================================================================
credit_sim.py  --  Pilot credit simulation engine
============================================================================

A decision-support simulator for the pharmacy credit pilot, NOT a prediction.

Three layers, deliberately kept apart (see SIM_CONFIG and the page that
renders it):

  Observed     measured from transaction history - segment, behavioural
               score, monthly spend, basket size, branch.
  Derived      business rules applied to observed data - risk grade and
               credit limit. Reproducible, inspectable, no black box.
  Assumption   unobservable future behaviour - take-up, package choice,
               default, recovery, repeat borrowing. NONE of these can be
               estimated from the data we hold: the pharmacy has never
               lent to these customers. Every one is a dashboard input.

Design notes
------------
1.  Risk grade comes from the behavioural score bands that were validated
    out-of-time (METHODOLOGY.md: score 80+ -> 83.7% of customers stayed
    active in >=5 of the next 6 months, 70-79 -> 63.5%, 60-69 -> 44.5%).
    Average basket size is deliberately NOT part of the risk grade: across
    the eligible pool its rank correlation with the behavioural score is
    -0.02 and with active months -0.01, so it separates nothing. It earns
    its place on the limit instead, through monthly spend (rho +0.57).
2.  Repayment is bullet at maturity - a 3-month 100 AZN loan returns 115
    AZN in month 3 and nothing before. Capital is therefore locked for the
    full term, which is what makes the capital question interesting.
3.  Nothing here touches CONFIG or pilot_ranking in loan_analytics.py.
============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#1b7f4b", "Medium": "#f2b134", "High": "#e8743b"}
TERMS = (1, 2, 3)


SIM_CONFIG = {
    # ------------------------------------------------ derived: risk grade
    # Anchored on validated score bands, not on producing equal groups.
    "RISK_SCORE_LOW": 80,            # 83.7% persistence in the holdout
    "RISK_SCORE_MEDIUM": 70,         # 63.5% persistence
    "LOW_REQUIRES_FREQUENT": True,   # a pure 80+ cut would call 53% of the pool Low

    # ---------------------------------------------- derived: credit limit
    "CREDIT_MIN": 100,
    "CREDIT_MAX": 300,
    "RISK_CAPS": {"Low": (200, 300), "Medium": (100, 200), "High": (100, 100)},
    # Affordability guardrail. A limit may not exceed this multiple of the
    # customer's observed monthly pharmacy spend. Without it the median
    # customer (40 AZN/month) is offered 4.1x their entire monthly spend.
    "AFFORDABILITY_MULTIPLE": 5.0,

    # ------------------------------------------------------- assumptions
    "PACKAGE_RATES": {1: 0.05, 2: 0.10, 3: 0.15},     # value added per term
    "TAKE_UP": {"Low": 0.25, "Medium": 0.15, "High": 0.05},
    "OFFER_RATE": {"Low": 1.00, "Medium": 1.00, "High": 0.20},
    "TERM_MIX": {"Low":    {1: 0.30, 2: 0.40, 3: 0.30},
                 "Medium": {1: 0.35, 2: 0.40, 3: 0.25},
                 "High":   {1: 0.50, 2: 0.35, 3: 0.15}},
    "DEFAULT_RATE": {"Low": 0.03, "Medium": 0.06, "High": 0.12},
    # Longer exposure may carry more risk; 1.0 keeps default independent of term.
    "DEFAULT_TERM_MULTIPLIER": {1: 1.0, 2: 1.0, 3: 1.0},
    "RECOVERY_RATE": 0.20,
    "REPEAT_RATE": 0.70,             # share of repaying principal lent again
    # Months a customer waits between repaying and borrowing again.
    # 0 = re-lends the same month the loan matures, so capital never idles;
    # 1 = one idle month between cycles. On a 1-month product this is the
    # single biggest lever on how hard the capital works.
    "REBORROW_GAP_MONTHS": 1,
    "ACQUISITION_MONTHS": 6,         # new borrowers are onboarded over this ramp
    "MONTHS": 12,
    "FUNDING_COST_ANNUAL": 0.0,
    "OPEX_PER_LOAN": 0.0,
    "CAPITAL_LIMIT": None,           # None = unconstrained, solve for the need
}


SCENARIOS = {
    "Conservative": {
        "TAKE_UP": {"Low": 0.15, "Medium": 0.08, "High": 0.02},
        "DEFAULT_RATE": {"Low": 0.06, "Medium": 0.12, "High": 0.20},
        "RECOVERY_RATE": 0.10,
        "REPEAT_RATE": 0.50,
    },
    "Base": {},
    "Aggressive": {
        "TAKE_UP": {"Low": 0.40, "Medium": 0.28, "High": 0.12},
        "OFFER_RATE": {"Low": 1.00, "Medium": 1.00, "High": 0.50},
        "DEFAULT_RATE": {"Low": 0.04, "Medium": 0.08, "High": 0.15},
        "REPEAT_RATE": 0.85,
    },
}


def merge_config(overrides: dict | None = None) -> dict:
    """Shallow-merge overrides onto the defaults, one level into dict values."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in SIM_CONFIG.items()}
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


# ===========================================================================
# DERIVED RULES
# ===========================================================================

def assign_risk_grade(features: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """Low / Medium / High from validated behaviour. Basket size plays no part."""
    c = merge_config(cfg)
    grade = pd.Series("High", index=features.index, dtype=object)
    grade[features["score"] >= c["RISK_SCORE_MEDIUM"]] = "Medium"
    top = features["score"] >= c["RISK_SCORE_LOW"]
    if c["LOW_REQUIRES_FREQUENT"]:
        top &= features["segment"].astype(str) == "Frequent"
    grade[top] = "Low"
    return pd.Categorical(grade, categories=RISK_ORDER, ordered=True)


def assign_credit_limit(features: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Position inside the risk band is set by the customer's monthly-spend
    percentile within that band, then capped by the affordability guardrail.
    A limit that falls below CREDIT_MIN means no offer is made at all.
    """
    c = merge_config(cfg)
    out = features.copy()
    if "risk_grade" not in out.columns:
        out["risk_grade"] = assign_risk_grade(out, cfg)

    base = pd.Series(np.nan, index=out.index, dtype="float64")
    for grade, (lo, hi) in c["RISK_CAPS"].items():
        m = out["risk_grade"].astype(str) == grade
        if not m.any():
            continue
        lo = max(lo, c["CREDIT_MIN"])
        hi = min(hi, c["CREDIT_MAX"])
        if hi <= lo:
            base[m] = lo
        else:
            pct = out.loc[m, "monthly_spend"].rank(pct=True)
            base[m] = lo + pct * (hi - lo)

    capped = np.minimum(base, c["AFFORDABILITY_MULTIPLE"] * out["monthly_spend"])
    out["credit_limit_uncapped"] = (base / 10).round() * 10
    out["credit_limit"] = (capped / 10).round() * 10
    out["limit_reduced"] = out["credit_limit"] < out["credit_limit_uncapped"]
    out["offerable"] = out["credit_limit"] >= c["CREDIT_MIN"]
    return out


def build_portfolio(scored: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Funnel per risk grade: eligible -> offerable -> offered -> takers."""
    c = merge_config(cfg)
    rows = []
    for grade in RISK_ORDER:
        g = scored[scored["risk_grade"].astype(str) == grade]
        ok = g[g["offerable"]]
        offered = len(ok) * c["OFFER_RATE"][grade]
        takers = offered * c["TAKE_UP"][grade]
        rows.append({
            "risk_grade": grade,
            "eligible": len(g),
            "offerable": len(ok),
            "offered": offered,
            "takers": takers,
            "avg_limit": float(ok["credit_limit"].mean()) if len(ok) else 0.0,
            "exposure_if_all_drew": float(ok["credit_limit"].sum()),
        })
    return pd.DataFrame(rows).set_index("risk_grade").reindex(RISK_ORDER)


# ===========================================================================
# ECONOMICS
# ===========================================================================

def break_even_default(rate: float, recovery: float) -> float:
    """d* where expected interest exactly offsets expected loss, per loan."""
    denominator = 1.0 + rate - recovery
    return rate / denominator if denominator > 0 else float("nan")


def annual_capital_return(rate: float, term: int) -> float:
    """Simple annual return on deployed capital, before losses."""
    return rate * (12.0 / term)


def package_economics(cfg: dict | None = None) -> pd.DataFrame:
    """Per-package view: the same gross return, very different risk tolerance."""
    c = merge_config(cfg)
    rows = []
    for term in TERMS:
        rate = c["PACKAGE_RATES"][term]
        rows.append({
            "term_months": term,
            "rate_pct": 100 * rate,
            "repayment_per_100": 100 * (1 + rate),
            "cycles_per_year": 12 / term,
            "gross_annual_return_pct": 100 * annual_capital_return(rate, term),
            "break_even_default_pct": 100 * break_even_default(rate, c["RECOVERY_RATE"]),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 12-MONTH LEDGER
# ===========================================================================

def simulate_year(portfolio: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Month-by-month money ledger with bullet repayment.

    A loan originated in month m with term t and principal P leaves the book
    as cash in month m, and in month m+t returns P*(1+r_t) if it performs or
    P*recovery if it does not. A share of performing principal is lent again
    REBORROW_GAP_MONTHS later, which is what recycles the capital: at a gap of
    0 the money goes straight back out the month it arrives, at 1 it idles for
    a month between cycles.
    """
    c = merge_config(cfg)
    months = int(c["MONTHS"])
    ramp = max(1, min(int(c["ACQUISITION_MONTHS"]), months))
    gap = max(0, int(c["REBORROW_GAP_MONTHS"]))
    recovery = c["RECOVERY_RATE"]
    capital_limit = c["CAPITAL_LIMIT"]

    # month -> list of (grade, term, principal, borrowers)
    originations: dict[int, list] = {m: [] for m in range(1, months + gap + 2)}
    maturities: dict[int, list] = {m: [] for m in range(1, months + max(TERMS) + 2)}

    for grade in RISK_ORDER:
        if grade not in portfolio.index:
            continue
        takers = float(portfolio.loc[grade, "takers"])
        limit = float(portfolio.loc[grade, "avg_limit"])
        if takers <= 0 or limit <= 0:
            continue
        for m in range(1, ramp + 1):
            n = takers / ramp
            for term, share in c["TERM_MIX"][grade].items():
                if share > 0:
                    originations[m].append((grade, term, n * share * limit, n * share))

    rows = []
    cash = float(capital_limit) if capital_limit else 0.0
    trough = cash
    outstanding = 0.0
    throttled_total = 0.0

    for m in range(1, months + 1):
        # Maturities are settled before new lending, so a loan repaid this month
        # can fund this month's originations when the re-borrow gap is zero.
        principal_in = interest_in = loss = repaid_borrowers = 0.0
        for grade, term, principal, borrowers in maturities[m]:
            d = min(c["DEFAULT_RATE"][grade] * c["DEFAULT_TERM_MULTIPLIER"][term], 1.0)
            performing = principal * (1 - d)
            defaulted = principal * d
            principal_in += performing + defaulted * recovery
            interest_in += performing * c["PACKAGE_RATES"][term]
            loss += defaulted * (1 - recovery)
            repaid_borrowers += borrowers * (1 - d)

            repeat_principal = performing * c["REPEAT_RATE"]
            back = m + gap
            if repeat_principal > 0 and back <= months:
                repeat_borrowers = borrowers * (1 - d) * c["REPEAT_RATE"]
                for t2, share in c["TERM_MIX"][grade].items():
                    if share > 0:
                        originations[back].append(
                            (grade, t2, repeat_principal * share, repeat_borrowers * share))

        available = cash + principal_in + interest_in
        wanted = sum(p for _, _, p, _ in originations[m])
        wanted_loans = sum(n for _, _, _, n in originations[m])
        # Writing a loan costs principal AND its processing fee, so the cap has
        # to cover both or the month closes overdrawn by the operating cost.
        wanted_cash = wanted + c["OPEX_PER_LOAN"] * wanted_loans
        scale = 1.0
        if capital_limit is not None and wanted_cash > available:
            scale = max(available, 0.0) / wanted_cash if wanted_cash > 0 else 0.0
            throttled_total += wanted * (1 - scale)

        disbursed = loans = 0.0
        for grade, term, principal, borrowers in originations[m]:
            principal, borrowers = principal * scale, borrowers * scale
            if principal <= 0:
                continue
            disbursed += principal
            loans += borrowers
            maturities[m + term].append((grade, term, principal, borrowers))

        opex = c["OPEX_PER_LOAN"] * loans
        outstanding += disbursed - (principal_in + loss)
        net_cash = -disbursed + principal_in + interest_in - opex
        cash += net_cash
        trough = min(trough, cash)

        rows.append({
            "month": m, "new_loans": loans, "disbursed": disbursed,
            "principal_repaid": principal_in, "interest_income": interest_in,
            "credit_loss": loss, "opex": opex, "borrowers_repaid": repaid_borrowers,
            "outstanding": max(outstanding, 0.0), "net_cash_flow": net_cash,
            "cumulative_cash": cash,
        })

    sim = pd.DataFrame(rows)
    sim["cumulative_interest"] = sim["interest_income"].cumsum()
    sim["cumulative_loss"] = sim["credit_loss"].cumsum()
    sim["cumulative_net"] = sim["cumulative_interest"] - sim["cumulative_loss"]

    required_capital = float(capital_limit) if capital_limit else abs(min(trough, 0.0))
    interest = sim["interest_income"].sum()
    losses = sim["credit_loss"].sum()
    opex_total = sim["opex"].sum()
    funding = required_capital * c["FUNDING_COST_ANNUAL"]
    disbursed_total = sim["disbursed"].sum()

    summary = {
        "borrowers": float(portfolio["takers"].sum()) if len(portfolio) else 0.0,
        "loans": sim["new_loans"].sum(),
        "disbursed": disbursed_total,
        "interest": interest,
        "losses": losses,
        "opex": opex_total,
        "funding_cost": funding,
        "net_contribution": interest - losses,
        "net_profit": interest - losses - opex_total - funding,
        "peak_outstanding": sim["outstanding"].max(),
        "required_capital": required_capital,
        # Cash needed on day one to fund the first month's cohort. Nothing has
        # matured yet, so month 1 is pure outflow.
        "month1_capital": float(sim["disbursed"].iloc[0]) if len(sim) else 0.0,
        "recycled": disbursed_total / required_capital if required_capital > 0 else 0.0,
        "return_on_capital": ((interest - losses - opex_total - funding) / required_capital
                              if required_capital > 0 else 0.0),
        "loss_rate": losses / disbursed_total if disbursed_total > 0 else 0.0,
        "unfunded_demand": throttled_total,
        "avg_loan": disbursed_total / sim["new_loans"].sum() if sim["new_loans"].sum() else 0.0,
    }
    return sim, summary


def run(scored: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Portfolio + 12-month ledger + summary in one call."""
    portfolio = build_portfolio(scored, cfg)
    sim, summary = simulate_year(portfolio, cfg)
    return portfolio, sim, summary


# ===========================================================================
# COMPARISON AND SENSITIVITY
# ===========================================================================

def scenario_table(scored: pd.DataFrame, base_cfg: dict | None = None) -> pd.DataFrame:
    """Conservative / Base / Aggressive side by side, on the current limits."""
    rows = []
    for name, overrides in SCENARIOS.items():
        cfg = merge_config({**(base_cfg or {}), **overrides})
        _, _, s = run(scored, cfg)
        rows.append({
            "scenario": name, "borrowers": s["borrowers"], "loans": s["loans"],
            "disbursed": s["disbursed"], "required_capital": s["required_capital"],
            "recycled": s["recycled"], "interest": s["interest"], "losses": s["losses"],
            "net_contribution": s["net_contribution"], "net_profit": s["net_profit"],
            "loss_rate_pct": 100 * s["loss_rate"],
            "return_on_capital_pct": 100 * s["return_on_capital"],
        })
    return pd.DataFrame(rows)


def sensitivity_grid(scored: pd.DataFrame, row_key: str, row_values, col_key: str,
                     col_values, metric: str = "net_contribution",
                     base_cfg: dict | None = None) -> pd.DataFrame:
    """
    Two-way sensitivity. `row_key`/`col_key` accept:
      'default_multiplier'  scale every grade's default rate
      'takeup_multiplier'   scale every grade's take-up
      'rate_1m'/'rate_2m'/'rate_3m'   set one package rate
      'recovery'            set the recovery rate
    """
    def apply(cfg: dict, key: str, value) -> dict:
        if key == "default_multiplier":
            cfg["DEFAULT_RATE"] = {k: min(v * value, 0.99)
                                   for k, v in cfg["DEFAULT_RATE"].items()}
        elif key == "takeup_multiplier":
            cfg["TAKE_UP"] = {k: min(v * value, 1.0) for k, v in cfg["TAKE_UP"].items()}
        elif key in ("rate_1m", "rate_2m", "rate_3m"):
            cfg["PACKAGE_RATES"] = {**cfg["PACKAGE_RATES"],
                                    int(key[5]): value}
        elif key == "recovery":
            cfg["RECOVERY_RATE"] = value
        else:
            raise ValueError(f"unknown sensitivity key: {key}")
        return cfg

    out = {}
    for rv in row_values:
        line = {}
        for cv in col_values:
            cfg = merge_config(base_cfg)
            cfg = apply(cfg, row_key, rv)
            cfg = apply(cfg, col_key, cv)
            _, _, s = run(scored, cfg)
            line[cv] = s[metric]
        out[rv] = line
    return pd.DataFrame(out).T


def break_even_default_multiplier(scored: pd.DataFrame, base_cfg: dict | None = None,
                                  metric: str = "net_contribution") -> float:
    """How many times the assumed default rates the portfolio can absorb."""
    lo, hi = 0.0, 40.0
    _, _, s0 = run(scored, merge_config(base_cfg))
    if s0[metric] <= 0:
        return 0.0
    for _ in range(45):
        mid = (lo + hi) / 2
        cfg = merge_config(base_cfg)
        cfg["DEFAULT_RATE"] = {k: min(v * mid, 0.99) for k, v in cfg["DEFAULT_RATE"].items()}
        _, _, s = run(scored, cfg)
        if s[metric] > 0:
            lo = mid
        else:
            hi = mid
    return lo


def branch_breakdown(scored: pd.DataFrame, cfg: dict | None = None,
                     branch_col: str = "primary_branch") -> pd.DataFrame:
    """
    Per-branch annual outcome, each branch run through the same ledger on its
    own customer mix, so branch totals reconcile with the portfolio.
    """
    c = merge_config(cfg)
    rows = []
    for branch, grp in scored.groupby(branch_col, observed=True):
        portfolio = build_portfolio(grp, c)
        _, s = simulate_year(portfolio, c)
        counts = grp["risk_grade"].value_counts()
        rows.append({
            "branch": branch,
            "eligible": len(grp),
            "offerable": int(grp["offerable"].sum()),
            "low": int(counts.get("Low", 0)),
            "medium": int(counts.get("Medium", 0)),
            "high": int(counts.get("High", 0)),
            "borrowers": s["borrowers"],
            "loans": s["loans"],
            "avg_limit": s["avg_loan"],
            "disbursed": s["disbursed"],
            "interest": s["interest"],
            "losses": s["losses"],
            "net_contribution": s["net_contribution"],
            "required_capital": s["required_capital"],
        })
    out = pd.DataFrame(rows)
    out["take_up_pct"] = 100 * out["borrowers"] / out["offerable"].clip(lower=1)
    return out
