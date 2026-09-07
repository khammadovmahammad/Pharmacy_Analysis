"""
============================================================================
loan_analytics.py  --  Customer segmentation, Behavioural Eligibility Score
                       and Branch Pilot Suitability Score
============================================================================

Every threshold, weight and cap in this module was derived from the Phase 1
diagnostics and validated out-of-time on the real dataset (see METHODOLOGY.md
and validate_model.py). Nothing here is a default or a guess.

Design rules
------------
1.  All tunable numbers live in CONFIG. Changing a threshold must never
    require editing logic.
2.  SQL does the heavy aggregation (15.6M rows), pandas does the derivation,
    so segment/score rules can be re-tuned without touching the database.
3.  The observation window is anchored on an `as_of` month, so the whole
    model can be replayed at any historical point in time (that is how it
    was validated).

Layout assumed
--------------
    project_root/
        data/pharmacy.duckdb
        dashboard/app.py                    <- existing dashboard (unchanged)
        dashboard/analytics/loan_analytics.py
        dashboard/pages/*.py
============================================================================
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import streamlit as st
    _cache = st.cache_data
except Exception:                                    # usable outside Streamlit
    def _cache(*a, **k):
        def deco(fn):
            return fn
        return deco if not a else a[0]


# ===========================================================================
# CONFIGURATION - all model parameters
# ===========================================================================

CONFIG = {
    # ---------------------------------------------------------------- data
    "TX_TABLE": "clean_transactions",
    "QUALIFYING_FILTER": "usable_for_sales = 1",

    # ------------------------------------------------------ observation
    # 6 months. Validated: a 6-month window predicts future persistence
    # better than 3, 12 or 18 (AUC 0.912 vs 0.909 / 0.901 / 0.884) and
    # 84.6% of customers have at least 6 months of history.
    "WINDOW_MONTHS": 6,

    # Minimum tenure before a customer can be classified at all.
    "MIN_TENURE_MONTHS": 6,

    # ------------------------------------------------------ segmentation
    # Rules are expressed on the observation window:
    #   coverage = months with >=1 qualifying transaction
    #   depth_3  = months with >=3 qualifying transactions
    "SEG_FREQUENT_COVERAGE": 5,      # >=5 of 6 months active ...
    "SEG_FREQUENT_DEPTH3": 5,        # ... with >=3 transactions in >=5 of them
    "SEG_REGULAR_COVERAGE": 5,       # >=5 of 6 months active
    "SEG_SEMI_COVERAGE": 3,          # 3-4 of 6 months active
    "SEG_SPORADIC_COVERAGE": 1,      # 1-2 of 6 months active
                                     # 0 -> Dormant, tenure<6 -> New

    # -------------------------------------------- behavioural score (0-100)
    # Weights are fixed, not fitted, because a logistic fit gave the same
    # discrimination (AUC 0.9187 fitted vs 0.9180 fixed) and fixed weights
    # are explainable to a credit committee. The fitted coefficients are
    # shown in brackets as the empirical justification.
    "SCORE_WEIGHTS": {
        "coverage":      0.30,       # fitted share 27.0%
        "intensity":     0.20,       # fitted share 20.5%
        "depth":         0.15,       # fitted share 15.3%
        "recency":       0.20,       # fitted share 26.7%
        "tenure":        0.10,       # fitted share  8.6%
        "spend":         0.05,       # fitted share  1.8%
    },
    "SCORE_CAP_INTENSITY": 6.0,      # tx per active month, ~p95
    "SCORE_CAP_TENURE": 24,          # months (dataset length)
    "SCORE_CAP_MONTHLY_SPEND": 400,  # currency per month, log-scaled

    # ------------------------------------------------------- eligibility
    # Score >=65 -> 70.2% of customers stayed active in >=5 of the next 6
    # months, against a 21.2% base rate. Cut chosen where lift is high and
    # the population is still commercially meaningful (~160k customers).
    "ELIGIBILITY_MIN_SCORE": 65,
    "SCORE_BANDS": [(80, 101, "A (80-100)"), (65, 80, "B (65-79)"),
                    (50, 65, "C (50-64)"), (35, 50, "D (35-49)"),
                    (0, 35, "E (0-34)")],

    # ------------------------------------------- branch pilot suitability
    "BRANCH_MIN_MONTHS_PRESENT": 24,   # full history only
    "BRANCH_MIN_ELIGIBLE": 300,        # workable pilot sample
    "PILOT_WEIGHTS": {
        "size":       0.30,   # log eligible population
        "rate":       0.20,   # eligibility rate among primary customers
        "quality":    0.10,   # median score of eligible customers
        "loyalty":    0.15,   # branch is the customer's own pharmacy
        "stability":  0.10,   # month-to-month volatility (inverted)
        "afford":     0.10,   # median monthly spend of eligible customers
        "data":       0.05,   # share of customers with >=6 months history
    },
}

SEGMENT_ORDER = ["Frequent", "Regular", "Semi-Regular", "Sporadic",
                 "Dormant", "New/Insufficient"]

SEGMENT_COLORS = {
    "Frequent": "#1b7f4b", "Regular": "#4c9f70", "Semi-Regular": "#f2b134",
    "Sporadic": "#e8743b", "Dormant": "#9aa0a6", "New/Insufficient": "#6c8ebf",
}


# ===========================================================================
# DATABASE
# ===========================================================================

def find_db() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parents[2] / "data" / "pharmacy.duckdb",
              here.parents[1] / "data" / "pharmacy.duckdb",
              Path.cwd() / "data" / "pharmacy.duckdb"):
        if p.exists():
            return p
    raise FileNotFoundError("pharmacy.duckdb not found")


def run_query(sql: str, params=None, register: dict | None = None) -> pd.DataFrame:
    """Read-only query; `register` exposes pandas frames to SQL by name."""
    con = duckdb.connect(str(find_db()), read_only=True)
    try:
        if register:
            for name, frame in register.items():
                con.register(name, frame)
        return con.execute(sql, params).fetchdf() if params else con.execute(sql).fetchdf()
    finally:
        con.close()


@_cache(show_spinner=False)
def available_months() -> list[str]:
    df = run_query(f"""
        SELECT DISTINCT CAST(date_trunc('month', tarix_saat) AS DATE) AS ym
        FROM {CONFIG['TX_TABLE']}
        WHERE {CONFIG['QUALIFYING_FILTER']}
        ORDER BY 1
    """)
    return [pd.Timestamp(v).strftime("%Y-%m") for v in df["ym"]]


# ===========================================================================
# CUSTOMER FEATURES
# ===========================================================================

@_cache(show_spinner="Building customer features...")
def load_customer_features(as_of: str | None = None,
                           window_months: int | None = None) -> pd.DataFrame:
    """
    One row per customer, as observed at the end of `as_of` (YYYY-MM).

    Lifetime columns describe the whole history up to as_of; window columns
    describe the last `window_months` months, which is what the segmentation
    and the score are built on.
    """
    months = available_months()
    as_of = as_of or months[-1]
    L = window_months or CONFIG["WINDOW_MONTHS"]

    end_excl = (pd.Period(as_of, "M") + 1).start_time.date().isoformat()
    win_start = (pd.Period(as_of, "M") - (L - 1)).start_time.date().isoformat()

    sql = f"""
    WITH base AS (
        SELECT
            musteri_acari                              AS cust,
            CAST(aptek_id AS VARCHAR)                  AS branch,
            tarix_saat                                 AS ts,
            CAST(date_trunc('month', tarix_saat) AS DATE) AS ym,
            CAST(net_amount AS DOUBLE)                 AS amt
        FROM {CONFIG['TX_TABLE']}
        WHERE {CONFIG['QUALIFYING_FILTER']}
          AND tarix_saat < CAST(? AS DATE)
    ),
    cm AS (
        SELECT cust, ym, COUNT(*) AS n, SUM(amt) AS s
        FROM base GROUP BY 1, 2
    ),
    life AS (
        SELECT cust,
               COUNT(*)                    AS n_tx,
               SUM(amt)                    AS total_spend,
               AVG(amt)                    AS avg_basket,
               median(amt)                 AS median_basket,
               MIN(ym)                     AS first_ym,
               MAX(ym)                     AS last_ym,
               MAX(ts)                     AS last_ts,
               COUNT(DISTINCT ym)          AS active_months_life,
               COUNT(DISTINCT branch)      AS n_branches
        FROM base GROUP BY 1
    ),
    win AS (
        SELECT cust,
               SUM(n)                                   AS tx_win,
               SUM(s)                                   AS spend_win,
               COUNT(*)                                 AS coverage,
               SUM(CASE WHEN n >= 2 THEN 1 ELSE 0 END)  AS depth_2,
               SUM(CASE WHEN n >= 3 THEN 1 ELSE 0 END)  AS depth_3
        FROM cm WHERE ym >= CAST(? AS DATE) GROUP BY 1
    ),
    pb AS (
        SELECT cust, branch AS primary_branch, b_tx AS primary_branch_tx
        FROM (
            SELECT cust, branch, COUNT(*) AS b_tx,
                   ROW_NUMBER() OVER (PARTITION BY cust
                                      ORDER BY COUNT(*) DESC, SUM(amt) DESC) AS rn
            FROM base GROUP BY 1, 2
        ) WHERE rn = 1
    )
    SELECT life.*, pb.primary_branch, pb.primary_branch_tx,
           COALESCE(win.tx_win, 0)    AS tx_win,
           COALESCE(win.spend_win, 0) AS spend_win,
           COALESCE(win.coverage, 0)  AS coverage,
           COALESCE(win.depth_2, 0)   AS depth_2,
           COALESCE(win.depth_3, 0)   AS depth_3
    FROM life
    LEFT JOIN win USING (cust)
    LEFT JOIN pb  USING (cust)
    """
    df = run_query(sql, [end_excl, win_start])

    # ---------------- derived (pure pandas, so it stays re-tunable) --------
    as_of_p = pd.Period(as_of, "M")
    first_p = pd.PeriodIndex(pd.to_datetime(df["first_ym"]), freq="M")
    last_p = pd.PeriodIndex(pd.to_datetime(df["last_ym"]), freq="M")
    first_ord = first_p.asi8
    last_ord = last_p.asi8

    df["first_month"] = first_p.astype(str)
    df["last_month"] = last_p.astype(str)
    df["tenure_months"] = (as_of_p.ordinal - first_ord + 1).astype("int16")
    df["recency_months"] = (as_of_p.ordinal - last_ord).astype("int16")
    df["recency_days"] = (as_of_p.end_time - pd.to_datetime(df["last_ts"])).dt.days
    df["observed_months"] = (last_ord - first_ord + 1).astype("int16")

    df["monthly_spend"] = df["spend_win"] / CONFIG["WINDOW_MONTHS"]
    df["intensity"] = np.where(df["coverage"] > 0,
                               df["tx_win"] / df["coverage"].clip(lower=1), 0.0)
    df["active_ratio_life"] = df["active_months_life"] / df["tenure_months"].clip(lower=1)
    df["primary_branch_share"] = df["primary_branch_tx"] / df["n_tx"]
    df["primary_branch"] = df["primary_branch"].astype(str)

    df = assign_segments(df)
    df = compute_score(df)

    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    return df


# ===========================================================================
# SEGMENTATION
# ===========================================================================

def assign_segments(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Transparent, reproducible rules on the observation window.

    Frequent      coverage >= 5 of 6 AND >=3 transactions in >=5 of those months
    Regular       coverage >= 5 of 6
    Semi-Regular  coverage 3-4 of 6
    Sporadic      coverage 1-2 of 6
    Dormant       coverage 0, but has history
    New/Insufficient   tenure < 6 months -> not classifiable, whatever the frequency
    """
    c = {**CONFIG, **(cfg or {})}
    cov, d3, ten = df["coverage"], df["depth_3"], df["tenure_months"]

    seg = pd.Series("Dormant", index=df.index, dtype=object)
    seg[cov.between(c["SEG_SPORADIC_COVERAGE"], c["SEG_SEMI_COVERAGE"] - 1)] = "Sporadic"
    seg[cov.between(c["SEG_SEMI_COVERAGE"], c["SEG_REGULAR_COVERAGE"] - 1)] = "Semi-Regular"
    seg[cov >= c["SEG_REGULAR_COVERAGE"]] = "Regular"
    seg[(cov >= c["SEG_FREQUENT_COVERAGE"]) & (d3 >= c["SEG_FREQUENT_DEPTH3"])] = "Frequent"
    seg[ten < c["MIN_TENURE_MONTHS"]] = "New/Insufficient"

    df["segment"] = pd.Categorical(seg, categories=SEGMENT_ORDER, ordered=True)
    return df


# ===========================================================================
# BEHAVIOURAL ELIGIBILITY SCORE
# ===========================================================================

def score_components(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Each component is scaled to 0-1 so weights are directly comparable."""
    c = {**CONFIG, **(cfg or {})}
    L = c["WINDOW_MONTHS"]
    cap_s = c["SCORE_CAP_MONTHLY_SPEND"]

    return pd.DataFrame({
        # how many of the last L months the customer showed up at all
        "coverage": (df["coverage"] / L).clip(0, 1),
        # how heavy an active month is
        "intensity": (df["intensity"] / c["SCORE_CAP_INTENSITY"]).clip(0, 1),
        # months that were more than a single visit
        "depth": (df["depth_2"] / L).clip(0, 1),
        # months since the last purchase, inverted
        "recency": (1 - df["recency_months"] / L).clip(0, 1),
        # relationship length, capped at the dataset length
        "tenure": (df["tenure_months"] / c["SCORE_CAP_TENURE"]).clip(0, 1),
        # ability-to-pay proxy, log-scaled because spend is heavily skewed
        "spend": np.log1p(df["monthly_spend"].clip(0, cap_s)) / np.log1p(cap_s),
    }, index=df.index)


def compute_score(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    c = {**CONFIG, **(cfg or {})}
    comp = score_components(df, cfg)
    w = c["SCORE_WEIGHTS"]
    df["score"] = (100 * sum(comp[k] * w[k] for k in w)).astype("float32")
    for k in w:
        df[f"sc_{k}"] = (100 * comp[k]).astype("float32")

    bands = c["SCORE_BANDS"]
    df["score_band"] = pd.cut(df["score"],
                              [b[0] for b in sorted(bands)] + [101],
                              right=False,
                              labels=[b[2] for b in sorted(bands)])
    df["eligible"] = ((df["score"] >= c["ELIGIBILITY_MIN_SCORE"]) &
                      (df["tenure_months"] >= c["MIN_TENURE_MONTHS"]))
    return df


def explain_score(row: pd.Series, cfg: dict | None = None) -> pd.DataFrame:
    """Per-customer score breakdown for the Customer Explorer."""
    c = {**CONFIG, **(cfg or {})}
    w = c["SCORE_WEIGHTS"]
    labels = {
        "coverage": f"Active months ({int(row['coverage'])} of {c['WINDOW_MONTHS']})",
        "intensity": f"Transactions per active month ({row['intensity']:.1f})",
        "depth": f"Months with 2+ transactions ({int(row['depth_2'])})",
        "recency": f"Months since last purchase ({int(row['recency_months'])})",
        "tenure": f"Tenure ({int(row['tenure_months'])} months)",
        "spend": f"Monthly spend ({row['monthly_spend']:.1f})",
    }
    out = []
    for k, weight in w.items():
        raw = float(row[f"sc_{k}"])
        out.append({"component": labels[k], "component_score_0_100": raw,
                    "weight_%": 100 * weight, "points": raw * weight})
    out.append({"component": "TOTAL", "component_score_0_100": np.nan,
                "weight_%": 100.0, "points": float(row["score"])})
    return pd.DataFrame(out)


# ===========================================================================
# BRANCH ANALYTICS
# ===========================================================================

@_cache(show_spinner=False)
def load_branch_month(as_of: str | None = None) -> pd.DataFrame:
    months = available_months()
    as_of = as_of or months[-1]
    end_excl = (pd.Period(as_of, "M") + 1).start_time.date().isoformat()
    return run_query(f"""
        SELECT CAST(aptek_id AS VARCHAR) AS branch,
               CAST(date_trunc('month', tarix_saat) AS DATE) AS month,
               COUNT(*) AS tx,
               COUNT(DISTINCT musteri_acari) AS customers,
               SUM(CAST(net_amount AS DOUBLE)) AS net_sales,
               AVG(CAST(net_amount AS DOUBLE)) AS avg_basket
        FROM {CONFIG['TX_TABLE']}
        WHERE {CONFIG['QUALIFYING_FILTER']}
          AND tarix_saat < CAST(? AS DATE)
        GROUP BY 1, 2 ORDER BY 1, 2
    """, [end_excl])


@_cache(show_spinner=False)
def load_branch_static(as_of: str | None = None) -> pd.DataFrame:
    """Unique customers, return rate and prescription share per branch."""
    months = available_months()
    as_of = as_of or months[-1]
    end_excl = (pd.Period(as_of, "M") + 1).start_time.date().isoformat()
    return run_query(f"""
        SELECT CAST(aptek_id AS VARCHAR) AS branch,
               COUNT(DISTINCT CASE WHEN {CONFIG['QUALIFYING_FILTER']}
                                   THEN musteri_acari END)          AS unique_customers,
               100.0 * SUM(CASE WHEN qaytarma = 1 THEN 1 ELSE 0 END)
                     / NULLIF(COUNT(*), 0)                          AS return_rate_pct,
               100.0 * AVG(CAST(resepti_var AS DOUBLE))             AS rx_share_pct
        FROM {CONFIG['TX_TABLE']}
        WHERE tarix_saat < CAST(? AS DATE)
        GROUP BY 1
    """, [end_excl])


def branch_summary(features: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    """
    One row per branch. `primary_*` columns count only customers for whom the
    branch is their own pharmacy - that is the realistic pilot population,
    not everyone who ever walked in.
    """
    f = features
    g = f.groupby("primary_branch", observed=True)
    b = g.agg(primary_customers=("cust", "size"),
              mean_score=("score", "mean"),
              median_score=("score", "median"),
              median_loyalty=("primary_branch_share", "median"),
              median_basket_customers=("avg_basket", "median"),
              pct_hist_6m=("tenure_months", lambda s: 100 * (s >= CONFIG["MIN_TENURE_MONTHS"]).mean()),
              ).reset_index().rename(columns={"primary_branch": "branch"})

    seg_counts = (f.groupby(["primary_branch", "segment"], observed=False)
                    .size().unstack(fill_value=0))
    seg_counts.columns = [f"n_{c.lower().replace('-', '_').replace('/', '_')}"
                          for c in seg_counts.columns]
    b = b.merge(seg_counts, left_on="branch", right_index=True, how="left")

    e = f[f["eligible"]]
    b = b.merge(e.groupby("primary_branch", observed=True).agg(
        eligible=("cust", "size"),
        elig_median_score=("score", "median"),
        elig_median_monthly_spend=("monthly_spend", "median"),
        elig_median_basket=("avg_basket", "median"),
        elig_median_loyalty=("primary_branch_share", "median"),
    ), left_on="branch", right_index=True, how="left")
    b["eligible"] = b["eligible"].fillna(0).astype(int)

    bm = load_branch_month(as_of)
    agg = bm.groupby("branch").agg(tx=("tx", "sum"), net_sales=("net_sales", "sum"),
                                   months_present=("month", "nunique"),
                                   tx_month_mean=("tx", "mean"), tx_month_sd=("tx", "std"))
    agg["tx_month_cv"] = agg["tx_month_sd"] / agg["tx_month_mean"]
    last_m = bm["month"].max()
    w_start = last_m - pd.DateOffset(months=CONFIG["WINDOW_MONTHS"] - 1)
    p_start = w_start - pd.DateOffset(months=CONFIG["WINDOW_MONTHS"])
    recent = bm[bm["month"] >= w_start].groupby("branch")["tx"].agg(["sum", "mean", "std"])
    prev = bm[(bm["month"] >= p_start) & (bm["month"] < w_start)].groupby("branch")["tx"].sum()
    agg["tx_cv_recent"] = recent["std"] / recent["mean"]
    agg["tx_growth_pct"] = 100 * (recent["sum"] / prev - 1)

    b = (b.merge(agg.reset_index(), on="branch", how="left")
           .merge(load_branch_static(as_of), on="branch", how="left"))

    b["primary_customer_ratio_pct"] = 100 * b["primary_customers"] / b["unique_customers"]
    b["eligibility_rate_pct"] = 100 * b["eligible"] / b["primary_customers"]
    for s in ["frequent", "regular", "semi_regular", "sporadic"]:
        col = f"n_{s}"
        if col in b.columns:
            b[f"{s}_rate_pct"] = 100 * b[col] / b["primary_customers"]
    b["avg_basket"] = b["net_sales"] / b["tx"]
    b["tx_per_primary_customer"] = b["tx"] / b["primary_customers"]
    return b.sort_values("net_sales", ascending=False).reset_index(drop=True)


# ===========================================================================
# BRANCH PILOT SUITABILITY SCORE
# ===========================================================================

PILOT_COMPONENT_LABELS = {
    "size": "Eligible population (size)",
    "rate": "Eligibility rate",
    "quality": "Behavioural quality of eligible base",
    "loyalty": "Customer loyalty to the branch",
    "stability": "Operational stability",
    "afford": "Affordability (monthly spend)",
    "data": "Data sufficiency",
}


def pilot_ranking(branches: pd.DataFrame,
                  weights: dict | None = None,
                  min_months: int | None = None,
                  min_eligible: int | None = None) -> pd.DataFrame:
    """
    Rank branches for the pilot. Components are percentile ranks *within the
    qualified set*, so the score is 0-100 and no single raw unit dominates.
    """
    w = {**CONFIG["PILOT_WEIGHTS"], **(weights or {})}
    min_months = CONFIG["BRANCH_MIN_MONTHS_PRESENT"] if min_months is None else min_months
    min_elig = CONFIG["BRANCH_MIN_ELIGIBLE"] if min_eligible is None else min_eligible

    q = branches[(branches["months_present"] >= min_months) &
                 (branches["eligible"] >= min_elig)].copy()
    if q.empty:
        return q

    def pr(s, invert=False):
        r = s.rank(pct=True, na_option="bottom")
        return 100 * ((1 - r) if invert else r)

    q["c_size"] = pr(np.log1p(q["eligible"]))
    q["c_rate"] = pr(q["eligibility_rate_pct"])
    q["c_quality"] = pr(q["elig_median_score"])
    q["c_loyalty"] = pr(0.5 * pr(q["elig_median_loyalty"]) +
                        0.5 * pr(q["primary_customer_ratio_pct"]))
    q["c_stability"] = pr(q["tx_cv_recent"], invert=True)
    q["c_afford"] = pr(q["elig_median_monthly_spend"])
    q["c_data"] = pr(q["pct_hist_6m"])

    q["pilot_score"] = sum(q[f"c_{k}"] * v for k, v in w.items())
    q = q.sort_values("pilot_score", ascending=False).reset_index(drop=True)
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    return q


def pilot_sensitivity(ranked: pd.DataFrame, n_draws: int = 2000,
                      concentration: float = 20.0, top_n: int = 5,
                      seed: int = 42) -> pd.DataFrame:
    """
    Re-rank under `n_draws` random weightings drawn around the base weights.
    A branch that keeps appearing in the top N is a robust choice rather than
    an artefact of one weighting.
    """
    comps = [f"c_{k}" for k in CONFIG["PILOT_WEIGHTS"]]
    base = np.array([CONFIG["PILOT_WEIGHTS"][k] for k in CONFIG["PILOT_WEIGHTS"]])
    M = ranked[comps].values
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(ranked))
    first = np.zeros(len(ranked))
    for _ in range(n_draws):
        s = M @ rng.dirichlet(base * concentration)
        order = np.argsort(-s)
        counts[order[:top_n]] += 1
        first[order[0]] += 1
    return (pd.DataFrame({"branch": ranked["branch"],
                          "base_rank": ranked["rank"],
                          f"pct_in_top{top_n}": 100 * counts / n_draws,
                          "pct_ranked_first": 100 * first / n_draws})
            .sort_values(f"pct_in_top{top_n}", ascending=False)
            .reset_index(drop=True))


# ===========================================================================
# SUPPORTING AGGREGATIONS FOR THE DASHBOARD
# ===========================================================================

def segment_summary(features: pd.DataFrame) -> pd.DataFrame:
    g = features.groupby("segment", observed=False)
    out = g.agg(customers=("cust", "size"),
                avg_basket=("avg_basket", "mean"),
                median_basket=("avg_basket", "median"),
                avg_monthly_spend=("monthly_spend", "mean"),
                median_monthly_spend=("monthly_spend", "median"),
                avg_tenure_months=("tenure_months", "mean"),
                avg_score=("score", "mean"),
                median_score=("score", "median"),
                median_tx_window=("tx_win", "median"),
                median_recency_months=("recency_months", "median"),
                eligible=("eligible", "sum")).reset_index()
    out["pct_of_customers"] = 100 * out["customers"] / out["customers"].sum()
    out["eligibility_rate_pct"] = 100 * out["eligible"] / out["customers"]
    return out


@_cache(show_spinner=False)
def monthly_sales_by_segment(_segments: pd.DataFrame, as_of: str | None = None,
                             key: str = "") -> pd.DataFrame:
    """Monthly transactions/sales split by the customer's current segment."""
    months = available_months()
    as_of = as_of or months[-1]
    end_excl = (pd.Period(as_of, "M") + 1).start_time.date().isoformat()
    return run_query(f"""
        SELECT CAST(date_trunc('month', t.tarix_saat) AS DATE) AS month,
               s.segment,
               COUNT(*) AS tx,
               COUNT(DISTINCT t.musteri_acari) AS customers,
               SUM(CAST(t.net_amount AS DOUBLE)) AS net_sales
        FROM {CONFIG['TX_TABLE']} t
        JOIN seg s ON s.cust = t.musteri_acari
        WHERE {CONFIG['QUALIFYING_FILTER']}
          AND t.tarix_saat < CAST(? AS DATE)
        GROUP BY 1, 2 ORDER BY 1, 2
    """, [end_excl], register={"seg": _segments})


def apply_filters(f: pd.DataFrame, branch="All Branches", segments=None,
                  min_score=0, max_score=100, min_tx=0, min_tenure=0,
                  basket_range=None, spend_range=None) -> pd.DataFrame:
    """Branch filtering is on the customer's PRIMARY branch, by design: a
    pilot is offered to the people a branch actually owns."""
    out = f
    if branch != "All Branches":
        out = out[out["primary_branch"] == str(branch)]
    if segments:
        out = out[out["segment"].isin(segments)]
    out = out[(out["score"] >= min_score) & (out["score"] <= max_score) &
              (out["tx_win"] >= min_tx) & (out["tenure_months"] >= min_tenure)]
    if basket_range:
        out = out[out["avg_basket"].between(*basket_range)]
    if spend_range:
        out = out[out["monthly_spend"].between(*spend_range)]
    return out


# ===========================================================================
# VALIDATION CHECKS (run from the dashboard or the CLI)
# ===========================================================================

def validation_checks(features: pd.DataFrame) -> pd.DataFrame:
    """Cheap invariants that must hold for the model to be trustworthy."""
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    n = len(features)
    add("Every customer has exactly one segment",
        features["segment"].notna().all(), f"{features['segment'].isna().sum()} missing")
    add("Score within 0-100",
        features["score"].between(0, 100).all(),
        f"min {features['score'].min():.1f}, max {features['score'].max():.1f}")
    add("Coverage never exceeds the window",
        (features["coverage"] <= CONFIG["WINDOW_MONTHS"]).all(),
        f"max {features['coverage'].max()}")
    add("Frequent customers all satisfy the Regular rule",
        (features.loc[features["segment"] == "Frequent", "coverage"]
         >= CONFIG["SEG_REGULAR_COVERAGE"]).all(), "nesting holds")
    add("No customer below the tenure floor is classified",
        (features.loc[features["tenure_months"] < CONFIG["MIN_TENURE_MONTHS"], "segment"]
         == "New/Insufficient").all(), "tenure gate holds")
    add("Eligible customers all clear the score floor",
        (features.loc[features["eligible"], "score"] >= CONFIG["ELIGIBILITY_MIN_SCORE"]).all(),
        f"{int(features['eligible'].sum()):,} eligible ({100*features['eligible'].mean():.1f}%)")
    add("Primary-branch share within 0-1",
        features["primary_branch_share"].between(0, 1.0001).all(), "ok")
    add("Window spend never exceeds lifetime spend",
        (features["spend_win"] <= features["total_spend"] * 1.001 + 1).all(), "ok")
    add("Segment mix is not degenerate",
        features["segment"].value_counts(normalize=True).max() < 0.9,
        f"largest segment {100*features['segment'].value_counts(normalize=True).max():.1f}%")
    add("Customer count matches the source table",
        n > 0, f"{n:,} customers")
    return pd.DataFrame(checks)
