"""
============================================================================
experiment.py  --  PHASE 1 DIAGNOSTICS
Pharmacy customer analytics / small-loan pilot project
============================================================================

PURPOSE
-------
This script does NOT decide anything. It only measures the dataset so that
segmentation thresholds, behavioural-score transformations/weights and the
branch pilot-selection rules can be derived from evidence instead of guesses.

It reads the same DuckDB database used by the existing Streamlit dashboard
(tables: clean_transactions, customer_segments) and produces:

    experiment_output/
        report_phase1.md          <-- compact report, PASTE THIS BACK
        customer_features.csv     <-- one row per customer (reused in Phase 2)
        customer_month.csv        <-- customer x month panel
        branch_metrics.csv        <-- one row per branch
        branch_month.csv          <-- branch x month panel
        segmentation_grid.csv     <-- threshold sensitivity grid
        cohort_retention.csv      <-- acquisition cohort retention matrix

HOW TO RUN
----------
    pip install duckdb pandas numpy
    python experiment.py                       # auto-detects data/pharmacy.duckdb
    python experiment.py --db /path/pharmacy.duckdb
    python experiment.py --out ./experiment_output

The report is written in Markdown and is intentionally compact (aggregates,
percentiles and counts only -- no raw customer records except anonymisable
top-N tables). Review it before sharing.

WHAT IS DELIBERATELY *NOT* DONE HERE
------------------------------------
No segment labels, no scores, no weights, no branch recommendation. Those are
Phase 2, after the numbers below are known.
============================================================================
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover
    sys.exit("duckdb is not installed. Run:  pip install duckdb pandas numpy")


# ===========================================================================
# CONFIGURATION  (everything tunable lives here)
# ===========================================================================

CONFIG = {
    # ---- source ----------------------------------------------------------
    "TX_TABLE": "clean_transactions",
    "SEG_TABLE": "customer_segments",

    "COL_CUSTOMER": "musteri_acari",
    "COL_BRANCH": "aptek_id",
    "COL_TIMESTAMP": "tarix_saat",
    "COL_NET": "net_amount",
    "COL_GROSS": "valid_gross_amount",
    "COL_DISCOUNT": "valid_discount_amount",
    "COL_USABLE": "usable_for_sales",     # 1 = real sale
    "COL_RETURN": "qaytarma",             # 1 = return / refund
    "COL_RX": "resepti_var",              # 1 = prescription
    "COL_CUST_TYPE": "customer_type",     # in customer_segments (Pension / Non-Pension)

    # ---- what counts as a "qualifying transaction" -----------------------
    # A qualifying transaction is the unit used for all frequency logic.
    "QUALIFYING_FILTER": "usable_for_sales = 1",

    # Customer keys that are placeholders rather than real people
    # (fill in after looking at section 3 of the report, then re-run).
    "EXCLUDE_CUSTOMER_KEYS": [],

    # ---- diagnostics parameters -----------------------------------------
    "PERCENTILES": [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    "WINDOWS": [3, 6, 12],                 # observation windows L (months)
    "MIN_TX_PER_MONTH": [1, 2, 3, 4],      # candidate k thresholds
    "COVERAGE_LEVELS": [0.5, 0.67, 0.8, 1.0],   # share of the L months with >= k tx
    "TOP_BRANCHES_IN_REPORT": 25,
    "TOP_CUSTOMERS_IN_REPORT": 15,
    "MAX_COHORTS_IN_REPORT": 18,
}


# ===========================================================================
# SMALL UTILITIES
# ===========================================================================

class Report:
    """Collects markdown text, prints a condensed version to the console."""

    def __init__(self):
        self.parts: list[str] = []

    def h1(self, text):
        self._add(f"\n# {text}\n", console=f"\n{'=' * 78}\n{text}\n{'=' * 78}")

    def h2(self, text):
        self._add(f"\n## {text}\n", console=f"\n--- {text} ---")

    def text(self, text):
        self._add(str(text) + "\n")

    def bullet(self, text):
        self._add(f"- {text}")

    def kv(self, key, value):
        self._add(f"- **{key}**: {value}")

    def code(self, text):
        self._add(f"```\n{text}\n```")

    def df(self, df: pd.DataFrame, max_rows: int = 60, floatfmt: str = "%.3f"):
        if df is None or len(df) == 0:
            self._add("_(empty)_")
            return
        shown = df.head(max_rows)
        with pd.option_context("display.max_columns", None,
                               "display.width", 250,
                               "display.float_format", lambda v: floatfmt % v):
            body = shown.to_string(index=False)
        note = "" if len(df) <= max_rows else f"\n_({len(df)} rows total, {max_rows} shown)_"
        self._add(f"```\n{body}\n```{note}")

    def warn(self, text):
        self._add(f"> **CHECK:** {text}")

    def _add(self, md, console=None):
        self.parts.append(md)
        print(console if console is not None else md)

    def save(self, path: Path):
        path.write_text("\n".join(self.parts), encoding="utf-8")


def describe(series: pd.Series, percentiles) -> dict:
    """Percentile profile of one numeric feature."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    out = {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
    }
    for p in percentiles:
        out[f"p{int(round(p * 100))}"] = float(s.quantile(p))
    out["max"] = float(s.max())
    out["zeros_%"] = float((s == 0).mean() * 100)
    return out


def profile_table(df: pd.DataFrame, cols, percentiles) -> pd.DataFrame:
    rows = {}
    for c in cols:
        if c in df.columns:
            d = describe(df[c], percentiles)
            if d:
                rows[c] = d
    return pd.DataFrame(rows).T.reset_index().rename(columns={"index": "feature"})


def month_index(s: pd.Series) -> pd.Series:
    """Calendar months as a contiguous integer index (for gap arithmetic)."""
    p = pd.to_datetime(s).dt.to_period("M")
    return p.dt.year * 12 + p.dt.month


def safe(section_name, report: Report):
    """Decorator-ish helper: run a section, never let it kill the whole run."""
    def wrapper(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            report.h2(f"[FAILED] {section_name}")
            report.code(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}")
            return None
    return wrapper


# ===========================================================================
# DATABASE ACCESS
# ===========================================================================

def find_db(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            sys.exit(f"Database not found: {p}")
        return p

    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / "pharmacy.duckdb",
        here.parent / "data" / "pharmacy.duckdb",
        here.parent.parent / "data" / "pharmacy.duckdb",
        Path.cwd() / "data" / "pharmacy.duckdb",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    sys.exit(
        "Could not auto-detect pharmacy.duckdb. Pass it explicitly:\n"
        "    python experiment.py --db /full/path/to/pharmacy.duckdb"
    )


def q(con, sql: str, params=None) -> pd.DataFrame:
    return con.execute(sql, params).fetchdf() if params else con.execute(sql).fetchdf()


def table_columns(con, table: str) -> list[str]:
    try:
        return q(con, f"PRAGMA table_info('{table}')")["name"].tolist()
    except Exception:  # noqa: BLE001
        return []


# ===========================================================================
# SECTION 0 - SCHEMA
# ===========================================================================

def section_schema(con, rep: Report) -> dict:
    rep.h1("0. SCHEMA AND DATA INVENTORY")

    tables = q(con, "SELECT table_name FROM information_schema.tables ORDER BY 1")
    rep.h2("0.1 Tables in the database")
    rep.df(tables, 50)

    info = {}
    for t in [CONFIG["TX_TABLE"], CONFIG["SEG_TABLE"]]:
        cols = table_columns(con, t)
        info[t] = cols
        rep.h2(f"0.2 Columns of `{t}`")
        if not cols:
            rep.warn(f"Table `{t}` not found or unreadable.")
            continue
        rep.df(q(con, f"PRAGMA table_info('{t}')")[["name", "type", "notnull"]], 80)
        rep.kv(f"{t} row count", f"{q(con, f'SELECT COUNT(*) c FROM {t}')['c'][0]:,}")

    tx_cols = info.get(CONFIG["TX_TABLE"], [])
    flags = {
        "has_usable": CONFIG["COL_USABLE"] in tx_cols,
        "has_return": CONFIG["COL_RETURN"] in tx_cols,
        "has_rx": CONFIG["COL_RX"] in tx_cols,
        "has_gross": CONFIG["COL_GROSS"] in tx_cols,
        "has_discount": CONFIG["COL_DISCOUNT"] in tx_cols,
        "has_segments": bool(info.get(CONFIG["SEG_TABLE"])),
        "has_cust_type": CONFIG["COL_CUST_TYPE"] in info.get(CONFIG["SEG_TABLE"], []),
        "tx_cols": tx_cols,
    }

    rep.h2("0.3 Optional columns detected")
    for k, v in flags.items():
        if k != "tx_cols":
            rep.kv(k, v)

    rep.h2("0.4 Sample rows (5) - REVIEW BEFORE SHARING")
    try:
        rep.df(q(con, f"SELECT * FROM {CONFIG['TX_TABLE']} LIMIT 5"), 5)
    except Exception as exc:  # noqa: BLE001
        rep.code(str(exc))

    return flags


# ===========================================================================
# SECTION 1 - VOLUME, QUALITY, TIME COVERAGE
# ===========================================================================

def build_views(con, flags):
    """Create the two working views used by every later section."""
    tx = CONFIG["TX_TABLE"]
    cust = f"NULLIF(TRIM(CAST({CONFIG['COL_CUSTOMER']} AS VARCHAR)), '')"
    ts = CONFIG["COL_TIMESTAMP"]

    ret_expr = (f"CAST(COALESCE({CONFIG['COL_RETURN']}, 0) AS INTEGER)"
                if flags["has_return"] else "0")
    rx_expr = (f"CAST(COALESCE({CONFIG['COL_RX']}, 0) AS INTEGER)"
               if flags["has_rx"] else "0")
    usable_expr = (f"CAST(COALESCE({CONFIG['COL_USABLE']}, 0) AS INTEGER)"
                   if flags["has_usable"] else "1")

    exclusions = ""
    if CONFIG["EXCLUDE_CUSTOMER_KEYS"]:
        keys = ", ".join("'" + str(k).replace("'", "''") + "'"
                         for k in CONFIG["EXCLUDE_CUSTOMER_KEYS"])
        exclusions = f"AND {cust} NOT IN ({keys})"

    # every row, customer key normalised
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW all_rows AS
        SELECT
            {cust}                                   AS cust,
            CAST({CONFIG['COL_BRANCH']} AS VARCHAR)  AS branch,
            CAST({ts} AS TIMESTAMP)                  AS ts,
            CAST({ts} AS DATE)                       AS d,
            CAST(date_trunc('month', {ts}) AS DATE)  AS ym,
            CAST({CONFIG['COL_NET']} AS DOUBLE)      AS net_amount,
            {usable_expr}                            AS usable,
            {ret_expr}                               AS is_return,
            {rx_expr}                                AS is_rx
        FROM {tx}
    """)

    # analysis base: real sales, identified customers only
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW tx AS
        SELECT * FROM all_rows
        WHERE usable = 1
          AND cust IS NOT NULL
          {exclusions}
    """)


def section_quality(con, rep: Report, flags) -> dict:
    rep.h1("1. VOLUME, DATA QUALITY, TIME COVERAGE")

    tot = q(con, """
        SELECT
            COUNT(*)                                   AS rows_total,
            SUM(usable)                                AS rows_usable,
            SUM(is_return)                             AS rows_return,
            SUM(CASE WHEN cust IS NULL THEN 1 ELSE 0 END) AS rows_no_customer,
            SUM(CASE WHEN branch IS NULL THEN 1 ELSE 0 END) AS rows_no_branch,
            SUM(CASE WHEN ts IS NULL THEN 1 ELSE 0 END)   AS rows_no_timestamp,
            SUM(CASE WHEN net_amount IS NULL THEN 1 ELSE 0 END) AS rows_null_amount,
            SUM(CASE WHEN net_amount < 0 THEN 1 ELSE 0 END)  AS rows_negative_amount,
            SUM(CASE WHEN net_amount = 0 THEN 1 ELSE 0 END)  AS rows_zero_amount,
            COUNT(DISTINCT cust)                       AS distinct_customers_all,
            COUNT(DISTINCT branch)                     AS distinct_branches_all,
            MIN(ts)                                    AS first_ts,
            MAX(ts)                                    AS last_ts
        FROM all_rows
    """).iloc[0]

    rep.h2("1.1 Row-level inventory (whole table)")
    for k, v in tot.items():
        rep.kv(k, f"{v:,}" if isinstance(v, (int, np.integer)) else v)

    n = int(tot["rows_total"])
    if n:
        rep.kv("share of rows WITHOUT a customer key (%)",
               round(100 * int(tot["rows_no_customer"]) / n, 2))
        rep.warn("If this share is high, the addressable pilot population is "
                 "limited to identified customers only - note it for sizing.")

    rep.h2("1.2 Qualifying transactions (usable_for_sales = 1 AND customer key present)")
    base = q(con, """
        SELECT COUNT(*) AS qualifying_tx,
               COUNT(DISTINCT cust) AS customers,
               COUNT(DISTINCT branch) AS branches,
               COUNT(DISTINCT ym) AS months,
               MIN(ym) AS first_month, MAX(ym) AS last_month,
               SUM(net_amount) AS net_sales,
               AVG(net_amount) AS avg_basket,
               median(net_amount) AS median_basket
        FROM tx
    """).iloc[0]
    for k, v in base.items():
        rep.kv(k, f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer)) else v)

    rep.h2("1.3 Duplicate suspicion (same customer + timestamp + branch + amount)")
    dup = q(con, """
        SELECT COUNT(*) AS duplicate_groups, COALESCE(SUM(c - 1), 0) AS extra_rows
        FROM (
            SELECT cust, ts, branch, net_amount, COUNT(*) AS c
            FROM all_rows GROUP BY 1,2,3,4 HAVING COUNT(*) > 1
        )
    """).iloc[0]
    rep.kv("duplicate groups", f"{dup['duplicate_groups']:,}")
    rep.kv("extra rows implied", f"{dup['extra_rows']:,}")

    rep.h2("1.4 Monthly coverage (qualifying transactions)")
    monthly = q(con, """
        SELECT ym AS month,
               COUNT(*) AS tx,
               COUNT(DISTINCT cust) AS customers,
               COUNT(DISTINCT branch) AS branches,
               SUM(net_amount) AS net_sales,
               AVG(net_amount) AS avg_basket
        FROM tx GROUP BY 1 ORDER BY 1
    """)
    rep.df(monthly, 40)
    rep.warn("Check for partial first/last months and for months where the "
             "branch count jumps - both distort frequency-based segmentation.")

    rep.h2("1.5 Basket value distribution (qualifying transactions)")
    basket = q(con, f"""
        SELECT {', '.join(f'quantile_cont(net_amount, {p}) AS p{int(p*100)}'
                          for p in CONFIG['PERCENTILES'])},
               AVG(net_amount) AS mean, MAX(net_amount) AS max
        FROM tx
    """)
    rep.df(basket.T.reset_index().rename(columns={"index": "stat", 0: "value"}), 20)

    if flags["has_return"]:
        rep.h2("1.6 Returns / refunds")
        ret = q(con, """
            SELECT SUM(is_return) AS return_rows,
                   COUNT(*) AS all_rows,
                   100.0 * SUM(is_return) / NULLIF(COUNT(*), 0) AS return_rate_pct,
                   COUNT(DISTINCT CASE WHEN is_return = 1 THEN cust END) AS customers_with_returns
            FROM all_rows
        """)
        rep.df(ret, 5)

    return {"months": int(base["months"]),
            "first_month": base["first_month"],
            "last_month": base["last_month"]}


# ===========================================================================
# SECTION 2 - CUSTOMER KEY SANITY / OUTLIERS
# ===========================================================================

def section_customer_keys(con, rep: Report):
    rep.h1("2. CUSTOMER KEY SANITY AND OUTLIERS")

    rep.h2("2.1 Most frequent customer keys (placeholder / B2B detection)")
    top = q(con, f"""
        SELECT cust,
               COUNT(*) AS tx,
               COUNT(DISTINCT branch) AS branches,
               COUNT(DISTINCT ym) AS active_months,
               SUM(net_amount) AS net_sales,
               AVG(net_amount) AS avg_basket
        FROM tx GROUP BY 1 ORDER BY tx DESC LIMIT {CONFIG['TOP_CUSTOMERS_IN_REPORT']}
    """)
    total_tx = q(con, "SELECT COUNT(*) c FROM tx")["c"][0]
    top["share_of_all_tx_%"] = 100 * top["tx"] / max(total_tx, 1)
    rep.df(top, CONFIG["TOP_CUSTOMERS_IN_REPORT"])
    rep.warn("A key with an implausible transaction count, or one active in "
             "very many branches, is probably a walk-in/anonymous placeholder "
             "or a wholesale account. Add it to CONFIG['EXCLUDE_CUSTOMER_KEYS'] "
             "and re-run before Phase 2.")

    rep.h2("2.2 Largest customers by spend")
    tops = q(con, f"""
        SELECT cust, COUNT(*) AS tx, SUM(net_amount) AS net_sales,
               AVG(net_amount) AS avg_basket, COUNT(DISTINCT ym) AS active_months
        FROM tx GROUP BY 1 ORDER BY net_sales DESC LIMIT {CONFIG['TOP_CUSTOMERS_IN_REPORT']}
    """)
    rep.df(tops, CONFIG["TOP_CUSTOMERS_IN_REPORT"])

    rep.h2("2.3 Concentration (Pareto)")
    conc = q(con, """
        WITH c AS (SELECT cust, SUM(net_amount) s FROM tx GROUP BY 1),
        r AS (SELECT s, ROW_NUMBER() OVER (ORDER BY s DESC) rn, COUNT(*) OVER () n,
                     SUM(s) OVER () tot FROM c)
        SELECT
            100.0 * SUM(CASE WHEN rn <= 0.01*n THEN s ELSE 0 END)/MAX(tot) AS top1pct_share,
            100.0 * SUM(CASE WHEN rn <= 0.05*n THEN s ELSE 0 END)/MAX(tot) AS top5pct_share,
            100.0 * SUM(CASE WHEN rn <= 0.10*n THEN s ELSE 0 END)/MAX(tot) AS top10pct_share,
            100.0 * SUM(CASE WHEN rn <= 0.20*n THEN s ELSE 0 END)/MAX(tot) AS top20pct_share
        FROM r
    """)
    rep.df(conc, 3)


# ===========================================================================
# SECTION 3 - CUSTOMER FEATURE TABLE
# ===========================================================================

def build_customer_features(con, rep: Report, flags) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep.h1("3. CUSTOMER-LEVEL FEATURE CONSTRUCTION")

    # ---- customer x month panel ------------------------------------------
    cust_month = q(con, """
        SELECT cust, ym,
               COUNT(*) AS n_tx,
               COUNT(DISTINCT d) AS n_days,
               SUM(net_amount) AS spend
        FROM tx GROUP BY 1, 2
    """)

    # ---- customer base ---------------------------------------------------
    base = q(con, """
        SELECT cust,
               MIN(ts) AS first_ts, MAX(ts) AS last_ts,
               MIN(ym) AS first_ym, MAX(ym) AS last_ym,
               COUNT(*) AS n_tx,
               COUNT(DISTINCT d) AS n_days,
               COUNT(DISTINCT ym) AS active_months,
               COUNT(DISTINCT branch) AS n_branches,
               SUM(net_amount) AS total_spend,
               AVG(net_amount) AS avg_basket,
               median(net_amount) AS median_basket,
               stddev_samp(net_amount) AS sd_basket,
               MIN(net_amount) AS min_basket,
               MAX(net_amount) AS max_basket
        FROM tx GROUP BY 1
    """)

    # ---- inter-purchase gaps (days) --------------------------------------
    try:
        gaps = q(con, """
            WITH days AS (SELECT DISTINCT cust, d FROM tx),
            g AS (
                SELECT cust,
                       date_diff('day', LAG(d) OVER (PARTITION BY cust ORDER BY d), d) AS gap
                FROM days
            )
            SELECT cust,
                   AVG(gap) AS mean_gap_days,
                   median(gap) AS median_gap_days,
                   MAX(gap) AS max_gap_days,
                   stddev_samp(gap) AS sd_gap_days
            FROM g WHERE gap IS NOT NULL GROUP BY 1
        """)
    except Exception:  # noqa: BLE001  -- pandas fallback if date_diff unsupported
        days = q(con, "SELECT DISTINCT cust, d FROM tx ORDER BY cust, d")
        days["d"] = pd.to_datetime(days["d"])
        days["gap"] = days.groupby("cust")["d"].diff().dt.days
        gaps = (days.dropna(subset=["gap"]).groupby("cust")["gap"]
                .agg(mean_gap_days="mean", median_gap_days="median",
                     max_gap_days="max", sd_gap_days="std").reset_index())

    # ---- branch loyalty ---------------------------------------------------
    cb = q(con, """
        SELECT cust, branch, COUNT(*) AS n_tx, SUM(net_amount) AS spend
        FROM tx GROUP BY 1, 2
    """)
    cb = cb.sort_values(["cust", "n_tx", "spend"], ascending=[True, False, False])
    primary = cb.groupby("cust", as_index=False).first()[["cust", "branch", "n_tx"]]
    primary.columns = ["cust", "primary_branch", "primary_branch_tx"]

    # ---- returns / prescription behaviour ---------------------------------
    beh = q(con, """
        SELECT cust,
               SUM(CASE WHEN is_return = 1 THEN 1 ELSE 0 END) AS return_tx,
               COUNT(*) AS all_tx,
               AVG(CAST(is_rx AS DOUBLE)) AS rx_share
        FROM all_rows WHERE cust IS NOT NULL GROUP BY 1
    """)

    f = (base
         .merge(gaps, on="cust", how="left")
         .merge(primary, on="cust", how="left")
         .merge(beh[["cust", "return_tx", "rx_share"]], on="cust", how="left"))

    # ---- pension flag -----------------------------------------------------
    if flags["has_cust_type"]:
        seg = q(con, f"""
            SELECT NULLIF(TRIM(CAST({CONFIG['COL_CUSTOMER']} AS VARCHAR)), '') AS cust,
                   {CONFIG['COL_CUST_TYPE']} AS customer_type
            FROM {CONFIG['SEG_TABLE']}
        """).drop_duplicates("cust")
        f = f.merge(seg, on="cust", how="left")

    # ---- derived time features -------------------------------------------
    for c in ["first_ts", "last_ts"]:
        f[c] = pd.to_datetime(f[c])
    f["first_ym"] = pd.to_datetime(f["first_ym"])
    f["last_ym"] = pd.to_datetime(f["last_ym"])

    dataset_end = f["last_ts"].max()
    dataset_end_mi = month_index(pd.Series([dataset_end]))[0]

    f["first_mi"] = month_index(f["first_ym"])
    f["last_mi"] = month_index(f["last_ym"])

    f["observed_months"] = f["last_mi"] - f["first_mi"] + 1          # first->last purchase
    f["window_months"] = dataset_end_mi - f["first_mi"] + 1          # first purchase->today
    f["tenure_days"] = (f["last_ts"] - f["first_ts"]).dt.days
    f["recency_days"] = (dataset_end - f["last_ts"]).dt.days

    f["active_ratio_observed"] = f["active_months"] / f["observed_months"]
    f["active_ratio_window"] = f["active_months"] / f["window_months"]
    f["tx_per_active_month"] = f["n_tx"] / f["active_months"]
    f["tx_per_observed_month"] = f["n_tx"] / f["observed_months"]
    f["tx_per_window_month"] = f["n_tx"] / f["window_months"]
    f["spend_per_active_month"] = f["total_spend"] / f["active_months"]
    f["spend_per_window_month"] = f["total_spend"] / f["window_months"]
    f["return_rate"] = (f["return_tx"] / f["n_tx"]).clip(upper=1)
    f["primary_branch_share"] = f["primary_branch_tx"] / f["n_tx"]

    # ---- monthly regularity from the panel --------------------------------
    cm = cust_month.copy()
    cm["ym"] = pd.to_datetime(cm["ym"])
    cm["mi"] = month_index(cm["ym"])
    cm = cm.sort_values(["cust", "mi"])

    monthly_stats = cm.groupby("cust").agg(
        tx_month_mean=("n_tx", "mean"),
        tx_month_sd=("n_tx", "std"),
        tx_month_min=("n_tx", "min"),
        tx_month_max=("n_tx", "max"),
        spend_month_mean=("spend", "mean"),
        spend_month_sd=("spend", "std"),
        spend_month_median=("spend", "median"),
    ).reset_index()

    cm["mi_gap"] = cm.groupby("cust")["mi"].diff()
    month_gaps = (cm.groupby("cust")["mi_gap"]
                  .agg(max_month_gap="max", mean_month_gap="mean").reset_index())

    f = f.merge(monthly_stats, on="cust", how="left").merge(month_gaps, on="cust", how="left")
    f["tx_month_cv"] = f["tx_month_sd"] / f["tx_month_mean"].replace(0, np.nan)
    f["spend_month_cv"] = f["spend_month_sd"] / f["spend_month_mean"].replace(0, np.nan)

    rep.kv("customers in feature table", f"{len(f):,}")
    rep.kv("dataset end (last transaction)", dataset_end)
    rep.kv("feature columns", len(f.columns))

    rep.h2("3.1 Missing values in engineered features (%)")
    miss = (f.isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False).reset_index()
    miss.columns = ["feature", "missing_%"]
    rep.df(miss, 40)
    rep.text("_Note: gap features are missing by construction for single-purchase "
             "customers; sd/CV features are missing for single-month customers._")

    return f, cm


# ===========================================================================
# SECTION 4 - DISTRIBUTIONS THAT DRIVE THE THRESHOLDS
# ===========================================================================

def section_distributions(f: pd.DataFrame, rep: Report):
    rep.h1("4. DISTRIBUTIONS OF SEGMENTATION-RELEVANT FEATURES")

    pcts = CONFIG["PERCENTILES"]

    rep.h2("4.1 Percentile profile")
    cols = ["n_tx", "n_days", "active_months", "observed_months", "window_months",
            "tenure_days", "recency_days", "active_ratio_observed", "active_ratio_window",
            "tx_per_active_month", "tx_per_observed_month", "tx_per_window_month",
            "tx_month_mean", "tx_month_max", "tx_month_cv",
            "avg_basket", "median_basket", "total_spend",
            "spend_per_active_month", "spend_per_window_month", "spend_month_cv",
            "mean_gap_days", "median_gap_days", "max_gap_days",
            "max_month_gap", "n_branches", "primary_branch_share", "return_rate"]
    rep.df(profile_table(f, cols, pcts), 40, floatfmt="%.2f")

    rep.h2("4.2 History length - how many customers can even be classified?")
    hist = (f["window_months"].value_counts().sort_index().reset_index())
    hist.columns = ["window_months", "customers"]
    hist["%"] = 100 * hist["customers"] / len(f)
    hist["cum_%"] = hist["%"].cumsum()
    rep.df(hist, 40, floatfmt="%.2f")
    rep.warn("Customers with window_months = 1 cannot be judged on recurrence "
             "and belong in 'New / Insufficient history' no matter their frequency.")

    rep.h2("4.3 Observed active months (months with >=1 qualifying transaction)")
    am = f["active_months"].value_counts().sort_index().reset_index()
    am.columns = ["active_months", "customers"]
    am["%"] = 100 * am["customers"] / len(f)
    am["cum_%"] = am["%"].cumsum()
    rep.df(am, 40, floatfmt="%.2f")

    rep.h2("4.4 Transactions per customer (frequency backbone)")
    tx = f["n_tx"].clip(upper=30).value_counts().sort_index().reset_index()
    tx.columns = ["n_tx (capped at 30)", "customers"]
    tx["%"] = 100 * tx["customers"] / len(f)
    tx["cum_%"] = tx["%"].cumsum()
    rep.df(tx, 35, floatfmt="%.2f")

    rep.h2("4.5 Average transactions per ACTIVE month - banding check")
    bands = pd.cut(f["tx_per_active_month"],
                   [0, 1.0001, 2, 3, 4, 6, 10, np.inf],
                   labels=["<=1", "1-2", "2-3", "3-4", "4-6", "6-10", "10+"])
    b = bands.value_counts().sort_index().reset_index()
    b.columns = ["tx_per_active_month", "customers"]
    b["%"] = 100 * b["customers"] / len(f)
    rep.df(b, 15, floatfmt="%.2f")

    rep.h2("4.6 Active-month ratio (activity coverage since acquisition)")
    rb = pd.cut(f["active_ratio_window"], [0, .2, .4, .5, .6, .8, .999, 1.0],
                labels=["0-20%", "20-40%", "40-50%", "50-60%", "60-80%", "80-99%", "100%"],
                include_lowest=True)
    r = rb.value_counts().sort_index().reset_index()
    r.columns = ["active_ratio_window", "customers"]
    r["%"] = 100 * r["customers"] / len(f)
    rep.df(r, 15, floatfmt="%.2f")

    rep.h2("4.7 Recency")
    rec = pd.cut(f["recency_days"], [-1, 30, 60, 90, 180, 365, np.inf],
                 labels=["0-30d", "31-60d", "61-90d", "91-180d", "181-365d", "365d+"])
    rr = rec.value_counts().sort_index().reset_index()
    rr.columns = ["recency", "customers"]
    rr["%"] = 100 * rr["customers"] / len(f)
    rep.df(rr, 10, floatfmt="%.2f")

    rep.h2("4.8 Feature correlation (Spearman) - guards against double counting in the score")
    score_inputs = ["n_tx", "active_months", "active_ratio_window", "tx_per_active_month",
                    "tx_per_window_month", "avg_basket", "spend_per_window_month",
                    "tenure_days", "recency_days", "tx_month_cv", "max_gap_days",
                    "primary_branch_share"]
    avail = [c for c in score_inputs if c in f.columns]
    corr = f[avail].corr(method="spearman").round(2).reset_index()
    rep.df(corr, 20, floatfmt="%.2f")
    rep.warn("Pairs above ~|0.8| should not both receive full weight in the "
             "behavioural score.")

    if "customer_type" in f.columns:
        rep.h2("4.9 Pension vs non-pension behaviour")
        g = f.groupby("customer_type").agg(
            customers=("cust", "count"),
            median_n_tx=("n_tx", "median"),
            median_active_months=("active_months", "median"),
            median_basket=("avg_basket", "median"),
            median_monthly_spend=("spend_per_window_month", "median"),
            median_recency=("recency_days", "median"),
        ).reset_index()
        rep.df(g, 10, floatfmt="%.2f")


# ===========================================================================
# SECTION 5 - THRESHOLD SENSITIVITY GRID
# ===========================================================================

def section_segmentation_grid(cm: pd.DataFrame, f: pd.DataFrame,
                              rep: Report, n_months: int) -> pd.DataFrame:
    """
    For each window L, minimum monthly transactions k, and coverage c:
    how many customers satisfy 'at least k transactions in at least c*L of the
    last L months', counting only customers whose history covers the window.
    This is the empirical basis for the Frequent / Regular / Semi-Regular rules.
    """
    rep.h1("5. SEGMENTATION THRESHOLD SENSITIVITY GRID")

    max_mi = cm["mi"].max()
    rows = []
    dists = []

    for L in [w for w in CONFIG["WINDOWS"] if w <= n_months]:
        start_mi = max_mi - L + 1
        eligible = f.loc[f["first_mi"] <= start_mi, "cust"]
        n_elig = len(eligible)
        win = cm[cm["mi"] >= start_mi]

        for k in CONFIG["MIN_TX_PER_MONTH"]:
            hit = (win.assign(ok=(win["n_tx"] >= k).astype(int))
                      .groupby("cust")["ok"].sum())
            hit = hit.reindex(eligible).fillna(0)

            d = hit.value_counts().sort_index()
            for months_hit, cnt in d.items():
                dists.append({"window_L": L, "min_tx_k": k,
                              "months_meeting_k": int(months_hit),
                              "customers": int(cnt),
                              "pct_of_eligible": 100 * cnt / max(n_elig, 1)})

            for c in CONFIG["COVERAGE_LEVELS"]:
                need = int(np.ceil(c * L))
                qualified = int((hit >= need).sum())
                rows.append({
                    "window_L": L,
                    "min_tx_per_month_k": k,
                    "coverage_c": c,
                    "months_required": need,
                    "customers_with_full_history": n_elig,
                    "qualifying_customers": qualified,
                    "pct_of_eligible": 100 * qualified / max(n_elig, 1),
                    "pct_of_all_customers": 100 * qualified / max(len(f), 1),
                })

    grid = pd.DataFrame(rows)
    dist = pd.DataFrame(dists)

    rep.h2("5.1 Qualifying-customer counts under every threshold combination")
    rep.text("Read as: with a window of L months, requiring at least k qualifying "
             "transactions in at least `months_required` of those months.")
    rep.df(grid, 200, floatfmt="%.2f")

    rep.h2("5.2 Distribution of 'how many of the last L months met the k threshold'")
    rep.df(dist, 200, floatfmt="%.2f")
    rep.warn("Look for a natural break: if the counts fall off a cliff between, "
             "say, k=2 and k=3, the strict '3 per month' definition may leave a "
             "pilot population too small to be commercially useful.")

    rep.h2("5.3 Population left after each history requirement")
    hist_rows = []
    for L in [w for w in CONFIG["WINDOWS"] if w <= n_months]:
        hist_rows.append({
            "min_history_months": L,
            "customers_with_history": int((f["window_months"] >= L).sum()),
            "pct_of_all": 100 * (f["window_months"] >= L).mean(),
            "their_share_of_tx_%": 100 * f.loc[f["window_months"] >= L, "n_tx"].sum() / f["n_tx"].sum(),
            "their_share_of_spend_%": 100 * f.loc[f["window_months"] >= L, "total_spend"].sum() / f["total_spend"].sum(),
        })
    rep.df(pd.DataFrame(hist_rows), 10, floatfmt="%.2f")

    return grid


# ===========================================================================
# SECTION 6 - BRANCH METRICS
# ===========================================================================

def section_branches(con, f: pd.DataFrame, cm: pd.DataFrame,
                     rep: Report, flags) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep.h1("6. BRANCH-LEVEL DIAGNOSTICS")

    branch = q(con, """
        SELECT branch,
               COUNT(*) AS tx,
               COUNT(DISTINCT cust) AS customers,
               COUNT(DISTINCT ym) AS active_months,
               MIN(ym) AS first_month, MAX(ym) AS last_month,
               SUM(net_amount) AS net_sales,
               AVG(net_amount) AS avg_basket,
               median(net_amount) AS median_basket
        FROM tx GROUP BY 1
    """)

    completeness = q(con, """
        SELECT branch,
               COUNT(*) AS rows_all,
               SUM(usable) AS rows_usable,
               SUM(is_return) AS rows_return,
               100.0 * SUM(CASE WHEN cust IS NULL THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0)
                   AS pct_rows_without_customer,
               100.0 * SUM(is_return) / NULLIF(COUNT(*),0) AS return_rate_pct,
               AVG(CAST(is_rx AS DOUBLE)) * 100 AS rx_share_pct
        FROM all_rows GROUP BY 1
    """)

    branch_month = q(con, """
        SELECT branch, ym AS month,
               COUNT(*) AS tx,
               COUNT(DISTINCT cust) AS customers,
               SUM(net_amount) AS net_sales
        FROM tx GROUP BY 1, 2 ORDER BY 1, 2
    """)

    stab = branch_month.groupby("branch").agg(
        months_present=("month", "nunique"),
        tx_month_mean=("tx", "mean"),
        tx_month_sd=("tx", "std"),
        cust_month_mean=("customers", "mean"),
        sales_month_sd=("net_sales", "std"),
        sales_month_mean=("net_sales", "mean"),
    ).reset_index()
    stab["tx_month_cv"] = stab["tx_month_sd"] / stab["tx_month_mean"]
    stab["sales_month_cv"] = stab["sales_month_sd"] / stab["sales_month_mean"]

    # loyalty: customers whose PRIMARY branch is this branch
    prim = (f.groupby("primary_branch")
              .agg(primary_customers=("cust", "count"),
                   primary_cust_median_tx=("n_tx", "median"),
                   primary_cust_median_basket=("avg_basket", "median"),
                   primary_cust_median_monthly_spend=("spend_per_window_month", "median"),
                   primary_cust_median_active_months=("active_months", "median"))
              .reset_index().rename(columns={"primary_branch": "branch"}))

    loyal = f.groupby("primary_branch")["primary_branch_share"].median().reset_index()
    loyal.columns = ["branch", "median_loyalty_share_of_primary_customers"]

    b = (branch.merge(completeness, on="branch", how="left")
                .merge(stab, on="branch", how="left")
                .merge(prim, on="branch", how="left")
                .merge(loyal, on="branch", how="left"))

    b["tx_per_customer"] = b["tx"] / b["customers"]
    b["customers_share_%"] = 100 * b["customers"] / b["customers"].sum()
    b["sales_share_%"] = 100 * b["net_sales"] / b["net_sales"].sum()
    b["primary_customer_ratio_%"] = 100 * b["primary_customers"] / b["customers"]

    # history depth available per branch (needed for scoring feasibility)
    depth = (f.groupby("primary_branch")
               .agg(pct_customers_with_6m=("window_months", lambda s: 100 * (s >= 6).mean()),
                    pct_customers_with_12m=("window_months", lambda s: 100 * (s >= 12).mean()))
               .reset_index().rename(columns={"primary_branch": "branch"}))
    b = b.merge(depth, on="branch", how="left")

    b = b.sort_values("net_sales", ascending=False)

    rep.h2("6.1 Branch table (sorted by net sales)")
    show = ["branch", "customers", "primary_customers", "primary_customer_ratio_%",
            "tx", "tx_per_customer", "net_sales", "avg_basket", "median_basket",
            "active_months", "tx_month_cv", "sales_month_cv",
            "median_loyalty_share_of_primary_customers",
            "pct_rows_without_customer", "return_rate_pct",
            "pct_customers_with_6m", "pct_customers_with_12m"]
    rep.df(b[[c for c in show if c in b.columns]],
           CONFIG["TOP_BRANCHES_IN_REPORT"], floatfmt="%.2f")
    rep.kv("total branches", len(b))
    rep.warn("`primary_customer_ratio_%` separates 'people who happened to shop "
             "here' from 'people for whom this is their pharmacy'. Only the "
             "second group is a realistic pilot population.")

    rep.h2("6.2 Branch size distribution")
    rep.df(profile_table(b, ["customers", "primary_customers", "tx", "net_sales",
                             "avg_basket", "tx_per_customer", "tx_month_cv",
                             "primary_customer_ratio_%"], CONFIG["PERCENTILES"]),
           20, floatfmt="%.2f")

    rep.h2("6.3 Branch continuity (branches not present in every month)")
    n_months_total = branch_month["month"].nunique()
    partial = b.loc[b["months_present"] < n_months_total,
                    ["branch", "months_present", "first_month", "last_month",
                     "customers", "net_sales"]]
    rep.kv("months in dataset", n_months_total)
    rep.kv("branches present in every month", int((b["months_present"] == n_months_total).sum()))
    rep.df(partial.sort_values("months_present"), 30, floatfmt="%.2f")
    rep.warn("Branches that opened or closed mid-period must not be ranked on "
             "the same growth/stability basis as full-history branches.")

    return b, branch_month


# ===========================================================================
# SECTION 7 - COHORT RETENTION
# ===========================================================================

def section_cohorts(cm: pd.DataFrame, f: pd.DataFrame, rep: Report) -> pd.DataFrame:
    rep.h1("7. ACQUISITION COHORT RETENTION")

    cm2 = cm.merge(f[["cust", "first_mi"]], on="cust", how="left")
    cm2["offset"] = cm2["mi"] - cm2["first_mi"]
    cohort_size = f.groupby("first_mi")["cust"].count()

    pivot = (cm2.groupby(["first_mi", "offset"])["cust"].nunique()
                .unstack(fill_value=0).sort_index())
    retention = pivot.div(cohort_size.reindex(pivot.index), axis=0) * 100

    labels = pd.PeriodIndex([pd.Period(year=(mi - 1) // 12, month=((mi - 1) % 12) + 1,
                                       freq="M") for mi in pivot.index]).astype(str)
    retention.index = labels
    retention.insert(0, "cohort_size", cohort_size.reindex(pivot.index).values)

    rep.h2("7.1 Retention by acquisition month (% of cohort active in month +N)")
    rep.df(retention.round(1).reset_index().rename(columns={"index": "cohort"}),
           CONFIG["MAX_COHORTS_IN_REPORT"], floatfmt="%.1f")
    rep.warn("Retention is the strongest single evidence of repayment-relevant "
             "stability; it also tells us how far back a reliable observation "
             "window can start.")

    return retention


# ===========================================================================
# SECTION 8 - PRELIMINARY ELIGIBILITY SIZING (counts only, no scoring)
# ===========================================================================

def section_sizing(f: pd.DataFrame, rep: Report, n_months: int):
    rep.h1("8. PRELIMINARY PILOT-POPULATION SIZING (counts only)")

    rep.text("These are raw feasibility counts under illustrative rule shapes. "
             "They are NOT the final segment definitions - they exist to show "
             "whether a given strictness leaves a workable pilot population.")

    combos = []
    for min_hist in [w for w in CONFIG["WINDOWS"] if w <= n_months]:
        for min_active_ratio in [0.5, 0.67, 0.8, 1.0]:
            for min_tx_pm in [1, 2, 3]:
                m = ((f["window_months"] >= min_hist)
                     & (f["active_ratio_window"] >= min_active_ratio)
                     & (f["tx_per_active_month"] >= min_tx_pm))
                sel = f.loc[m]
                combos.append({
                    "min_history_months": min_hist,
                    "min_active_ratio": min_active_ratio,
                    "min_tx_per_active_month": min_tx_pm,
                    "customers": len(sel),
                    "pct_of_all": 100 * len(sel) / len(f),
                    "median_monthly_spend": sel["spend_per_window_month"].median(),
                    "median_basket": sel["avg_basket"].median(),
                    "median_recency_days": sel["recency_days"].median(),
                    "branches_covered": sel["primary_branch"].nunique(),
                })
    sizing = pd.DataFrame(combos)
    rep.df(sizing, 200, floatfmt="%.2f")

    rep.h2("8.2 Where would those customers sit? (top branches under a mid-strict rule)")
    mid_hist = min(6, n_months)
    m = ((f["window_months"] >= mid_hist)
         & (f["active_ratio_window"] >= 0.67)
         & (f["tx_per_active_month"] >= 1))
    by_branch = (f.loc[m].groupby("primary_branch")
                 .agg(candidate_customers=("cust", "count"),
                      median_monthly_spend=("spend_per_window_month", "median"),
                      median_basket=("avg_basket", "median"))
                 .reset_index()
                 .sort_values("candidate_customers", ascending=False))
    total_by_branch = f.groupby("primary_branch")["cust"].count().rename("all_primary_customers")
    by_branch = by_branch.merge(total_by_branch, left_on="primary_branch", right_index=True, how="left")
    by_branch["candidate_rate_%"] = 100 * by_branch["candidate_customers"] / by_branch["all_primary_customers"]
    rep.df(by_branch, CONFIG["TOP_BRANCHES_IN_REPORT"], floatfmt="%.2f")
    rep.warn("Note whether the branch with the most candidates is the same as "
             "the branch with the highest candidate RATE - if not, the pilot "
             "ranking genuinely needs a weighted score rather than a sort.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Phase 1 diagnostics for the pharmacy loan-pilot analysis")
    ap.add_argument("--db", default=None, help="path to pharmacy.duckdb")
    ap.add_argument("--out", default="experiment_output", help="output directory")
    args = ap.parse_args()

    db_path = find_db(args.db)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = Report()
    rep.text(f"# PHASE 1 DIAGNOSTIC REPORT\n")
    rep.kv("generated", datetime.now().strftime("%Y-%m-%d %H:%M"))
    rep.kv("database", str(db_path))
    rep.kv("qualifying transaction filter", CONFIG["QUALIFYING_FILTER"])
    rep.kv("excluded customer keys", CONFIG["EXCLUDE_CUSTOMER_KEYS"] or "none")

    con = duckdb.connect(str(db_path), read_only=True)
    run = safe("section", rep)

    try:
        flags = section_schema(con, rep)
        if not flags:
            raise RuntimeError("schema inspection failed")

        build_views(con, flags)

        meta = run(section_quality, con, rep, flags) or {"months": 12}
        n_months = int(meta.get("months") or 12)

        run(section_customer_keys, con, rep)

        result = run(build_customer_features, con, rep, flags)
        if result is None:
            raise RuntimeError("customer feature construction failed - cannot continue")
        f, cm = result

        run(section_distributions, f, rep)
        grid = run(section_segmentation_grid, cm, f, rep, n_months)
        branch_out = run(section_branches, con, f, cm, rep, flags)
        retention = run(section_cohorts, cm, f, rep)
        run(section_sizing, f, rep, n_months)

        # ---- exports ------------------------------------------------------
        rep.h1("9. FILES WRITTEN")
        f.to_csv(out_dir / "customer_features.csv", index=False)
        cm.to_csv(out_dir / "customer_month.csv", index=False)
        if grid is not None:
            grid.to_csv(out_dir / "segmentation_grid.csv", index=False)
        if branch_out is not None:
            branch_out[0].to_csv(out_dir / "branch_metrics.csv", index=False)
            branch_out[1].to_csv(out_dir / "branch_month.csv", index=False)
        if retention is not None:
            retention.to_csv(out_dir / "cohort_retention.csv")
        for p in sorted(out_dir.glob("*.csv")):
            rep.bullet(f"{p.name}  ({p.stat().st_size / 1024:,.0f} KB)")

        rep.h2("NEXT STEP")
        rep.text("Send back `report_phase1.md`. Thresholds, score components, "
                 "weights and the branch ranking method will be derived from "
                 "these numbers in Phase 2 - nothing is fixed before then.")

    finally:
        report_path = out_dir / "report_phase1.md"
        rep.save(report_path)
        con.close()
        print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
