# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Streamlit + DuckDB analytics project over pharmacy loyalty transaction data. There is no git repo, no `requirements.txt`/`pyproject.toml`, and no test suite — dependencies exist only inside `.venv` (streamlit, duckdb, pandas, numpy, plotly, pyarrow, altair, pydeck). Install/replicate the environment by inspecting `.venv\Lib\site-packages` if `pip freeze` is needed.

## Commands

Run everything from the project root using the project's venv interpreter (`.venv\Scripts\python.exe`), or activate it first with `.venv\Scripts\Activate.ps1`.

```powershell
# Launch the dashboard (multipage Streamlit app)
streamlit run dashboard/app.py

# Rebuild the raw `transactions` table from the source CSV
# (expects data/raw/(Copy) loyalty_satislar.csv, pipe-delimited)
python scripts/01_build_and_validate.py

# Build clean_transactions / customer_segments / monthly_branch_metrics
# from the raw transactions table (run after 01_build_and_validate.py)
python scripts/04_build_analytics_tables.py

# Re-run Phase 1 diagnostics that justify the thresholds in loan_analytics.CONFIG
# (writes experiment_output/report_phase1.md and several CSVs)
python scripts/experiment.py

# Out-of-time validation of the segmentation/score model
# (writes validation_output/holdout_report.md) - run after every data refresh
python dashboard/validate_model.py --outcome 6 --window 6
```

There is no lint/test command configured — verify changes by running the affected script or by launching the dashboard and checking the relevant page.

## Data pipeline

`data/raw/(Copy) loyalty_satislar.csv` → `scripts/01_build_and_validate.py` (creates `transactions` in `data/pharmacy.duckdb`, casts/cleans raw columns, prints validation output) → `scripts/04_build_analytics_tables.py` (creates `clean_transactions`, `customer_segments`, `monthly_branch_metrics`). Everything downstream (the dashboard and `loan_analytics.py`) reads only `clean_transactions` (and, for the pension analysis, `customer_segments`) — never `transactions` directly.

Source columns are in Azerbaijani and keep their original names throughout the codebase:
- `aptek_id` = branch id, `musteri_acari` = customer key, `tarix_saat` = transaction timestamp, `cek_id` = receipt/transaction id
- `meblegh` = gross amount, `endirim_meblegi` = discount (signed, negative), `qaytarma` = return flag (1 = returned), `resepti_var` = prescription flag

`clean_transactions` cleaning rules (see `scripts/04_build_analytics_tables.py`): one row per `cek_id` (dedup by `ROW_NUMBER()`), a transaction is a "financial anomaly" if `meblegh <= 0` or `abs(discount) > meblegh`, and `usable_for_sales = 1` only when `qaytarma = 0 AND financial_anomaly = 0`. All derived sales columns (`valid_gross_amount`, `valid_discount_amount`, `net_amount`) are zeroed out for non-usable rows rather than filtered — queries must gate on `usable_for_sales = 1` explicitly rather than assuming the table is pre-filtered. `customer_segments.customer_type` is "Pension" when a customer's lifetime discount rate exceeds 50%, else "Non-Pension" (unrelated to the loan-pilot segmentation model below).

`scripts/02_investigate_source_rules.py` and `scripts/exp.py` are ad hoc one-off investigation scripts, not part of the regular pipeline.

## Other raw data

`data/raw/Payonix_rank_branch.xlsx` is not read by any script — treat it as reference-only, outside the `transactions` → `clean_transactions` pipeline. It has 7 sheets:
- `Sheet3`, `26`, `25`, `24` — raw cash/POS transaction extracts (`MonthAndYear`, `IDSurrogatePK`, `Pharmacy`, `Operation Type`, `Amount`).
- `cash_pos` — monthly cash vs. POS (card) sales volume per branch (`Year`, `Month`, `branch`, `CASH_AZN`, `POS_AZN`, `Rank`).
- `info` — branch reference data: id, name, network/chain (`sebeke`), city/district (`seher_rayon`), a Baku flag, and `Region`.
- `Pilot_branches_scoring` — an external, point-in-time export whose columns (`pilot_score`, `eligible`, `eligibility_rate_pct`, `primary_customer_ratio_pct`, `tx_cv_recent`, `tx_growth_pct`, ...) mirror `pilot_ranking()`/`branch_summary()` in `loan_analytics.py` — it's a saved snapshot of the model's own output, not independent ground truth.

## Two independent analytical layers in the dashboard

1. **Descriptive sales dashboard** — the main body of `dashboard/app.py`. Queries `clean_transactions`/`customer_segments` directly via short-lived read-only `duckdb.connect()` calls (see `run_query`), all wrapped in `@st.cache_data`. Covers company/branch KPIs, monthly trends, prescription split, pension split, and branch ranking, filtered by a sidebar month range + branch selector.

2. **Loan-pilot customer scoring model** — `dashboard/analytics/loan_analytics.py`, driving the "Loan Pilot Overview" block appended near the end of `app.py` (visually marked off with `PASTE INTO app.py` comments — that block was spliced in after the original dashboard was written) plus `dashboard/pages/1`–`7`. This is the part most likely to need changes:
   - **`CONFIG`** is the single source of truth for every threshold/weight/cap (segmentation cutoffs, score component weights, eligibility cutoff, branch pilot weights). Change behavior by editing `CONFIG`, never by hardcoding numbers in logic — every value in it is annotated with the empirical justification from `experiment_output/report_phase1.md`.
   - **`load_customer_features(as_of)`** builds one row per customer as of the end of a given month: SQL aggregates raw transactions (lifetime + a trailing `WINDOW_MONTHS`-month window), pandas derives ratios/scores on top. Because everything is anchored on `as_of`, the whole model can be replayed at any historical cutoff — this is what `validate_model.py` and the "As-of month" sidebar selector both rely on.
   - **`assign_segments`** → rule-based segment (Frequent/Regular/Semi-Regular/Sporadic/Dormant/New-Insufficient) from coverage + depth in the observation window.
   - **`score_components` / `compute_score`** → the 0–100 behavioural eligibility score, a fixed-weight sum of six 0–1 components (see `CONFIG["SCORE_WEIGHTS"]`); `eligible` requires both a score floor and a tenure floor.
   - **`branch_summary` / `pilot_ranking` / `pilot_sensitivity`** → per-branch rollups of the customer features, filtered to branches with enough history/eligible customers, ranked by a percentile-based weighted score (`CONFIG["PILOT_WEIGHTS"]`); `pilot_sensitivity` re-ranks under random weight perturbations to check robustness.
   - **`validation_checks`** → cheap structural invariants (score bounds, segment nesting, etc.), rendered live on the Model Validation page and reusable anywhere the model output needs a sanity check.
   - `dashboard/analytics/dashboard_ui.py` holds the shared sidebar (`sidebar()` — re-runs the whole model at the selected `as_of` month, not just a display filter) and small formatting helpers used by every page in `dashboard/pages/`.
   - `dashboard/validate_model.py` independently reimplements the feature/scoring pipeline from raw SQL (via `frame_at`) to validate `assign_segments`/`compute_score` out-of-time against actual future behavior (AUC, lift-by-band, gains tables). If segmentation/scoring logic in `loan_analytics.py` changes, check whether the corresponding logic in `validate_model.py` needs the same change.

Every `dashboard/pages/*.py` file does `sys.path.append(str(Path(__file__).resolve().parents[1]))` before importing from `analytics.*` — Streamlit pages are executed with `dashboard/` as the working context but not automatically on `sys.path`.

## Conventions

- Multi-line SQL/expression formatting throughout the codebase (especially `app.py` and `04_build_analytics_tables.py`) uses one clause per line with heavy vertical spacing — match this style in that file rather than condensing.
- `loan_analytics.py` and its pages are denser/more compact by contrast (per the "Design rules" docstring: SQL for aggregation, pandas for derivation, no logic embedded in raw numbers). Match whichever style is local to the file being edited.
- DuckDB connections are always opened `read_only=True` and closed in a `finally` block for the dashboard's query paths — follow this pattern for any new query helper.
