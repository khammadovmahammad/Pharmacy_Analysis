"""
============================================================================
validate_model.py  --  out-of-time validation of the segmentation and score
============================================================================

Rebuilds the model as it would have looked at a past cut-off, then compares
its output with what customers actually did afterwards. Run this after every
data refresh; if the lift table stops being monotonic or AUC drops materially,
the thresholds in loan_analytics.CONFIG need revisiting.

    python validate_model.py                    # default: 6-month outcome window
    python validate_model.py --outcome 6 --window 6

Outputs a printed report and validation_output/holdout_report.md
============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from analytics.loan_analytics import CONFIG, assign_segments, compute_score, find_db


def load_panel() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Customer x month matrices of transaction counts and spend."""
    con = duckdb.connect(str(find_db()), read_only=True)
    try:
        cm = con.execute(f"""
            SELECT musteri_acari AS cust,
                   CAST(date_trunc('month', tarix_saat) AS DATE) AS ym,
                   COUNT(*) AS n, SUM(CAST(net_amount AS DOUBLE)) AS s
            FROM {CONFIG['TX_TABLE']}
            WHERE {CONFIG['QUALIFYING_FILTER']}
            GROUP BY 1, 2
        """).fetchdf()
    finally:
        con.close()

    cust_codes, cust_index = pd.factorize(cm["cust"])
    ym = pd.PeriodIndex(pd.to_datetime(cm["ym"]), freq="M")
    m0 = ym.asi8.min()
    j = ym.asi8 - m0
    n_m = int(j.max()) + 1

    TX = np.zeros((len(cust_index), n_m), dtype=np.int16)
    SP = np.zeros((len(cust_index), n_m), dtype=np.float32)
    TX[cust_codes, j] = cm["n"].values
    SP[cust_codes, j] = cm["s"].values
    months = [str(pd.Period(ordinal=m0 + k, freq="M")) for k in range(n_m)]
    return TX, SP, months, cust_index.values


def frame_at(TX, SP, cut, window):
    obs, sp = TX[:, :cut], SP[:, :cut]
    act = obs > 0
    has = act.any(1)
    first = np.where(has, act.argmax(1), 10**6)
    last = np.where(has, cut - 1 - act[:, ::-1].argmax(1), -1)
    w, s = obs[:, cut - window:cut], sp[:, cut - window:cut]
    df = pd.DataFrame({
        "cust": np.arange(TX.shape[0]),
        "coverage": (w > 0).sum(1),
        "depth_2": (w >= 2).sum(1),
        "depth_3": (w >= 3).sum(1),
        "tx_win": w.sum(1),
        "spend_win": s.sum(1),
        "n_tx": obs.sum(1),
        "total_spend": sp.sum(1),
        "tenure_months": np.where(has, cut - first, 0),
        "recency_months": np.where(has, cut - 1 - last, 99),
    })
    df["monthly_spend"] = df["spend_win"] / window
    df["intensity"] = np.where(df["coverage"] > 0,
                               df["tx_win"] / df["coverage"].clip(lower=1), 0.0)
    df["avg_basket"] = df["total_spend"] / df["n_tx"].clip(lower=1)
    df["primary_branch_share"] = 1.0
    return compute_score(assign_segments(df)), has


def auc(score: np.ndarray, y: np.ndarray) -> float:
    r = pd.Series(score).rank().values
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", type=int, default=6, help="outcome window in months")
    ap.add_argument("--window", type=int, default=CONFIG["WINDOW_MONTHS"])
    ap.add_argument("--out", default="validation_output")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)
    lines = []

    def say(text=""):
        print(text)
        lines.append(str(text))

    TX, SP, months, _ = load_panel()
    n_m = TX.shape[1]
    cut = n_m - args.outcome
    say(f"# Holdout validation\n")
    say(f"- months available: {months[0]} .. {months[-1]} ({n_m})")
    say(f"- model built on data up to: {months[cut-1]}")
    say(f"- outcome observed over: {months[cut]} .. {months[-1]}")
    say(f"- observation window: {args.window} months\n")

    df, has = frame_at(TX, SP, cut, args.window)
    fut = (TX[:, cut:] > 0).sum(1)
    persist = fut >= max(args.outcome - 1, 1)
    judgeable = has & (df["tenure_months"].values >= CONFIG["MIN_TENURE_MONTHS"])

    say(f"- customers judgeable at cut-off: {int(judgeable.sum()):,}")
    say(f"- base rate (active in >= {max(args.outcome-1,1)} of {args.outcome} months): "
        f"{100*persist[judgeable].mean():.2f}%\n")

    say("## Segments vs actual future behaviour\n")
    t = (pd.DataFrame({"segment": df["segment"], "persist": persist,
                       "future_months": fut, "judgeable": judgeable})
         .groupby("segment", observed=False)
         .agg(customers=("persist", "size"),
              pct_active_next=("future_months", lambda s: 100 * (s > 0).mean()),
              mean_future_active_months=("future_months", "mean"),
              pct_persisted=("persist", lambda s: 100 * s.mean())).reset_index())
    say("```\n" + t.to_string(index=False, float_format=lambda v: f"{v:9.2f}") + "\n```\n")

    say("## Score bands vs actual future behaviour (gains table)\n")
    g = (pd.DataFrame({"band": df["score_band"], "persist": persist, "fm": fut})
         .loc[judgeable].groupby("band", observed=False)
         .agg(customers=("persist", "size"),
              pct_persisted=("persist", lambda s: 100 * s.mean()),
              mean_future_active_months=("fm", "mean")).reset_index())
    say("```\n" + g.to_string(index=False, float_format=lambda v: f"{v:9.2f}") + "\n```\n")
    monotonic = g["pct_persisted"].is_monotonic_increasing
    say(f"- lift monotonic across bands: **{monotonic}**")

    a = auc(df["score"].values[judgeable], persist[judgeable])
    say(f"- AUC of the behavioural score: **{a:.4f}**")
    say(f"- AUC of transaction count alone: {auc(df['tx_win'].values[judgeable], persist[judgeable]):.4f}")
    say(f"- AUC of coverage alone: {auc(df['coverage'].values[judgeable], persist[judgeable]):.4f}\n")

    say("## Eligibility cut-off\n")
    rows = []
    for thr in [50, 55, 60, 65, 70, 75, 80]:
        m = judgeable & (df["score"].values >= thr)
        rows.append(dict(min_score=thr, customers=int(m.sum()),
                         pct_of_judgeable=100 * m.sum() / judgeable.sum(),
                         pct_persisted=100 * persist[m].mean(),
                         mean_future_active_months=fut[m].mean()))
    say("```\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:9.2f}")
        + "\n```\n")

    say("## Alternative observation windows (AUC on transaction count)\n")
    for L in [3, 6, 12, 18]:
        if cut - L < 0:
            continue
        tx = TX[:, cut - L:cut].sum(1).astype(float)
        say(f"- last {L:>2} months: {auc(tx[judgeable], persist[judgeable]):.4f}")

    (out_dir / "holdout_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten: {out_dir / 'holdout_report.md'}")


if __name__ == "__main__":
    main()
