"""
Out-of-sample evaluation gate for the "Work" TODO in hmm.py:

    - Compare model to:
        (1) a persistence model
        (2) a random walk baseline
        (3) a simple moving-average trend filter
        (4) a positive Sharpe on a paper trading strategy
    - If HMM beats all of the above on out-of-sample data, notify and save

Usage (drop into hmm.py after Phase 7's rolling window / `new_results` is built):

    from evaluation import evaluate_and_maybe_save
    evaluate_and_maybe_save(model, new_results, bull_regimes, volatility_threshold, window_size)
"""
import datetime as dt
import json
import os

import joblib
import numpy as np

from functions import sharpe_ratio
from baselines import persistence_signal, random_walk_baseline_sharpe, ma_trend_signal


def evaluate_and_maybe_save(model, new_results, bull_regimes, volatility_threshold,
                             window_size, ma_window=50, n_random_sims=500, out_dir="models"):
    """
    Compares the HMM's walk-forward Sharpe (`new_results['Strategy_Returns']`,
    already Signal.shift(1) * Returns from Phase 7) against three baselines
    computed on the *same* `new_results` frame, so every comparison uses the
    same out-of-sample dates and the same Returns series.

    The "persistence" baseline here is a control group: always bullish /
    always fully invested, with no regime detection at all (equivalent to
    buy-and-hold). It answers "does selectively sitting out bearish-labeled
    days actually help, or would you have done just as well staying in the
    market the whole time?"

    `new_results` must already have 'Signal', 'Returns', 'Close', and
    'Strategy_Returns' columns (Phase 7 in hmm.py produces all of these).

    Prints a pass/fail console banner (the requested "notify"). On a pass,
    saves the fitted model + a metrics JSON under `out_dir` (the requested
    "save"). Returns a dict of the computed Sharpe ratios either way, so you
    can inspect/plot them even on a fail.
    """
    hmm_sharpe = sharpe_ratio(new_results["Strategy_Returns"])

    persistence_returns = persistence_signal(new_results["Returns"]) * new_results["Returns"]
    persistence_sharpe = sharpe_ratio(persistence_returns)

    rw_mean, rw_p95 = random_walk_baseline_sharpe(new_results["Returns"], n_sims=n_random_sims)

    ma_returns = ma_trend_signal(new_results["Close"], window=ma_window).shift(1) * new_results["Returns"]
    ma_sharpe = sharpe_ratio(ma_returns)

    results = {
        "hmm_sharpe": hmm_sharpe,
        "persistence_sharpe": persistence_sharpe,
        "random_walk_mean": rw_mean,
        "random_walk_p95": rw_p95,
        "ma_sharpe": ma_sharpe,
    }

    def _beats(baseline_sharpe):
        return not np.isnan(hmm_sharpe) and hmm_sharpe > (
            baseline_sharpe if not np.isnan(baseline_sharpe) else -np.inf
        )

    beats_all = (
        not np.isnan(hmm_sharpe)
        and hmm_sharpe > 0
        and _beats(persistence_sharpe)
        and hmm_sharpe > rw_p95
        and _beats(ma_sharpe)
    )

    print("\n── Out-of-sample baseline comparison ──────────────")
    print(f"  HMM (walk-forward)      : {hmm_sharpe:+.4f}")
    print(f"  Always invested (ctrl)  : {persistence_sharpe:+.4f}")
    print(f"  Random walk (p95)       : {rw_p95:+.4f}  (mean: {rw_mean:+.4f})")
    print(f"  MA trend filter         : {ma_sharpe:+.4f}")
    print("────────────────────────────────────────────────────")

    if beats_all:
        print(f"✅ HMM cleared all baselines out-of-sample (Sharpe {hmm_sharpe:.4f}). Saving model.\n")
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(model, f"{out_dir}/hmm_{stamp}.pkl")
        with open(f"{out_dir}/hmm_{stamp}_metrics.json", "w") as f:
            json.dump({
                **results,
                "bull_regimes": [int(b) for b in bull_regimes],
                "volatility_threshold": volatility_threshold,
                "window_size": window_size,
                "ma_window": ma_window,
                "as_of": stamp,
            }, f, indent=2)
    else:
        print(f"❌ Did not clear all baselines (HMM Sharpe {hmm_sharpe:.4f}).\n")

    return results
