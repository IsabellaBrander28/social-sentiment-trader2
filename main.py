#!/usr/bin/env python3
"""
main.py — Run a daily sentiment-driven backtest end-to-end.

Pipeline
--------
1) Pull prices (yfinance via data_loader.find_price_data).
2) Load/score news -> daily sentiment signals (sentiment_scorer.load_and_score_sentiment).
3) Backtest with costs, risk exits, (optional) trend filter / cooldown.
4) Compute metrics (metrics.performance_metrics) with optional benchmark alignment.
5) Save trades CSV + plot images to ./results/.

Usage
-----
# simplest run (uses defaults: AAPL, 1y, 1d, data/example_news.csv)
python main.py

# change ticker and news file
python main.py --ticker AMD --news_csv data/example_news.csv

# add trend filter and 3-day cooldown
python main.py --use_trend_filter --sma_window 20 --cooldown_days 3

# set a train/test split date for walk-forward reporting
python main.py --split_date 2025-06-01

Outputs
-------
- results/<ticker>_trades.csv      (rows where position changed)
- results/equity.png               (equity vs benchmark)
- results/drawdown.png             (drawdown)
- results/rolling_sharpe.png       (rolling Sharpe)
- results/metrics_<ticker>.csv     (overall or train/test metrics)
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Dict, Any

import pandas as pd

from data_loader import find_price_data
from sentiment_scorer import load_and_score_sentiment
from backtester import Backtester


# ------------------------------- CLI -------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run sentiment-driven backtest.")
    # data
    p.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol (e.g., AAPL).")
    p.add_argument("--period", type=str, default="1y", help="yfinance period (e.g., 1y, 2y, 5y).")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval (e.g., 1d, 1h).")
    p.add_argument("--news_csv", type=str, default="data/example_news.csv",
                   help="Path to news CSV (must have 'date' + 'headline' or 'score').")

    # sentiment knobs (match sentiment_scorer.py)
    p.add_argument("--use_zscore", action="store_true", default=True,
                   help="Use rolling z-score thresholds (recommended).")
    p.add_argument("--no_use_zscore", dest="use_zscore", action="store_false",
                   help="Use absolute thresholds instead of z-scores.")
    p.add_argument("--z_lookback", type=int, default=20, help="Rolling window for z-score.")
    p.add_argument("--z_buy", type=float, default=0.5, help="Z-score BUY threshold.")
    p.add_argument("--z_sell", type=float, default=-0.5, help="Z-score SELL threshold.")
    p.add_argument("--decay_half_life", type=int, default=3, help="EWMA half-life for sentiment decay.")
    p.add_argument("--apply_lag", action="store_true", default=True, help="Apply t+1 execution lag.")
    p.add_argument("--no_apply_lag", dest="apply_lag", action="store_false",
                   help="Disable t+1 lag (not recommended).")

    # backtester knobs
    p.add_argument("--transaction_cost", type=float, default=0.0005, help="Per trade cost (e.g., 0.0005 = 5 bps).")
    p.add_argument("--slippage", type=float, default=0.0005, help="Per trade slippage (fraction).")
    p.add_argument("--stop_loss_pct", type=float, default=0.10, help="Stop loss threshold (fraction).")
    p.add_argument("--take_profit_pct", type=float, default=0.10, help="Take profit threshold (fraction).")
    p.add_argument("--use_trend_filter", action="store_true", default=False,
                   help="BUY only if price>SMA and SELL only if price<SMA.")
    p.add_argument("--sma_window", type=int, default=20, help="SMA window for trend filter.")
    p.add_argument("--cooldown_days", type=int, default=0, help="Bars to wait after a trade signal.")
    p.add_argument("--vol_target", type=float, default=None, nargs="?",
                   help="Target annualized vol (e.g., 0.10). Leave empty to disable.")
    p.add_argument("--vol_window", type=int, default=20, help="Lookback for realized vol (if vol_target used).")
    p.add_argument("--max_leverage", type=float, default=3.0, help="Cap leverage for vol targeting.")

    # train/test split
    p.add_argument("--split_date", type=str, default=None,
                   help="Date like YYYY-MM-DD to split train/test (optional).")

    # misc
    p.add_argument("--results_dir", type=str, default="results", help="Output directory.")
    return p


# ---------------------------- main workflow -------------------------- #

def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the full backtest and save artifacts; return metrics dict."""
    os.makedirs(args.results_dir, exist_ok=True)

    # 1) Load price data
    prices = find_price_data(args.ticker, period=args.period, interval=args.interval)
    prices = prices.sort_values("date").reset_index(drop=True)

    # 2) Load + score sentiment -> daily signals (t+1 lag handled inside)
    sentiment = load_and_score_sentiment(
        args.news_csv,
        # CSV parsing
        date_col="date",
        text_col="headline",          # change to 'title' if your CSV uses that
        score_col="score",            # if present in CSV we'll use it directly
        # resampling/aggregation
        agg="mean",
        resample="D",
        fill_missing_dates=False,     # we trade only on real-news days
        smoothing_window=None,
        # signal method
        use_zscore=args.use_zscore,
        z_lookback=args.z_lookback,
        z_buy=args.z_buy,
        z_sell=args.z_sell,
        buy_threshold=0.05,           # only used if use_zscore=False
        sell_threshold=-0.05,
        # decay + lag
        decay_half_life=args.decay_half_life,
        apply_lag=args.apply_lag,
        dayfirst=True
    )

    # Diagnostics: confirm signals look sane
    print("\nSignal counts (after lag):")
    try:
        print(sentiment["signal"].value_counts(dropna=False))
    except Exception:
        print("[warn] sentiment DataFrame missing 'signal' column?")

    print("\nFirst 10 signal rows:")
    print(sentiment.head(10))

    # 3) Backtest
    bt = Backtester(
        prices,
        sentiment,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        vol_target=args.vol_target,
        vol_window=args.vol_window,
        max_leverage=args.max_leverage,
        results_dir=args.results_dir,
        use_trend_filter=args.use_trend_filter,
        sma_window=args.sma_window,
        cooldown_days=args.cooldown_days
    )

    # benchmark = buy & hold on the same asset
    bench_returns = prices.set_index("date")["close"].pct_change().fillna(0.0)

    results = bt.results(benchmark_returns=bench_returns, train_test_split=args.split_date)

    # 4) Print metrics (overall or train/test)
    def _print_block(title: str, d: Dict[str, float]) -> None:
        print(f"\n{title}")
        for k in ["CAGR", "Ann Vol", "Sharpe", "Sortino", "Max Drawdown",
                  "Calmar", "Hit Ratio", "Turnover", "Alpha (ann)", "Info Ratio"]:
            if k in d:
                v = d[k]
                if isinstance(v, (int, float)):
                    print(f"{k}: {v:.4f}" if k not in ("Hit Ratio",) else f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")

    if args.split_date and "overall" in results:
        _print_block("Performance (OVERALL):", results["overall"])
        _print_block("Performance (TRAIN):", results["train"])
        _print_block("Performance (TEST):", results["test"])
        metrics_to_save = {
            "overall": results["overall"],
            "train": results["train"],
            "test": results["test"],
        }
    else:
        _print_block("Performance Metrics:", results["metrics"])
        metrics_to_save = {"overall": results["metrics"]}

    # 5) Save trades CSV
    trades_name = f"{args.ticker.lower()}_trades.csv"
    bt.save_trades(trades_name)

    # 6) Save plots
    eq_ax = bt.plot_equity_vs_benchmark(benchmark_returns=bench_returns, show=False)
    eq_ax.figure.savefig(os.path.join(args.results_dir, "equity.png"), dpi=150, bbox_inches="tight")

    dd_ax = bt.plot_drawdown(show=False)
    dd_ax.figure.savefig(os.path.join(args.results_dir, "drawdown.png"), dpi=150, bbox_inches="tight")

    rs_ax = bt.plot_rolling_sharpe(window=21, show=False)
    rs_ax.figure.savefig(os.path.join(args.results_dir, "rolling_sharpe.png"), dpi=150, bbox_inches="tight")

    # 7) Save metrics to CSV (flatten dict)
    # If train/test provided, we'll write a tidy two-level CSV; else a single row.
    metrics_path = os.path.join(args.results_dir, f"metrics_{args.ticker.lower()}.csv")
    if "train" in metrics_to_save:
        # create a tidy DataFrame with section label
        rows = []
        for tag, d in metrics_to_save.items():
            row = {"section": tag}
            row.update(d)
            rows.append(row)
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
    else:
        pd.DataFrame([metrics_to_save["overall"]]).to_csv(metrics_path, index=False)

    print(f"\n[i] saved trades  -> {os.path.join(args.results_dir, trades_name)}")
    print(f"[i] saved plots   -> {args.results_dir}/equity.png, drawdown.png, rolling_sharpe.png")
    print(f"[i] saved metrics -> {metrics_path}")

    return results


# ------------------------------- entry -------------------------------- #

if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    run(args)
