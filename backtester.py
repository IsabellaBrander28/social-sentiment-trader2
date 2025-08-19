"""
Backtester for daily, signal-driven trading strategies.

Features
--------
- Robust merge of prices & signals (date-safe, tz-naive)
- Stateful position handling (BUY = long, SELL = short, HOLD = flat)
- Stop-loss / take-profit exits (percentage from entry)
- Transaction cost + slippage per trade
- Optional volatility targeting (scale exposure to target annual vol)
- Optional trend filter (SMA-based) and post-trade cooldown
- Performance metrics via `metrics.performance_metrics`
- Optional train/test split for walk-forward reporting
- Trade log export
- Plots: equity vs benchmark, drawdown, rolling Sharpe

Expected inputs
---------------
price_df : DataFrame with columns ['date', 'close']
signal_df: DataFrame with columns ['date', 'signal'] where signal ∈ {'BUY','SELL','HOLD'}
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import performance_metrics


# ------------------ helper utilities (date/index safe) ------------------ #

def _to_naive_datetime_index(values) -> pd.DatetimeIndex:
    """
    Convert array/Series/Index of dates to a tz-naive DatetimeIndex.

    Notes
    -----
    - Coerces invalid entries to NaT (never raises on bad parses).
    - Strips timezone if present.

    Returns
    -------
    pd.DatetimeIndex
    """
    dt = pd.to_datetime(values, errors="coerce")
    if isinstance(dt, pd.Series):
        idx = pd.DatetimeIndex(dt)
    elif isinstance(dt, pd.DatetimeIndex):
        idx = dt
    else:
        idx = pd.DatetimeIndex(dt)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx


def _align_series_to_dates(series: pd.Series,
                           dates_index: pd.DatetimeIndex,
                           fill_value: float = 0.0) -> pd.Series:
    """
    Align an arbitrary Series to a target DatetimeIndex.

    Behaviour
    ---------
    - If `series` has no DatetimeIndex but same length as `dates_index`,
      align by position.
    - Else, attempt to convert its index to DatetimeIndex and reindex.
    - Fills missing with `fill_value`.

    Returns
    -------
    pd.Series
    """
    s = pd.Series(series).copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        if len(s) == len(dates_index):
            s.index = dates_index
        else:
            try:
                s.index = _to_naive_datetime_index(s.index)
            except Exception:
                s = pd.Series(s.values, index=dates_index)
    else:
        s.index = _to_naive_datetime_index(s.index)
    return s.reindex(dates_index).fillna(fill_value)


# ------------------------------- backtester ------------------------------ #

class Backtester:
    """
    Stateful daily backtester for long/short/flat signals.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must contain ['date','close'].
    signal_df : pd.DataFrame
        Must contain ['date','signal'] with signal ∈ {'BUY','SELL','HOLD'}.
        IMPORTANT: Apply t+1 lag in your signal generator to avoid look-ahead.
    transaction_cost : float, default 0.0005
        Proportional per-trade cost (e.g., 0.0005 = 5 bps).
    slippage : float, default 0.0005
        Additional per-trade impact/slippage (fractional, e.g., 0.0005).
    stop_loss_pct : float, default 0.10
        Stop loss threshold (10% below entry for longs; above entry for shorts).
    take_profit_pct : float, default 0.10
        Take profit threshold (symmetric to stop by default).
    vol_target : float | None, default None
        If set (e.g., 0.10 for 10%/yr), scale exposure to target annual vol.
    vol_window : int, default 20
        Lookback window (trading days) for realized vol (if vol_target set).
    max_leverage : float, default 3.0
        Cap on leverage when using vol targeting.
    results_dir : str, default "results"
        Directory for CSV outputs (trades).
    use_trend_filter : bool, default False
        If True, only BUY when price > SMA(sma_window), SELL when price < SMA.
    sma_window : int, default 20
        Window for SMA trend filter (used only if use_trend_filter=True).
    cooldown_days : int, default 0
        Minimum gap (in bars) between successive non-HOLD signals. 0 disables.

    Attributes
    ----------
    df : pd.DataFrame
        Backtest table with positions, returns, PnL, costs, etc.
    """

    def __init__(self,
                 price_df: pd.DataFrame,
                 signal_df: pd.DataFrame,
                 transaction_cost: float = 0.0005,
                 slippage: float = 0.0005,
                 stop_loss_pct: float = 0.10,
                 take_profit_pct: float = 0.10,
                 vol_target: Optional[float] = None,
                 vol_window: int = 20,
                 max_leverage: float = 3.0,
                 results_dir: str = "results",
                 use_trend_filter: bool = False,
                 sma_window: int = 20,
                 cooldown_days: int = 0) -> None:

        # --- store inputs ---
        self.price_df = price_df.copy()
        self.signal_df = signal_df.copy()

        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.stop_loss_pct = float(max(0.0, stop_loss_pct))      # guard against negative inputs
        self.take_profit_pct = float(max(0.0, take_profit_pct))  # guard against negative inputs

        self.vol_target = vol_target
        self.vol_window = int(vol_window)
        self.max_leverage = float(max_leverage)

        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.use_trend_filter = bool(use_trend_filter)
        self.sma_window = int(sma_window)
        self.cooldown_days = int(max(0, cooldown_days))

        self.df: Optional[pd.DataFrame] = None
        self._prepare()

    # ------------------------------------------------------------------ #
    def _prepare(self) -> None:
        """
        Merge prices & signals, optionally apply trend/cooldown filters,
        build positions statefully with stop/take exits, and compute PnL.
        """
        # --- ensure required columns & tz-naive dates ---
        price = self.price_df[['date', 'close']].copy()
        sig = self.signal_df[['date', 'signal']].copy()

        price['date'] = _to_naive_datetime_index(price['date'])
        sig['date'] = _to_naive_datetime_index(sig['date'])

        # --- merge and sort ---
        df = pd.merge(price, sig, on='date', how='inner').sort_values('date').reset_index(drop=True)
        df['signal'] = df['signal'].fillna('HOLD')

        # --- optional trend filter (simple SMA gate) ---
        if self.use_trend_filter and self.sma_window > 1:
            df['sma'] = df['close'].rolling(self.sma_window).mean()
            cond_buy = (df['signal'] == 'BUY') & (df['close'] > df['sma'])
            cond_sell = (df['signal'] == 'SELL') & (df['close'] < df['sma'])
            df['signal'] = np.where(cond_buy, 'BUY', np.where(cond_sell, 'SELL', 'HOLD'))

        # --- optional cooldown to reduce flip-flops ---
        if self.cooldown_days > 0:
            last_sig_idx = -10**9
            sig_list = df['signal'].tolist()
            for i, s in enumerate(sig_list):
                if s in ('BUY', 'SELL'):
                    if i - last_sig_idx <= self.cooldown_days:
                        sig_list[i] = 'HOLD'
                    else:
                        last_sig_idx = i
            df['signal'] = sig_list

        # --------------------- stateful position engine --------------------- #
        df['entry_price'] = np.nan
        position = 0      # +1 long, 0 flat, -1 short
        entry_px = None   # price at which current position opened

        for i, row in df.iterrows():
            s = row['signal']
            px = float(row['close'])

            # Open/reverse positions on signals (t+1 lag should be done upstream)
            if s == 'BUY' and position <= 0:
                position = 1
                entry_px = px
            elif s == 'SELL' and position >= 0:
                position = -1
                entry_px = px

            # Risk exits: stop-loss / take-profit relative to entry
            if position != 0 and entry_px is not None:
                chg = (px - entry_px) / entry_px
                if position == 1 and (chg <= -self.stop_loss_pct or chg >= self.take_profit_pct):
                    position, entry_px = 0, None
                elif position == -1 and (chg >= self.stop_loss_pct or chg <= -self.take_profit_pct):
                    position, entry_px = 0, None

            df.at[i, 'position'] = position
            df.at[i, 'entry_price'] = entry_px

        # ------------------------ returns & PnL block ----------------------- #
        df['ret'] = df['close'].pct_change().fillna(0.0)           # raw asset returns
        df['pos_prev'] = df['position'].shift(1).fillna(0.0)       # realized pos for period
        df['trade'] = (df['pos_prev'] != df['position']).astype(float)  # trade indicator

        # Optional volatility targeting (scale exposure)
        if self.vol_target is not None:
            daily_vol = df['ret'].rolling(self.vol_window).std()
            daily_vol = daily_vol.fillna(df['ret'].std())           # fallback early
            ann_vol = daily_vol * np.sqrt(252.0)

            leverage = (self.vol_target / ann_vol).clip(upper=self.max_leverage)
            df['exposure'] = df['pos_prev'] * leverage              # signed exposure
        else:
            df['exposure'] = df['pos_prev']                         # +/-1 or 0

        # Strategy return = exposure * asset return − trading costs (per switch)
        roundtrip_cost = (self.transaction_cost + self.slippage)
        df['strategy_ret'] = df['exposure'] * df['ret'] - df['trade'] * roundtrip_cost

        self.df = df

    # ------------------------------------------------------------------ #
    def results(self,
                benchmark_returns: Optional[pd.Series] = None,
                train_test_split: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute metrics and (optionally) train/test split metrics.

        Parameters
        ----------
        benchmark_returns : pd.Series, optional
            Benchmark percent returns aligned to dates (e.g., buy & hold).
            If provided, alpha and info ratio are computed.
        train_test_split : str, optional
            Date string (e.g., '2025-06-01'). If provided, returns:
              {'df', 'overall', 'train', 'test'}
            Else returns:
              {'df', 'metrics'}

        Returns
        -------
        dict
        """
        if self.df is None:
            raise RuntimeError("Backtest not prepared. Call _prepare() first.")

        df = self.df.copy()
        dates = _to_naive_datetime_index(df['date'])

        # strategy returns as a date-indexed Series (for robust alignment)
        strat = pd.Series(df['strategy_ret'].values, index=dates)

        # align benchmark to same dates (safe)
        bench = None
        if benchmark_returns is not None:
            bench = _align_series_to_dates(benchmark_returns, dates, fill_value=0.0)

        overall = performance_metrics(strat, benchmark_returns=bench)

        # Optional walk-forward split
        if train_test_split:
            split_dt = pd.to_datetime(train_test_split)
            if getattr(split_dt, "tzinfo", None) is not None:
                split_dt = split_dt.tz_localize(None)

            mask_train = strat.index <= split_dt
            mask_test = strat.index > split_dt

            strat_train = strat[mask_train]
            strat_test = strat[mask_test]
            bench_train = bench[mask_train] if bench is not None else None
            bench_test = bench[mask_test] if bench is not None else None

            metrics_train = performance_metrics(strat_train, benchmark_returns=bench_train)
            metrics_test = performance_metrics(strat_test, benchmark_returns=bench_test)

            return {'df': df, 'overall': overall, 'train': metrics_train, 'test': metrics_test}

        return {'df': df, 'metrics': overall}

    # ------------------------------------------------------------------ #
    def save_trades(self, filename: str = "trades.csv") -> pd.DataFrame:
        """
        Save trade rows (where position changed) to CSV.

        Returns
        -------
        pd.DataFrame
            The trades DataFrame that was written to disk.
        """
        if self.df is None:
            raise RuntimeError("No backtest data to save. Run results() first or call _prepare().")

        trades = self.df[self.df['trade'] == 1.0].copy()
        outpath = os.path.join(self.results_dir, filename)
        trades.to_csv(outpath, index=False)
        print(f"[i] saved trades -> {outpath}")
        return trades

    # ------------------------------- plots -------------------------------- #

    def plot_equity_vs_benchmark(self,
                                 benchmark_returns: Optional[pd.Series] = None,
                                 show: bool = True):
        """
        Plot cumulative strategy return and (optionally) benchmark.

        Notes
        -----
        Robust alignment; tolerant to different indices on the benchmark.
        """
        if self.df is None:
            raise RuntimeError("No data to plot. Run results() first or call _prepare().")

        df = self.df.copy()
        dates = _to_naive_datetime_index(df['date'])
        strat_cum = (1.0 + pd.Series(df['strategy_ret'].values, index=dates)).cumprod()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, strat_cum.values, label='Strategy')

        if benchmark_returns is not None:
            bench = _align_series_to_dates(benchmark_returns, dates, fill_value=0.0)
            bench_cum = (1.0 + bench).cumprod()
            ax.plot(dates, bench_cum.values, label='Benchmark (Buy & Hold)', linestyle='--')

        ax.set_title("Equity Curve")
        ax.set_ylabel("Cumulative Return (x)")
        ax.legend()
        if show:
            plt.show()
        return ax

    def plot_drawdown(self, show: bool = True):
        """
        Plot drawdown series (cum_return / cum_max - 1).
        """
        if self.df is None:
            raise RuntimeError("No data to plot. Run results() first or call _prepare().")

        dates = _to_naive_datetime_index(self.df['date'])
        cum = (1.0 + pd.Series(self.df['strategy_ret'].values, index=dates)).cumprod()
        dd = cum / cum.cummax() - 1.0

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(dates, dd.values, 0.0, alpha=0.35)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        if show:
            plt.show()
        return ax

    def plot_rolling_sharpe(self, window: int = 21, show: bool = True):
        """
        Plot annualised rolling Sharpe. Uses min_periods to get early values,
        avoids div/0, and drops NaNs before plotting.
        """
        if self.df is None:
            raise RuntimeError("No data to plot. Run results() first or call _prepare().")

        dates = _to_naive_datetime_index(self.df['date'])
        r = pd.Series(self.df['strategy_ret'].values, index=dates)

        # Allow early estimates; replace 0 std with NaN to avoid inf
        roll_mean = r.rolling(window=window, min_periods=max(5, window // 2)).mean()
        roll_std = r.rolling(window=window, min_periods=max(5, window // 2)).std()
        roll_std = roll_std.replace(0.0, np.nan)

        roll_sharpe = (roll_mean / roll_std) * np.sqrt(252.0)
        roll_sharpe = roll_sharpe.dropna()

        fig, ax = plt.subplots(figsize=(10, 3))
        if not roll_sharpe.empty:
            ax.plot(roll_sharpe.index, roll_sharpe.values, label=f"Rolling Sharpe ({window}d)")
        else:
            ax.text(0.02, 0.6, "Not enough data for rolling Sharpe",
                    transform=ax.transAxes)
        ax.axhline(0.0, linestyle='--', linewidth=1.0)
        ax.set_title("Rolling Sharpe Ratio")
        ax.legend(loc="best")
        if show:
            plt.show()
        return ax
