"""
data_loader.py

Utilities to fetch price data for a single ticker using yfinance.

Key features
------------
- Thin wrapper around yfinance with clear, stable output schema.
- Returns a DataFrame with columns: ['date', 'close'].
- Optional start/end or period/interval controls.
- Optional timezone normalization (tz-naive by default).
- Optional CSV export.

Example
-------
from data_loader import find_price_data

prices = find_price_data(
    ticker="AAPL",
    period="1y",
    interval="1d",
    auto_adjust=True,
    tz=None,                              # tz-naive datetimes (default)
    save_csv="data/aapl_prices.csv"
)
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import yfinance as yf


def _to_tz_naive(dt_index: pd.DatetimeIndex, tz: Optional[str]) -> pd.DatetimeIndex:
    """
    Normalize a DatetimeIndex to either tz-naive or a specific timezone (then strip tz).

    Behavior
    --------
    - If tz is None: return tz-naive (no timezone info).
    - If tz is provided: convert to that timezone, then strip tz to keep downstream merges simple.

    Notes
    -----
    yfinance typically returns a tz-aware index (e.g., US/Eastern).
    """
    idx = pd.DatetimeIndex(dt_index)
    # Ensure tz-aware first (yfinance usually is)
    if idx.tz is None:
        # Assume UTC if tz info is missing (conservative fallback)
        idx = idx.tz_localize("UTC")

    if tz:
        idx = idx.tz_convert(tz)
    # Strip timezone to keep everything tz-naive in the project
    return idx.tz_localize(None)


def find_price_data(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = True,
    tz: Optional[str] = None,
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical prices via yfinance and return a tidy DataFrame.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'AAPL').
    period : str, default '1y'
        yfinance period string (e.g., '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max').
        Ignored if `start` is provided.
    interval : str, default '1d'
        yfinance interval (e.g., '1d', '1h', '30m', '15m').
    start : str, optional
        Start date (YYYY-MM-DD). If provided, `period` is ignored.
    end : str, optional
        End date (YYYY-MM-DD). If not provided, defaults to today.
    auto_adjust : bool, default True
        If True, returns adjusted Close; if False, raw Close.
    tz : str, optional
        Target timezone to convert to (e.g., 'Europe/London').
        We strip tz info afterward to keep project tz-naive.
    save_csv : str, optional
        If provided, path to write the CSV.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'close'] sorted ascending by date.

    Raises
    ------
    RuntimeError
        If no data is returned by yfinance.
    """
    # Build the yfinance Ticker and fetch history
    tkr = yf.Ticker(ticker)

    # Use start/end if provided; else fall back to period/interval
    if start:
        hist = tkr.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)
    else:
        hist = tkr.history(period=period, interval=interval, auto_adjust=auto_adjust)

    if hist is None or hist.empty:
        raise RuntimeError(
            f"No data found for {ticker} with "
            f"{'start='+str(start)+' ' if start else ''}"
            f"{'end='+str(end)+' ' if end else ''}"
            f"period={period} interval={interval}"
        )

    # Ensure we have the Close/Adj Close (yfinance standardizes column to 'Close' when auto_adjust=True)
    if "Close" not in hist.columns:
        raise RuntimeError("Downloaded dataframe missing 'Close' column.")

    # Reset index to expose the DatetimeIndex as a column
    hist = hist.reset_index()

    # Normalize date column name (yfinance uses 'Date' after reset_index)
    if "Date" not in hist.columns:
        raise RuntimeError("Downloaded dataframe missing 'Date' column after reset_index().")

    # Normalize/standardize the datetime column => tz-naive
    date_idx = _to_tz_naive(pd.to_datetime(hist["Date"]), tz)

    df = pd.DataFrame({
        "date": date_idx,
        "close": pd.to_numeric(hist["Close"], errors="coerce")
    })

    # Drop rows with missing close values and ensure unique, sorted dates
    df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Optional CSV export
    if save_csv:
        df.to_csv(save_csv, index=False)

    return df