"""
sentiment_scorer.py

Convert a news CSV into daily sentiment scores and trading signals.

Features
--------
- Uses precomputed 'score' column if present; otherwise computes VADER sentiment from text.
- Robust CSV parsing (configurable date/text columns; UK day-first parsing).
- Optional resampling/aggregation to daily.
- News gating: trade only on days with *actual* headlines (configurable min count).
- Short-memory decay so sentiment fades toward neutral.
- Rolling z-score thresholding (default) for robust, relative signals.
- t+1 lag to avoid look-ahead bias.

Returns
-------
pd.DataFrame with columns: ['date', 'score', 'signal']
"""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd

# VADER is optional at runtime if you already have a 'score' column in the CSV.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False


def load_and_score_sentiment(
    csv_path: str,
    *,
    date_col: str = "date",
    text_col: str = "headline",        # change to 'title' if your CSV uses that
    score_col: str = "score",          # if present, we will use it directly
    # aggregation / resampling
    agg: Literal["mean", "median", "sum"] = "mean",
    resample: Optional[str] = "D",     # 'D' daily, 'W' weekly; None = groupby exact date
    fill_missing_dates: bool = False,  # if True, create calendar rows with NaN score
    smoothing_window: Optional[int] = None,  # extra simple smoothing (usually not needed)
    # signal method
    use_zscore: bool = True,           # recommended: relative thresholds via z-score
    buy_threshold: float = 0.05,       # used only if use_zscore=False (absolute score)
    sell_threshold: float = -0.05,
    # z-score params
    z_lookback: int = 20,
    z_buy: float = 0.5,
    z_sell: float = -0.5,
    # decay / lag
    decay_half_life: Optional[int] = 3,   # EWMA half-life for short persistence; None to disable
    apply_lag: bool = True,               # shift signals by 1 period (t+1 execution)
    # gating
    min_news_count: int = 1,              # require at least this many headlines to allow a trade
    # parsing
    dayfirst: bool = True                 # UK-style date parsing
) -> pd.DataFrame:
    """
    Load a news CSV and return daily sentiment scores + trading signals.
    """
    # ------------------ 1) Load & basic validation ------------------ #
    df = pd.read_csv(csv_path, parse_dates=[date_col], dayfirst=dayfirst)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in CSV.")

    have_score = score_col in df.columns and pd.api.types.is_numeric_dtype(df[score_col])
    have_text = text_col in df.columns

    if not have_score and not have_text:
        raise ValueError(
            f"Neither numeric '{score_col}' nor text '{text_col}' columns found. "
            f"Provide one of them."
        )

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # ------------------ 2) Ensure per-row sentiment score ------------- #
    if have_score:
        df["score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    else:
        if not _HAS_VADER:
            raise RuntimeError(
                f"VADER not available and '{score_col}' not present. "
                f"Install vaderSentiment or provide a '{score_col}' column."
            )
        analyser = SentimentIntensityAnalyzer()

        def _score_text(s) -> float:
            if pd.isna(s):
                return 0.0
            return float(analyser.polarity_scores(str(s))["compound"])

        df["score"] = df[text_col].apply(_score_text)

    # ------------------ 3) Aggregate / resample to daily -------------- #
    if resample:
        ser = df.set_index(date_col)["score"].resample(resample)
        if agg == "mean":
            daily = ser.mean().to_frame(name="score")
        elif agg == "median":
            daily = ser.median().to_frame(name="score")
        elif agg == "sum":
            daily = ser.sum().to_frame(name="score")
        else:
            raise ValueError("agg must be 'mean', 'median', or 'sum'.")
        daily.index.name = "date"
        daily = daily.reset_index()

        if fill_missing_dates:
            # Calendar rows (NaN score) for days without news
            daily = daily.set_index("date").asfreq(resample).reset_index()
    else:
        if agg == "mean":
            daily = df.groupby(date_col)["score"].mean().reset_index(name="score")
        elif agg == "median":
            daily = df.groupby(date_col)["score"].median().reset_index(name="score")
        elif agg == "sum":
            daily = df.groupby(date_col)["score"].sum().reset_index(name="score")
        else:
            raise ValueError("agg must be 'mean', 'median', or 'sum'.")

    daily = daily.sort_values("date").reset_index(drop=True)

    # ------------------ 4) News gating (real headlines only) ---------- #
    if resample:
        counts = (
            df.set_index(date_col)["score"]
              .resample(resample).count()
              .rename("news_count")
              .to_frame()
              .reset_index()
              .rename(columns={date_col: "date"})
        )
    else:
        counts = (
            df.groupby(date_col)["score"]
              .count().rename("news_count")
              .reset_index().rename(columns={date_col: "date"})
        )

    daily = daily.merge(counts, on="date", how="left")
    daily["news_count"] = daily["news_count"].fillna(0).astype(int)

    # Optional simple smoothing (before decay)
    if smoothing_window and smoothing_window > 1:
        daily["score"] = daily["score"].rolling(smoothing_window, min_periods=1).mean()

    # ------------------ 5) Decay toward neutral ----------------------- #
    if decay_half_life and decay_half_life > 0:
        # EWMA over available rows. If fill_missing_dates=True, this includes calendar days.
        daily["score"] = daily["score"].ewm(halflife=decay_half_life, adjust=False).mean()

    # ------------------ 6) Signal generation -------------------------- #
    if use_zscore:
        rolling_mean = daily["score"].rolling(z_lookback, min_periods=max(5, z_lookback // 2)).mean()
        rolling_std = daily["score"].rolling(z_lookback, min_periods=max(5, z_lookback // 2)).std()
        rolling_std = rolling_std.replace(0.0, np.nan)

        daily["zscore"] = (daily["score"] - rolling_mean) / rolling_std

        def _signal_from_z(z: float, n: int) -> str:
            if pd.isna(z) or n < min_news_count:
                return "HOLD"
            if z >= z_buy:
                return "BUY"
            if z <= z_sell:
                return "SELL"
            return "HOLD"

        daily["signal"] = [_signal_from_z(z, n) for z, n in zip(daily["zscore"], daily["news_count"])]
    else:
        def _signal_from_abs(s: float, n: int) -> str:
            if pd.isna(s) or n < min_news_count:
                return "HOLD"
            if s >= buy_threshold:
                return "BUY"
            if s <= sell_threshold:
                return "SELL"
            return "HOLD"

        daily["signal"] = [_signal_from_abs(s, n) for s, n in zip(daily["score"], daily["news_count"])]

    # ------------------ 7) Apply t+1 lag ------------------------------ #
    if apply_lag:
        daily["signal"] = daily["signal"].shift(1).fillna("HOLD")

    return daily[["date", "score", "signal"]]
