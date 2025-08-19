# Imports
import numpy as np
import pandas as pd
from typing import Dict, Optional

def performance_metrics(
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series]=None,
        weights:Optional[pd.DataFrame]=None,
        periods_per_year:int=252,
        rf:float=0.0
) -> Dict[str,float]:
    """
     Compute common performance metrics for a trading strategy.

     Parameters
     ----------
     strategy_returns : pd.Series
         Series of periodic strategy returns (e.g., daily log/percent returns).
     benchmark_returns : pd.Series, optional
         Series of benchmark returns (e.g., buy & hold) aligned to the same dates.
         If provided, alpha and information ratio are computed.
     weights : pd.DataFrame, optional
         Portfolio weights per asset (rows = dates, cols = assets).
         If provided, turnover is computed as average absolute change in weights.
     periods_per_year : int, default=252
         Number of periods per year (252 for daily, 12 for monthly).
     rf : float, default=0.0
         Annualized risk-free rate. Adjusts Sharpe and Sortino.

     Returns
     -------
     Dict[str, float]
         Dictionary of performance metrics:
         - CAGR: Compound Annual Growth Rate
         - Ann Vol: Annualized volatility
         - Sharpe: Annualized Sharpe ratio
         - Sortino: Annualized Sortino ratio
         - Max Drawdown: Worst peak-to-trough drop
         - Calmar: CAGR / |Max Drawdown|
         - Hit Ratio: % of positive return periods
         - Turnover: Avg portfolio turnover (if weights provided)
         - Alpha (ann): Annualized alpha vs benchmark
         - Info Ratio: Information ratio vs benchmark
     """
    # Ensure numeric, drop NaNs
    rr=strategy_returns.dropna().astype(float)
    if rr.empty:
        return{}

    # Cumulative returns and total time span (in years)
    cumulative=(1+rr).cumprod()
    total_years=len(rr)/periods_per_year

    #CAGR (compound annual growth rate)
    cagr=cumulative.iloc[-1]**(1/total_years)-1 if total_years>0 else 0.0

    #Annualised Volatility (stdev * sqrt(periods_per_year))
    ann_vol=rr.std(ddof=1)*np.sqrt(periods_per_year)

    #Sharpe Ratio
    excess_ret=rr - rf / periods_per_year
    sharpe=(excess_ret.mean()*periods_per_year) / ann_vol if ann_vol !=0 else np.nan

    #Sortino Ratio (uses downside deviation only)
    downside=excess_ret[excess_ret<0].std(ddof=1)*np.sqrt(periods_per_year)
    sortino=(excess_ret.mean()*periods_per_year)/downside if downside !=0 else np.nan

    #Max Drawdown (peak to trough decline)
    running_max=cumulative.cummax()
    drawdown=cumulative/running_max - 1.0
    max_dd=drawdown.min()

    #Calmar Ratio (CAGR / |MAX DD|)
    calmar=cagr/abs(max_dd) if max_dd !=0 else np.nan

    #Hit Ratio (% of periods with positive returns)
    hit_ratio=100.0*rr.gt(0).sum()/len(rr) if len(rr)>0 else np.nan

    #Turnover (if weights provided: mean absolute change in allocations)
    turnover=None
    if weights is not None:
        turnover=weights.diff().abs().sum(axis=1).mean()

    #Alpha & Information Ratio vs Benchmark
    alpha, info_ratio = None, None
    if benchmark_returns is not None:
        aligned, bench = rr.align(benchmark_returns, join='inner')
        excess = aligned - bench
        alpha=(excess.mean()*periods_per_year) # annualised alpha
        te=excess.std(ddof=1)*np.sqrt(periods_per_year)
        info_ratio=alpha/te if te!=0 else np.nan

    return{
        'CAGR':float(cagr),
        'Ann Vol':float(ann_vol),
        'Sharpe':float(sharpe),
        'Sortino':float(sortino),
        'Max Drawdown':float(max_dd),
        'Calmar':float(calmar),
        'Hit Ratio':float(hit_ratio),
        'Turnover':float(turnover) if turnover is not None else np.nan,
        'Alpha (ann)':float(alpha) if alpha is not None else np.nan,
        'Info Ratio':float(info_ratio) if info_ratio is not None else np.nan
    }




