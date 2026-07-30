"""
alpha_omega.py  —  Alpha-Omega Crossover Engine
================================================

Computes:
  - Alpha (long-term momentum) = rate of return over ALPHA_WINDOW
  - Omega (short-term momentum) = rate of return over OMEGA_WINDOW
  - Crossover Signal = Omega - Alpha (standardized)
  - EVT tail risk overlay = path-signature extreme value overlay

Signal interpretation:
  - Positive crossover → short-term momentum > long-term momentum → BUY signal
  - Negative crossover → short-term momentum < long-term momentum → SELL signal
  - EVT overlay: overweight crossover signals with low tail risk, underweight high tail risk
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from raw prices."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_alpha_omega(
    prices: pd.Series,
    alpha_window: int = 252,
    omega_window: int = 63
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Alpha (LT momentum) and Omega (ST momentum).
    
    Returns:
        alpha: Long-term momentum (annualized rate of return over alpha_window)
        omega: Short-term momentum (annualized rate of return over omega_window)
        crossover: Omega - Alpha (raw difference)
    """
    returns = compute_log_returns(prices)
    
    if len(returns) < alpha_window:
        return pd.Series(index=returns.index), pd.Series(index=returns.index), pd.Series(index=returns.index)
    
    # Alpha: rate of return over alpha_window (annualized)
    alpha = returns.rolling(alpha_window).mean() * 252
    
    # Omega: rate of return over omega_window (annualized)
    omega = returns.rolling(omega_window).mean() * 252
    
    # Crossover = Omega - Alpha
    crossover = omega - alpha
    
    return alpha, omega, crossover


def compute_tail_risk_overlay(
    prices: pd.Series,
    window: int = 63,
    threshold_quantile: float = 0.95
) -> Dict:
    """
    Compute EVT tail risk overlay for crossover signals.
    
    Returns a full tail risk assessment including the tail index (xi).
    """
    returns = compute_log_returns(prices)
    if len(returns) < window + 20:
        return {
            "tail_index": np.nan,
            "risk_factor": 1.0,
            "threshold": np.nan,
            "exceedances": 0,
            "sigma": np.nan,
            "xi": np.nan
        }
    
    # Use recent window for tail risk
    recent_returns = returns.iloc[-window:]
    
    # Fit GPD to negative returns (downside risk)
    neg_returns = -recent_returns.values
    neg_returns = neg_returns[neg_returns > 0]
    
    if len(neg_returns) < 20:
        return {
            "tail_index": np.nan,
            "risk_factor": 1.0,
            "threshold": np.nan,
            "exceedances": len(neg_returns),
            "sigma": np.nan,
            "xi": np.nan
        }
    
    threshold = np.quantile(neg_returns, threshold_quantile)
    exceedances = neg_returns[neg_returns > threshold] - threshold
    
    if len(exceedances) < 10:
        return {
            "tail_index": np.nan,
            "risk_factor": 1.0,
            "threshold": threshold,
            "exceedances": len(exceedances),
            "sigma": np.nan,
            "xi": np.nan
        }
    
    try:
        xi, loc, sigma = genpareto.fit(exceedances, floc=0)
        
        # Risk factor: higher tail index = higher risk = reduce signal confidence
        # Xi > 0.3 = heavy tail → reduce signal by 30%
        # Xi > 0.5 = very heavy tail → reduce signal by 50%
        if np.isnan(xi):
            risk_factor = 1.0
        elif xi > 0.5:
            risk_factor = 0.5
        elif xi > 0.3:
            risk_factor = 0.7
        elif xi > 0.1:
            risk_factor = 0.9
        else:
            risk_factor = 1.0
        
        return {
            "tail_index": xi,  # This is the key metric we want
            "xi": xi,          # Keep as backup
            "risk_factor": risk_factor,
            "threshold": threshold,
            "exceedances": len(exceedances),
            "sigma": sigma
        }
    except Exception as e:
        return {
            "tail_index": np.nan,
            "risk_factor": 1.0,
            "threshold": threshold,
            "exceedances": len(exceedances),
            "sigma": np.nan,
            "xi": np.nan
        }


def compute_crossover_signal(
    prices: pd.Series,
    alpha_window: int = 252,
    omega_window: int = 63,
    evt_threshold: float = 0.95
) -> Dict:
    """
    Complete crossover analysis with EVT overlay.
    
    Returns:
        alpha: Latest Alpha value
        omega: Latest Omega value
        crossover: Omega - Alpha
        z_score: Standardized crossover across universe
        tail_index: EVT tail index (xi)
        tail_risk: Risk factor (1.0 = no reduction, 0.5 = 50% reduction)
        signal: Adjusted signal strength
        action: BUY/SELL/HOLD recommendation
        exceedances: Number of exceedances for EVT
    """
    alpha, omega, crossover = compute_alpha_omega(prices, alpha_window, omega_window)
    
    if crossover.empty or len(crossover) == 0:
        return {
            "alpha": np.nan,
            "omega": np.nan,
            "crossover": np.nan,
            "z_score": np.nan,
            "tail_index": np.nan,
            "tail_risk": 1.0,
            "signal": 0.0,
            "action": "INSUFFICIENT DATA",
            "exceedances": 0
        }
    
    # Get latest values
    latest_alpha = alpha.iloc[-1] if not pd.isna(alpha.iloc[-1]) else 0
    latest_omega = omega.iloc[-1] if not pd.isna(omega.iloc[-1]) else 0
    latest_crossover = crossover.iloc[-1] if not pd.isna(crossover.iloc[-1]) else 0
    
    # Compute tail risk overlay (using a longer window for stability)
    tail_risk = compute_tail_risk_overlay(
        prices, 
        window=min(omega_window * 2, 252), 
        threshold_quantile=evt_threshold
    )
    
    # Get tail index - ensure it's a float, not None
    tail_index = tail_risk.get("tail_index", np.nan)
    if tail_index is None:
        tail_index = np.nan
    
    risk_factor = tail_risk.get("risk_factor", 1.0)
    if risk_factor is None:
        risk_factor = 1.0
    
    # Signal = crossover * risk_factor (adjust for tail risk)
    signal = latest_crossover * risk_factor
    
    # Determine action (will be refined by trainer with z-scores)
    if abs(signal) < 0.01:
        action = "HOLD"
    elif signal > 0:
        action = "BUY"
    else:
        action = "SELL"
    
    return {
        "alpha": latest_alpha,
        "omega": latest_omega,
        "crossover": latest_crossover,
        "z_score": np.nan,  # Will be set by trainer
        "tail_index": tail_index,  # The actual tail index (xi)
        "tail_risk": risk_factor,
        "signal": signal,
        "action": action,
        "exceedances": tail_risk.get("exceedances", 0),
        "threshold": tail_risk.get("threshold", np.nan),
        "sigma": tail_risk.get("sigma", np.nan)
    }


def compute_cross_sectional_zscore(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Compute cross-sectional z-scores within a universe.
    """
    values = np.array([v for v in scores.values() if not np.isnan(v)])
    if len(values) < 2:
        return {t: 0.0 for t in scores.keys()}
    
    mean = np.mean(values)
    std = np.std(values)
    if std == 0 or np.isnan(std):
        return {t: 0.0 for t in scores.keys()}
    
    return {t: (scores[t] - mean) / std if not np.isnan(scores[t]) else 0.0 
            for t in scores.keys()}
