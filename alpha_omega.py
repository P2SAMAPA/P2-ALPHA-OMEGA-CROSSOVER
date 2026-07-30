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
    window: int = 252,  # Increased from 63 to 252 for more data
    threshold_quantile: float = 0.90  # Lowered from 0.95 to get more exceedances
) -> Dict:
    """
    Compute EVT tail risk overlay for crossover signals.
    
    Returns a full tail risk assessment including the tail index (xi).
    """
    returns = compute_log_returns(prices)
    
    # Use a longer window for EVT
    evt_window = min(window * 2, len(returns) - 1)
    if len(returns) < evt_window + 20:
        return {
            "tail_index": 0.0,  # Default to moderate tail
            "risk_factor": 1.0,
            "threshold": np.nan,
            "exceedances": 0,
            "sigma": np.nan,
            "xi": 0.0
        }
    
    # Use recent data for tail risk
    recent_returns = returns.iloc[-evt_window:]
    
    # Fit GPD to negative returns (downside risk)
    neg_returns = -recent_returns.values
    neg_returns = neg_returns[neg_returns > 0]
    
    if len(neg_returns) < 50:
        return {
            "tail_index": 0.0,
            "risk_factor": 1.0,
            "threshold": np.nan,
            "exceedances": len(neg_returns),
            "sigma": np.nan,
            "xi": 0.0
        }
    
    threshold = np.quantile(neg_returns, threshold_quantile)
    exceedances = neg_returns[neg_returns > threshold] - threshold
    
    # If we have at least 5 exceedances, try to fit
    if len(exceedances) < 5:
        # Use a fallback: estimate tail index from the data
        # Use Hill estimator as fallback
        sorted_neg = np.sort(neg_returns)[::-1]
        k = min(20, len(sorted_neg) // 10)
        if k > 5:
            log_returns_neg = np.log(sorted_neg[:k] / sorted_neg[k])
            hill_xi = np.mean(log_returns_neg)
            return {
                "tail_index": max(0.0, min(0.5, hill_xi)),  # Clamp between 0 and 0.5
                "risk_factor": 1.0 - 0.3 * max(0, min(1, hill_xi / 0.5)),
                "threshold": threshold,
                "exceedances": len(exceedances),
                "sigma": np.std(exceedances) if len(exceedances) > 0 else np.nan,
                "xi": hill_xi
            }
        return {
            "tail_index": 0.0,
            "risk_factor": 1.0,
            "threshold": threshold,
            "exceedances": len(exceedances),
            "sigma": np.nan,
            "xi": 0.0
        }
    
    try:
        xi, loc, sigma = genpareto.fit(exceedances, floc=0)
        
        # Clamp xi to reasonable values
        if np.isnan(xi) or np.isinf(xi):
            xi = 0.0
        xi = max(-0.5, min(0.8, xi))  # Clamp between -0.5 and 0.8
        
        # Risk factor: higher tail index = higher risk = reduce signal confidence
        if xi > 0.5:
            risk_factor = 0.5
        elif xi > 0.3:
            risk_factor = 0.7
        elif xi > 0.1:
            risk_factor = 0.9
        else:
            risk_factor = 1.0
        
        return {
            "tail_index": xi,
            "xi": xi,
            "risk_factor": risk_factor,
            "threshold": threshold,
            "exceedances": len(exceedances),
            "sigma": sigma
        }
    except Exception:
        # Fallback: estimate using moments
        if len(exceedances) > 5:
            # Use moment estimator for GPD
            mean_ex = np.mean(exceedances)
            var_ex = np.var(exceedances)
            if mean_ex > 0 and var_ex > 0:
                xi_est = 0.5 * (1 - (mean_ex ** 2) / var_ex)
                xi_est = max(-0.5, min(0.8, xi_est))
                return {
                    "tail_index": xi_est,
                    "risk_factor": 1.0 - 0.3 * max(0, min(1, xi_est / 0.5)),
                    "threshold": threshold,
                    "exceedances": len(exceedances),
                    "sigma": np.std(exceedances),
                    "xi": xi_est
                }
        
        return {
            "tail_index": 0.0,
            "risk_factor": 1.0,
            "threshold": threshold,
            "exceedances": len(exceedances),
            "sigma": np.nan,
            "xi": 0.0
        }


def compute_crossover_signal(
    prices: pd.Series,
    alpha_window: int = 252,
    omega_window: int = 63,
    evt_threshold: float = 0.90  # Lowered from 0.95
) -> Dict:
    """
    Complete crossover analysis with EVT overlay.
    """
    alpha, omega, crossover = compute_alpha_omega(prices, alpha_window, omega_window)
    
    if crossover.empty or len(crossover) == 0:
        return {
            "alpha": 0.0,
            "omega": 0.0,
            "crossover": 0.0,
            "z_score": 0.0,
            "tail_index": 0.0,
            "tail_risk": 1.0,
            "signal": 0.0,
            "action": "INSUFFICIENT DATA",
            "exceedances": 0
        }
    
    # Get latest values
    latest_alpha = alpha.iloc[-1] if not pd.isna(alpha.iloc[-1]) else 0
    latest_omega = omega.iloc[-1] if not pd.isna(omega.iloc[-1]) else 0
    latest_crossover = crossover.iloc[-1] if not pd.isna(crossover.iloc[-1]) else 0
    
    # Compute tail risk overlay using a longer window
    tail_risk = compute_tail_risk_overlay(
        prices, 
        window=252,  # Use 252 days for EVT
        threshold_quantile=evt_threshold
    )
    
    # Get tail index - ensure it's a float
    tail_index = tail_risk.get("tail_index", 0.0)
    if tail_index is None or np.isnan(tail_index):
        tail_index = 0.0
    
    risk_factor = tail_risk.get("risk_factor", 1.0)
    if risk_factor is None or np.isnan(risk_factor):
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
        "z_score": 0.0,  # Will be set by trainer
        "tail_index": tail_index,
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
