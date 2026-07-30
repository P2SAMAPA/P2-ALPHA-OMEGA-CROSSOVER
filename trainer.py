"""
trainer.py  —  Orchestrator for Alpha-Omega Crossover pipeline
===============================================================
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

import config
from data_manager import load_master_data, validate_data
from alpha_omega import (
    compute_crossover_signal, 
    compute_cross_sectional_zscore,
    compute_alpha_omega
)
from push_results import upload_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """
    Run the full Alpha-Omega Crossover pipeline.
    """
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — will skip HuggingFace upload.")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("🔄 Loading master data from HuggingFace...")
    try:
        prices_df, macro_df = load_master_data(token)
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(
        f"✅ Loaded {len(prices_df)} days, "
        f"{len(prices_df.columns)} ETFs, "
        f"{len(macro_df.columns)} macro cols"
    )

    run_date = datetime.now().strftime("%Y-%m-%d")

    # ── Results containers ────────────────────────────────────────────────────
    results_tab1 = {
        "run_date": run_date,
        "universes": {}
    }

    results_tab2 = {
        "run_date": run_date,
        "universes": {}
    }

    # ── Process each universe ─────────────────────────────────────────────────
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Processing universe: {universe_name}")

        available = [t for t in tickers if t in prices_df.columns]
        logger.info(f"   Available: {len(available)}/{len(tickers)}")

        if not available:
            continue

        # Store results with full data
        ticker_signals = {}
        ticker_alpha = {}
        ticker_omega = {}
        ticker_crossover = {}
        ticker_tail_index = {}
        ticker_tail_risk = {}
        ticker_exceedances = {}
        ticker_signal = {}

        # ── Compute for each ticker ────────────────────────────────────────────
        for ticker in available:
            logger.info(f"   Computing {ticker}...")
            prices = prices_df[ticker]
            
            result = compute_crossover_signal(
                prices,
                alpha_window=config.ALPHA_WINDOW,
                omega_window=config.OMEGA_WINDOW,
                evt_threshold=config.EVT_THRESHOLD_QUANTILE
            )
            
            if result.get("action") != "INSUFFICIENT DATA":
                ticker_signals[ticker] = result.get("signal", 0)
                ticker_alpha[ticker] = result.get("alpha", 0)
                ticker_omega[ticker] = result.get("omega", 0)
                ticker_crossover[ticker] = result.get("crossover", 0)
                # Store tail_index - ensure it's a float
                tail_idx = result.get("tail_index", np.nan)
                if tail_idx is None:
                    tail_idx = np.nan
                ticker_tail_index[ticker] = tail_idx
                ticker_tail_risk[ticker] = result.get("tail_risk", 1.0)
                ticker_exceedances[ticker] = result.get("exceedances", 0)
                ticker_signal[ticker] = result.get("signal", 0)
                
                logger.info(f"      {ticker}: tail_index = {tail_idx:.4f}")

        # ── Cross-sectional z-scores ──────────────────────────────────────────
        if ticker_signals:
            z_scores = compute_cross_sectional_zscore(ticker_signals)
            
            # Determine action based on z-score and tail risk
            actions = {}
            for ticker, z in z_scores.items():
                tail_risk = ticker_tail_risk.get(ticker, 1.0)
                # Adjust z-score by tail risk
                adjusted_z = z * tail_risk
                
                if adjusted_z > config.CROSSOVER_THRESHOLD:
                    actions[ticker] = "STRONG BUY"
                elif adjusted_z > 0.2:
                    actions[ticker] = "BUY"
                elif adjusted_z > -0.2:
                    actions[ticker] = "HOLD"
                elif adjusted_z > -config.CROSSOVER_THRESHOLD:
                    actions[ticker] = "REDUCE"
                else:
                    actions[ticker] = "STRONG SELL"
            
            # Top 5 buys
            top_buys = sorted(
                [(t, z_scores[t]) for t in z_scores if not np.isnan(z_scores[t])],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Top 5 sells
            top_sells = sorted(
                [(t, z_scores[t]) for t in z_scores if not np.isnan(z_scores[t])],
                key=lambda x: x[1]
            )[:5]
            
            # Build Tab 1 with FULL data
            results_tab1["universes"][universe_name] = {
                "top_buys": [
                    {"ticker": t, "z_score": z} for t, z in top_buys
                ],
                "top_sells": [
                    {"ticker": t, "z_score": z} for t, z in top_sells
                ],
                "full_scores": {
                    t: {
                        "z_score": z_scores.get(t, 0),
                        "alpha": ticker_alpha.get(t, 0),
                        "omega": ticker_omega.get(t, 0),
                        "crossover": ticker_crossover.get(t, 0),
                        "tail_index": ticker_tail_index.get(t, np.nan),  # MUST be here
                        "tail_risk": ticker_tail_risk.get(t, 1.0),
                        "exceedances": ticker_exceedances.get(t, 0),
                        "action": actions.get(t, "HOLD")
                    }
                    for t in ticker_signals.keys()
                }
            }
            
            # Build Tab 2
            results_tab2["universes"][universe_name] = {
                "full_ranking": [
                    {
                        "ticker": t,
                        "z_score": z_scores.get(t, 0),
                        "alpha": ticker_alpha.get(t, 0),
                        "omega": ticker_omega.get(t, 0),
                        "crossover": ticker_crossover.get(t, 0),
                        "tail_index": ticker_tail_index.get(t, np.nan),  # MUST be here
                        "tail_risk": ticker_tail_risk.get(t, 1.0),
                        "exceedances": ticker_exceedances.get(t, 0),
                        "action": actions.get(t, "HOLD")
                    }
                    for t in ticker_signals.keys()
                ]
            }

    # ── Save JSON files ──────────────────────────────────────────────────────
    logger.info("\n💾 Saving JSON results...")

    tab1_path = f"alpha_omega_{run_date}.json"
    tab2_path = f"alpha_omega_breakdown_{run_date}.json"

    with open(tab1_path, "w") as f:
        json.dump(results_tab1, f, indent=2)

    with open(tab2_path, "w") as f:
        json.dump(results_tab2, f, indent=2)

    logger.info(f"   Saved: {tab1_path}")
    logger.info(f"   Saved: {tab2_path}")

    # ── Upload to HuggingFace ───────────────────────────────────────────────
    if token:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            upload_results(tab1_path, tab2_path, token)
        except Exception as e:
            logger.error(f"   Upload failed: {e}")
    else:
        logger.info("\n📤 Skipping upload (no HF_TOKEN)")

    return {"tab1": results_tab1, "tab2": results_tab2}


if __name__ == "__main__":
    run_trainer()
