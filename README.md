# P2-ALPHA-OMEGA-CROSSOVER

**Alpha-Omega Crossover Engine — Momentum Regime Detection with EVT Overlay**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine identifies momentum regime shifts by comparing **long-term momentum (Alpha)** with **short-term momentum (Omega)** , with a tail risk overlay from Extreme Value Theory.

### Theory

**Alpha** (long-term momentum) = rate of return over 252 days (slow trend)
**Omega** (short-term momentum) = rate of return over 63 days (fast trend)

When **Omega crosses above Alpha**, momentum is accelerating → **BUY** signal
When **Omega crosses below Alpha**, momentum is decelerating → **SELL** signal

**EVT Risk Overlay:** Reduces signal confidence in high tail-risk environments.

### Signal Interpretation

| Condition | Signal | Action |
|-----------|--------|--------|
| Omega > Alpha + z > 0.5 | STRONG BUY | Add exposure |
| Omega > Alpha + z > 0.2 | BUY | Increase exposure |
| Omega ≈ Alpha | HOLD | Maintain position |
| Omega < Alpha + z < -0.2 | REDUCE | Decrease exposure |
| Omega < Alpha + z < -0.5 | STRONG SELL | Exit position |

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| ALPHA_WINDOW | Long-term momentum window | 252 days |
| OMEGA_WINDOW | Short-term momentum window | 63 days |
| CROSSOVER_THRESHOLD | z-score threshold for signal strength | 0.5 |
| EVT_THRESHOLD_QUANTILE | Quantile for EVT overlay | 0.95 |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ALPHA-OMEGA-CROSSOVER
cd P2-ALPHA-OMEGA-CROSSOVER
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday via .github/workflows/daily.yml.

Required secret: HF_TOKEN

text

---

## Summary

This new repo follows the **exact same structure** as your Hilbert and Rough-EVT repos:

| Component | Purpose |
|-----------|---------|
| **config.py** | All hyperparameters (universes, windows, thresholds) |
| **data_manager.py** | Loads data from HuggingFace (identical to Hilbert) |
| **alpha_omega.py** | Core engine: Alpha/Omega crossover + EVT overlay |
| **trainer.py** | Orchestrates: load → compute → JSON → upload |
| **push_results.py** | HuggingFace upload wrapper |
| **streamlit_app.py** | Professional two-tab dashboard |
| **.github/workflows/daily.yml** | Scheduled runs |

The dashboard shows:
- **Tab 1**: Top Buys (Omega > Alpha) and Top Sells (Omega < Alpha)
- **Tab 2**: Full breakdown with all metrics
