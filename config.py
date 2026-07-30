"""
config.py  —  Configuration for Alpha-Omega Crossover Engine
=============================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - MACRO_SIGNALS: macro columns with weights and regime signs
  - ALPHA_WINDOW: long-term momentum window (slow)
  - OMEGA_WINDOW: short-term momentum window (fast)
  - CROSSOVER_THRESHOLD: z-score threshold for signal generation
  - EVT parameters: for tail risk overlay
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""  # set via env var HF_TOKEN, or inline for local dev

DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-alpha-omega-crossover-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Alpha-Omega Configuration ───────────────────────────────────────────────

# Momentum windows
ALPHA_WINDOW = 252      # Long-term momentum (slow) — ~1 year
OMEGA_WINDOW = 63       # Short-term momentum (fast) — ~3 months

# Signal thresholds
CROSSOVER_THRESHOLD = 0.5   # z-score threshold for signal strength
MIN_DATA_POINTS = 100       # Minimum data points for reliable signal

# ── EVT Risk Overlay ─────────────────────────────────────────────────────────

EVT_THRESHOLD_QUANTILE = 0.95   # Quantile for POT threshold
RETURN_PERIOD_YEARS = 100       # 1-in-100-year event
SIGNATURE_DEPTH = 2             # Path signature depth (lighter than full EVT)
LOOKAHEAD_DAYS = 5              # Forward horizon


# ── Macro Signals ────────────────────────────────────────────────────────────
# Format: (column_name, display_name, weight, regime_sign)
# regime_sign: +1 = risk-on, -1 = risk-off

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

# Backward-compatible names for data_manager.py
MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
