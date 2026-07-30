import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfApi
from datetime import date, timedelta
import config
import os
import numpy as np

st.set_page_config(page_title="Alpha-Omega Crossover Engine", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#1a1a2e;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
           padding-left:0.5rem;border-left:5px solid #2ecc71}
.uni-title-sell{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
                padding-left:0.5rem;border-left:5px solid #e74c3c}
.buy-card{background:linear-gradient(135deg,#1a472a 0%,#2d6a4f 60%,#40916c 100%);
          color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
          box-shadow:0 6px 20px rgba(39,174,96,0.3)}
.sell-card{background:linear-gradient(135deg,#4a1a1a 0%,#6a2d2d 60%,#914040 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(231,76,60,0.3)}
.hold-card{background:linear-gradient(135deg,#2c3e50 0%,#4a5d6a 60%,#5d7a8a 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(44,62,80,0.3)}
.best-card{background:linear-gradient(135deg,#0d47a1 0%,#1565c0 60%,#1e88e5 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(21,101,192,0.4)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.badge-buy{background:#27ae60;border-radius:6px;padding:2px 12px;font-size:0.75rem;
           font-weight:700;color:white}
.badge-sell{background:#e74c3c;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-hold{background:#f39c12;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-best{background:#1a237e;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.tail-badge{background:#8e44ad;border-radius:6px;padding:2px 8px;font-size:0.7rem;
            font-weight:700;color:white}
.tail-badge-heavy{background:#c0392b;border-radius:6px;padding:2px 8px;font-size:0.7rem;
                  font-weight:700;color:white}
.tail-badge-moderate{background:#f39c12;border-radius:6px;padding:2px 8px;font-size:0.7rem;
                     font-weight:700;color:white}
.tail-badge-thin{background:#27ae60;border-radius:6px;padding:2px 8px;font-size:0.7rem;
                 font-weight:700;color:white}
.composite-badge{background:#1a237e;border-radius:6px;padding:2px 8px;font-size:0.7rem;
                 font-weight:700;color:white}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">〜 Alpha-Omega Crossover Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Short-term momentum (Omega) crossing long-term momentum (Alpha) · '
    'EVT tail risk overlay · Composite scoring · Cross-sectional z-scores</div>',
    unsafe_allow_html=True)

HF_TOKEN = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
RESULTS_REPO = config.RESULTS_REPO

US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def next_trading_day() -> str:
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d.strftime("%B %d, %Y")

def action_badge(action: str) -> str:
    if "BUY" in action:
        return f'<span class="badge-buy">🟢 {action}</span>'
    elif "SELL" in action:
        return f'<span class="badge-sell">🔴 {action}</span>'
    else:
        return f'<span class="badge-hold">🟡 {action}</span>'

def tail_badge(xi) -> str:
    if xi is None or pd.isna(xi) or np.isnan(xi):
        return f'<span class="tail-badge">ξ = N/A</span>'
    if xi > 0.5:
        return f'<span class="tail-badge-heavy">ξ = {xi:.3f} (VERY HEAVY)</span>'
    elif xi > 0.3:
        return f'<span class="tail-badge-heavy">ξ = {xi:.3f} (HEAVY)</span>'
    elif xi > 0.0:
        return f'<span class="tail-badge-moderate">ξ = {xi:.3f} (MODERATE)</span>'
    else:
        return f'<span class="tail-badge-thin">ξ = {xi:.3f} (THIN)</span>'

def composite_badge(score: float) -> str:
    if score > 0.6:
        return f'<span class="badge-best">⭐ TOP</span>'
    elif score > 0.3:
        return f'<span class="badge-buy">✅ GOOD</span>'
    elif score > -0.3:
        return f'<span class="badge-hold">⚖️ NEUTRAL</span>'
    elif score > -0.6:
        return f'<span class="badge-sell">⚠️ POOR</span>'
    else:
        return f'<span class="badge-sell">🔴 WORST</span>'

def safe_float(val, default=0.0):
    """Safely convert to float, handling None and NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default

def compute_composite_score(z_score, tail_index, tail_risk, exceedances):
    """
    Compute a composite score combining all metrics.
    
    Components:
    1. z-score: higher = better (positive momentum)
    2. tail_index: lower = better (less extreme risk)
    3. tail_risk: higher = better (more confidence)
    4. exceedances: higher = better (more data reliability)
    
    Returns: score between -1 and 1
    """
    # Normalize z-score (already roughly normalized)
    z_comp = z_score
    
    # Tail index: lower is better (negative = bounded downside)
    # Map: ξ > 0.5 → -1, ξ = 0 → 0, ξ < -0.3 → +1
    if np.isnan(tail_index):
        tail_comp = 0
    else:
        tail_comp = -np.clip(tail_index / 0.5, -1, 1)
    
    # Tail risk: higher is better (more confidence)
    # Map: 100% → +1, 50% → 0, 0% → -1
    tail_risk_comp = (safe_float(tail_risk, 1.0) - 0.5) * 2
    tail_risk_comp = np.clip(tail_risk_comp, -1, 1)
    
    # Exceedances: more is better (more data)
    # Map: 20+ → +1, 10 → 0, 0 → -1
    exceed_comp = np.clip((safe_float(exceedances, 10) - 10) / 10, -1, 1)
    
    # Weighted composite (weights sum to 1)
    # z-score: 40%, tail_index: 30%, tail_risk: 20%, exceedances: 10%
    composite = (0.40 * z_comp + 
                 0.30 * tail_comp + 
                 0.20 * tail_risk_comp + 
                 0.10 * exceed_comp)
    
    return np.clip(composite, -1, 1)

def get_card_class(composite_score):
    """Get CSS class based on composite score."""
    if composite_score > 0.5:
        return "best-card"
    elif composite_score > 0.1:
        return "buy-card"
    elif composite_score > -0.1:
        return "hold-card"
    elif composite_score > -0.5:
        return "sell-card"
    else:
        return "sell-card"

def get_emoji(composite_score):
    """Get emoji based on composite score."""
    if composite_score > 0.6:
        return "⭐"
    elif composite_score > 0.3:
        return "🟢"
    elif composite_score > -0.3:
        return "🟡"
    elif composite_score > -0.6:
        return "🟠"
    else:
        return "🔴"

@st.cache_data(ttl=3600)
def list_repo_files():
    if not HF_TOKEN:
        st.sidebar.warning("⚠️ HF_TOKEN not set")
        return []
    try:
        api = HfApi(token=HF_TOKEN)
        return api.list_repo_files(repo_id=RESULTS_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        return []


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f], reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json_from_hf(path):
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set"}
    try:
        api = HfApi(token=HF_TOKEN)
        content = api.hf_hub_download(repo_id=RESULTS_REPO, filename=path, repo_type="dataset", token=HF_TOKEN)
        with open(content, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 〜 Alpha-Omega")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Alpha (LT):** {config.ALPHA_WINDOW}d")
st.sidebar.markdown(f"**Omega (ST):** {config.OMEGA_WINDOW}d")
st.sidebar.markdown(f"**EVT Threshold:** {config.EVT_THRESHOLD_QUANTILE:.0%}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Composite Score Weights:**")
st.sidebar.markdown("  • z-score: 40%")
st.sidebar.markdown("  • Tail Index: 30%")
st.sidebar.markdown("  • Tail Risk: 20%")
st.sidebar.markdown("  • Exceedances: 10%")
st.sidebar.markdown("---")
st.sidebar.markdown("**Macro signals:**")
for col, desc, w, sign in config.MACRO_SIGNALS:
    arrow = "↑risk-on" if sign > 0 else "↑risk-off"
    st.sidebar.markdown(f"  • {col} ({arrow}, w={w:.0%})")

# ── Load data ─────────────────────────────────────────────────────────────────
files = list_repo_files()
if not files:
    st.error("No files found. Run trainer.py first.")
    st.info(f"Looking in: {RESULTS_REPO}")
    st.stop()

tab1_path = find_latest(files, "alpha_omega_")
tab2_path = find_latest(files, "alpha_omega_breakdown_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json_from_hf(tab1_path)
if "error" in data1:
    st.error(f"Error: {data1['error']}")
    st.stop()

data2 = load_json_from_hf(tab2_path) if tab2_path else None
universes1 = data1.get("universes", {})
universes2 = data2.get("universes", {}) if data2 and "error" not in data2 else None

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")
st.sidebar.success(f"✅ {len(universes1)} universes")

tab1, tab2, tab3 = st.tabs(["🏆 Composite Ranking", "🔍 Full Breakdown", "📊 Buy/Sell Signals"])

UNIVERSE_ORDER = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED": "🌐 Combined",
}

ntd = next_trading_day()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - COMPOSITE RANKING (GREEN TO RED)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Composite Ranking — Best to Worst (Green → Red)")

    with st.expander("📖 How Composite Score Works", expanded=True):
        st.markdown("""
**Composite Score combines all metrics into a single actionable number:**

| Component | Weight | Why |
|-----------|--------|-----|
| **z-score** | 40% | Primary signal — positive momentum |
| **Tail Index (ξ)** | 30% | Risk profile — lower is better |
| **Tail Risk** | 20% | Signal confidence — higher is better |
| **Exceedances** | 10% | Data quality — more is better |

**Score Interpretation:**
- **> 0.6** ⭐ **TOP** — Best risk/reward, strong buy
- **0.3 to 0.6** 🟢 **GOOD** — Favorable risk/reward
- **-0.3 to 0.3** 🟡 **NEUTRAL** — Balanced, hold
- **-0.6 to -0.3** 🟠 **POOR** — Unfavorable, reduce
- **< -0.6** 🔴 **WORST** — Worst risk/reward, strong sell

**Color Coding:** Green (best) → Yellow (neutral) → Red (worst)
        """)

    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        full_scores = uni_data.get("full_scores", {})
        
        if not full_scores:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        
        # Compute composite scores for all ETFs
        ranked_etfs = []
        for ticker, info in full_scores.items():
            z_score = safe_float(info.get("z_score", 0))
            tail_index = info.get("tail_index")
            if tail_index is None:
                tail_index = np.nan
            tail_index = safe_float(tail_index, 0)
            tail_risk = safe_float(info.get("tail_risk", 1.0))
            exceedances = safe_float(info.get("exceedances", 0))
            
            composite = compute_composite_score(z_score, tail_index, tail_risk, exceedances)
            
            ranked_etfs.append({
                "ticker": ticker,
                "z_score": z_score,
                "tail_index": tail_index,
                "tail_risk": tail_risk,
                "exceedances": exceedances,
                "composite": composite,
                "action": info.get("action", "HOLD"),
                "alpha": safe_float(info.get("alpha", 0)),
                "omega": safe_float(info.get("omega", 0))
            })
        
        # Sort by composite score (highest = best)
        ranked_etfs = sorted(ranked_etfs, key=lambda x: x["composite"], reverse=True)
        
        st.markdown(f'<div class="uni-title">{label} — Ranked by Composite Score</div>', unsafe_allow_html=True)
        
        # Display in 4 columns
        cols = st.columns(4)
        for idx, etf in enumerate(ranked_etfs[:4]):  # Show top 4 in cards
            ticker = etf["ticker"]
            composite = etf["composite"]
            z_score = etf["z_score"]
            tail_index = etf["tail_index"]
            tail_risk = etf["tail_risk"]
            action = etf["action"]
            
            card_class = get_card_class(composite)
            emoji = get_emoji(composite)
            badge = composite_badge(composite)
            tail_badge_html = tail_badge(tail_index)
            
            with cols[idx]:
                st.markdown(f"""
<div class="{card_class}">
  <div class="ticker">{emoji} {ticker}</div>
  <div class="score">Composite = {composite:+.3f}</div>
  <div class="score">{badge}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">{tail_badge_html}</div>
  <div class="score">Tail Risk = {tail_risk*100:.0f}%</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        
        # Full ranking table with color coding
        with st.expander(f"📋 Full Composite Ranking — {label}"):
            rows = []
            for idx, etf in enumerate(ranked_etfs):
                # Determine row color
                comp = etf["composite"]
                if comp > 0.3:
                    color = "#27ae60"  # Green
                elif comp > 0:
                    color = "#f1c40f"  # Yellow
                elif comp > -0.3:
                    color = "#e67e22"  # Orange
                else:
                    color = "#e74c3c"  # Red
                
                rows.append({
                    "Rank": idx + 1,
                    "ETF": etf["ticker"],
                    "Composite Score": round(comp, 4),
                    "z-score": round(etf["z_score"], 4),
                    "Tail Index (ξ)": round(etf["tail_index"], 4),
                    "Tail Risk": f"{etf['tail_risk']*100:.0f}%",
                    "Exceedances": int(etf["exceedances"]),
                    "Action": etf["action"],
                    "Alpha": round(etf["alpha"], 4),
                    "Omega": round(etf["omega"], 4)
                })
            
            df_rank = pd.DataFrame(rows)
            
            # Apply color formatting
            def color_rank(val):
                if isinstance(val, (int, float)):
                    if val <= 3:
                        return 'background-color: #27ae60; color: white;'
                    elif val <= 6:
                        return 'background-color: #2ecc71; color: white;'
                    elif val <= 10:
                        return 'background-color: #f1c40f; color: black;'
                    elif val <= 15:
                        return 'background-color: #e67e22; color: white;'
                    else:
                        return 'background-color: #e74c3c; color: white;'
                return ''
            
            def color_composite(val):
                if val > 0.3:
                    return 'background-color: #27ae60; color: white;'
                elif val > 0:
                    return 'background-color: #f1c40f; color: black;'
                elif val > -0.3:
                    return 'background-color: #e67e22; color: white;'
                else:
                    return 'background-color: #e74c3c; color: white;'
            
            styled_df = df_rank.style.applymap(color_rank, subset=['Rank']).applymap(color_composite, subset=['Composite Score'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Summary stats
            st.caption(f"**Summary:** Best: {ranked_etfs[0]['ticker']} ({ranked_etfs[0]['composite']:+.3f}) | Worst: {ranked_etfs[-1]['ticker']} ({ranked_etfs[-1]['composite']:+.3f})")
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · Composite = z-score(40%) + Tail Index(30%) + Tail Risk(20%) + Exceedances(10%)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - FULL BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Full Signal Breakdown with Composite Score")

    if not universes2:
        st.warning("Breakdown data not found.")
        st.stop()

    for universe_name in UNIVERSE_ORDER:
        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        uni_data = universes2.get(universe_name, {})
        ranking = uni_data.get("full_ranking", [])

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        if not ranking:
            st.info(f"No data for {universe_name}")
            st.divider()
            continue

        rows = []
        for item in ranking:
            z_score = safe_float(item.get("z_score", 0))
            tail_idx = item.get("tail_index")
            if tail_idx is None:
                tail_idx = np.nan
            tail_idx = safe_float(tail_idx, 0)
            tail_risk = safe_float(item.get("tail_risk", 1.0))
            exceedances = safe_float(item.get("exceedances", 0))
            
            # Compute composite score
            composite = compute_composite_score(z_score, tail_idx, tail_risk, exceedances)
            
            rows.append({
                "ETF": item.get("ticker", ""),
                "Composite Score": round(composite, 4),
                "z-score": round(z_score, 4),
                "Alpha (252d)": round(safe_float(item.get("alpha", 0)), 4),
                "Omega (63d)": round(safe_float(item.get("omega", 0)), 4),
                "Crossover": round(safe_float(item.get("crossover", 0)), 4),
                "Tail Index (ξ)": round(tail_idx, 4),
                "Tail Risk": f"{tail_risk*100:.0f}%",
                "Exceedances": int(exceedances),
                "Action": item.get("action", "HOLD")
            })

        df = pd.DataFrame(rows).sort_values("Composite Score", ascending=False)
        
        # Color formatting for composite
        def color_composite_col(val):
            if isinstance(val, (int, float)):
                if val > 0.3:
                    return 'background-color: #27ae60; color: white;'
                elif val > 0:
                    return 'background-color: #f1c40f; color: black;'
                elif val > -0.3:
                    return 'background-color: #e67e22; color: white;'
                else:
                    return 'background-color: #e74c3c; color: white;'
            return ''
        
        styled_df = df.style.applymap(color_composite_col, subset=['Composite Score'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data2.get('run_date','?')} · Alpha = {config.ALPHA_WINDOW}d · Omega = {config.OMEGA_WINDOW}d")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - BUY/SELL SIGNALS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Buy/Sell Signals Summary")

    st.markdown("""
    ### Quick Action Guide
    
    | Score Range | Action | Color |
    |-------------|--------|-------|
    | **> 0.6** | ⭐ STRONG BUY | 🟢 Green |
    | **0.3 - 0.6** | ✅ BUY | 🟢 Light Green |
    | **0 - 0.3** | ⚖️ HOLD | 🟡 Yellow |
    | **-0.3 - 0** | ⚖️ HOLD with Caution | 🟠 Orange |
    | **-0.6 - -0.3** | ⚠️ REDUCE | 🟠 Orange-Red |
    | **< -0.6** | 🔴 STRONG SELL | 🔴 Red |
    """)

    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        full_scores = uni_data.get("full_scores", {})
        
        if not full_scores:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        
        # Compute composite scores
        ranked_etfs = []
        for ticker, info in full_scores.items():
            z_score = safe_float(info.get("z_score", 0))
            tail_idx = info.get("tail_index")
            if tail_idx is None:
                tail_idx = np.nan
            tail_idx = safe_float(tail_idx, 0)
            tail_risk = safe_float(info.get("tail_risk", 1.0))
            exceedances = safe_float(info.get("exceedances", 0))
            
            composite = compute_composite_score(z_score, tail_idx, tail_risk, exceedances)
            
            ranked_etfs.append({
                "ticker": ticker,
                "composite": composite,
                "action": info.get("action", "HOLD")
            })
        
        ranked_etfs = sorted(ranked_etfs, key=lambda x: x["composite"], reverse=True)
        
        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)
        
        # Split into buy, hold, sell
        buys = [e for e in ranked_etfs if e["composite"] > 0.1]
        holds = [e for e in ranked_etfs if -0.1 <= e["composite"] <= 0.1]
        sells = [e for e in ranked_etfs if e["composite"] < -0.1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🟢 BUY Signals")
            if buys:
                for e in buys[:5]:
                    st.markdown(f"**{e['ticker']}** — {e['composite']:+.3f} ({e['action']})")
            else:
                st.caption("No buy signals")
        
        with col2:
            st.markdown("#### 🟡 HOLD Signals")
            if holds:
                for e in holds[:5]:
                    st.markdown(f"**{e['ticker']}** — {e['composite']:+.3f} ({e['action']})")
            else:
                st.caption("No hold signals")
        
        with col3:
            st.markdown("#### 🔴 SELL Signals")
            if sells:
                for e in sells[:5]:
                    st.markdown(f"**{e['ticker']}** — {e['composite']:+.3f} ({e['action']})")
            else:
                st.caption("No sell signals")
        
        st.divider()
    
    st.caption(f"Run date: {data1.get('run_date','?')} · Composite scoring across all metrics")
