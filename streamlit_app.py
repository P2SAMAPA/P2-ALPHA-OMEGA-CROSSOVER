import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfApi
from datetime import date, timedelta
import config
import os

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
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.metric-box{background:#f8f9fa;border-radius:10px;padding:0.8rem;margin:0.3rem 0;
            border-left:4px solid #2ecc71}
.metric-label{font-size:0.75rem;color:#666;text-transform:uppercase;letter-spacing:0.5px}
.metric-value{font-size:1.1rem;font-weight:700;color:#1a1a2e}
.badge-buy{background:#27ae60;border-radius:6px;padding:2px 12px;font-size:0.75rem;
           font-weight:700;color:white}
.badge-sell{background:#e74c3c;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-hold{background:#f39c12;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">〜 Alpha-Omega Crossover Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Short-term momentum (Omega) crossing long-term momentum (Alpha) · '
    'EVT tail risk overlay · Cross-sectional z-scores</div>',
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

tab1, tab2 = st.tabs(["🏆 Top Buys & Sells", "🔍 Full Signal Breakdown"])

UNIVERSE_ORDER = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED": "🌐 Combined",
}

ntd = next_trading_day()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top Buys & Sells — Crossover Signals")

    with st.expander("📖 How Alpha-Omega Works", expanded=True):
        st.markdown("""
**Alpha-Omega Crossover** identifies momentum regime shifts:

| Metric | What it is | Signal |
|--------|-----------|--------|
| **Alpha** | Long-term momentum (252d) | Slow trend |
| **Omega** | Short-term momentum (63d) | Fast trend |
| **Crossover** | Omega - Alpha | Positive = momentum accelerating |
| **EVT Overlay** | Tail risk adjustment | Reduces signal in high-risk environments |

**Crossover Interpretation:**
- **Omega > Alpha** → Short-term momentum exceeds long-term → **BUY** (momentum accelerating)
- **Omega < Alpha** → Short-term momentum lags long-term → **SELL** (momentum decelerating)
- **EVT Risk Overlay** → Heavy tails reduce signal confidence
        """)

    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        top_buys = uni_data.get("top_buys", [])
        top_sells = uni_data.get("top_sells", [])
        
        if not top_buys and not top_sells:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        
        # Show Top Buys
        st.markdown(f'<div class="uni-title">🟢 {label} — Top Buys (Omega > Alpha)</div>', unsafe_allow_html=True)
        if top_buys:
            cols = st.columns(3)
            for idx, item in enumerate(top_buys[:3]):
                ticker = item["ticker"]
                z_score = item["z_score"]
                full_data = uni_data.get("full_scores", {}).get(ticker, {})
                action = full_data.get("action", "HOLD")
                alpha = full_data.get("alpha", 0)
                omega = full_data.get("omega", 0)
                tail_risk = full_data.get("tail_risk", 1.0)
                
                with cols[idx]:
                    st.markdown(f"""
<div class="buy-card">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">α = {alpha:.3f} | ω = {omega:.3f}</div>
  <div class="score">Tail Risk = {(1-tail_risk)*100:.0f}% reduction</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No BUY signals in this universe")

        # Show Top Sells
        st.markdown(f'<div class="uni-title-sell">🔴 {label} — Top Sells (Omega < Alpha)</div>', unsafe_allow_html=True)
        if top_sells:
            cols = st.columns(3)
            for idx, item in enumerate(top_sells[:3]):
                ticker = item["ticker"]
                z_score = item["z_score"]
                full_data = uni_data.get("full_scores", {}).get(ticker, {})
                action = full_data.get("action", "HOLD")
                alpha = full_data.get("alpha", 0)
                omega = full_data.get("omega", 0)
                
                with cols[idx]:
                    st.markdown(f"""
<div class="sell-card">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">α = {alpha:.3f} | ω = {omega:.3f}</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No SELL signals in this universe")

        # Full ranking
        with st.expander(f"📋 Full ranking — {label}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    rows.append({
                        "ETF": t,
                        "z-score": round(info.get("z_score", 0), 4),
                        "Alpha": round(info.get("alpha", 0), 4),
                        "Omega": round(info.get("omega", 0), 4),
                        "Crossover": round(info.get("crossover", 0), 4),
                        "Tail Index": round(info.get("tail_index", 0), 4),
                        "Action": info.get("action", "HOLD")
                    })
                df_rank = pd.DataFrame(rows).sort_values("z-score", ascending=False)
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · Alpha = {config.ALPHA_WINDOW}d · Omega = {config.OMEGA_WINDOW}d · EVT overlay")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Full Signal Breakdown")

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
            rows.append({
                "ETF": item.get("ticker", ""),
                "z-score": round(item.get("z_score", 0), 4),
                "Alpha (252d)": round(item.get("alpha", 0), 4),
                "Omega (63d)": round(item.get("omega", 0), 4),
                "Crossover": round(item.get("crossover", 0), 4),
                "Tail Index": round(item.get("tail_index", 0), 4),
                "Tail Risk": f"{item.get('tail_risk', 1.0)*100:.0f}%",
                "Action": item.get("action", "HOLD")
            })

        df = pd.DataFrame(rows).sort_values("z-score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data2.get('run_date','?')} · Alpha = {config.ALPHA_WINDOW}d · Omega = {config.OMEGA_WINDOW}d")
