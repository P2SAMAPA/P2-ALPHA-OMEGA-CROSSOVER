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

US_HOLIDAY
