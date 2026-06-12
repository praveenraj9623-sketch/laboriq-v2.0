from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

APP_ROOT = Path(__file__).resolve().parent
sys.path.append(str(APP_ROOT))

from src.config import REPORTS_DIR, MODELS_DIR, DEFAULT_CANDIDATES_PATH, DEFAULT_SKILLS_PATH, PROJECT_ROOT
from src.skill_extractor import SkillExtractor
from src.data_ingestion import load_taxonomy
from src.adzuna_client import AdzunaSearchSpec, fetch_with_cache
from src.recommender import recommend_skills_for_role
from src.modeling import predict_role

REFRESH_PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "refresh_dashboard_pipeline.py"
ADZUNA_OUTPUT_PATH = PROJECT_ROOT / "data" / "external" / "adzuna_jobs.csv"

# ─────────────────────────────────────────────
#  BRAND CONSTANTS
# ─────────────────────────────────────────────
PRIMARY   = "#2563EB"
SECONDARY = "#38BDF8"
TEAL      = "#14B8A6"
AMBER     = "#F59E0B"
ROSE      = "#FB7185"
EMERALD   = "#10B981"
VIOLET    = "#8B5CF6"
SLATE     = "#E5F0FF"
MUTED     = "#9CB3D1"
BG        = "#0D1B2E"
CARD_BG   = "#1E3A5F"
BORDER    = "#31527A"

CHART_COLORS = [PRIMARY, SECONDARY, TEAL, AMBER, EMERALD, ROSE,
                "#A78BFA", "#F472B6", "#FB923C", "#22D3EE"]

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Inter, sans-serif", color=SLATE, size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=18, r=18, t=52, b=24),
    title_font=dict(size=14, color=SLATE, family="IBM Plex Mono, monospace"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=MUTED, size=11)),
    colorway=CHART_COLORS,
    xaxis=dict(gridcolor="rgba(156,179,209,0.13)", zerolinecolor="rgba(156,179,209,0.22)"),
    yaxis=dict(gridcolor="rgba(156,179,209,0.13)", zerolinecolor="rgba(156,179,209,0.22)"),
)

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LaborIQ — Labor Market Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  CSS INJECTION  (total production-grade revamp)
# ─────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        /* ── CSS Custom Properties ─────────────────── */
        :root {{
          --primary:   {PRIMARY};
          --secondary: {SECONDARY};
          --teal:      {TEAL};
          --amber:     {AMBER};
          --rose:      {ROSE};
          --emerald:   {EMERALD};
          --violet:    {VIOLET};
          --slate:     {SLATE};
          --muted:     {MUTED};
          --bg:        {BG};
          --border:    rgba(49,82,122,0.8);
          --glow-blue: rgba(37,99,235,0.35);
          --glow-teal: rgba(20,184,166,0.25);
          --radius:    13px;
          --card-shadow: 0 10px 40px rgba(0,0,0,0.32), 0 0 0 1px rgba(37,99,235,0.07);
        }}

        /* ── Animations ────────────────────────────── */
        @keyframes gradientFlow {{
          0%   {{ background-position: 0% 50%;   }}
          50%  {{ background-position: 100% 50%; }}
          100% {{ background-position: 0% 50%;   }}
        }}
        @keyframes pulseDot {{
          0%,100% {{ opacity:1; transform:scale(1);    }}
          50%      {{ opacity:.5; transform:scale(.8); }}
        }}
        @keyframes pulseGlow {{
          0%,100% {{ box-shadow: 0 0 0 0 rgba(37,99,235,.45); }}
          50%     {{ box-shadow: 0 0 0 10px rgba(37,99,235,0); }}
        }}
        @keyframes slideUp {{
          from {{ opacity:0; transform:translateY(18px); }}
          to   {{ opacity:1; transform:translateY(0);    }}
        }}
        @keyframes shimmerSweep {{
          0%   {{ transform:translateX(-120%) skewX(-18deg); }}
          100% {{ transform:translateX(220%)  skewX(-18deg); }}
        }}
        @keyframes borderPulse {{
          0%,100% {{ border-color: rgba(37,99,235,.42); }}
          50%      {{ border-color: rgba(56,189,248,.72); }}
        }}
        @keyframes countIn {{
          from {{ opacity:0; transform:translateY(10px); }}
          to   {{ opacity:1; transform:translateY(0);    }}
        }}
        @keyframes rotateGlow {{
          0%   {{ filter: hue-rotate(0deg);   }}
          100% {{ filter: hue-rotate(360deg); }}
        }}

        /* ── Global reset / base ───────────────────── */
        html, body, [class*="css"] {{
          font-family: 'Inter', sans-serif !important;
          -webkit-font-smoothing: antialiased;
        }}

        /* ── Custom scrollbar ──────────────────────── */
        ::-webkit-scrollbar               {{ width:5px; height:5px; }}
        ::-webkit-scrollbar-track         {{ background:rgba(7,16,29,.9); border-radius:10px; }}
        ::-webkit-scrollbar-thumb         {{ background:rgba(37,99,235,.5); border-radius:10px; }}
        ::-webkit-scrollbar-thumb:hover   {{ background:rgba(56,189,248,.75); }}

        /* ── Page background ───────────────────────── */
        .stApp {{
          background:
            radial-gradient(ellipse at 8% 0%,   rgba(37,99,235,.26)  0%, transparent 36%),
            radial-gradient(ellipse at 92% 4%,   rgba(20,184,166,.18) 0%, transparent 30%),
            radial-gradient(ellipse at 50% 100%, rgba(139,92,246,.12) 0%, transparent 42%),
            radial-gradient(ellipse at 20% 60%,  rgba(37,99,235,.07)  0%, transparent 28%),
            linear-gradient(180deg, #060f1c 0%, {BG} 28%, #09152a 72%, #060f1c 100%);
        }}

        .block-container {{
          padding: 1.6rem 2.4rem 4rem;
          max-width: 1560px;
          animation: slideUp .4s ease-out;
        }}

        /* ── Streamlit top header ──────────────────── */
        .stApp > header {{
          background: rgba(6,15,28,.92) !important;
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border-bottom: 1px solid rgba(49,82,122,.35);
        }}

        /* ── Sidebar ───────────────────────────────── */
        section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, #060d1b 0%, #091524 60%, #06101e 100%) !important;
          border-right: 1px solid rgba(49,82,122,.45) !important;
        }}
        section[data-testid="stSidebar"] * {{ color: #cfe2ff !important; }}
        section[data-testid="stSidebar"] .stRadio label {{
          font-size: .83rem;
          font-weight: 600;
          padding: .4rem .65rem;
          border-radius: 8px;
          transition: all .17s ease;
          border: 1px solid transparent;
          display: block;
        }}
        section[data-testid="stSidebar"] .stRadio label:hover {{
          background: rgba(37,99,235,.15);
          border-color: rgba(37,99,235,.32);
          color: #fff !important;
        }}
        section[data-testid="stSidebar"] hr {{
          border-color: rgba(49,82,122,.5) !important;
          margin: .75rem 0 !important;
        }}

        /* ── stMetric cards ────────────────────────── */
        div[data-testid="stMetric"] {{
          background: linear-gradient(145deg,
            rgba(28,54,92,.92) 0%,
            rgba(12,28,58,.96) 100%);
          border: 1px solid rgba(49,82,122,.65);
          border-top: 1px solid rgba(56,189,248,.22);
          border-left: 3px solid {PRIMARY};
          padding: 1rem 1.25rem .9rem;
          border-radius: var(--radius);
          box-shadow: var(--card-shadow);
          transition: transform .2s ease, box-shadow .2s ease;
          position: relative;
          overflow: hidden;
        }}
        div[data-testid="stMetric"]::before {{
          content: "";
          position: absolute;
          top: 0; left: 0; right: 0;
          height: 1px;
          background: linear-gradient(90deg, transparent, rgba(56,189,248,.55), transparent);
        }}
        div[data-testid="stMetric"]::after {{
          content: "";
          position: absolute;
          top: 0; left: -100%;
          width: 55%; height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,.03), transparent);
          transform: skewX(-15deg);
          animation: shimmerSweep 3.5s ease-in-out infinite;
        }}
        div[data-testid="stMetric"]:hover {{
          transform: translateY(-2px);
          box-shadow: 0 18px 52px rgba(0,0,0,.36), 0 0 28px rgba(37,99,235,.14);
          animation: borderPulse 2s ease-in-out infinite;
        }}
        div[data-testid="stMetricLabel"] p {{
          font-size: .69rem !important;
          font-weight: 700 !important;
          text-transform: uppercase !important;
          letter-spacing: .09em !important;
          color: #6e99c0 !important;
          font-family: 'IBM Plex Mono', monospace !important;
        }}
        div[data-testid="stMetricValue"] {{
          font-size: 1.65rem !important;
          font-weight: 700 !important;
          color: #f0f8ff !important;
          font-family: 'IBM Plex Mono', monospace !important;
          animation: countIn .55s ease-out;
        }}
        div[data-testid="stMetricDelta"] {{
          font-size: .73rem !important;
          font-family: 'IBM Plex Mono', monospace !important;
        }}

        /* ── Hero banner ───────────────────────────── */
        .lm-banner {{
          background: linear-gradient(-45deg,
            #060e1b 0%,
            #0f2b61 28%,
            #0e5e75 58%,
            #0c4a45 80%,
            #091828 100%);
          background-size: 400% 400%;
          animation: gradientFlow 9s ease infinite;
          border: 1px solid rgba(56,189,248,.2);
          border-radius: 16px;
          padding: 2.3rem 2.7rem;
          margin-bottom: 1.9rem;
          position: relative;
          overflow: hidden;
          box-shadow:
            0 36px 88px rgba(0,0,0,.4),
            0 0 0 1px rgba(37,99,235,.1),
            inset 0 1px 0 rgba(255,255,255,.07);
        }}
        .lm-banner::before {{
          content: "";
          position: absolute;
          top: -70px; right: -70px;
          width: 300px; height: 300px;
          background: radial-gradient(circle, rgba(56,189,248,.16) 0%, transparent 68%);
          border-radius: 50%;
          animation: pulseGlow 5s ease-in-out infinite;
        }}
        .lm-banner::after {{
          content: "";
          position: absolute;
          bottom: -50px; left: 32%;
          width: 240px; height: 240px;
          background: radial-gradient(circle, rgba(20,184,166,.13) 0%, transparent 65%);
          border-radius: 50%;
        }}
        .lm-banner-shimmer {{
          position: absolute;
          top: 0; left: -100%;
          width: 60%; height: 100%;
          background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255,255,255,.04) 50%,
            transparent 100%);
          transform: skewX(-12deg);
          animation: shimmerSweep 6s ease-in-out infinite 1.5s;
        }}
        .lm-banner-title {{
          font-size: 1.9rem;
          font-weight: 800;
          color: #fff;
          line-height: 1.15;
          margin: 0 0 .5rem;
          font-family: 'IBM Plex Mono', monospace;
          letter-spacing: -.025em;
          text-shadow: 0 0 48px rgba(56,189,248,.3);
          position: relative; z-index: 1;
        }}
        .lm-banner-sub {{
          color: rgba(255,255,255,.76);
          font-size: .91rem;
          line-height: 1.65;
          max-width: 650px;
          margin: 0 0 .9rem;
          position: relative; z-index: 1;
        }}
        .lm-badge {{
          display: inline-flex;
          align-items: center;
          gap: .3rem;
          background: rgba(255,255,255,.1);
          backdrop-filter: blur(8px);
          border: 1px solid rgba(255,255,255,.17);
          border-radius: 999px;
          padding: .24rem .85rem;
          font-size: .74rem;
          font-weight: 700;
          color: #fff;
          margin: 0 .3rem .3rem 0;
          font-family: 'IBM Plex Mono', monospace;
          transition: background .2s, border-color .2s;
          position: relative; z-index: 1;
        }}
        .lm-badge:hover {{
          background: rgba(37,99,235,.28);
          border-color: rgba(56,189,248,.5);
        }}
        .lm-badge-live {{
          background: rgba(16,185,129,.18);
          border-color: rgba(16,185,129,.45);
        }}
        .lm-badge-live .dot {{
          display: inline-block;
          width: 6px; height: 6px;
          background: {EMERALD};
          border-radius: 50%;
          animation: pulseDot 1.4s ease-in-out infinite;
        }}

        /* ── Section labels ────────────────────────── */
        .lm-section-title {{
          font-size: .68rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: .13em;
          color: #5f8fb0;
          font-family: 'IBM Plex Mono', monospace;
          margin: 1.25rem 0 .8rem;
          padding-bottom: .5rem;
          border-bottom: 1px solid rgba(49,82,122,.4);
          display: flex;
          align-items: center;
          gap: .55rem;
        }}
        .lm-section-title::before {{
          content: "";
          display: inline-block;
          width: 3px; height: 15px;
          background: linear-gradient(180deg, {PRIMARY}, {TEAL});
          border-radius: 3px;
          flex-shrink: 0;
        }}

        /* ── Insight / KPI cards ───────────────────── */
        .upgrade-card {{
          background: linear-gradient(148deg,
            rgba(28,54,92,.82) 0%,
            rgba(11,26,52,.88) 100%);
          border: 1px solid rgba(49,82,122,.62);
          border-left: 3px solid {PRIMARY};
          border-radius: var(--radius);
          padding: 1.05rem 1.15rem;
          min-height: 96px;
          box-shadow: 0 12px 36px rgba(0,0,0,.24),
            inset 0 1px 0 rgba(255,255,255,.04);
          transition: transform .22s ease, box-shadow .22s ease, border-left-color .22s ease;
          position: relative;
          overflow: hidden;
        }}
        .upgrade-card::after {{
          content: "";
          position: absolute;
          top: 0; left: -100%;
          width: 60%; height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,.04), transparent);
          transform: skewX(-12deg);
          transition: left .65s ease;
        }}
        .upgrade-card:hover {{
          transform: translateY(-3px);
          box-shadow: 0 20px 56px rgba(0,0,0,.32), 0 0 28px rgba(37,99,235,.15);
          border-left-color: {SECONDARY};
        }}
        .upgrade-card:hover::after {{ left: 160%; }}
        .upgrade-card .num {{
          color: #f0f8ff;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 1.52rem;
          font-weight: 700;
          line-height: 1.1;
          text-shadow: 0 0 22px rgba(56,189,248,.18);
        }}
        .upgrade-card .label {{
          color: #6e99c0;
          font-size: .7rem;
          font-weight: 700;
          letter-spacing: .08em;
          text-transform: uppercase;
          margin-top: .32rem;
          font-family: 'IBM Plex Mono', monospace;
        }}
        .upgrade-card .note {{
          color: #a2bdd6;
          font-size: .76rem;
          margin-top: .3rem;
          line-height: 1.4;
        }}

        /* ── Chips ─────────────────────────────────── */
        .lm-chip-green {{
          display:inline-block;
          background:rgba(16,185,129,.12); color:#6ee7b7;
          border:1px solid rgba(16,185,129,.35); border-radius:999px;
          padding:.2rem .72rem; font-size:.76rem; font-weight:600;
          margin:.18rem; font-family:'IBM Plex Mono',monospace;
          transition:background .15s;
        }}
        .lm-chip-blue {{
          display:inline-block;
          background:rgba(37,99,235,.13); color:#93c5fd;
          border:1px solid rgba(96,165,250,.36); border-radius:999px;
          padding:.2rem .72rem; font-size:.76rem; font-weight:600;
          margin:.18rem; font-family:'IBM Plex Mono',monospace;
        }}
        .lm-chip-amber {{
          display:inline-block;
          background:rgba(245,158,11,.13); color:#fcd34d;
          border:1px solid rgba(245,158,11,.36); border-radius:999px;
          padding:.2rem .72rem; font-size:.76rem; font-weight:600;
          margin:.18rem; font-family:'IBM Plex Mono',monospace;
        }}
        .lm-chip-rose {{
          display:inline-block;
          background:rgba(251,113,133,.13); color:#fda4af;
          border:1px solid rgba(251,113,133,.36); border-radius:999px;
          padding:.2rem .72rem; font-size:.76rem; font-weight:600;
          margin:.18rem; font-family:'IBM Plex Mono',monospace;
        }}

        /* ── Info / warn boxes ─────────────────────── */
        .lm-info {{
          background: rgba(37,99,235,.1);
          border-left: 3px solid {PRIMARY};
          color: #bfdbfe;
          padding: .88rem 1.1rem;
          border-radius: 0 10px 10px 0;
          margin: .8rem 0;
          font-size: .86rem;
          line-height: 1.65;
          position: relative;
        }}
        .lm-warn {{
          background: rgba(245,158,11,.1);
          border-left: 3px solid {AMBER};
          color: #fde68a;
          padding: .88rem 1.1rem;
          border-radius: 0 10px 10px 0;
          margin: .8rem 0;
          font-size: .86rem;
          line-height: 1.65;
        }}

        /* ── Gradient separator ────────────────────── */
        .lm-divider {{
          height: 1px;
          background: linear-gradient(90deg,
            transparent 0%,
            rgba(37,99,235,.5) 30%,
            rgba(20,184,166,.5) 70%,
            transparent 100%);
          margin: 1.6rem 0;
          border: none;
        }}

        /* ── Tabs ──────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
          background: rgba(8,18,34,.85);
          border-radius: 11px;
          padding: .3rem .3rem;
          gap: .18rem;
          border: 1px solid rgba(49,82,122,.38);
          backdrop-filter: blur(12px);
        }}
        .stTabs [data-baseweb="tab"] {{
          background: transparent !important;
          color: #7a9cc5 !important;
          border-radius: 9px !important;
          font-size: .82rem !important;
          font-weight: 600 !important;
          padding: .45rem 1.05rem !important;
          border: 1px solid transparent !important;
          transition: all .18s ease !important;
          font-family: 'Inter', sans-serif !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
          background: rgba(37,99,235,.14) !important;
          color: #dbeafe !important;
          border-color: rgba(37,99,235,.3) !important;
        }}
        .stTabs [aria-selected="true"] {{
          background: linear-gradient(135deg, rgba(37,99,235,.34), rgba(20,184,166,.2)) !important;
          color: #fff !important;
          border-color: rgba(56,189,248,.38) !important;
          box-shadow: 0 0 18px rgba(37,99,235,.18) !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{ display:none !important; }}

        /* ── Buttons ───────────────────────────────── */
        .stButton > button {{
          background: linear-gradient(135deg, #1d4ed8, #1a3fb5) !important;
          color: #fff !important;
          border: 1px solid rgba(96,165,250,.38) !important;
          border-radius: 10px !important;
          font-weight: 600 !important;
          font-size: .85rem !important;
          padding: .52rem 1.45rem !important;
          font-family: 'Inter', sans-serif !important;
          transition: all .22s ease !important;
          box-shadow: 0 4px 16px rgba(37,99,235,.28) !important;
        }}
        .stButton > button:hover {{
          background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
          box-shadow: 0 8px 28px rgba(37,99,235,.42) !important;
          transform: translateY(-1px) !important;
          border-color: rgba(96,165,250,.58) !important;
        }}
        .stButton > button[kind="primary"] {{
          background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1a3fb5 100%) !important;
          box-shadow: 0 4px 22px rgba(37,99,235,.38), 0 0 0 1px rgba(37,99,235,.18) !important;
        }}
        .stButton > button[kind="primary"]:hover {{
          box-shadow: 0 8px 34px rgba(37,99,235,.52), 0 0 0 1px rgba(56,189,248,.4) !important;
        }}

        /* ── Input fields ──────────────────────────── */
        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] > div {{
          background: rgba(7,16,30,.78) !important;
          border-color: rgba(49,82,122,.68) !important;
          border-radius: 9px !important;
          box-shadow: none !important;
          transition: border-color .18s, box-shadow .18s !important;
        }}
        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within,
        div[data-baseweb="select"]:focus-within > div {{
          border-color: {PRIMARY} !important;
          box-shadow: 0 0 0 2px rgba(37,99,235,.22) !important;
        }}
        input:invalid, textarea:invalid {{
          box-shadow: none !important;
          border-color: rgba(49,82,122,.68) !important;
        }}
        input, textarea {{ color: {SLATE} !important; }}

        /* ── Form labels ───────────────────────────── */
        label,
        .stSelectbox label, .stTextInput label, .stTextArea label,
        .stMultiSelect label, .stSlider label, .stCheckbox label,
        .stNumberInput label {{
          color: #7b9dbd !important;
          font-size: .78rem !important;
          font-weight: 600 !important;
          letter-spacing: .04em !important;
          text-transform: uppercase !important;
          font-family: 'IBM Plex Mono', monospace !important;
        }}

        /* ── DataFrames ────────────────────────────── */
        .stDataFrame {{
          border-radius: var(--radius) !important;
          overflow: hidden !important;
          border: 1px solid rgba(49,82,122,.48) !important;
          box-shadow: 0 8px 28px rgba(0,0,0,.22) !important;
        }}
        .stDataFrame table {{ font-size:.82rem !important; }}
        .stDataFrame thead tr th {{
          background: rgba(8,18,38,.95) !important;
          color: #93c5fd !important;
          font-weight: 700 !important;
          font-family: 'IBM Plex Mono', monospace !important;
          font-size: .7rem !important;
          text-transform: uppercase !important;
          letter-spacing: .07em !important;
          border-bottom: 1px solid rgba(49,82,122,.55) !important;
        }}
        .stDataFrame tbody tr:hover td {{
          background: rgba(37,99,235,.07) !important;
        }}

        /* ── Multiselect tags ──────────────────────── */
        [data-baseweb="tag"] {{
          background: rgba(37,99,235,.24) !important;
          border: 1px solid rgba(96,165,250,.38) !important;
          border-radius: 6px !important;
        }}
        [data-baseweb="tag"] span {{ color: #bfdbfe !important; }}

        /* ── Expanders ─────────────────────────────── */
        details {{
          background: rgba(10,20,38,.72) !important;
          border: 1px solid rgba(49,82,122,.38) !important;
          border-radius: 10px !important;
        }}
        details summary {{
          color: #93c5fd !important;
          font-weight: 600 !important;
          font-size: .84rem !important;
          padding: .6rem .85rem !important;
        }}

        /* ── Download button ───────────────────────── */
        .stDownloadButton > button {{
          background: rgba(20,184,166,.14) !important;
          border: 1px solid rgba(20,184,166,.42) !important;
          color: #5eead4 !important;
        }}
        .stDownloadButton > button:hover {{
          background: rgba(20,184,166,.26) !important;
          box-shadow: 0 6px 22px rgba(20,184,166,.22) !important;
        }}

        /* ── Alerts ────────────────────────────────── */
        .stSuccess {{
          background: rgba(16,185,129,.1) !important;
          border: 1px solid rgba(16,185,129,.3) !important;
          border-radius: 10px !important;
          color: #6ee7b7 !important;
        }}
        .stError {{
          background: rgba(251,113,133,.1) !important;
          border: 1px solid rgba(251,113,133,.3) !important;
          border-radius: 10px !important;
        }}
        .stWarning {{
          background: rgba(245,158,11,.1) !important;
          border: 1px solid rgba(245,158,11,.3) !important;
          border-radius: 10px !important;
        }}

        /* ── Slider ────────────────────────────────── */
        .stSlider [data-baseweb="slider"] [role="slider"] {{
          background: {PRIMARY} !important;
          box-shadow: 0 0 10px rgba(37,99,235,.45) !important;
        }}

        /* ── Progress bar ──────────────────────────── */
        .stProgress > div > div {{
          background: linear-gradient(90deg, {PRIMARY}, {TEAL}) !important;
          border-radius: 999px !important;
        }}

        /* ── Caption ───────────────────────────────── */
        .stCaption {{ color: #4e7296 !important; font-size: .73rem !important; }}

        /* ── Code blocks ───────────────────────────── */
        code {{
          background: rgba(8,18,36,.82) !important;
          border: 1px solid rgba(49,82,122,.38) !important;
          border-radius: 5px !important;
          color: #93c5fd !important;
          font-family: 'IBM Plex Mono', monospace !important;
        }}
        pre code {{
          background: transparent !important;
          border: none !important;
        }}

        /* ── Priority pills ────────────────────────── */
        .priority-high   {{color:#fda4af;background:rgba(251,113,133,.14);padding:3px 10px;border-radius:999px;font-weight:700;font-size:.76rem;border:1px solid rgba(251,113,133,.34);}}
        .priority-medium {{color:#fcd34d;background:rgba(245,158,11,.14);padding:3px 10px;border-radius:999px;font-weight:700;font-size:.76rem;border:1px solid rgba(245,158,11,.34);}}
        .priority-low    {{color:#93c5fd;background:rgba(37,99,235,.14);padding:3px 10px;border-radius:999px;font-weight:700;font-size:.76rem;border:1px solid rgba(96,165,250,.34);}}
        .priority-ok     {{color:#6ee7b7;background:rgba(16,185,129,.12);padding:3px 10px;border-radius:999px;font-weight:600;font-size:.76rem;border:1px solid rgba(16,185,129,.34);}}

        /* ── Sidebar branding ──────────────────────── */
        .sb-brand {{
          font-family: 'IBM Plex Mono', monospace;
          font-size: 1.05rem;
          font-weight: 700;
          color: #60a5fa !important;
          letter-spacing: .1em;
          text-transform: uppercase;
          margin-bottom: .12rem;
          text-shadow: 0 0 22px rgba(96,165,250,.4);
        }}
        .sb-hex {{
          font-size: 1.3rem;
          margin-right: .3rem;
          vertical-align: middle;
        }}
        .sb-tag {{
          font-size: .66rem;
          color: #456680 !important;
          font-family: 'IBM Plex Mono', monospace;
          letter-spacing: .06em;
          text-transform: uppercase;
        }}

        /* ── 3D chart wrapper ──────────────────────── */
        .chart3d-wrap {{
          background: rgba(10,22,42,.72);
          border: 1px solid rgba(49,82,122,.46);
          border-radius: 14px;
          padding: .4rem;
          box-shadow: 0 14px 44px rgba(0,0,0,.3);
          overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ─────────────────────────────────────────────
#  3D CHART HELPERS
# ─────────────────────────────────────────────
_3D_SCENE = dict(
    bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        backgroundcolor="rgba(10,22,44,.8)",
        gridcolor="rgba(56,189,248,.13)",
        showbackground=True,
        zerolinecolor="rgba(56,189,248,.22)",
        tickfont=dict(color=MUTED, size=9),
    ),
    yaxis=dict(
        backgroundcolor="rgba(10,22,44,.8)",
        gridcolor="rgba(56,189,248,.13)",
        showbackground=True,
        zerolinecolor="rgba(56,189,248,.22)",
        tickfont=dict(color=MUTED, size=9),
    ),
    zaxis=dict(
        backgroundcolor="rgba(10,22,44,.8)",
        gridcolor="rgba(56,189,248,.13)",
        showbackground=True,
        zerolinecolor="rgba(56,189,248,.22)",
        tickfont=dict(color=MUTED, size=9),
    ),
)


def scene_axis_title(text: str) -> dict:
    return dict(text=text, font=dict(color=SLATE, size=11, family="IBM Plex Mono"))

def plot_3d_salary_scatter(sal_df: pd.DataFrame) -> go.Figure | None:
    """3D scatter: role × experience × avg salary LPA."""
    required = {"role_label", "experience_level", "avg_mid_lpa"}
    if sal_df.empty or not required.issubset(sal_df.columns):
        return None
    exp_order = {"Entry": 0, "Junior": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Principal": 5, "Unknown": 6}
    df = sal_df.copy()
    df["exp_num"] = df["experience_level"].map(exp_order).fillna(2)
    df["avg_mid_lpa"] = pd.to_numeric(df["avg_mid_lpa"], errors="coerce")
    df = df.dropna(subset=["avg_mid_lpa"])
    if df.empty:
        return None

    size_col = df["postings"].clip(1) / df["postings"].max() * 16 + 7 if "postings" in df.columns else 12

    fig = go.Figure(data=[go.Scatter3d(
        x=df["exp_num"],
        y=df["role_label"],
        z=df["avg_mid_lpa"],
        mode="markers",
        marker=dict(
            size=size_col,
            color=df["avg_mid_lpa"],
            colorscale=[[0, SECONDARY], [0.45, TEAL], [1.0, EMERALD]],
            opacity=0.88,
            line=dict(width=1, color="rgba(255,255,255,.2)"),
            showscale=True,
            colorbar=dict(
                title=dict(text="LPA", font=dict(color=MUTED, family="IBM Plex Mono", size=10)),
                tickfont=dict(color=MUTED, size=9),
                thickness=10,
            ),
        ),
        text=df.apply(
            lambda r: f"<b>{r['role_label']}</b><br>{r['experience_level']}<br>₹{r['avg_mid_lpa']:.1f} LPA",
            axis=1,
        ),
        hoverinfo="text",
    )])
    scene = dict(_3D_SCENE)
    scene["xaxis"] = dict(
        scene["xaxis"],
        title=scene_axis_title("Experience"),
        tickvals=list(exp_order.values()),
        ticktext=list(exp_order.keys()),
    )
    scene["yaxis"] = dict(scene["yaxis"], title=scene_axis_title("Role Family"))
    scene["zaxis"] = dict(scene["zaxis"], title=scene_axis_title("Avg Salary (LPA)"))
    fig.update_layout(
        **{**PLOTLY_LAYOUT, "margin": dict(l=0, r=0, t=52, b=0)},
        height=500,
        title="⬡ 3D Salary Intelligence  ·  Role × Experience × Compensation",
        scene=scene,
    )
    return fig


def plot_3d_forecast_surface(forecasts: pd.DataFrame) -> go.Figure | None:
    """3D surface: skill demand over forecast months."""
    required = {"skill", "forecast_month", "forecast_mentions"}
    if forecasts.empty or not required.issubset(forecasts.columns):
        return None
    top_skills = (
        forecasts.groupby("skill")["forecast_mentions"].sum().nlargest(14).index.tolist()
    )
    df_top = forecasts[forecasts["skill"].isin(top_skills)].copy()
    try:
        pivot = df_top.pivot_table(
            index="skill", columns="forecast_month",
            values="forecast_mentions", aggfunc="mean",
        ).fillna(0)
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            return None
        z_data = pivot.values
        x_months = list(pivot.columns)
        y_skills = list(pivot.index)

        fig = go.Figure(data=[go.Surface(
            z=z_data,
            x=x_months,
            y=y_skills,
            colorscale=[
                [0.0,  "rgba(13,27,46,.95)"],
                [0.25, "rgba(37,99,235,.88)"],
                [0.55, "rgba(56,189,248,.88)"],
                [0.8,  "rgba(20,184,166,.88)"],
                [1.0,  "rgba(16,185,129,.92)"],
            ],
            opacity=0.92,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="rgba(255,255,255,.5)", project_z=True),
            ),
            lighting=dict(ambient=0.75, diffuse=0.65, roughness=0.45, specular=0.5, fresnel=0.2),
            lightposition=dict(x=200, y=200, z=1000),
        )])
        scene = dict(_3D_SCENE)
        scene["xaxis"] = dict(scene["xaxis"], title=scene_axis_title("Forecast Month"))
        scene["yaxis"] = dict(scene["yaxis"], title=scene_axis_title("Skill"))
        scene["zaxis"] = dict(scene["zaxis"], title=scene_axis_title("Predicted Mentions"))
        fig.update_layout(
            **{**PLOTLY_LAYOUT, "margin": dict(l=0, r=0, t=52, b=0)},
            height=520,
            title="⬡ 3D Skill Demand Surface  ·  Skill × Month × Forecast Mentions",
            scene=scene,
        )
        return fig
    except Exception:
        return None


def plot_3d_role_skill_cloud(skills_by_role: pd.DataFrame) -> go.Figure | None:
    """3D scatter cloud: role × skill rank × mention volume."""
    required = {"role_label", "skill", "mentions"}
    if skills_by_role.empty or not required.issubset(skills_by_role.columns):
        return None
    try:
        top_roles = (
            skills_by_role.groupby("role_label")["mentions"].sum().nlargest(7).index.tolist()
        )
        df = skills_by_role[skills_by_role["role_label"].isin(top_roles)].copy()
        df = (
            df.groupby("role_label", group_keys=False)
            .apply(lambda g: g.nlargest(6, "mentions"))
            .reset_index(drop=True)
        )
        if df.empty:
            return None
        role_list = list(df["role_label"].unique())
        df["role_num"] = df["role_label"].map({r: i for i, r in enumerate(role_list)})
        df["skill_rank"] = (
            df.groupby("role_label")["mentions"].rank(ascending=False, method="first")
        )
        fig = go.Figure()
        for i, role in enumerate(role_list):
            rdf = df[df["role_label"] == role]
            color = CHART_COLORS[i % len(CHART_COLORS)]
            fig.add_trace(go.Scatter3d(
                x=[i] * len(rdf),
                y=rdf["skill_rank"],
                z=rdf["mentions"],
                mode="markers+text",
                name=role[:22],
                marker=dict(
                    size=11, color=color, opacity=0.87,
                    line=dict(width=1, color="rgba(255,255,255,.18)"),
                ),
                text=rdf["skill"],
                textfont=dict(size=8, color="rgba(255,255,255,.72)"),
                hovertext=rdf.apply(
                    lambda r: f"<b>{r['skill']}</b><br>{r['mentions']:,} mentions", axis=1
                ),
                hoverinfo="text+name",
            ))
        scene = dict(_3D_SCENE)
        scene["xaxis"] = dict(
            scene["xaxis"],
            title=scene_axis_title("Role Family"),
            tickvals=list(range(len(role_list))),
            ticktext=[r[:18] for r in role_list],
        )
        scene["yaxis"] = dict(scene["yaxis"], title=scene_axis_title("Skill Rank (within role)"))
        scene["zaxis"] = dict(scene["zaxis"], title=scene_axis_title("Mentions"))
        fig.update_layout(
            **{
                **PLOTLY_LAYOUT,
                "margin": dict(l=0, r=0, t=52, b=0),
                "legend": dict(
                    bgcolor="rgba(10,22,44,.85)",
                    bordercolor=BORDER,
                    borderwidth=1,
                    font=dict(color=MUTED, size=9),
                ),
            },
            height=480,
            title="⬡ 3D Skill Demand Cloud  ·  Role × Rank × Volume",
            scene=scene,
        )
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS  (all original logic preserved)
# ─────────────────────────────────────────────
def banner(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    badge_html = ""
    for b in (badges or []):
        if "live" in b.lower() or "api:" in b.lower():
            badge_html += (
                f'<span class="lm-badge lm-badge-live">'
                f'<span class="dot"></span>{b}</span>'
            )
        else:
            badge_html += f'<span class="lm-badge">{b}</span>'
    st.markdown(
        f"""
        <div class="lm-banner">
          <div class="lm-banner-shimmer"></div>
          <div class="lm-banner-title">{title}</div>
          <p class="lm-banner-sub">{subtitle}</p>
          <div style="margin-top:.65rem">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_cards(cards: list[tuple[str, str, str]]) -> None:
    if not cards:
        return
    cols = st.columns(len(cards))
    for col, (value, label, note) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="upgrade-card">
              <div class="num">{value}</div>
              <div class="label">{label}</div>
              <div class="note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def section(title: str = "") -> None:
    if title:
        st.markdown(
            f'<div class="lm-section-title">{title}</div>',
            unsafe_allow_html=True,
        )


def divider() -> None:
    st.markdown('<div class="lm-divider"></div>', unsafe_allow_html=True)


def info_box(text: str, kind: str = "info") -> None:
    cls = "lm-info" if kind == "info" else "lm-warn"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def styled_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker_line_width=0)
    return fig


def safe_n(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique(dropna=True)) if not df.empty and col in df.columns else 0


def safe_mean(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df[col], errors="coerce").mean()) if not df.empty and col in df.columns else 0.0


def safe_sum(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df[col], errors="coerce").sum()) if not df.empty and col in df.columns else 0.0


def run_command_with_live_logs(command: list[str], cwd: Path) -> int:
    log_box = st.empty()
    logs: list[str] = []
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    if process.stdout:
        for line in process.stdout:
            logs.append(line)
            log_box.code("".join(logs[-120:]), language="text")
    process.wait()
    return process.returncode


def value_count_frame(df: pd.DataFrame, col: str, n: int = 50) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    out = df[col].fillna("N/A").astype(str).value_counts().head(n).reset_index()
    out.columns = [col, "count"]
    return out


def default_display_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "job_id", "posted_date", "company", "job_title", "location",
        "employment_type", "experience_level", "experience_source",
        "salary_min_lpa", "salary_max_lpa", "salary_mid_lpa",
        "salary_source", "salary_is_predicted",
        "source_query", "redirect_url", "job_description",
    ]
    return [c for c in preferred if c in df.columns]


def normalize_source_value(value: object) -> str:
    s = str(value or "").strip().lower()
    if not s or s in {"nan", "none"}:
        return "unknown"
    if "adzuna" in s:
        return "adzuna_api"
    if "india" in s or "naukri" in s or "indian_job_market_2025" in s:
        return "indian_job_market_2025"
    if "synthetic" in s or "sample" in s:
        return "synthetic_sample"
    if "kaggle" in s:
        return "kaggle"
    return s


def apply_source_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if df.empty or "source" not in df.columns:
        return df.copy()
    source_norm = df["source"].apply(normalize_source_value)
    scope = scope or "Real Data Only"
    if scope == "All Sources":
        mask = pd.Series(True, index=df.index)
    elif scope == "Real Data Only":
        mask = ~source_norm.isin(["synthetic_sample", "sample", "unknown"])
    elif scope == "India 2025 Dataset":
        mask = source_norm.eq("indian_job_market_2025")
    elif scope == "Live Adzuna API":
        mask = source_norm.eq("adzuna_api")
    elif scope == "Synthetic Sample":
        mask = source_norm.eq("synthetic_sample")
    else:
        mask = pd.Series(True, index=df.index)
    return df[mask].copy()


def explode_skill_mentions_for_reports(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "extracted_skills" not in df.columns:
        return pd.DataFrame(columns=["job_id", "posted_date", "location", "role_label", "skill"])
    rows = []
    for _, row in df.iterrows():
        raw_skills = row.get("extracted_skills", "")
        if pd.isna(raw_skills):
            raw_skills = ""
        skills = [
            s.strip()
            for s in str(raw_skills).split("|")
            if s and s.strip() and s.strip().lower() not in {"nan", "none", "<na>"}
        ]
        for skill in skills:
            rows.append({
                "job_id":     row.get("job_id"),
                "posted_date": row.get("posted_date"),
                "location":   row.get("location"),
                "role_label": row.get("role_label"),
                "skill":      skill,
            })
    return pd.DataFrame(rows)


def build_filtered_report_views(
    active_jobs: pd.DataFrame, original_reports: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    views: dict[str, pd.DataFrame] = {}
    df = active_jobs.copy()

    if df.empty:
        for name in [
            "role_demand", "location_demand", "top_skills", "skills_by_role",
            "skills_by_location", "monthly_role_demand", "salary_by_role_experience",
            "source_mix", "top_companies",
        ]:
            views[name] = pd.DataFrame()
        return views

    if "role_label" in df.columns:
        views["role_demand"] = (
            df.groupby("role_label", dropna=False).size()
            .reset_index(name="postings")
            .sort_values("postings", ascending=False)
        )

    if "location" in df.columns:
        agg_dict = {"job_id": "count"} if "job_id" in df.columns else {}
        if "salary_mid_lpa" in df.columns:
            agg_dict["salary_mid_lpa"] = "mean"
        if "company" in df.columns:
            agg_dict["company"] = pd.Series.nunique
        if agg_dict:
            loc = df.groupby("location", dropna=False).agg(agg_dict).reset_index()
            rename = {"job_id": "postings", "salary_mid_lpa": "avg_salary_lpa", "company": "companies"}
            loc = loc.rename(columns=rename)
        else:
            loc = df.groupby("location", dropna=False).size().reset_index(name="postings")
        views["location_demand"] = loc.sort_values("postings", ascending=False)

    skill_fact = explode_skill_mentions_for_reports(df)
    if not skill_fact.empty:
        views["top_skills"] = (
            skill_fact.groupby("skill").size().reset_index(name="mentions")
            .sort_values("mentions", ascending=False)
        )
        if "role_label" in skill_fact.columns:
            views["skills_by_role"] = (
                skill_fact.groupby(["role_label", "skill"], dropna=False)
                .size().reset_index(name="mentions")
                .sort_values(["role_label", "mentions"], ascending=[True, False])
            )
        if "location" in skill_fact.columns:
            views["skills_by_location"] = (
                skill_fact.groupby(["location", "skill"], dropna=False)
                .size().reset_index(name="mentions")
                .sort_values(["location", "mentions"], ascending=[True, False])
            )
    else:
        views["top_skills"] = pd.DataFrame(columns=["skill", "mentions"])
        views["skills_by_role"] = pd.DataFrame(columns=["role_label", "skill", "mentions"])
        views["skills_by_location"] = pd.DataFrame(columns=["location", "skill", "mentions"])

    if {"posted_date", "role_label"}.issubset(df.columns):
        tmp = df.copy()
        tmp["posted_date"] = pd.to_datetime(tmp["posted_date"], errors="coerce")
        tmp = tmp.dropna(subset=["posted_date"])
        if not tmp.empty:
            tmp["month"] = tmp["posted_date"].dt.to_period("M").astype(str)
            views["monthly_role_demand"] = (
                tmp.groupby(["month", "role_label"], dropna=False)
                .size().reset_index(name="postings")
                .sort_values(["month", "role_label"])
            )

    if {"role_label", "experience_level", "salary_mid_lpa"}.issubset(df.columns):
        sal_df = df.copy()
        for col in ["salary_min_lpa", "salary_max_lpa", "salary_mid_lpa"]:
            if col in sal_df.columns:
                sal_df[col] = pd.to_numeric(sal_df[col], errors="coerce")
        sal_group = sal_df.dropna(subset=["salary_mid_lpa"]).groupby(
            ["role_label", "experience_level"], dropna=False
        )
        views["salary_by_role_experience"] = sal_group.agg(
            postings=(
                "job_id" if "job_id" in sal_df.columns else "salary_mid_lpa",
                "count",
            ),
            avg_min_lpa=(
                "salary_min_lpa" if "salary_min_lpa" in sal_df.columns else "salary_mid_lpa",
                "mean",
            ),
            avg_max_lpa=(
                "salary_max_lpa" if "salary_max_lpa" in sal_df.columns else "salary_mid_lpa",
                "mean",
            ),
            avg_mid_lpa=("salary_mid_lpa", "mean"),
        ).reset_index()

    if "source" in df.columns:
        sm = df.copy()
        if "salary_mid_lpa" in sm.columns:
            sm["salary_mid_lpa"] = pd.to_numeric(sm["salary_mid_lpa"], errors="coerce")
        source_agg: dict = {"postings": ("source", "size")}
        if "company" in sm.columns:
            source_agg["companies"] = ("company", "nunique")
        if "location" in sm.columns:
            source_agg["locations"] = ("location", "nunique")
        if "salary_mid_lpa" in sm.columns:
            source_agg["avg_salary_lpa"] = ("salary_mid_lpa", "mean")
        views["source_mix"] = sm.groupby("source", dropna=False).agg(**source_agg).reset_index()

    if "company" in df.columns:
        company_agg: dict = {"postings": ("company", "size")}
        if "role_label" in df.columns:
            company_agg["role_variety"] = ("role_label", "nunique")
        views["top_companies"] = (
            df.groupby("company", dropna=False).agg(**company_agg).reset_index()
            .sort_values("postings", ascending=False)
        )

    for key in [
        "role_demand", "location_demand", "top_skills", "skills_by_role",
        "skills_by_location", "monthly_role_demand", "salary_by_role_experience",
        "source_mix", "top_companies",
    ]:
        if key not in views:
            views[key] = original_reports.get(key, pd.DataFrame())

    return views


def normalize_skill_token(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def canonicalize_user_skills_with_taxonomy(skills: list[str], taxonomy: pd.DataFrame) -> list[str]:
    alias_map: dict[str, str] = {}
    if not taxonomy.empty:
        for _, row in taxonomy.iterrows():
            canonical = str(row.get("canonical_skill", "")).strip()
            if not canonical:
                continue
            for alias in [canonical] + str(row.get("aliases", "")).split("|"):
                alias = str(alias).strip()
                if alias and alias.lower() not in {"nan", "none", "null"}:
                    alias_map[normalize_skill_token(alias)] = canonical
    fallback_aliases = {
        "nlp": "Natural Language Processing",
        "naturallanguageprocessing": "Natural Language Processing",
        "ml": "Machine Learning",
        "machinelearning": "Machine Learning",
        "sklearn": "Scikit-learn",
        "scikitlearn": "Scikit-learn",
        "tfidf": "TF-IDF",
        "rag": "RAG",
        "llm": "LLM",
        "genai": "Generative AI",
        "sql": "SQL",
        "python": "Python",
    }
    for k, v in fallback_aliases.items():
        alias_map.setdefault(k, v)
    canonicalized = []
    for skill in skills:
        token = normalize_skill_token(skill)
        canonicalized.append(alias_map.get(token, str(skill).strip()))
    return sorted({s for s in canonicalized if s})


def salary_coverage(df: pd.DataFrame) -> dict[str, int | float]:
    if df.empty:
        return {"rows": 0, "salary_rows": 0, "missing_rows": 0, "coverage_pct": 0.0}
    salary_cols = [c for c in ["salary_min_lpa", "salary_max_lpa", "salary_mid_lpa"] if c in df.columns]
    if not salary_cols:
        return {"rows": len(df), "salary_rows": 0, "missing_rows": len(df), "coverage_pct": 0.0}
    salary_matrix = df[salary_cols].apply(pd.to_numeric, errors="coerce")
    has_salary = salary_matrix.notna().any(axis=1)
    salary_rows = int(has_salary.sum())
    rows = int(len(df))
    return {
        "rows": rows,
        "salary_rows": salary_rows,
        "missing_rows": rows - salary_rows,
        "coverage_pct": round((salary_rows / rows * 100), 2) if rows else 0.0,
    }


def experience_coverage(df: pd.DataFrame) -> dict[str, int | float]:
    if df.empty or "experience_level" not in df.columns:
        return {"rows": len(df), "known_rows": 0, "unknown_rows": len(df), "coverage_pct": 0.0}
    exp = df["experience_level"].fillna("Unknown").astype(str).str.strip()
    known = ~exp.str.lower().isin(["", "none", "nan", "unknown", "missing"])
    known_rows = int(known.sum())
    rows = int(len(df))
    return {
        "rows": rows,
        "known_rows": known_rows,
        "unknown_rows": rows - known_rows,
        "coverage_pct": round((known_rows / rows * 100), 2) if rows else 0.0,
    }


def format_p_value(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(value):
        return "N/A"
    if value == 0:
        return "< 1e-6"
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def classifier_confidence(pred: dict) -> float:
    conf = pred.get("confidence", None)
    if conf is not None:
        try:
            return float(conf)
        except (TypeError, ValueError):
            pass
    top_roles = pred.get("top_roles") or []
    if top_roles and isinstance(top_roles, list):
        try:
            return float(top_roles[0].get("probability", 0.0))
        except (TypeError, ValueError, AttributeError):
            return 0.0
    return 0.0


def re_escape_for_contains(query: str) -> str:
    import re
    return re.escape(str(query).strip().lower())


def searchable_filter(df: pd.DataFrame, query: str, columns: list[str]) -> pd.DataFrame:
    if df.empty or not query:
        return df
    available = [c for c in columns if c in df.columns]
    if not available:
        return df
    pattern = re_escape_for_contains(query)
    mask = (
        df[available].fillna("").astype(str).agg(" ".join, axis=1)
        .str.lower().str.contains(pattern, regex=True, na=False)
    )
    return df[mask]


def display_row_control(default: str = "200", key: str = "row_limit") -> int | None:
    label_options = ["50", "100", "200", "500", "All"]
    label = st.selectbox(
        "Rows to display",
        label_options,
        index=label_options.index(default) if default in label_options else 2,
        key=key,
    )
    return None if label == "All" else int(label)


def show_dataframe_with_limit(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    height: int = 480,
    key: str = "df_limit",
    default: str = "200",
) -> None:
    display_df = df[columns] if columns else df
    limit = display_row_control(default=default, key=key)
    if limit is None:
        shown = display_df
        st.caption(f"Showing all {len(display_df):,} rows")
    else:
        shown = display_df.head(limit)
        st.caption(f"Showing {len(shown):,} of {len(display_df):,} rows")
    st.dataframe(shown, use_container_width=True, height=height)


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_reports():
    def csv(name):
        p = REPORTS_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return {
        "jobs":                       csv("processed_job_postings.csv"),
        "role_demand":                csv("role_demand.csv"),
        "location_demand":            csv("location_demand.csv"),
        "top_skills":                 csv("top_skills.csv"),
        "skills_by_role":             csv("skills_by_role.csv"),
        "skills_by_location":         csv("skills_by_location.csv"),
        "monthly_role_demand":        csv("monthly_role_demand.csv"),
        "monthly_skill_demand":       csv("monthly_skill_demand.csv"),
        "skill_trends":               csv("skill_trends.csv"),
        "skill_forecasts":            csv("skill_forecasts.csv"),
        "salary_by_role_experience":  csv("salary_by_role_experience.csv"),
        "data_quality":               csv("data_quality_report.csv"),
        "source_mix":                 csv("source_mix.csv"),
        "statistical_tests":          csv("statistical_tests.csv"),
        "role_skill_similarity":      csv("role_skill_similarity.csv"),
        "top_companies":              csv("top_companies.csv"),
    }


@st.cache_resource(show_spinner=False)
def load_models():
    rp = MODELS_DIR / "role_classifier.joblib"
    sp = MODELS_DIR / "salary_model.joblib"
    return {
        "role":   joblib.load(rp) if rp.exists() else None,
        "salary": joblib.load(sp) if sp.exists() else None,
    }


with st.spinner("⬡  Loading platform data…"):
    reports = load_reports()
    models  = load_models()

jobs = reports["jobs"]

if st.session_state.pop("dashboard_refresh_success", False):
    st.success("✓  Dashboard rebuilt from latest data. Reports refreshed.")
if st.session_state.pop("dashboard_refresh_failed", False):
    st.error("✗  Dashboard refresh failed. Check logs.")

if jobs.empty:
    st.error("Reports not found. Run: `python pipelines/run_pipeline.py`")
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><span class="sb-hex">⬡</span> LaborIQ</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-tag">Labor Market Intelligence Platform</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠  Executive Overview",
            "📊  Labor Market Analytics",
            "🔬  Skill Extraction & NLP",
            "📈  Workforce Demand Forecasting",
            "🔢  SciPy Statistical Insights",
            "🌐  Live Adzuna API Ingestion",
            "🎯  Candidate Skill Gap Advisor",
            "🗃️  Job Data Explorer",
            "⚙️  Data Quality & Model Metrics",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="sb-tag">DATA SCOPE</div>', unsafe_allow_html=True)
    data_scope = st.selectbox(
        "Dataset filter",
        ["Real Data Only", "All Sources", "India 2025 Dataset", "Live Adzuna API", "Synthetic Sample"],
        index=0,
        help="Use Real Data Only for recruiter demos.",
        label_visibility="collapsed",
    )

    jobs = apply_source_scope(jobs, data_scope)
    reports = {**reports, **build_filtered_report_views(jobs, reports)}
    st.caption(f"Active scope: {data_scope} · {len(jobs):,} rows")

    st.divider()
    st.markdown('<div class="sb-tag">PLATFORM STATUS</div>', unsafe_allow_html=True)
    st.metric("Processed Jobs",  f"{len(jobs):,}")
    st.metric("Role Families",   f"{safe_n(jobs, 'role_label'):,}")
    st.metric("Locations",       f"{safe_n(jobs, 'location'):,}")

    if ADZUNA_OUTPUT_PATH.exists():
        try:
            az = pd.read_csv(ADZUNA_OUTPUT_PATH)
            st.metric("Adzuna Live Rows", f"{len(az):,}")
        except Exception:
            pass

    st.divider()
    st.markdown('<div class="sb-tag">TECH STACK</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:.72rem;color:#4e7296;line-height:1.85;
                    font-family:'IBM Plex Mono',monospace;">
        Python · SQL · DuckDB<br>
        SciPy · scikit-learn · Pandas<br>
        NLP · TF-IDF · Taxonomy<br>
        Adzuna API · Streamlit · Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  PAGE: EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────
if "Executive Overview" in page:
    banner(
        "⬡ Labor Market Intelligence Platform",
        "A Lightcast-style workforce analytics engine. Ingests live job postings, extracts skills with a "
        "controlled taxonomy, maps occupations, models salary baselines, and forecasts future skill demand.",
        [f"{len(jobs):,} postings", f"{safe_n(jobs, 'role_label'):,} role families",
         "MLflow tracked", "API: live"],
    )

    insight_cards([
        (f"{len(jobs):,}",               "Active postings",   data_scope),
        (f"{safe_n(jobs, 'company'):,}",  "Hiring companies",  "deduplicated view"),
        (f"{safe_n(jobs, 'location'):,}", "Locations",         "market coverage"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Job Postings",    f"{len(jobs):,}")
    c2.metric("Role Families",   f"{safe_n(jobs, 'role_label'):,}")
    c3.metric("Locations",       f"{safe_n(jobs, 'location'):,}")
    avg_s = safe_mean(jobs, "salary_mid_lpa")
    c4.metric("Avg Salary (LPA)", f"₹ {avg_s:.1f}" if avg_s else "—")
    c5.metric("Companies",        f"{safe_n(jobs, 'company'):,}")

    info_box(
        f"Active dataset scope: <b>{data_scope}</b>. Use the sidebar filter to switch between "
        "Real Data Only, India 2025, Live Adzuna API, Synthetic Sample, or All Sources.",
    )

    divider()
    col1, col2 = st.columns(2)

    with col1:
        section("ROLE DEMAND DISTRIBUTION")
        rd = reports["role_demand"]
        if not rd.empty:
            rd_sorted = rd.sort_values("postings", ascending=True)
            fig = px.bar(
                rd_sorted, x="postings", y="role_label", orientation="h",
                color="postings",
                color_continuous_scale=[[0, "#1e3a6e"], [0.5, SECONDARY], [1, PRIMARY]],
                text="postings",
            )
            fig.update_traces(textposition="outside", textfont_size=11)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(
                **PLOTLY_LAYOUT, title="Job Postings by Role Family",
                xaxis_title="Postings", yaxis_title="", height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("TOP IN-DEMAND SKILLS")
        ts = reports["top_skills"]
        if not ts.empty:
            fig2 = px.bar(
                ts.head(15), x="mentions", y="skill", orientation="h",
                color="mentions",
                color_continuous_scale=[[0, "#0e3a35"], [0.5, TEAL], [1, EMERALD]],
                text="mentions",
            )
            fig2.update_traces(textposition="outside", textfont_size=11)
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(
                **PLOTLY_LAYOUT, title="Most Mentioned Skills Across All Postings",
                xaxis_title="Mentions", yaxis_title="", height=340,
            )
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        section("DATA SOURCE MIX")
        sm = reports["source_mix"]
        if not sm.empty:
            fig3 = px.pie(
                sm, names="source", values="postings",
                color_discrete_sequence=CHART_COLORS, hole=0.5,
            )
            fig3.update_traces(
                textposition="inside", textinfo="percent+label",
                marker_line_width=2, marker_line_color="#07111e",
                pull=[0.04] * len(sm),
            )
            fig3.update_layout(
                **PLOTLY_LAYOUT, title="Static + Live API Contribution", height=300,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("SALARY VS ROLE")
        sal = reports["salary_by_role_experience"]
        if not sal.empty:
            sal_agg = (
                sal.groupby("role_label")["avg_mid_lpa"].mean()
                .reset_index().sort_values("avg_mid_lpa", ascending=True)
            )
            fig4 = px.bar(
                sal_agg, x="avg_mid_lpa", y="role_label", orientation="h",
                color="avg_mid_lpa",
                color_continuous_scale=[[0, "#3a2800"], [0.5, AMBER], [1, "#fde68a"]],
                text=sal_agg["avg_mid_lpa"].apply(lambda v: f"₹{v:.1f}"),
            )
            fig4.update_coloraxes(showscale=False)
            fig4.update_traces(textposition="outside", textfont_size=11)
            fig4.update_layout(
                **PLOTLY_LAYOUT, title="Average Salary Midpoint by Role (LPA)",
                xaxis_title="Avg Mid LPA", yaxis_title="", height=300,
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── 3D role-skill cloud ──────────────────────────────
    sbr = reports.get("skills_by_role", pd.DataFrame())
    fig_3d = plot_3d_role_skill_cloud(sbr)
    if fig_3d is not None:
        divider()
        section("3D SKILL DEMAND CLOUD  ·  ROLE × RANK × VOLUME")
        st.markdown(
            '<div class="chart3d-wrap">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        info_box(
            "Interactive 3D view — click & drag to rotate, scroll to zoom. "
            "Each axis encodes a distinct signal: role family (X), rank within role (Y), "
            "and raw mention frequency (Z). Bubble size is uniform to avoid distraction.",
        )

    divider()
    top_role  = reports["role_demand"].sort_values("postings", ascending=False).iloc[0]["role_label"] if not reports["role_demand"].empty else "—"
    top_skill = reports["top_skills"].iloc[0]["skill"] if not reports["top_skills"].empty else "—"
    info_box(
        f"🔍 <b>Platform Insight:</b> <b>{top_role}</b> is the most in-demand role family. "
        f"<b>{top_skill}</b> is the most-mentioned skill. Salary data shows strong stratification "
        f"by role — validated by a statistically significant Kruskal-Wallis test (SciPy page).",
    )
    st.download_button(
        "⬇ Download role demand CSV",
        reports["role_demand"].to_csv(index=False).encode(),
        "role_demand.csv", "text/csv",
    )


# ─────────────────────────────────────────────
#  PAGE: LABOR MARKET ANALYTICS
# ─────────────────────────────────────────────
elif "Labor Market Analytics" in page:
    banner(
        "📊 Labor Market Intelligence & Analytics",
        "DuckDB SQL-powered analytics across roles, locations, companies, and salary distributions.",
        ["DuckDB SQL", "Pandas", "Plotly"],
    )

    insight_cards([
        (f"{len(jobs):,}",                              "Rows analyzed",    data_scope),
        (f"{safe_n(jobs, 'location'):,}",               "Locations",        "hiring footprint"),
        (f"₹ {safe_mean(jobs, 'salary_mid_lpa'):.1f}", "Avg salary LPA",   "salary-available rows"),
    ])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺 Location Demand", "💰 Salary Intelligence", "🏢 Company Hiring", "📅 Trend Over Time"]
    )

    with tab1:
        section("LOCATION-WISE HIRING DEMAND")
        ld = reports["location_demand"]
        if not ld.empty:
            col1, col2 = st.columns([1.6, 1])
            with col1:
                top_locs = ld.head(20).sort_values("postings", ascending=True)
                fig = px.bar(
                    top_locs, x="postings", y="location", orientation="h",
                    color="postings",
                    color_continuous_scale=[[0, "#0d1e44"], [0.5, SECONDARY], [1, PRIMARY]],
                    text="postings",
                )
                fig.update_coloraxes(showscale=False)
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    **PLOTLY_LAYOUT, title="Top 20 Hiring Locations",
                    height=500, xaxis_title="Postings", yaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                section("VOLUME VS SALARY SCATTER")
                if "avg_salary_lpa" in ld.columns:
                    top10 = ld.dropna(subset=["avg_salary_lpa"]).sort_values(
                        "avg_salary_lpa", ascending=False
                    ).head(10)
                    fig2 = px.scatter(
                        top10, x="postings", y="avg_salary_lpa", text="location",
                        color="avg_salary_lpa",
                        color_continuous_scale=[[0, "#3a2800"], [1, AMBER]],
                        size="postings",
                    )
                    fig2.update_coloraxes(showscale=False)
                    fig2.update_traces(textposition="top center", textfont_size=9)
                    fig2.update_layout(
                        **PLOTLY_LAYOUT, title="Volume vs Avg Salary by Location",
                        xaxis_title="Postings", yaxis_title="Avg Salary LPA", height=400,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            st.download_button(
                "⬇ Download location data CSV",
                ld.to_csv(index=False).encode(),
                "location_demand.csv", "text/csv",
            )

    with tab2:
        section("SALARY INTELLIGENCE BY ROLE & EXPERIENCE")
        sal = reports["salary_by_role_experience"]
        if not sal.empty:
            fig = px.bar(
                sal, x="role_label", y="avg_mid_lpa", color="experience_level",
                barmode="group", color_discrete_sequence=CHART_COLORS,
                text=sal["avg_mid_lpa"].apply(
                    lambda v: f"₹{v:.1f}" if pd.notna(v) else ""
                ),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Average Salary Midpoint by Role × Experience Level",
                xaxis_title="Role", yaxis_title="Avg Mid LPA (₹)", height=440,
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                top_paying = sal.sort_values("avg_mid_lpa", ascending=False).head(10)
                fig2 = px.bar(
                    top_paying, x="avg_mid_lpa", y="role_label", orientation="h",
                    color="experience_level", color_discrete_sequence=CHART_COLORS,
                )
                fig2.update_layout(
                    **PLOTLY_LAYOUT, title="Highest-Paying Role × Experience Combos",
                    xaxis_title="Avg Mid LPA", yaxis_title="", height=320,
                )
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.dataframe(
                    sal.style.background_gradient(subset=["avg_mid_lpa"], cmap="Blues"),
                    use_container_width=True, height=320,
                )

            # ── 3D salary scatter ────────────────────────
            fig_3d_sal = plot_3d_salary_scatter(sal)
            if fig_3d_sal is not None:
                divider()
                section("3D SALARY INTELLIGENCE  ·  ROLE × EXPERIENCE × COMPENSATION")
                st.markdown('<div class="chart3d-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_3d_sal, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                info_box(
                    "Each sphere represents a role × experience combination. "
                    "Sphere size encodes posting volume; colour encodes average salary. "
                    "Drag to rotate — the Z-axis is compensation in LPA.",
                )

            st.download_button(
                "⬇ Download salary data CSV",
                sal.to_csv(index=False).encode(),
                "salary_by_role_experience.csv", "text/csv",
            )

    with tab3:
        section("TOP HIRING COMPANIES")
        tc = reports["top_companies"]
        if not tc.empty:
            col1, col2 = st.columns(2)
            with col1:
                top15 = tc.head(15).sort_values("postings", ascending=True)
                fig = px.bar(
                    top15, x="postings", y="company", orientation="h",
                    color="postings",
                    color_continuous_scale=[[0, "#0e3a35"], [1, TEAL]],
                )
                fig.update_coloraxes(showscale=False)
                fig.update_layout(
                    **PLOTLY_LAYOUT, title="Top 15 Companies by Job Postings",
                    xaxis_title="Postings", yaxis_title="", height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if "role_variety" in tc.columns:
                    fig2 = px.scatter(
                        tc.head(20), x="postings", y="role_variety", text="company",
                        color="role_variety",
                        color_continuous_scale=[[0, "#2e1b5e"], [1, VIOLET]],
                    )
                    fig2.update_coloraxes(showscale=False)
                    fig2.update_traces(textposition="top center", textfont_size=9)
                    fig2.update_layout(
                        **PLOTLY_LAYOUT, title="Postings vs Role Diversity per Company",
                        xaxis_title="Total Postings", yaxis_title="Distinct Roles", height=400,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            st.download_button(
                "⬇ Download company data CSV",
                tc.to_csv(index=False).encode(),
                "top_companies.csv", "text/csv",
            )

    with tab4:
        section("MONTHLY ROLE DEMAND TREND")
        mrd = reports["monthly_role_demand"]
        if not mrd.empty and "month" in mrd.columns:
            roles_available = sorted(mrd["role_label"].dropna().unique())
            selected = st.multiselect(
                "Roles to display", roles_available, default=roles_available[:5]
            )
            view = mrd[mrd["role_label"].isin(selected)] if selected else mrd
            fig = px.line(
                view, x="month", y="postings", color="role_label",
                markers=True, color_discrete_sequence=CHART_COLORS,
            )
            fig.update_traces(line_width=2.4, marker_size=6)
            fig.update_layout(
                **PLOTLY_LAYOUT, title="Monthly Role Demand Over Time",
                xaxis_title="Month", yaxis_title="Postings", height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "⬇ Download monthly trend CSV",
                mrd.to_csv(index=False).encode(),
                "monthly_role_demand.csv", "text/csv",
            )


# ─────────────────────────────────────────────
#  PAGE: SKILL EXTRACTION & NLP
# ─────────────────────────────────────────────
elif "Skill Extraction" in page:
    banner(
        "🔬 Skill Extraction & Occupation Mapping Engine",
        "Taxonomy-driven, explainable NLP. Every skill match is traceable to a canonical term, "
        "alias, and character position — no black-box embeddings.",
        ["TF-IDF", "NLP", "Taxonomy", "scikit-learn", "Logistic Regression"],
    )

    insight_cards([
        (f"{safe_n(jobs, 'role_label'):,}", "Role families",    "classifier targets"),
        (
            f"{reports['top_skills'].head(20)['skill'].nunique() if not reports['top_skills'].empty else 0:,}",
            "Top skills", "taxonomy demand",
        ),
        (f"{len(jobs):,}", "Descriptions", "ready for extraction"),
    ])

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Live Extractor Demo", "🗂 Occupation Mapping", "📊 Skills by Role"]
    )

    with tab1:
        section("PASTE ANY JOB DESCRIPTION")
        col_demo, col_out = st.columns([1.1, 1])

        with col_demo:
            text = st.text_area(
                "Job description text",
                value=(
                    "We are hiring a Data Scientist to work with Python, SQL, SciPy, and NLP.\n"
                    "Responsibilities include building machine learning models, statistical analysis,\n"
                    "data visualization, and communicating insights to business stakeholders.\n"
                    "Experience with scikit-learn, TF-IDF, forecasting, and DuckDB is a plus."
                ),
                height=200,
                label_visibility="collapsed",
            )

        with col_out:
            taxonomy   = load_taxonomy(DEFAULT_SKILLS_PATH)
            extractor  = SkillExtractor(taxonomy)
            extracted  = extractor.extract_matches(text)

            st.markdown(f"**{len(extracted)} skills extracted**")
            chip_html = "".join(
                f'<span class="lm-chip-blue">{m.canonical_skill}</span>'
                for m in extracted
            )
            st.markdown(chip_html, unsafe_allow_html=True)

        if extracted:
            divider()
            st.markdown("**Full extraction detail** *(canonical term · alias matched · char position)*")
            ext_df = pd.DataFrame([m.__dict__ for m in extracted])
            st.dataframe(ext_df, use_container_width=True, height=240)
            st.download_button(
                "⬇ Download extraction detail CSV",
                ext_df.to_csv(index=False).encode(),
                "skill_extraction_detail.csv", "text/csv",
            )

        if models["role"] is not None:
            divider()
            section("ML ROLE CLASSIFIER PREDICTION")
            pred = predict_role(models["role"], text)
            conf = classifier_confidence(pred)

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Predicted Role", pred.get("predicted_role", "—"))
            cc2.metric("Confidence",     f"{conf*100:.1f}%" if conf else "—")
            cc3.metric("Model",          "TF-IDF + Logistic Regression")

            top_roles = pred.get("top_roles") or []
            if top_roles:
                st.markdown("**Top role probabilities**")
                prob_df = pd.DataFrame(top_roles)
                st.dataframe(prob_df, use_container_width=True, height=130)
                st.download_button(
                    "⬇ Download role probabilities CSV",
                    prob_df.to_csv(index=False).encode(),
                    "role_probabilities.csv", "text/csv",
                )

            metrics_path = REPORTS_DIR / "model_metrics.json"
            macro_f1_text = "available in the Model Metrics page"
            if metrics_path.exists():
                try:
                    _m = json.loads(metrics_path.read_text())
                    macro_f1 = float(_m.get("role_classifier", {}).get("macro_f1", 0))
                    if macro_f1:
                        macro_f1_text = f"{macro_f1*100:.1f}% macro F1"
                except Exception:
                    pass

            info_box(
                "The classifier uses TF-IDF text features from job descriptions and titles, then "
                f"predicts the closest role family. Current validation: <b>{macro_f1_text}</b>.",
            )

    with tab2:
        section("OCCUPATION MAPPING SAMPLE OUTPUT")
        map_cols = ["job_title", "role_label", "occupation_family",
                    "occupation_confidence", "mapping_method"]
        avail = [c for c in map_cols if c in jobs.columns]
        occ_df = jobs[avail].drop_duplicates().head(40)
        st.dataframe(occ_df, use_container_width=True, height=420)
        st.download_button(
            "⬇ Download occupation mapping CSV",
            occ_df.to_csv(index=False).encode(),
            "occupation_mapping.csv", "text/csv",
        )

        if "mapping_method" in jobs.columns:
            mm = value_count_frame(jobs, "mapping_method")
            fig = px.pie(
                mm, names="mapping_method", values="count",
                color_discrete_sequence=CHART_COLORS, hole=0.42,
            )
            fig.update_layout(
                **PLOTLY_LAYOUT, title="Mapping Method Distribution", height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("TOP SKILLS REQUIRED PER ROLE")
        sbr = reports["skills_by_role"]
        if not sbr.empty:
            roles = sorted(sbr["role_label"].dropna().unique())
            chosen = st.selectbox("Select a role family", roles)
            role_skills = (
                sbr[sbr["role_label"] == chosen]
                .sort_values("mentions", ascending=True).head(15)
            )
            fig = px.bar(
                role_skills, x="mentions", y="skill", orientation="h",
                color="mentions",
                color_continuous_scale=[[0, "#0d1e44"], [1, PRIMARY]],
                text="mentions",
            )
            fig.update_coloraxes(showscale=False)
            fig.update_traces(textposition="outside")
            fig.update_layout(
                **PLOTLY_LAYOUT, title=f"Top Skills for {chosen}",
                xaxis_title="Mentions", yaxis_title="", height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            if not reports["skills_by_location"].empty:
                section("SKILL × LOCATION HEATMAP")
                sbl = reports["skills_by_location"]
                loc_pivot = sbl.pivot_table(
                    index="location", columns="skill",
                    values="mentions", aggfunc="sum",
                ).fillna(0)
                if not loc_pivot.empty:
                    fig2 = px.imshow(
                        loc_pivot.head(12),
                        color_continuous_scale=["#061224", PRIMARY, SECONDARY],
                        aspect="auto",
                    )
                    fig2.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Skill Demand Heatmap by Location", height=380,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            st.download_button(
                "⬇ Download skills-by-role CSV",
                sbr.to_csv(index=False).encode(),
                "skills_by_role.csv", "text/csv",
            )


# ─────────────────────────────────────────────
#  PAGE: WORKFORCE DEMAND FORECASTING
# ─────────────────────────────────────────────
elif "Forecasting" in page:
    banner(
        "📈 Workforce Demand Forecasting & Emerging Skill Trends",
        "6-month skill demand forecasts using SciPy-optimized lag regression. "
        "Trend signals classify skills as Emerging, Growing, Stable, or Declining.",
        ["scipy.optimize.minimize", "Lag Features", "L2 Regularization"],
    )

    forecast_rows = len(reports["skill_forecasts"]) if not reports["skill_forecasts"].empty else 0
    trend_rows    = len(reports["skill_trends"])    if not reports["skill_trends"].empty    else 0
    insight_cards([
        (f"{forecast_rows:,}",                                "Forecast rows",   "6-month outlook"),
        (f"{trend_rows:,}",                                   "Trend signals",   "skill momentum"),
        (f"{safe_n(reports['skill_forecasts'], 'skill'):,}",  "Forecast skills", "tracked"),
    ])

    tab1, tab2 = st.tabs(["📅 Skill Demand Forecast", "🌱 Emerging Skills Radar"])

    with tab1:
        forecasts = reports["skill_forecasts"]
        if forecasts.empty:
            st.warning("No forecast data found. Run `python pipelines/run_pipeline.py`.")
        else:
            all_skills = sorted(forecasts["skill"].unique())
            c1, c2 = st.columns([2, 1])
            with c1:
                selected = st.multiselect(
                    "Select skills to forecast", all_skills, default=all_skills[:8]
                )
            with c2:
                show_table = st.checkbox("Show forecast table", value=False)

            view = forecasts[forecasts["skill"].isin(selected)] if selected else forecasts
            fig = px.line(
                view, x="forecast_month", y="forecast_mentions", color="skill",
                markers=True, color_discrete_sequence=CHART_COLORS, line_shape="spline",
            )
            fig.update_traces(line_width=2.6, marker_size=7)
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="6-Month Skill Demand Forecast (SciPy optimized lag regression)",
                xaxis_title="Month", yaxis_title="Predicted Mentions", height=440,
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_table:
                st.dataframe(view, use_container_width=True)
                st.download_button(
                    "⬇ Download forecast table CSV",
                    view.to_csv(index=False).encode(),
                    "skill_forecasts.csv", "text/csv",
                )

            # ── 3D forecast surface ──────────────────────
            fig_3d_fc = plot_3d_forecast_surface(forecasts)
            if fig_3d_fc is not None:
                divider()
                section("3D SKILL DEMAND SURFACE  ·  SKILL × MONTH × MENTIONS")
                st.markdown('<div class="chart3d-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_3d_fc, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                info_box(
                    "Surface height = predicted mention frequency per month. "
                    "Peaks indicate high-demand skills in that forecast window. "
                    "Rotate to compare skill trajectories across time simultaneously.",
                )

            info_box(
                "Forecast method: <code>scipy.optimize.minimize</code> (BFGS) fits a regularized regression "
                "on lag-1, lag-2, and 3-month rolling average features. "
                "L2 penalty prevents coefficient explosion on small samples. "
                "MAE is reported per skill in the table.",
            )
            info_box(
                "Forecasts are <b>directional planning estimates</b>, not guaranteed market predictions. "
                "Sudden spikes can happen when recent source coverage changes. "
                "Use them to identify skills worth monitoring, then validate with more live postings.",
                kind="warn",
            )

    with tab2:
        trends = reports["skill_trends"]
        if trends.empty:
            st.warning("No trend data found.")
        else:
            min_val = pd.to_numeric(
                trends.get("total_mentions", pd.Series([1])), errors="coerce"
            ).fillna(1).max()
            min_mentions = st.slider(
                "Minimum total mentions for trend display",
                min_value=1,
                max_value=max(1, int(min_val)),
                value=min(30, max(1, int(min_val))),
                help="Filters noisy skills with very low support.",
            )
            trends_display = trends[
                pd.to_numeric(trends["total_mentions"], errors="coerce").fillna(0) >= min_mentions
            ].copy()
            info_box(
                f"Hiding skills below <b>{min_mentions}</b> total mentions to reduce noise.",
            )

            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(
                    trends_display, x="linear_slope", y="growth_score",
                    color="trend_label", size="total_mentions",
                    hover_name="skill", hover_data=["last_observed_month"],
                    color_discrete_map={
                        "Emerging":  EMERALD,
                        "Growing":   SECONDARY,
                        "Stable":    AMBER,
                        "Declining": ROSE,
                    },
                )
                fig.add_vline(x=0, line_dash="dash", line_color=MUTED, line_width=1)
                fig.add_hline(y=1, line_dash="dash", line_color=MUTED, line_width=1)
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Skill Trend Signal Map  (size = total mentions)",
                    xaxis_title="Linear Slope (momentum)",
                    yaxis_title="Growth Score", height=440,
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                tc = trends_display["trend_label"].value_counts().reset_index()
                tc.columns = ["label", "count"]
                color_map = {
                    "Emerging": EMERALD, "Growing": SECONDARY,
                    "Stable": AMBER,     "Declining": ROSE,
                }
                fig2 = px.bar(
                    tc, x="count", y="label", orientation="h",
                    color="label", color_discrete_map=color_map, text="count",
                )
                fig2.update_traces(showlegend=False, textposition="outside")
                fig2.update_layout(
                    **PLOTLY_LAYOUT, title="Skills by Trend Category",
                    xaxis_title="Skill count", yaxis_title="", height=260,
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("**Top 10 Emerging Skills**")
                emerging = (
                    trends_display[trends_display["trend_label"] == "Emerging"]
                    .sort_values("growth_score", ascending=False).head(10)
                )
                chip_html = "".join(
                    f'<span class="lm-chip-green">{r["skill"]}</span>'
                    for _, r in emerging.iterrows()
                )
                st.markdown(chip_html or "*No emerging skills detected.*", unsafe_allow_html=True)

                st.markdown("**Top Declining Skills**")
                declining = (
                    trends_display[trends_display["trend_label"] == "Declining"]
                    .sort_values("growth_score").head(8)
                )
                chip_html2 = "".join(
                    f'<span class="lm-chip-rose">{r["skill"]}</span>'
                    for _, r in declining.iterrows()
                )
                st.markdown(chip_html2 or "*None detected.*", unsafe_allow_html=True)

            st.download_button(
                "⬇ Download trend signals CSV",
                trends_display.to_csv(index=False).encode(),
                "skill_trends.csv", "text/csv",
            )


# ─────────────────────────────────────────────
#  PAGE: SCIPY STATISTICAL INSIGHTS
# ─────────────────────────────────────────────
elif "SciPy" in page:
    banner(
        "🔢 SciPy Statistical Validation Layer",
        "Every insight is statistically validated — not just observed. Four hypothesis tests + "
        "cosine role similarity, all using SciPy directly (not through scikit-learn).",
        [
            "scipy.stats.spearmanr", "scipy.stats.kruskal",
            "scipy.stats.chi2_contingency", "scipy.stats.mannwhitneyu",
            "scipy.spatial.distance.cosine", "scipy.optimize.minimize",
        ],
    )

    tests = reports["statistical_tests"]
    sim   = reports["role_skill_similarity"]

    sig_total = (
        int((tests["result"] == "statistically_significant").sum())
        if not tests.empty and "result" in tests.columns else 0
    )
    insight_cards([
        (f"{len(tests):,}",   "Tests run",     "SciPy validation"),
        (f"{sig_total:,}",    "Significant",   "p < 0.05"),
        (f"{len(sim):,}",     "Role pairs",    "cosine similarity"),
    ])

    tab1, tab2 = st.tabs(["📐 Hypothesis Tests", "🔗 Role Skill Similarity"])

    with tab1:
        if tests.empty:
            st.warning("Run `python pipelines/run_pipeline.py` to generate statistical test output.")
        else:
            sig_count  = int((tests["result"] == "statistically_significant").sum())
            nsig_count = len(tests) - sig_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Tests Run",               len(tests))
            col2.metric("Significant (p < 0.05)",  sig_count,  delta="findings confirmed")
            col3.metric("Not Significant",          nsig_count)

            rc = tests["result"].value_counts().reset_index()
            rc.columns = ["result", "count"]
            color_map_r = {
                "statistically_significant":     EMERALD,
                "not_statistically_significant":  ROSE,
                "insufficient_data":              MUTED,
            }
            col_chart, col_table = st.columns([1, 1.6])
            with col_chart:
                fig = px.pie(
                    rc, names="result", values="count", hole=0.54,
                    color="result", color_discrete_map=color_map_r,
                )
                fig.update_traces(
                    textposition="inside", textinfo="percent+label",
                    marker_line_width=2, marker_line_color="#07111e",
                    pull=[0.05] * len(rc),
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT, title="Test Result Distribution", height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_table:
                st.markdown("**All SciPy Statistical Tests**")
                display_tests = tests[
                    ["test_name", "business_question", "statistic", "p_value",
                     "result", "scipy_function"]
                ].copy()
                display_tests["p_value_display"] = display_tests["p_value"].apply(format_p_value)
                display_tests = display_tests[
                    ["test_name", "business_question", "statistic",
                     "p_value_display", "result", "scipy_function"]
                ].rename(columns={"p_value_display": "p_value"})
                st.dataframe(display_tests, use_container_width=True, height=280)
                st.download_button(
                    "⬇ Download statistical tests CSV",
                    display_tests.to_csv(index=False).encode(),
                    "statistical_tests.csv", "text/csv",
                )

            divider()
            st.markdown("**Business interpretation per test**")
            for _, row in tests.iterrows():
                sig_chip = (
                    '<span class="lm-chip-green">✓ Significant</span>'
                    if row["result"] == "statistically_significant"
                    else '<span class="lm-chip-rose">✗ Not significant</span>'
                )
                func_chip = f'<span class="lm-chip-blue">{row["scipy_function"]}</span>'
                st.markdown(
                    f'{sig_chip} {func_chip} &nbsp; **{row["test_name"]}** — '
                    f'{row["business_question"]}<br>'
                    f'<span style="color:{MUTED};font-size:.83rem">{row["interpretation"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

    with tab2:
        if sim.empty:
            st.warning("Role skill similarity requires skill extraction to be completed.")
        else:
            st.markdown(
                "Role-to-role cosine similarity from role×skill frequency vectors. "
                "Higher = more overlapping skill demands."
            )
            col1, col2 = st.columns([1.4, 1])
            with col1:
                top_sim = sim.head(20)
                fig = px.bar(
                    top_sim,
                    x="cosine_similarity",
                    y=top_sim.apply(lambda r: f"{r['role_a']} ↔ {r['role_b']}", axis=1),
                    orientation="h",
                    color="cosine_similarity",
                    color_continuous_scale=[[0, "#0d1e44"], [1, PRIMARY]],
                    text=top_sim["cosine_similarity"].apply(lambda v: f"{v:.3f}"),
                )
                fig.update_coloraxes(showscale=False)
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Most Similar Role Pairs  (1 − scipy.spatial.distance.cosine)",
                    xaxis_title="Cosine Similarity", yaxis_title="", height=440,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Top 10 most similar role pairs**")
                sim_df = sim[
                    ["role_a", "role_b", "cosine_similarity", "shared_top_skills"]
                ].head(10)
                st.dataframe(sim_df, use_container_width=True, height=360)
                st.download_button(
                    "⬇ Download similarity CSV",
                    sim_df.to_csv(index=False).encode(),
                    "role_skill_similarity.csv", "text/csv",
                )


# ─────────────────────────────────────────────
#  PAGE: LIVE ADZUNA API INGESTION
# ─────────────────────────────────────────────
elif "Adzuna" in page:
    banner(
        "🌐 Live Adzuna API Ingestion",
        "Fetch real job postings from the Adzuna job search API, track data quality "
        "(salary source, experience source), and rebuild all dashboard reports.",
        ["Adzuna REST API", "Live Ingestion", "API: live"],
    )

    current_api_rows = 0
    if ADZUNA_OUTPUT_PATH.exists():
        try:
            current_api_rows = len(pd.read_csv(ADZUNA_OUTPUT_PATH))
        except Exception:
            pass
    insight_cards([
        (f"{current_api_rows:,}", "Saved API rows", "local Adzuna file"),
        ("3",                     "Workflow steps", "fetch, inspect, rebuild"),
        ("Live",                  "API status",     "credential based"),
    ])

    tab_fetch, tab_current, tab_rebuild = st.tabs(
        ["① Fetch Live Jobs", "② View Saved API Data", "③ Rebuild Dashboard Reports"]
    )

    with tab_fetch:
        queries_text = st.text_area(
            "Search queries (one per line)",
            value=(
                "data scientist\ndata analyst\nbusiness analyst\ndata engineer\n"
                "machine learning engineer\nai engineer\nnlp engineer"
            ),
            height=160,
        )
        c1, c2, c3, c4 = st.columns(4)
        location         = c1.text_input("Location",     value="India")
        country          = c2.text_input("Country code", value="in")
        pages            = c3.number_input("Pages per query",   1, 5,  3)
        results_per_page = c4.number_input("Results / page",  10, 50, 25, step=5)
        max_days_old     = st.slider("Max posting age (days)", 1, 90, 90)
        force_refresh    = st.checkbox("Force refresh (bypass cache)", value=False)

        queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
        q1, q2, q3 = st.columns(3)
        q1.metric("Queries",           len(queries))
        q2.metric("API hits estimated", len(queries) * int(pages))
        q3.metric("Max raw rows",       f"{len(queries) * int(pages) * int(results_per_page):,}")

        info_box(
            "Adzuna does not guarantee salary or experience data in every posting. "
            "This app captures <b>salary_source</b> and <b>experience_source</b> so missing "
            "or model-derived values are transparent in all reports.",
        )

        if st.button("Fetch Adzuna Jobs", type="secondary"):
            if not queries:
                st.warning("Enter at least one query.")
            else:
                specs = [
                    AdzunaSearchSpec(
                        query=q, location=location,
                        pages=int(pages), max_days_old=int(max_days_old),
                    )
                    for q in queries
                ]
                try:
                    with st.spinner("Fetching live jobs…"):
                        fetched = fetch_with_cache(
                            specs=specs,
                            output_path=ADZUNA_OUTPUT_PATH,
                            country=country,
                            results_per_page=int(results_per_page),
                            force_refresh=force_refresh,
                        )
                    st.success(f"Saved {len(fetched):,} deduplicated rows to `{ADZUNA_OUTPUT_PATH}`")
                    st.session_state["adzuna_fetch_completed"] = True
                    st.session_state["adzuna_last_row_count"]  = len(fetched)

                    sal_cov = salary_coverage(fetched)
                    exp_cov = experience_coverage(fetched)
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("API rows saved",    f"{len(fetched):,}")
                    c2.metric("Unique companies",   f"{fetched['company'].nunique():,}" if "company" in fetched.columns else "—")
                    c3.metric("Unique locations",   f"{fetched['location'].nunique():,}" if "location" in fetched.columns else "—")
                    c4.metric("Salary rows",         f"{sal_cov['salary_rows']:,}", delta=f"{sal_cov['coverage_pct']}%")
                    c5.metric("Known experience",   f"{exp_cov['known_rows']:,}",   delta=f"{exp_cov['coverage_pct']}%")

                    preview_cols = default_display_columns(fetched)
                    show_dataframe_with_limit(
                        fetched, columns=preview_cols, height=360,
                        key="adzuna_fetch_preview_rows_to_display", default="100",
                    )
                except Exception as exc:
                    st.error(f"Adzuna fetch failed: {exc}")

    with tab_current:
        if ADZUNA_OUTPUT_PATH.exists():
            try:
                az = pd.read_csv(ADZUNA_OUTPUT_PATH)
                st.success(f"File: `{ADZUNA_OUTPUT_PATH.name}` · {len(az):,} rows")

                sal_cov = salary_coverage(az)
                exp_cov = experience_coverage(az)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total API rows",     f"{len(az):,}")
                c2.metric("Companies",           f"{az['company'].nunique():,}" if "company" in az.columns else "—")
                c3.metric("Locations",           f"{az['location'].nunique():,}" if "location" in az.columns else "—")
                c4.metric("Rows with salary",   f"{sal_cov['salary_rows']:,}", delta=f"{sal_cov['coverage_pct']}% coverage")
                c5.metric("Known experience",   f"{exp_cov['known_rows']:,}",   delta=f"{exp_cov['coverage_pct']}% coverage")

                info_box(
                    "Use the row selector below to show 50, 100, 200, 500, or all rows. "
                    "Salary may be missing because many Adzuna listings do not publish compensation.",
                )

                f1, f2, f3 = st.columns(3)
                query_filter = salary_source_filter = experience_filter = []
                if "source_query" in az.columns:
                    query_filter = f1.multiselect(
                        "Filter by API search query",
                        sorted(az["source_query"].dropna().astype(str).unique()),
                    )
                if "salary_source" in az.columns:
                    salary_source_filter = f2.multiselect(
                        "Filter by salary source",
                        sorted(az["salary_source"].dropna().astype(str).unique()),
                    )
                if "experience_level" in az.columns:
                    experience_filter = f3.multiselect(
                        "Filter by experience level",
                        sorted(az["experience_level"].dropna().astype(str).unique()),
                    )

                search = st.text_input("Search title / company / location / description", "").strip().lower()
                filtered = az.copy()
                if query_filter and "source_query" in filtered.columns:
                    filtered = filtered[filtered["source_query"].astype(str).isin(query_filter)]
                if salary_source_filter and "salary_source" in filtered.columns:
                    filtered = filtered[filtered["salary_source"].astype(str).isin(salary_source_filter)]
                if experience_filter and "experience_level" in filtered.columns:
                    filtered = filtered[filtered["experience_level"].astype(str).isin(experience_filter)]
                filtered = searchable_filter(
                    filtered, search,
                    ["job_title", "company", "location", "job_description", "source_query"],
                )

                display_cols = default_display_columns(filtered)
                show_dataframe_with_limit(
                    filtered, columns=display_cols, height=520,
                    key="adzuna_saved_rows_to_display", default="200",
                )
                st.download_button(
                    "⬇ Download filtered API rows as CSV",
                    filtered.to_csv(index=False).encode(),
                    "adzuna_filtered_export.csv", "text/csv",
                )

                with st.expander("Data source audit: why salary / experience can be missing"):
                    st.markdown(
                        """
                        - **salary_source = api** — Adzuna returned salary fields directly.
                        - **salary_source = missing** — The posting did not expose salary through the API.
                        - **experience_source = text_derived** — Inferred Entry/Mid/Senior/Lead from description phrases.
                        - **experience_source = missing** — No reliable experience signal found.
                        """
                    )
                    a1, a2 = st.columns(2)
                    if "salary_source" in az.columns:
                        a1.dataframe(value_count_frame(az, "salary_source"), use_container_width=True, height=180)
                    if "experience_source" in az.columns:
                        a2.dataframe(value_count_frame(az, "experience_source"), use_container_width=True, height=180)

            except Exception as exc:
                st.error(f"Could not read Adzuna CSV: {exc}")
        else:
            st.warning("No Adzuna CSV found yet. Use the Fetch tab above first.")

    with tab_rebuild:
        st.markdown(
            "This runs the full backend refresh: combines Adzuna data with the India job-market "
            "dataset, recreates the portfolio dataset, re-runs cleaning, skill extraction, "
            "occupation mapping, ML training, SciPy validation, forecasting, and exports fresh reports."
        )
        with st.expander("What this runs"):
            st.code(
                "python scripts/combine_external_data.py\n"
                "python scripts/create_portfolio_dataset.py\n"
                "python pipelines/run_pipeline.py "
                "--external data/external/combined_real_job_data_portfolio.csv",
                language="bash",
            )
        if not REFRESH_PIPELINE_SCRIPT.exists():
            info_box(
                f"Refresh script not found at `{REFRESH_PIPELINE_SCRIPT}`. "
                "Create it before using this button.",
                kind="warn",
            )
        rebuild = st.button(
            "Rebuild Dashboard Reports", type="primary",
            disabled=not REFRESH_PIPELINE_SCRIPT.exists(),
        )
        if rebuild:
            st.warning("This may take a few minutes. Do not close the tab.")
            with st.spinner("Running backend refresh…"):
                rc = run_command_with_live_logs(
                    [sys.executable, str(REFRESH_PIPELINE_SCRIPT)], cwd=PROJECT_ROOT
                )
            if rc == 0:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state["dashboard_refresh_success"] = True
                st.rerun()
            else:
                st.session_state["dashboard_refresh_failed"] = True
                st.error("Refresh failed. Check logs.")


# ─────────────────────────────────────────────
#  PAGE: CANDIDATE SKILL GAP ADVISOR
# ─────────────────────────────────────────────
elif "Skill Gap" in page:
    banner(
        "🎯 Candidate Skill Gap Advisor",
        "Enter your current skills, choose a target role, and get a prioritised upskilling roadmap "
        "combining role demand frequency with emerging-trend signals.",
        ["Skill Gap Analysis", "Trend-Weighted Recommendations"],
    )

    insight_cards([
        (f"{safe_n(jobs, 'role_label'):,}",                                          "Target roles",     "career paths"),
        (f"{len(reports['skill_trends']) if not reports['skill_trends'].empty else 0:,}", "Trend signals", "priority weighting"),
        (f"{len(reports['skills_by_role']) if not reports['skills_by_role'].empty else 0:,}", "Role-skill rows", "recommendation base"),
    ])

    col_input, col_results = st.columns([1, 1.4])

    with col_input:
        section("YOUR PROFILE")
        target_roles = sorted(jobs["role_label"].dropna().unique())
        default_idx  = target_roles.index("Data Scientist") if "Data Scientist" in target_roles else 0
        target_role  = st.selectbox("Target role", target_roles, index=default_idx)

        try:
            candidate     = pd.read_csv(DEFAULT_CANDIDATES_PATH).iloc[0]
            default_skills = candidate["current_skills"].replace("|", ", ")
        except Exception:
            default_skills = "Python, SQL, Machine Learning"

        current_skills_text = st.text_area(
            "Your current skills (comma or | separated)",
            value=default_skills, height=120,
        )
        raw_current_skills = [
            s.strip()
            for part in current_skills_text.split("|")
            for s in part.split(",") if s.strip()
        ]
        try:
            skill_taxonomy_for_gap = load_taxonomy(DEFAULT_SKILLS_PATH)
            current_skills = canonicalize_user_skills_with_taxonomy(
                raw_current_skills, skill_taxonomy_for_gap
            )
        except Exception:
            current_skills = raw_current_skills

        st.caption(
            f"{len(raw_current_skills)} skills entered · "
            f"{len(current_skills)} canonical after alias matching"
        )
        with st.expander("View normalized skills used for matching"):
            st.write(", ".join(current_skills) if current_skills else "No skills entered")

        st.button("Analyse My Skill Gap", type="primary")

    with col_results:
        section("SKILL GAP RECOMMENDATIONS")
        recs = recommend_skills_for_role(
            target_role, current_skills,
            reports["skills_by_role"], reports["skill_trends"], top_n=20,
        )

        if recs.empty:
            st.warning("No recommendations found for the selected role.")
        else:
            high   = recs[recs["priority"] == "High"]
            medium = recs[recs["priority"] == "Medium"]
            strong = recs[recs["priority"] == "Already Strong"]

            s1, s2, s3 = st.columns(3)
            s1.metric("High Priority Gaps",  len(high),   delta="Learn these first")
            s2.metric("Medium Priority Gaps", len(medium))
            s3.metric("Already Strong",       len(strong), delta="✓ Keep proof ready")

            prio_order = {"High": 0, "Medium": 1, "Low": 2, "Already Strong": 3}
            recs_sorted = recs.sort_values("priority", key=lambda s: s.map(prio_order))
            color_map = {
                "High": ROSE, "Medium": AMBER, "Low": SECONDARY, "Already Strong": EMERALD,
            }
            fig = px.bar(
                recs_sorted.head(20),
                x="mentions", y="skill", orientation="h",
                color="priority", color_discrete_map=color_map, text="priority",
            )
            fig.update_traces(textposition="inside", textfont_size=10)
            fig.update_layout(
                **PLOTLY_LAYOUT, title=f"Skill Gap for {target_role}",
                xaxis_title="Role Demand (mentions)", yaxis_title="", height=440,
            )
            st.plotly_chart(fig, use_container_width=True)

    if not recs.empty:
        divider()
        st.markdown("**Full recommendation table with trend signals**")
        st.dataframe(recs, use_container_width=True, height=280)
        st.download_button(
            "⬇ Download skill gap recommendations CSV",
            recs.to_csv(index=False).encode(),
            "skill_gap_recommendations.csv", "text/csv",
        )
        if not high.empty:
            info_box(
                f"🔥 <b>Top 3 priority skills to learn for {target_role}:</b> "
                + " · ".join(f"<b>{r['skill']}</b>" for _, r in high.head(3).iterrows()),
                kind="warn",
            )


# ─────────────────────────────────────────────
#  PAGE: JOB DATA EXPLORER
# ─────────────────────────────────────────────
elif "Job Data Explorer" in page:
    banner(
        "🗃️ Job Data Explorer",
        "Filter, search, and export the full processed job posting dataset.",
    )

    insight_cards([
        (f"{len(jobs):,}",               "Rows available", data_scope),
        (f"{safe_n(jobs, 'company'):,}",  "Companies",      "searchable"),
        (f"{safe_n(jobs, 'role_label'):,}", "Role filters", "ready"),
    ])

    c1, c2, c3 = st.columns(3)
    role_filter = c1.multiselect("Role family", sorted(jobs["role_label"].dropna().unique()))
    loc_filter  = c2.multiselect("Location",    sorted(jobs["location"].dropna().unique()))
    exp_filter  = c3.multiselect(
        "Experience level",
        sorted(jobs["experience_level"].dropna().unique()) if "experience_level" in jobs.columns else [],
    )

    view = jobs.copy()
    if role_filter:
        view = view[view["role_label"].isin(role_filter)]
    if loc_filter:
        view = view[view["location"].isin(loc_filter)]
    if exp_filter and "experience_level" in view.columns:
        view = view[view["experience_level"].isin(exp_filter)]

    search_q = st.text_input("Search job title / company / description", "").strip().lower()
    view = searchable_filter(view, search_q, ["job_title", "company", "job_description", "location"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered postings", f"{len(view):,}")
    m2.metric("Companies",          f"{safe_n(view, 'company'):,}")
    m3.metric(
        "Avg Salary",
        f"₹{safe_mean(view, 'salary_mid_lpa'):.1f} LPA" if safe_mean(view, "salary_mid_lpa") else "—",
    )

    explorer_cols = [
        "job_id", "posted_date", "company", "job_title", "location",
        "role_label", "experience_level", "salary_mid_lpa",
        "salary_source", "extracted_skills", "source",
    ]
    show_cols = [c for c in explorer_cols if c in view.columns]
    show_dataframe_with_limit(
        view, columns=show_cols, height=520,
        key="job_explorer_rows_to_display", default="500",
    )
    st.download_button(
        "⬇ Download filtered dataset as CSV",
        view.to_csv(index=False).encode(),
        "filtered_jobs_export.csv", "text/csv",
    )


# ─────────────────────────────────────────────
#  PAGE: DATA QUALITY & MODEL METRICS
# ─────────────────────────────────────────────
elif "Data Quality" in page:
    sal_cov_header = salary_coverage(jobs)
    exp_cov_header = experience_coverage(jobs)

    banner(
        "⚙️ Data Quality & Model Metrics",
        "End-to-end transparency: data quality report, ML classification metrics, and salary model performance.",
        ["scikit-learn", "F1 Score", "MAE", "R²"],
    )

    insight_cards([
        (f"{sal_cov_header['coverage_pct']}%", "Salary coverage",     "audited source"),
        (f"{exp_cov_header['coverage_pct']}%", "Experience coverage", "known rows"),
        ("2",                                   "Tracked models",      "role + salary"),
    ])

    tab1, tab2 = st.tabs(["📋 Data Quality Report", "🤖 Model Performance"])

    with tab1:
        dq = reports["data_quality"]
        if dq.empty:
            st.warning("No data quality report found.")
        else:
            st.dataframe(dq, use_container_width=True, height=400)
            st.download_button(
                "⬇ Download data quality report CSV",
                dq.to_csv(index=False).encode(),
                "data_quality_report.csv", "text/csv",
            )

        divider()
        section("SOURCE, SALARY, AND EXPERIENCE TRANSPARENCY")
        q1, q2, q3 = st.columns(3)
        sal_cov = salary_coverage(jobs)
        exp_cov = experience_coverage(jobs)
        q1.metric("Rows with salary",      f"{sal_cov['salary_rows']:,}", delta=f"{sal_cov['coverage_pct']}% coverage")
        q2.metric("Rows missing salary",   f"{sal_cov['missing_rows']:,}")
        q3.metric("Known experience rows", f"{exp_cov['known_rows']:,}", delta=f"{exp_cov['coverage_pct']}% coverage")

        a1, a2, a3 = st.columns(3)
        if "source" in jobs.columns:
            a1.markdown("**Data source mix**")
            a1.dataframe(value_count_frame(jobs, "source"), use_container_width=True, height=200)
        if "salary_source" in jobs.columns:
            a2.markdown("**Salary source audit**")
            a2.dataframe(value_count_frame(jobs, "salary_source"), use_container_width=True, height=200)
        if "experience_source" in jobs.columns:
            a3.markdown("**Experience source audit**")
            a3.dataframe(value_count_frame(jobs, "experience_source"), use_container_width=True, height=200)

        info_box(
            "This project intentionally tracks missing/API/model-derived fields rather than hiding them. "
            "That is a production-style data quality practice and gives a clear answer if asked "
            "why some live API rows do not contain salary.",
        )

    with tab2:
        metrics_path = REPORTS_DIR / "model_metrics.json"
        if not metrics_path.exists():
            st.warning("Model metrics JSON not found. Run the pipeline first.")
        else:
            metrics = json.loads(metrics_path.read_text())
            rc_m  = metrics.get("role_classifier", {})
            sal_m = metrics.get("salary_model", {})

            section("ROLE CLASSIFIER  ·  TF-IDF + LOGISTIC REGRESSION")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy",    f"{rc_m.get('accuracy', 0)*100:.1f}%")
            col2.metric("Macro F1",    f"{rc_m.get('macro_f1', 0)*100:.1f}%")
            col3.metric("Weighted F1", f"{rc_m.get('weighted_f1', 0)*100:.1f}%")

            info_box(
                "Leakage check: the role classifier uses <b>text_for_model</b> as input and "
                "<b>role_label</b> only as the target label. The target column is not passed as a feature.",
            )

            cr = rc_m.get("classification_report", {})
            cr_rows = [
                {"role": k, **{kk: round(vv, 3) for kk, vv in v.items()}}
                for k, v in cr.items() if isinstance(v, dict)
            ]
            if cr_rows:
                cr_df = pd.DataFrame(cr_rows)
                if "precision" in cr_df.columns:
                    fig = px.bar(
                        cr_df[cr_df["role"].str.len() < 30],
                        x="role", y=["precision", "recall", "f1-score"],
                        barmode="group", color_discrete_sequence=CHART_COLORS,
                    )
                    fig.update_layout(
                        **PLOTLY_LAYOUT, title="Per-Class Classifier Performance",
                        xaxis_title="Role", yaxis_title="Score", height=380,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cr_df, use_container_width=True)
                st.download_button(
                    "⬇ Download classification report CSV",
                    cr_df.to_csv(index=False).encode(),
                    "classification_report.csv", "text/csv",
                )

            divider()
            section("SALARY MODEL  ·  RIDGE REGRESSION")
            sc1, sc2 = st.columns(2)
            sc1.metric("MAE (LPA)", f"{sal_m.get('mae_lpa', 0):.2f}")
            sc2.metric("R²",        f"{sal_m.get('r2', 0):.3f}")
            trained_salary_rows = salary_coverage(jobs)["salary_rows"]
            info_box(
                f"{sal_m.get('prediction_note', '')} "
                f"Salary model trained and evaluated only on salary-available rows "
                f"(<b>{trained_salary_rows:,}</b> in the active dataset scope). "
                "Rows without published salary are excluded from supervised training.",
                kind="warn",
            )
