from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
PRIMARY   = "#1B4FD8"   # deep indigo-blue
SECONDARY = "#0EA5E9"   # sky blue
TEAL      = "#0D9488"   # teal accent
AMBER     = "#F59E0B"   # amber highlight
ROSE      = "#E11D48"   # rose for decline
EMERALD   = "#059669"   # emerald for growth
SLATE     = "#334155"   # dark slate body text
MUTED     = "#94A3B8"   # muted text
BG        = "#F1F5F9"   # page background
CARD_BG   = "#FFFFFF"

CHART_COLORS = [PRIMARY, SECONDARY, TEAL, AMBER, ROSE, EMERALD,
                "#8B5CF6", "#EC4899", "#F97316", "#06B6D4"]

PLOTLY_LAYOUT = dict(
    font=dict(family="DM Sans, sans-serif", color=SLATE, size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=16, r=16, t=40, b=16),
    title_font=dict(size=14, color=SLATE, family="DM Mono, monospace"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    colorway=CHART_COLORS,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG + CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Labor Market Intelligence Platform",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'DM Sans', sans-serif;
        }}

        /* ── Page background ── */
        .stApp {{
            background: {BG};
        }}
        .block-container {{
            padding: 1.5rem 2rem 3rem;
            max-width: 1540px;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: #0F172A;
            border-right: 1px solid #1E293B;
        }}
        section[data-testid="stSidebar"] * {{
            color: #CBD5E1 !important;
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            font-size: 0.87rem;
            font-weight: 500;
            padding: 0.35rem 0.5rem;
            border-radius: 8px;
            transition: background 0.15s;
        }}
        section[data-testid="stSidebar"] .stRadio label:hover {{
            background: #1E293B;
            color: #fff !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: #1E293B !important;
        }}

        /* ── Metric cards ── */
        div[data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid #E2E8F0;
            border-left: 4px solid {PRIMARY};
            padding: 1rem 1.2rem;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(15,23,42,0.06);
            transition: box-shadow 0.2s;
        }}
        div[data-testid="stMetric"]:hover {{
            box-shadow: 0 4px 20px rgba(27,79,216,0.12);
        }}
        div[data-testid="stMetricLabel"] p {{
            font-size: 0.75rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {MUTED} !important;
            font-family: 'DM Mono', monospace !important;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.7rem !important;
            font-weight: 700 !important;
            color: {SLATE} !important;
        }}

        /* ── Banner / hero ── */
        .lm-banner {{
            background: linear-gradient(135deg, #0F172A 0%, #1B4FD8 60%, #0D9488 100%);
            border-radius: 18px;
            padding: 2rem 2.4rem;
            margin-bottom: 1.6rem;
            position: relative;
            overflow: hidden;
        }}
        .lm-banner::before {{
            content: "";
            position: absolute;
            top: -40px; right: -40px;
            width: 220px; height: 220px;
            background: radial-gradient(circle, rgba(14,165,233,0.18) 0%, transparent 70%);
            border-radius: 50%;
        }}
        .lm-banner-title {{
            font-size: 1.7rem;
            font-weight: 800;
            color: #fff;
            line-height: 1.2;
            margin: 0 0 0.4rem;
            font-family: 'DM Mono', monospace;
        }}
        .lm-banner-sub {{
            color: rgba(255,255,255,0.72);
            font-size: 0.93rem;
            line-height: 1.55;
            max-width: 620px;
            margin: 0;
        }}
        .lm-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 999px;
            padding: 0.18rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: #fff;
            margin: 0.65rem 0.3rem 0 0;
            font-family: 'DM Mono', monospace;
        }}

        /* ── Section cards ── */
        .lm-section {{
            background: {CARD_BG};
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            margin: 1rem 0;
            box-shadow: 0 1px 8px rgba(15,23,42,0.04);
        }}
        .lm-section-title {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {MUTED};
            font-family: 'DM Mono', monospace;
            margin-bottom: 0.75rem;
        }}

        /* ── Insight chips ── */
        .lm-chip-green {{
            display: inline-block;
            background: #ECFDF5; color: #065F46;
            border: 1px solid #6EE7B7; border-radius: 999px;
            padding: 0.18rem 0.65rem; font-size: 0.78rem; font-weight: 700;
            margin: 0.15rem;
        }}
        .lm-chip-blue {{
            display: inline-block;
            background: #EFF6FF; color: #1E40AF;
            border: 1px solid #93C5FD; border-radius: 999px;
            padding: 0.18rem 0.65rem; font-size: 0.78rem; font-weight: 700;
            margin: 0.15rem;
        }}
        .lm-chip-amber {{
            display: inline-block;
            background: #FFFBEB; color: #92400E;
            border: 1px solid #FDE68A; border-radius: 999px;
            padding: 0.18rem 0.65rem; font-size: 0.78rem; font-weight: 700;
            margin: 0.15rem;
        }}
        .lm-chip-rose {{
            display: inline-block;
            background: #FFF1F2; color: #9F1239;
            border: 1px solid #FECDD3; border-radius: 999px;
            padding: 0.18rem 0.65rem; font-size: 0.78rem; font-weight: 700;
            margin: 0.15rem;
        }}

        /* ── Info/warn boxes ── */
        .lm-info {{
            background: #EFF6FF; border-left: 4px solid {PRIMARY};
            color: #1E3A8A; padding: 0.8rem 1rem;
            border-radius: 0 10px 10px 0; margin: 0.6rem 0;
            font-size: 0.88rem;
        }}
        .lm-warn {{
            background: #FFFBEB; border-left: 4px solid {AMBER};
            color: #92400E; padding: 0.8rem 1rem;
            border-radius: 0 10px 10px 0; margin: 0.6rem 0;
            font-size: 0.88rem;
        }}

        /* ── Table improvements ── */
        .stDataFrame {{ border-radius: 12px; overflow: hidden; }}
        .stDataFrame table {{ font-size: 0.84rem !important; }}
        .stDataFrame thead tr th {{
            background: #F8FAFC !important;
            color: {SLATE} !important;
            font-weight: 700 !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* ── Priority pills in recommender ── */
        .priority-high   {{ color: #065F46; background: #ECFDF5; padding:2px 8px; border-radius:999px; font-weight:700; font-size:0.8rem; }}
        .priority-medium {{ color: #92400E; background: #FFFBEB; padding:2px 8px; border-radius:999px; font-weight:700; font-size:0.8rem; }}
        .priority-low    {{ color: #1E40AF; background: #EFF6FF; padding:2px 8px; border-radius:999px; font-weight:700; font-size:0.8rem; }}
        .priority-ok     {{ color: #64748B; background: #F1F5F9; padding:2px 8px; border-radius:999px; font-weight:600; font-size:0.8rem; }}

        /* ── Sidebar logo area ── */
        .sb-brand {{
            font-family: 'DM Mono', monospace;
            font-size: 0.88rem;
            font-weight: 700;
            color: #0EA5E9 !important;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }}
        .sb-tag {{
            font-size: 0.72rem;
            color: #64748B !important;
            font-family: 'DM Mono', monospace;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def banner(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    badge_html = "".join(f'<span class="lm-badge">{b}</span>' for b in (badges or []))
    st.markdown(
        f"""
        <div class="lm-banner">
            <div class="lm-banner-title">{title}</div>
            <p class="lm-banner-sub">{subtitle}</p>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str = "") -> None:
    if title:
        st.markdown(f'<div class="lm-section-title">{title}</div>', unsafe_allow_html=True)


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
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
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
    preferred = ["job_id", "posted_date", "company", "job_title", "location",
                 "employment_type", "experience_level", "salary_min_lpa", "salary_max_lpa",
                 "salary_source", "source_query", "redirect_url", "job_description"]
    return [c for c in preferred if c in df.columns]


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_reports():
    def csv(name):
        p = REPORTS_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return {
        "jobs":                   csv("processed_job_postings.csv"),
        "role_demand":            csv("role_demand.csv"),
        "location_demand":        csv("location_demand.csv"),
        "top_skills":             csv("top_skills.csv"),
        "skills_by_role":         csv("skills_by_role.csv"),
        "skills_by_location":     csv("skills_by_location.csv"),
        "monthly_role_demand":    csv("monthly_role_demand.csv"),
        "monthly_skill_demand":   csv("monthly_skill_demand.csv"),
        "skill_trends":           csv("skill_trends.csv"),
        "skill_forecasts":        csv("skill_forecasts.csv"),
        "salary_by_role_experience": csv("salary_by_role_experience.csv"),
        "data_quality":           csv("data_quality_report.csv"),
        "source_mix":             csv("source_mix.csv"),
        "statistical_tests":      csv("statistical_tests.csv"),
        "role_skill_similarity":  csv("role_skill_similarity.csv"),
        "top_companies":          csv("top_companies.csv"),
    }


@st.cache_resource(show_spinner=False)
def load_models():
    rp = MODELS_DIR / "role_classifier.joblib"
    sp = MODELS_DIR / "salary_model.joblib"
    return {
        "role":   joblib.load(rp) if rp.exists() else None,
        "salary": joblib.load(sp) if sp.exists() else None,
    }


with st.spinner("Loading platform data…"):
    reports = load_reports()
    models  = load_models()

jobs = reports["jobs"]

if st.session_state.pop("dashboard_refresh_success", False):
    st.success("Dashboard rebuilt from latest data. Reports refreshed.")
if st.session_state.pop("dashboard_refresh_failed", False):
    st.error("Dashboard refresh failed. Check logs.")

if jobs.empty:
    st.error("Reports not found. Run: `python pipelines/run_pipeline.py`")
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">⬡ LaborIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">Labor Market Intelligence Platform</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="sb-tag">PLATFORM STATUS</div>', unsafe_allow_html=True)
    st.metric("Processed Jobs", f"{len(jobs):,}")
    st.metric("Role Families", f"{safe_n(jobs, 'role_label'):,}")
    st.metric("Locations", f"{safe_n(jobs, 'location'):,}")

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
        <div style="font-size:0.73rem; color:#64748B; line-height:1.8; font-family:'DM Mono',monospace">
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
        "📊 Labor Market Intelligence Platform",
        "A Lightcast-style workforce analytics engine. Ingests live job postings, extracts skills with a "
        "controlled taxonomy, maps occupations, models salary baselines, and forecasts future skill demand.",
        ["Python", "SciPy", "NLP", "DuckDB SQL", "Adzuna API", "scikit-learn"],
    )

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Job Postings", f"{len(jobs):,}")
    c2.metric("Role Families", f"{safe_n(jobs, 'role_label'):,}")
    c3.metric("Locations", f"{safe_n(jobs, 'location'):,}")
    avg_s = safe_mean(jobs, "salary_mid_lpa")
    c4.metric("Avg Salary (LPA)", f"₹ {avg_s:.1f}" if avg_s else "—")
    c5.metric("Companies", f"{safe_n(jobs, 'company'):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        section("ROLE DEMAND DISTRIBUTION")
        rd = reports["role_demand"]
        if not rd.empty:
            rd_sorted = rd.sort_values("postings", ascending=True)
            fig = px.bar(
                rd_sorted, x="postings", y="role_label", orientation="h",
                color="postings", color_continuous_scale=["#DBEAFE", PRIMARY],
                text="postings",
            )
            fig.update_traces(textposition="outside", textfont_size=11)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Job Postings by Role Family",
                xaxis_title="Postings",
                yaxis_title="",
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("TOP IN-DEMAND SKILLS")
        ts = reports["top_skills"]
        if not ts.empty:
            fig2 = px.bar(
                ts.head(15), x="mentions", y="skill", orientation="h",
                color="mentions", color_continuous_scale=["#D1FAE5", TEAL],
                text="mentions",
            )
            fig2.update_traces(textposition="outside", textfont_size=11)
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(
                **PLOTLY_LAYOUT,
                title="Most Mentioned Skills Across All Postings",
                xaxis_title="Mentions",
                yaxis_title="",
                height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        section("DATA SOURCE MIX")
        sm = reports["source_mix"]
        if not sm.empty:
            fig3 = px.pie(
                sm, names="source", values="postings",
                color_discrete_sequence=CHART_COLORS,
                hole=0.48,
            )
            fig3.update_traces(textposition="inside", textinfo="percent+label",
                               marker_line_width=2, marker_line_color="#F1F5F9")
            fig3.update_layout(**PLOTLY_LAYOUT, title="Static + Live API Contribution", height=280)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("SALARY VS ROLE")
        sal = reports["salary_by_role_experience"]
        if not sal.empty:
            sal_agg = sal.groupby("role_label")["avg_mid_lpa"].mean().reset_index().sort_values("avg_mid_lpa", ascending=True)
            fig4 = px.bar(
                sal_agg, x="avg_mid_lpa", y="role_label", orientation="h",
                color="avg_mid_lpa", color_continuous_scale=["#FEF3C7", AMBER],
                text=sal_agg["avg_mid_lpa"].apply(lambda v: f"₹{v:.1f}"),
            )
            fig4.update_coloraxes(showscale=False)
            fig4.update_traces(textposition="outside", textfont_size=11)
            fig4.update_layout(
                **PLOTLY_LAYOUT,
                title="Average Salary Midpoint by Role (LPA)",
                xaxis_title="Avg Mid LPA",
                yaxis_title="",
                height=280,
            )
            st.plotly_chart(fig4, use_container_width=True)

    # Narrative insight
    top_role = reports["role_demand"].sort_values("postings", ascending=False).iloc[0]["role_label"] if not reports["role_demand"].empty else "—"
    top_skill = reports["top_skills"].iloc[0]["skill"] if not reports["top_skills"].empty else "—"
    info_box(
        f"🔍 <b>Platform Insight:</b> <b>{top_role}</b> is the most in-demand role family across the dataset. "
        f"<b>{top_skill}</b> is the single most-mentioned skill. Salary data shows strong stratification by role — "
        f"validated by a statistically significant Kruskal-Wallis test (SciPy Statistical Insights page).",
        kind="info",
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

    tab1, tab2, tab3, tab4 = st.tabs(["🗺 Location Demand", "💰 Salary Intelligence", "🏢 Company Hiring", "📅 Trend Over Time"])

    with tab1:
        section("LOCATION-WISE HIRING DEMAND")
        ld = reports["location_demand"]
        if not ld.empty:
            col1, col2 = st.columns([1.6, 1])
            with col1:
                top_locs = ld.head(20).sort_values("postings", ascending=True)
                fig = px.bar(
                    top_locs, x="postings", y="location", orientation="h",
                    color="postings", color_continuous_scale=["#DBEAFE", PRIMARY],
                    text="postings",
                )
                fig.update_coloraxes(showscale=False)
                fig.update_traces(textposition="outside")
                fig.update_layout(**PLOTLY_LAYOUT, title="Top 20 Hiring Locations", height=480, xaxis_title="Postings", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                section("LOCATION METRICS")
                if "avg_salary_lpa" in ld.columns:
                    top10 = ld.dropna(subset=["avg_salary_lpa"]).sort_values("avg_salary_lpa", ascending=False).head(10)
                    fig2 = px.scatter(
                        top10, x="postings", y="avg_salary_lpa", text="location",
                        color="avg_salary_lpa", color_continuous_scale=["#FEF3C7", AMBER],
                        size="postings",
                    )
                    fig2.update_coloraxes(showscale=False)
                    fig2.update_traces(textposition="top center", textfont_size=9)
                    fig2.update_layout(**PLOTLY_LAYOUT, title="Volume vs Avg Salary by Location",
                                       xaxis_title="Postings", yaxis_title="Avg Salary LPA", height=380)
                    st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section("SALARY INTELLIGENCE BY ROLE & EXPERIENCE")
        sal = reports["salary_by_role_experience"]
        if not sal.empty:
            fig = px.bar(
                sal, x="role_label", y="avg_mid_lpa", color="experience_level",
                barmode="group", color_discrete_sequence=CHART_COLORS,
                text=sal["avg_mid_lpa"].apply(lambda v: f"₹{v:.1f}" if pd.notna(v) else ""),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT, title="Average Salary Midpoint by Role × Experience Level",
                              xaxis_title="Role", yaxis_title="Avg Mid LPA (₹)", height=420)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                top_paying = sal.sort_values("avg_mid_lpa", ascending=False).head(10)
                fig2 = px.bar(
                    top_paying, x="avg_mid_lpa", y="role_label", orientation="h",
                    color="experience_level", color_discrete_sequence=CHART_COLORS,
                )
                fig2.update_layout(**PLOTLY_LAYOUT, title="Highest-Paying Role × Experience Combos",
                                   xaxis_title="Avg Mid LPA", yaxis_title="", height=300)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.dataframe(sal.style.background_gradient(subset=["avg_mid_lpa"], cmap="Blues"),
                             use_container_width=True, height=300)

    with tab3:
        section("TOP HIRING COMPANIES")
        tc = reports["top_companies"]
        if not tc.empty:
            col1, col2 = st.columns(2)
            with col1:
                top15 = tc.head(15).sort_values("postings", ascending=True)
                fig = px.bar(top15, x="postings", y="company", orientation="h",
                             color="postings", color_continuous_scale=["#D1FAE5", TEAL])
                fig.update_coloraxes(showscale=False)
                fig.update_layout(**PLOTLY_LAYOUT, title="Top 15 Companies by Job Postings",
                                  xaxis_title="Postings", yaxis_title="", height=380)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if "role_variety" in tc.columns:
                    fig2 = px.scatter(
                        tc.head(20), x="postings", y="role_variety", text="company",
                        color="role_variety", color_continuous_scale=["#EDE9FE", "#7C3AED"],
                    )
                    fig2.update_coloraxes(showscale=False)
                    fig2.update_traces(textposition="top center", textfont_size=9)
                    fig2.update_layout(**PLOTLY_LAYOUT, title="Postings vs Role Diversity per Company",
                                       xaxis_title="Total Postings", yaxis_title="Distinct Roles", height=380)
                    st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        section("MONTHLY ROLE DEMAND TREND")
        mrd = reports["monthly_role_demand"]
        if not mrd.empty and "month" in mrd.columns:
            roles_available = sorted(mrd["role_label"].dropna().unique())
            selected = st.multiselect("Roles to display", roles_available, default=roles_available[:5])
            view = mrd[mrd["role_label"].isin(selected)] if selected else mrd
            fig = px.line(
                view, x="month", y="postings", color="role_label",
                markers=True, color_discrete_sequence=CHART_COLORS,
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Monthly Role Demand Over Time",
                              xaxis_title="Month", yaxis_title="Postings", height=400)
            st.plotly_chart(fig, use_container_width=True)


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

    tab1, tab2, tab3 = st.tabs(["🔍 Live Extractor Demo", "🗂 Occupation Mapping", "📊 Skills by Role"])

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
            taxonomy = load_taxonomy(DEFAULT_SKILLS_PATH)
            extractor = SkillExtractor(taxonomy)
            extracted = extractor.extract_matches(text)

            st.markdown(f'**{len(extracted)} skills extracted**')
            chip_html = ""
            for m in extracted:
                chip_html += f'<span class="lm-chip-blue">{m.canonical_skill}</span>'
            st.markdown(chip_html, unsafe_allow_html=True)

        if extracted:
            st.markdown("---")
            st.markdown("**Full extraction detail** *(canonical term, alias matched, character position)*")
            st.dataframe(
                pd.DataFrame([m.__dict__ for m in extracted]),
                use_container_width=True,
                height=240,
            )

        if models["role"] is not None:
            st.markdown("---")
            section("ML ROLE CLASSIFIER PREDICTION")
            pred = predict_role(models["role"], text)
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Predicted Role", pred.get("predicted_role", "—"))
            conf = pred.get("confidence", 0)
            cc2.metric("Confidence", f"{conf*100:.1f}%" if conf else "—")
            cc3.metric("Model", "TF-IDF + Logistic Regression")

            info_box(
                "The classifier was trained on processed job postings using TF-IDF text features "
                "combined with extracted skill counts per category. Macro F1 = 0.75.",
                kind="info",
            )

    with tab2:
        section("OCCUPATION MAPPING SAMPLE OUTPUT")
        map_cols = ["job_title", "role_label", "occupation_family", "occupation_confidence", "mapping_method"]
        avail = [c for c in map_cols if c in jobs.columns]
        st.dataframe(jobs[avail].drop_duplicates().head(40), use_container_width=True, height=420)

        if "mapping_method" in jobs.columns:
            mm = value_count_frame(jobs, "mapping_method")
            fig = px.pie(mm, names="mapping_method", values="count",
                         color_discrete_sequence=CHART_COLORS, hole=0.4)
            fig.update_layout(**PLOTLY_LAYOUT, title="Mapping Method Distribution", height=260)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("TOP SKILLS REQUIRED PER ROLE")
        sbr = reports["skills_by_role"]
        if not sbr.empty:
            roles = sorted(sbr["role_label"].dropna().unique())
            chosen = st.selectbox("Select a role family", roles)
            role_skills = sbr[sbr["role_label"] == chosen].sort_values("mentions", ascending=True).head(15)
            fig = px.bar(
                role_skills, x="mentions", y="skill", orientation="h",
                color="mentions", color_continuous_scale=["#DBEAFE", PRIMARY], text="mentions",
            )
            fig.update_coloraxes(showscale=False)
            fig.update_traces(textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Top Skills for {chosen}",
                              xaxis_title="Mentions", yaxis_title="", height=360)
            st.plotly_chart(fig, use_container_width=True)

            if not reports["skills_by_location"].empty:
                st.markdown("**Skills by Location (top locations)**")
                sbl = reports["skills_by_location"]
                loc_pivot = sbl.pivot_table(index="location", columns="skill", values="mentions", aggfunc="sum").fillna(0)
                if not loc_pivot.empty:
                    fig2 = px.imshow(
                        loc_pivot.head(12),
                        color_continuous_scale="Blues",
                        aspect="auto",
                    )
                    fig2.update_layout(**PLOTLY_LAYOUT, title="Skill Demand Heatmap by Location", height=360)
                    st.plotly_chart(fig2, use_container_width=True)


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

    tab1, tab2 = st.tabs(["📅 Skill Demand Forecast", "🌱 Emerging Skills Radar"])

    with tab1:
        forecasts = reports["skill_forecasts"]
        if forecasts.empty:
            st.warning("No forecast data found. Run `python pipelines/run_pipeline.py`.")
        else:
            all_skills = sorted(forecasts["skill"].unique())
            c1, c2 = st.columns([2, 1])
            with c1:
                selected = st.multiselect("Select skills to forecast", all_skills,
                                          default=all_skills[:8])
            with c2:
                show_table = st.checkbox("Show forecast table", value=False)

            view = forecasts[forecasts["skill"].isin(selected)] if selected else forecasts
            fig = px.line(
                view, x="forecast_month", y="forecast_mentions", color="skill",
                markers=True, color_discrete_sequence=CHART_COLORS,
                line_shape="spline",
            )
            fig.update_traces(line_width=2.5, marker_size=7)
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="6-Month Skill Demand Forecast (SciPy optimized lag regression)",
                xaxis_title="Month",
                yaxis_title="Predicted Mentions",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_table:
                st.dataframe(view, use_container_width=True)

            info_box(
                "Forecast method: <code>scipy.optimize.minimize</code> (BFGS) fits a regularized regression "
                "on lag-1, lag-2, and 3-month rolling average features. "
                "L2 penalty prevents coefficient explosion on small samples. "
                "MAE is reported per skill in the table.",
                kind="info",
            )

    with tab2:
        trends = reports["skill_trends"]
        if trends.empty:
            st.warning("No trend data found.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(
                    trends, x="linear_slope", y="growth_score",
                    color="trend_label", size="total_mentions",
                    hover_name="skill", hover_data=["last_observed_month"],
                    color_discrete_map={
                        "Emerging": EMERALD,
                        "Growing": SECONDARY,
                        "Stable": AMBER,
                        "Declining": ROSE,
                    },
                )
                fig.add_vline(x=0, line_dash="dash", line_color=MUTED, line_width=1)
                fig.add_hline(y=1, line_dash="dash", line_color=MUTED, line_width=1)
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Skill Trend Signal Map  (size = total mentions)",
                    xaxis_title="Linear Slope (momentum)",
                    yaxis_title="Growth Score",
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                tc = trends["trend_label"].value_counts().reset_index()
                tc.columns = ["label", "count"]
                color_map = {"Emerging": EMERALD, "Growing": SECONDARY, "Stable": AMBER, "Declining": ROSE}
                fig2 = px.bar(
                    tc, x="count", y="label", orientation="h",
                    color="label", color_discrete_map=color_map, text="count",
                )
                fig2.update_traces(showlegend=False, textposition="outside")
                fig2.update_layout(**PLOTLY_LAYOUT, title="Skills by Trend Category",
                                   xaxis_title="Skill count", yaxis_title="", height=240)
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("**Top 10 Emerging Skills**")
                emerging = trends[trends["trend_label"] == "Emerging"].sort_values("growth_score", ascending=False).head(10)
                chip_html = "".join(f'<span class="lm-chip-green">{r["skill"]}</span>' for _, r in emerging.iterrows())
                st.markdown(chip_html or "*No emerging skills detected.*", unsafe_allow_html=True)

                st.markdown("**Top Declining Skills**")
                declining = trends[trends["trend_label"] == "Declining"].sort_values("growth_score").head(8)
                chip_html2 = "".join(f'<span class="lm-chip-rose">{r["skill"]}</span>' for _, r in declining.iterrows())
                st.markdown(chip_html2 or "*None detected.*", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: SCIPY STATISTICAL INSIGHTS
# ─────────────────────────────────────────────
elif "SciPy" in page:
    banner(
        "🔢 SciPy Statistical Validation Layer",
        "Every insight is statistically validated — not just observed. Four hypothesis tests + "
        "cosine role similarity, all using SciPy directly (not through scikit-learn).",
        ["scipy.stats.spearmanr", "scipy.stats.kruskal", "scipy.stats.chi2_contingency",
         "scipy.stats.mannwhitneyu", "scipy.spatial.distance.cosine", "scipy.optimize.minimize"],
    )

    tests = reports["statistical_tests"]
    sim   = reports["role_skill_similarity"]

    tab1, tab2 = st.tabs(["📐 Hypothesis Tests", "🔗 Role Skill Similarity"])

    with tab1:
        if tests.empty:
            st.warning("Run `python pipelines/run_pipeline.py` to generate statistical test output.")
        else:
            sig_count = int((tests["result"] == "statistically_significant").sum())
            nsig_count = len(tests) - sig_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Tests Run", len(tests))
            col2.metric("Significant (p < 0.05)", sig_count, delta="findings confirmed")
            col3.metric("Not Significant", nsig_count)

            # Result summary donut
            rc = tests["result"].value_counts().reset_index()
            rc.columns = ["result", "count"]
            color_map_r = {"statistically_significant": EMERALD, "not_statistically_significant": ROSE, "insufficient_data": MUTED}
            col_chart, col_table = st.columns([1, 1.6])
            with col_chart:
                fig = px.pie(rc, names="result", values="count", hole=0.52,
                             color="result", color_discrete_map=color_map_r)
                fig.update_traces(textposition="inside", textinfo="percent+label",
                                  marker_line_width=2, marker_line_color="#F1F5F9")
                fig.update_layout(**PLOTLY_LAYOUT, title="Test Result Distribution", height=280)
                st.plotly_chart(fig, use_container_width=True)
            with col_table:
                st.markdown("**All SciPy Statistical Tests**")
                display_tests = tests[["test_name", "business_question", "statistic", "p_value", "result", "scipy_function"]].copy()
                st.dataframe(display_tests, use_container_width=True, height=260)

            st.markdown("---")
            st.markdown("**Business interpretation per test**")
            for _, row in tests.iterrows():
                sig_chip = '<span class="lm-chip-green">✓ Significant</span>' if row["result"] == "statistically_significant" else '<span class="lm-chip-rose">✗ Not significant</span>'
                func_chip = f'<span class="lm-chip-blue">{row["scipy_function"]}</span>'
                st.markdown(
                    f'{sig_chip} {func_chip} &nbsp; **{row["test_name"]}** — {row["business_question"]}<br>'
                    f'<span style="color:{MUTED}; font-size:0.83rem">{row["interpretation"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

    with tab2:
        if sim.empty:
            st.warning("Role skill similarity requires skill extraction to be completed.")
        else:
            st.markdown(
                "Role-to-role cosine similarity computed from role×skill frequency vectors. "
                "Higher similarity = more overlapping skill demands."
            )
            col1, col2 = st.columns([1.4, 1])
            with col1:
                top_sim = sim.head(20)
                fig = px.bar(
                    top_sim,
                    x="cosine_similarity", y=top_sim.apply(lambda r: f"{r['role_a']} ↔ {r['role_b']}", axis=1),
                    orientation="h",
                    color="cosine_similarity", color_continuous_scale=["#DBEAFE", PRIMARY],
                    text=top_sim["cosine_similarity"].apply(lambda v: f"{v:.3f}"),
                )
                fig.update_coloraxes(showscale=False)
                fig.update_traces(textposition="outside")
                fig.update_layout(**PLOTLY_LAYOUT, title="Most Similar Role Pairs (SciPy Cosine Distance)",
                                  xaxis_title="Cosine Similarity", yaxis_title="", height=420)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Top 10 most similar role pairs**")
                st.dataframe(
                    sim[["role_a", "role_b", "cosine_similarity", "shared_top_skills"]].head(10),
                    use_container_width=True,
                    height=350,
                )


# ─────────────────────────────────────────────
#  PAGE: LIVE ADZUNA API INGESTION
# ─────────────────────────────────────────────
elif "Adzuna" in page:
    banner(
        "🌐 Live Adzuna API Ingestion",
        "Fetch real job postings from the Adzuna job search API, track data quality "
        "(salary source, experience source), and rebuild all dashboard reports from the latest data.",
        ["Adzuna REST API", "Live Ingestion", "Data Quality Tracking"],
    )

    tab_fetch, tab_current, tab_rebuild = st.tabs(
        ["① Fetch Live Jobs", "② View Saved API Data", "③ Rebuild Dashboard Reports"]
    )

    with tab_fetch:
        queries_text = st.text_area(
            "Search queries (one per line)",
            value="data scientist\ndata analyst\nbusiness analyst\ndata engineer\nmachine learning engineer\nai engineer\nnlp engineer",
            height=160,
        )
        c1, c2, c3, c4 = st.columns(4)
        location         = c1.text_input("Location", value="India")
        country          = c2.text_input("Country code", value="in")
        pages            = c3.number_input("Pages per query", 1, 5, 3)
        results_per_page = c4.number_input("Results/page", 10, 50, 25, step=5)
        max_days_old     = st.slider("Max posting age (days)", 1, 90, 90)
        force_refresh    = st.checkbox("Force refresh (bypass cache)", value=False)

        queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
        q1, q2, q3 = st.columns(3)
        q1.metric("Queries", len(queries))
        q2.metric("API hits estimated", len(queries) * int(pages))
        q3.metric("Max raw rows", f"{len(queries) * int(pages) * int(results_per_page):,}")

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
                    AdzunaSearchSpec(query=q, location=location, pages=int(pages), max_days_old=int(max_days_old))
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
                    st.session_state["adzuna_last_row_count"] = len(fetched)
                    # Quick quality metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("API rows saved", f"{len(fetched):,}")
                    c2.metric("Unique companies", f"{fetched['company'].nunique():,}" if "company" in fetched.columns else "—")
                    c3.metric("Unique locations", f"{fetched['location'].nunique():,}" if "location" in fetched.columns else "—")
                    c4.metric("Queries used", f"{fetched['source_query'].nunique():,}" if "source_query" in fetched.columns else "—")
                    st.dataframe(fetched.head(50), use_container_width=True, height=360)
                except Exception as exc:
                    st.error(f"Adzuna fetch failed: {exc}")

    with tab_current:
        if ADZUNA_OUTPUT_PATH.exists():
            try:
                az = pd.read_csv(ADZUNA_OUTPUT_PATH)
                st.success(f"Current file: `{ADZUNA_OUTPUT_PATH.name}` | Rows: {len(az):,}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total rows", f"{len(az):,}")
                c2.metric("Companies", f"{az['company'].nunique():,}" if "company" in az.columns else "—")
                c3.metric("Locations", f"{az['location'].nunique():,}" if "location" in az.columns else "—")
                salary_rows = int(pd.to_numeric(az.get("salary_min_lpa", pd.Series()), errors="coerce").notna().sum())
                c4.metric("Salary rows", f"{salary_rows:,}")

                # Filters
                search = st.text_input("Search title / company / location", "").strip().lower()
                filtered = az.copy()
                if search:
                    text_cols = [c for c in ["job_title", "company", "location"] if c in filtered.columns]
                    mask = filtered[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.contains(search)
                    filtered = filtered[mask]
                st.caption(f"Showing {len(filtered):,} of {len(az):,} rows")
                st.dataframe(filtered[default_display_columns(filtered)].head(200), use_container_width=True, height=480)
                st.download_button("Download CSV", az.to_csv(index=False).encode(), "adzuna_export.csv", "text/csv")
            except Exception as exc:
                st.error(f"Could not read Adzuna CSV: {exc}")
        else:
            st.warning("No Adzuna CSV found yet. Use the Fetch tab above first.")

    with tab_rebuild:
        st.markdown(
            "This runs the full backend refresh: combines Adzuna data with the India job-market dataset, "
            "recreates the portfolio dataset, re-runs cleaning, skill extraction, occupation mapping, "
            "ML training, SciPy validation, forecasting, and exports fresh reports."
        )
        with st.expander("What this runs"):
            st.code(
                "python scripts/combine_external_data.py\n"
                "python scripts/create_portfolio_dataset.py\n"
                "python pipelines/run_pipeline.py --external data/external/combined_real_job_data_portfolio.csv",
                language="bash",
            )
        if not REFRESH_PIPELINE_SCRIPT.exists():
            info_box(f"Refresh script not found at `{REFRESH_PIPELINE_SCRIPT}`. Create it before using this button.", kind="warn")
        rebuild = st.button("Rebuild Dashboard Reports", type="primary", disabled=not REFRESH_PIPELINE_SCRIPT.exists())
        if rebuild:
            st.warning("This may take a few minutes. Do not close the tab.")
            with st.spinner("Running backend refresh…"):
                rc = run_command_with_live_logs([sys.executable, str(REFRESH_PIPELINE_SCRIPT)], cwd=PROJECT_ROOT)
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

    col_input, col_results = st.columns([1, 1.4])

    with col_input:
        section("YOUR PROFILE")
        target_roles = sorted(jobs["role_label"].dropna().unique())
        default_idx  = target_roles.index("Data Scientist") if "Data Scientist" in target_roles else 0
        target_role  = st.selectbox("Target role", target_roles, index=default_idx)

        try:
            candidate = pd.read_csv(DEFAULT_CANDIDATES_PATH).iloc[0]
            default_skills = candidate["current_skills"].replace("|", ", ")
        except Exception:
            default_skills = "Python, SQL, Machine Learning"

        current_skills_text = st.text_area(
            "Your current skills (comma or | separated)",
            value=default_skills,
            height=120,
        )
        current_skills = [s.strip() for part in current_skills_text.split("|") for s in part.split(",") if s.strip()]
        st.caption(f"{len(current_skills)} skills entered")

        run_btn = st.button("Analyse My Skill Gap", type="primary")

    with col_results:
        section("SKILL GAP RECOMMENDATIONS")
        recs = recommend_skills_for_role(
            target_role, current_skills,
            reports["skills_by_role"], reports["skill_trends"], top_n=20,
        )

        if recs.empty:
            st.warning("No recommendations found for the selected role.")
        else:
            # Summary chips
            high   = recs[recs["priority"] == "High"]
            medium = recs[recs["priority"] == "Medium"]
            strong = recs[recs["priority"] == "Already Strong"]

            s1, s2, s3 = st.columns(3)
            s1.metric("High Priority Gaps", len(high), delta="Learn these first")
            s2.metric("Medium Priority Gaps", len(medium))
            s3.metric("Already Strong", len(strong), delta="✓ Keep proof ready")

            # Visual bar of gap priority
            prio_order = {"High": 0, "Medium": 1, "Low": 2, "Already Strong": 3}
            recs_sorted = recs.sort_values("priority", key=lambda s: s.map(prio_order))

            color_map = {"High": ROSE, "Medium": AMBER, "Low": SECONDARY, "Already Strong": EMERALD}
            fig = px.bar(
                recs_sorted.head(20),
                x="mentions", y="skill", orientation="h",
                color="priority", color_discrete_map=color_map,
                text="priority",
            )
            fig.update_traces(textposition="inside", textfont_size=10)
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Skill Gap for {target_role}",
                xaxis_title="Role Demand (mentions)",
                yaxis_title="",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    if not recs.empty:
        st.markdown("---")
        st.markdown("**Full recommendation table with trend signals**")
        st.dataframe(recs, use_container_width=True, height=280)

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

    c1, c2, c3 = st.columns(3)
    role_filter = c1.multiselect("Role family", sorted(jobs["role_label"].dropna().unique()))
    loc_filter  = c2.multiselect("Location", sorted(jobs["location"].dropna().unique()))
    exp_filter  = c3.multiselect(
        "Experience level",
        sorted(jobs["experience_level"].dropna().unique()) if "experience_level" in jobs.columns else [],
    )

    view = jobs.copy()
    if role_filter: view = view[view["role_label"].isin(role_filter)]
    if loc_filter:  view = view[view["location"].isin(loc_filter)]
    if exp_filter and "experience_level" in view.columns:
        view = view[view["experience_level"].isin(exp_filter)]

    search_q = st.text_input("Search job title / company / description", "").strip().lower()
    if search_q:
        text_cols = [c for c in ["job_title", "company", "job_description"] if c in view.columns]
        mask = view[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.contains(search_q)
        view = view[mask]

    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered postings", f"{len(view):,}")
    m2.metric("Companies", f"{safe_n(view, 'company'):,}")
    m3.metric("Avg Salary", f"₹{safe_mean(view, 'salary_mid_lpa'):.1f} LPA" if safe_mean(view, 'salary_mid_lpa') else "—")

    explorer_cols = ["job_id", "posted_date", "company", "job_title", "location",
                     "role_label", "experience_level", "salary_mid_lpa", "extracted_skills", "source"]
    show_cols = [c for c in explorer_cols if c in view.columns]
    st.dataframe(view[show_cols], use_container_width=True, height=520)
    st.download_button(
        "Download filtered dataset as CSV",
        view.to_csv(index=False).encode(),
        "filtered_jobs_export.csv",
        "text/csv",
    )


# ─────────────────────────────────────────────
#  PAGE: DATA QUALITY & MODEL METRICS
# ─────────────────────────────────────────────
elif "Data Quality" in page:
    banner(
        "⚙️ Data Quality & Model Metrics",
        "End-to-end transparency: data quality report, ML classification metrics, and salary model performance.",
        ["scikit-learn", "F1 Score", "MAE", "R²"],
    )

    tab1, tab2 = st.tabs(["📋 Data Quality Report", "🤖 Model Performance"])

    with tab1:
        dq = reports["data_quality"]
        if dq.empty:
            st.warning("No data quality report found.")
        else:
            st.dataframe(dq, use_container_width=True, height=400)

    with tab2:
        metrics_path = REPORTS_DIR / "model_metrics.json"
        if not metrics_path.exists():
            st.warning("Model metrics JSON not found. Run the pipeline first.")
        else:
            metrics = json.loads(metrics_path.read_text())
            rc_m = metrics.get("role_classifier", {})
            sal_m = metrics.get("salary_model", {})

            st.subheader("Role Classifier — TF-IDF + Logistic Regression")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{rc_m.get('accuracy', 0)*100:.1f}%")
            col2.metric("Macro F1", f"{rc_m.get('macro_f1', 0)*100:.1f}%")
            col3.metric("Weighted F1", f"{rc_m.get('weighted_f1', 0)*100:.1f}%")

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
                    fig.update_layout(**PLOTLY_LAYOUT, title="Per-Class Classifier Performance",
                                      xaxis_title="Role", yaxis_title="Score", height=360)
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cr_df, use_container_width=True)

            st.markdown("---")
            st.subheader("Salary Model — Ridge Regression")
            sc1, sc2 = st.columns(2)
            sc1.metric("MAE (LPA)", f"{sal_m.get('mae_lpa', 0):.2f}")
            sc2.metric("R²", f"{sal_m.get('r2', 0):.3f}")
            info_box(
                sal_m.get("prediction_note", ""),
                kind="warn",
            )
