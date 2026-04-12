"""
Customer Churn Prediction System — Streamlit Dashboard
BIS405 Graduation Project | Analytics Team Alpha
Sprint 4 | Group 3: Maha, Hadeel, Ghadeer
"""
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS matching navy/dark theme ────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F3864 0%, #2E75B6 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label { color: white !important; }

    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #1F3864, #2E75B6);
        padding: 18px 24px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.7rem; font-weight: 700; }
    .main-header p  { color: #BDD7EE; margin: 4px 0 0 0; font-size: 0.9rem; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border: 1px solid #DEE2E6;
        border-top: 4px solid #1F3864;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-card .kpi-value { font-size: 2rem; font-weight: 700; color: #1F3864; }
    .kpi-card .kpi-label { font-size: 0.82rem; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Risk badges */
    .risk-high   { background:#C00000; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-medium { background:#ED7D31; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-low    { background:#375623; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }

    /* Section headings */
    .section-head {
        background: #1F3864;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1rem;
        margin: 16px 0 10px 0;
    }

    /* Table styling */
    .metric-table { width:100%; border-collapse:collapse; }
    .metric-table th { background:#1F3864; color:white; padding:8px; text-align:center; }
    .metric-table td { padding:7px 10px; border-bottom:1px solid #eee; text-align:center; }
    .metric-table tr:nth-child(even) { background:#F8F9FA; }

    /* Streamlit overrides */
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    div[data-testid="metric-container"] { background:#F8F9FA; border-radius:8px; padding:10px; }
    .stButton button { background:#1F3864; color:white; border:none; font-weight:600; }
    .stButton button:hover { background:#2E75B6; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Navigation")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Executive Dashboard",
        "🔮  Churn Predictor",
        "📈  Advanced Analytics",
        "👥  Customer Segmentation",
        "🎯  Model Performance",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**BIS405 — Sprint 4**")
    st.markdown("Analytics Team Alpha")
    st.markdown("Group 3")
    st.markdown("---")
    st.markdown("**Team**")
    st.markdown("Maha · Hadeel · Ghadeer")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("XGBoost · AUC 0.9536")

# ── Route to pages ─────────────────────────────────────────────────────────
if "Executive" in page:
    from pages import page1_executive
    page1_executive.show()
elif "Predictor" in page:
    from pages import page2_predictor
    page2_predictor.show()
elif "Analytics" in page:
    from pages import page3_analytics
    page3_analytics.show()
elif "Segmentation" in page:
    from pages import page4_segmentation
    page4_segmentation.show()
elif "Performance" in page:
    from pages import page5_performance
    page5_performance.show()
