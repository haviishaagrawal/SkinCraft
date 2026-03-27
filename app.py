"""
app.py — Main Streamlit Application Entry Point
Havisha's Personalised Skincare Recommendation Platform
Analytical Dashboard: Descriptive | Diagnostic | Predictive | Prescriptive
"""

import streamlit as st
import pandas as pd
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinIQ — Skincare Analytics Dashboard",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #FAFAFA; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C3E50 0%, #3D5A80 100%);
    }
    [data-testid="stSidebar"] * { color: #F0F4F8 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E8ECF0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px 8px 0 0;
        padding: 0 18px;
        font-weight: 500;
    }

    /* Section headers */
    h2 { color: #2C3E50; }
    h3 { color: #3D5A80; }
    h4 { color: #457B9D; margin-top: 0.5rem; }

    /* Divider */
    hr { margin: 1rem 0; border-color: #E8ECF0; }
</style>
""", unsafe_allow_html=True)

# ── Import modules ────────────────────────────────────────────────────────────
from preprocessing import load_data
from model_trainer import train_all_models
import tab_descriptive
import tab_diagnostic
import tab_clustering
import tab_association
import tab_predictive
import tab_prescriptive
import tab_upload

# ── Data & Model loading (cached) ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    data_path = "skincare_survey_data.csv"
    if not os.path.exists(data_path):
        st.error(f"❌ Dataset not found at `{data_path}`. Please ensure the CSV is in the app directory.")
        st.stop()
    return load_data(data_path)


@st.cache_resource(show_spinner=False)
def get_models(df):
    return train_all_models(df, force=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧴 SkinIQ")
    st.markdown("**Skincare Analytics Platform**")
    st.markdown("*Powered by Havisha's Survey Data*")
    st.divider()

    st.markdown("### 📌 About")
    st.markdown(
        "This dashboard applies **four layers of analytics** "
        "on 2,000 Indian skincare survey respondents:\n\n"
        "- 📊 **Descriptive** — What happened?\n"
        "- 🔍 **Diagnostic** — Why did it happen?\n"
        "- 🤖 **Predictive** — What will happen?\n"
        "- 💡 **Prescriptive** — What should we do?"
    )
    st.divider()

    st.markdown("### 🧮 Algorithms Used")
    st.markdown(
        "- 🎯 **Classification** — Random Forest\n"
        "- 🗂️ **Clustering** — K-Means\n"
        "- 🛒 **Association Rules** — Apriori\n"
        "- 💰 **Regression** — Gradient Boosting"
    )
    st.divider()

    st.markdown("### 📦 Dataset Info")

    # Load data to show stats
    try:
        _df = get_data()
        st.markdown(f"- **Respondents:** {len(_df):,}")
        st.markdown(f"- **Features:** {_df.shape[1]}")
        st.markdown(f"- **Signup Rate:** {_df['will_signup'].mean()*100:.1f}%")
        st.markdown(f"- **Avg Spend:** ₹{_df['monthly_skincare_spend'].mean():,.0f}")
    except Exception:
        st.markdown("*(Loading...)*")

    st.divider()
    st.caption("Built for Havisha's Skincare Platform | 2025")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#2C3E50,#3D5A80);
                padding:22px 28px;border-radius:12px;margin-bottom:16px">
        <h1 style="color:#FFFFFF;margin:0;font-size:1.9rem">
            🧴 SkinIQ — Personalised Skincare Analytics
        </h1>
        <p style="color:#B0C4DE;margin:6px 0 0 0;font-size:0.95rem">
            Data-driven intelligence for India's skincare market | 2,000 Respondents | 4-Layer Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Load data & models ────────────────────────────────────────────────────────
with st.spinner("🔄 Loading dataset and training models (first load may take ~30 seconds)..."):
    df     = get_data()
    models = get_models(df)

clf_meta     = models['classification']
cluster_meta = models['clustering']
reg_meta     = models['regression']

# ── Tab navigation ────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Descriptive",
    "🔍 Diagnostic",
    "🎯 Segments",
    "🛒 Associations",
    "🤖 Predictions",
    "💡 Prescriptive",
    "📥 New Data Upload"
])

with tabs[0]:
    tab_descriptive.render(df)

with tabs[1]:
    tab_diagnostic.render(df)

with tabs[2]:
    tab_clustering.render(df, cluster_meta)

with tabs[3]:
    tab_association.render(df)

with tabs[4]:
    tab_predictive.render(df, clf_meta, reg_meta)

with tabs[5]:
    tab_prescriptive.render(df, clf_meta, cluster_meta, reg_meta)

with tabs[6]:
    tab_upload.render(df, clf_meta, cluster_meta, reg_meta)
