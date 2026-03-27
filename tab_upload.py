"""
tab_upload.py — New Customer Data Upload & Prediction Tab
Havisha's Skincare Recommendation Platform
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import io

from model_trainer import (
    predict_new_classification, predict_cluster,
    predict_spend, CLUSTER_NAMES, CLUSTER_ACTIONS, CLUSTER_CHANNELS
)
from preprocessing import (
    CLASSIFICATION_FEATURES, REGRESSION_FEATURES, CLUSTERING_FEATURES
)


def signup_band(prob):
    if prob >= 0.65:
        return '🔴 High Intent'
    elif prob >= 0.35:
        return '🟡 Medium Intent'
    else:
        return '🟢 Low Intent / Nurture'


def spend_label(spend):
    if spend >= 3000:
        return '💎 Premium (₹3,000+)'
    elif spend >= 1500:
        return '🥈 Mid-Range'
    else:
        return '🥉 Budget'


REQUIRED_COLUMNS = [
    'age', 'gender', 'city_tier', 'occupation', 'monthly_income_band',
    'skin_type', 'skin_tone', 'climate_zone', 'water_hardness', 'known_allergies',
    'concern_acne_breakouts', 'concern_pigmentation_dark_spots', 'concern_open_pores',
    'concern_dullness', 'concern_dark_circles', 'concern_uneven_skin_tone',
    'concern_dryness_dehydration', 'concern_anti_ageing_wrinkles',
    'concern_sensitivity_redness', 'concern_tan_removal',
    'total_concerns', 'routine_steps',
    'uses_cleanser', 'uses_moisturiser', 'uses_sunscreen', 'uses_serum',
    'uses_acne_treatment', 'uses_brightening', 'uses_under_eye_cream', 'uses_ayurvedic',
    'ingredient_awareness_score', 'psychographic_type', 'brand_openness',
    'digital_content_hrs_week', 'online_shopping_freq',
    'past_product_failure', 'failure_reason', 'current_satisfaction_score',
    'preferred_format', 'preferred_brand_origin', 'importance_natural_ingredients',
    'purchase_decision_driver', 'premium_willingness', 'platform_appeal_score'
]


def render(df_train: pd.DataFrame, clf_meta: dict, cluster_meta: dict, reg_meta: dict):
    st.markdown("## 📥 New Customer Data Upload & Prediction")
    st.markdown(
        "Upload a CSV of new survey respondents. The system will automatically predict "
        "**signup probability**, assign a **customer persona**, and forecast **monthly spend** — "
        "giving you a ready-to-use enriched lead list for your marketing team."
    )

    # ── Download template ─────────────────────────────────────────────────────
    st.markdown("#### 📄 Step 1 — Download the Data Template")
    st.markdown(
        "Use the same column format as your original survey data. "
        "Download a sample template (5 rows from training data) to get started."
    )
    template = df_train[REQUIRED_COLUMNS].head(5).copy()
    csv_bytes = template.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Template CSV",
        data=csv_bytes,
        file_name="new_customers_template.csv",
        mime="text/csv"
    )

    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("#### 📂 Step 2 — Upload New Customer CSV")
    uploaded_file = st.file_uploader(
        "Upload CSV file with new respondent data",
        type=['csv'],
        help="Must follow the same column structure as the training data."
    )

    if uploaded_file is None:
        st.info(
            "👆 Upload a CSV file above to begin predictions. "
            "The file should contain survey responses from new potential customers."
        )

        # Show what columns are expected
        with st.expander("📋 Expected Columns"):
            st.write(REQUIRED_COLUMNS)
        return

    # ── Load & validate ───────────────────────────────────────────────────────
    try:
        df_new = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
        return

    st.success(f"✅ File uploaded: **{uploaded_file.name}** — {len(df_new):,} rows, {df_new.shape[1]} columns")

    # Check missing columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_new.columns]
    if missing_cols:
        st.warning(
            f"⚠️ Missing {len(missing_cols)} expected column(s): `{', '.join(missing_cols[:10])}...`\n\n"
            "Predictions will proceed but accuracy may be reduced. Missing columns will be filled with defaults."
        )
        for c in missing_cols:
            df_new[c] = 0

    # Fill numeric nulls with medians from training
    for col in df_new.select_dtypes(include=[np.number]).columns:
        if df_new[col].isnull().any():
            df_new[col] = df_new[col].fillna(df_train[col].median() if col in df_train else 0)

    # Fill categorical nulls with mode
    for col in df_new.select_dtypes(include=['object']).columns:
        if df_new[col].isnull().any():
            mode = df_new[col].mode()
            df_new[col] = df_new[col].fillna(mode[0] if len(mode) > 0 else 'Unknown')

    st.divider()

    # ── Run predictions ───────────────────────────────────────────────────────
    st.markdown("#### 🤖 Step 3 — Running Predictions")
    with st.spinner("Running classification, clustering, and regression models..."):
        try:
            signup_probs   = predict_new_classification(df_new, clf_meta)
            cluster_labels = predict_cluster(df_new, cluster_meta)
            pred_spends    = predict_spend(df_new, reg_meta)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
            return

    # ── Build results dataframe ───────────────────────────────────────────────
    id_col = 'respondent_id' if 'respondent_id' in df_new.columns else None
    results = pd.DataFrame()

    if id_col:
        results['Respondent ID']  = df_new[id_col].values
    else:
        results['Respondent #']   = [f"NEW_{i+1:04d}" for i in range(len(df_new))]

    # Carry through key profile columns if available
    profile_cols = ['age', 'gender', 'city_tier', 'skin_type', 'occupation',
                    'psychographic_type', 'monthly_income_band']
    for col in profile_cols:
        if col in df_new.columns:
            results[col.replace('_', ' ').title()] = df_new[col].values

    results['Signup Probability (%)'] = (signup_probs * 100).round(1)
    results['Signup Intent']          = [signup_band(p) for p in signup_probs]
    results['Customer Persona']       = [CLUSTER_NAMES.get(c, f"Cluster {c}") for c in cluster_labels]
    results['Predicted Spend (₹)']    = pred_spends.round(0).astype(int)
    results['Spend Tier']             = [spend_label(s) for s in pred_spends]
    results['Recommended Action']     = [CLUSTER_ACTIONS.get(c, '—') for c in cluster_labels]
    results['Best Channel']           = [CLUSTER_CHANNELS.get(c, '—') for c in cluster_labels]

    st.divider()

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    st.markdown("#### 📊 Prediction Summary")
    high_intent   = (signup_probs >= 0.65).sum()
    medium_intent = ((signup_probs >= 0.35) & (signup_probs < 0.65)).sum()
    low_intent    = (signup_probs < 0.35).sum()
    avg_prob      = signup_probs.mean() * 100
    avg_spend_pred = pred_spends.mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Respondents",   f"{len(df_new):,}")
    m2.metric("🔴 High Intent",      f"{high_intent:,}")
    m3.metric("🟡 Medium Intent",    f"{medium_intent:,}")
    m4.metric("Avg Signup Prob",     f"{avg_prob:.1f}%")
    m5.metric("Avg Predicted Spend", f"₹{avg_spend_pred:,.0f}")

    st.divider()

    # ── Visualisations ────────────────────────────────────────────────────────
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Signup Intent Distribution**")
        intent_counts = results['Signup Intent'].value_counts().reset_index()
        intent_counts.columns = ['Intent', 'Count']
        fig = px.pie(intent_counts, names='Intent', values='Count',
                     color_discrete_sequence=['#E63946', '#F4A261', '#2A9D8F'],
                     hole=0.4)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with v2:
        st.markdown("**Customer Persona Distribution**")
        persona_counts = results['Customer Persona'].value_counts().reset_index()
        persona_counts.columns = ['Persona', 'Count']
        fig = px.bar(persona_counts, x='Count', y='Persona', orientation='h',
                     color='Count', color_continuous_scale='Blues',
                     text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Signup probability distribution ───────────────────────────────────────
    st.markdown("**Signup Probability Distribution — New Leads**")
    fig = px.histogram(
        pd.DataFrame({'Signup Probability (%)': signup_probs * 100}),
        x='Signup Probability (%)', nbins=25,
        color_discrete_sequence=['#457B9D']
    )
    fig.add_vline(x=65, line_dash='dash', line_color='red',
                  annotation_text='High Intent Threshold (65%)')
    fig.add_vline(x=35, line_dash='dash', line_color='orange',
                  annotation_text='Medium Intent Threshold (35%)')
    fig.update_layout(xaxis_title='Predicted Signup Probability (%)',
                      yaxis_title='Number of Leads', margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown("#### 📋 Full Results Table")
    st.dataframe(results, use_container_width=True)

    st.divider()

    # ── Download enriched CSV ─────────────────────────────────────────────────
    st.markdown("#### ⬇️ Step 4 — Download Enriched Lead List")
    out_csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Scored Lead List (CSV)",
        data=out_csv,
        file_name="scored_leads.csv",
        mime="text/csv"
    )

    st.success(
        "✅ **Enriched lead list ready.** Share this with your marketing team. "
        "Prioritise 🔴 High Intent leads for immediate outreach. "
        "Use the 'Recommended Action' and 'Best Channel' columns to personalise your campaigns."
    )

    st.divider()

    # ── High priority leads focus ─────────────────────────────────────────────
    st.markdown("#### 🚨 High-Priority Leads — Ready for Immediate Outreach")
    high_leads = results[results['Signup Intent'] == '🔴 High Intent'].copy()
    if len(high_leads) == 0:
        st.info("No high-intent leads found in this batch. Consider adjusting the threshold.")
    else:
        high_leads = high_leads.sort_values('Signup Probability (%)', ascending=False)
        st.markdown(f"**{len(high_leads):,} leads** with ≥65% predicted signup probability:")
        st.dataframe(high_leads, use_container_width=True)

        hot_csv = high_leads.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Download {len(high_leads):,} High-Priority Leads",
            data=hot_csv,
            file_name="high_priority_leads.csv",
            mime="text/csv"
        )
