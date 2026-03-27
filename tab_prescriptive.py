"""
tab_prescriptive.py — Prescriptive Analysis Tab
Havisha's Skincare Recommendation Platform
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from model_trainer import CLUSTER_NAMES, CLUSTER_ACTIONS, CLUSTER_CHANNELS


CLUSTER_COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']

PRESCRIPTIVE_MATRIX = {
    # (cluster_id, signup_band) → action
    (0, 'High'):    {'action': 'Priority onboarding — personalised skin quiz + 20% welcome offer',
                     'offer':  'Premium Starter Bundle', 'discount': '20%',
                     'urgency': '🔴 Act Now'},
    (0, 'Medium'):  {'action': 'Retargeting with before/after content + free skin report',
                     'offer':  'Free Skin Analysis Report', 'discount': '10%',
                     'urgency': '🟡 Warm Lead'},
    (0, 'Low'):     {'action': 'Brand awareness content — ingredient education series',
                     'offer':  'Educational Newsletter', 'discount': '0%',
                     'urgency': '🟢 Nurture'},
    (1, 'High'):    {'action': 'Ayurvedic trust-building content + Nykaa-style dermat review',
                     'offer':  'Ayurvedic Glow Bundle', 'discount': '15%',
                     'urgency': '🔴 Act Now'},
    (1, 'Medium'):  {'action': 'Ingredient transparency campaign + free herbal sample kit',
                     'offer':  'Herbal Sample Kit', 'discount': '5%',
                     'urgency': '🟡 Warm Lead'},
    (1, 'Low'):     {'action': 'Long-form blog content about Ayurvedic actives',
                     'offer':  'Blog + Social Content', 'discount': '0%',
                     'urgency': '🟢 Nurture'},
    (2, 'High'):    {'action': '₹99 trial starter kit — remove price barrier immediately',
                     'offer':  '₹99 Starter Kit', 'discount': '₹99 flat',
                     'urgency': '🔴 Act Now'},
    (2, 'Medium'):  {'action': 'Referral programme — bring 2 friends, get ₹200 credit',
                     'offer':  'Referral Credit Programme', 'discount': 'Referral',
                     'urgency': '🟡 Warm Lead'},
    (2, 'Low'):     {'action': 'College ambassador outreach + offline event sampling',
                     'offer':  'Campus Sampling', 'discount': '0%',
                     'urgency': '🟢 Nurture'},
    (3, 'High'):    {'action': 'Dermat-endorsed premium anti-ageing bundle + WhatsApp follow-up',
                     'offer':  'Anti-Ageing Premium Bundle', 'discount': '12%',
                     'urgency': '🔴 Act Now'},
    (3, 'Medium'):  {'action': 'Email series — "Science of ageing skin" + serum trial',
                     'offer':  'Serum Trial + Email Series', 'discount': '8%',
                     'urgency': '🟡 Warm Lead'},
    (3, 'Low'):     {'action': 'Retargeting via email — clinical proof + testimonials',
                     'offer':  'Clinical Testimonials Content', 'discount': '0%',
                     'urgency': '🟢 Nurture'},
    (4, 'High'):    {'action': '2-step minimalist routine kit — simplicity as the pitch',
                     'offer':  'Minimalist 2-Step Kit', 'discount': '10%',
                     'urgency': '🔴 Act Now'},
    (4, 'Medium'):  {'action': 'Push notification — "Your 2-min routine is ready"',
                     'offer':  'Routine Notification Campaign', 'discount': '5%',
                     'urgency': '🟡 Warm Lead'},
    (4, 'Low'):     {'action': 'Re-engagement email with quiz — "Find your skin match"',
                     'offer':  'Skin Match Quiz', 'discount': '0%',
                     'urgency': '🟢 Nurture'},
}


def signup_band(prob: float) -> str:
    if prob >= 0.65:
        return 'High'
    elif prob >= 0.35:
        return 'Medium'
    else:
        return 'Low'


def spend_tier(spend: float) -> str:
    if spend >= 3000:
        return '💎 Premium (₹3,000+)'
    elif spend >= 1500:
        return '🥈 Mid-Range (₹1,500–₹3,000)'
    else:
        return '🥉 Budget (<₹1,500)'


def render(df: pd.DataFrame, clf_meta: dict, cluster_meta: dict, reg_meta: dict):
    st.markdown("## 💡 Prescriptive Analysis — What Should We Do?")
    st.markdown(
        "Combining classification scores, cluster personas, and spend predictions "
        "into a **concrete marketing action plan** for each customer segment."
    )

    df_c = cluster_meta['df_clustered'].copy()

    # ── Predict signup probability and spend for full dataset ─────────────────
    from model_trainer import predict_new_classification, predict_spend
    from preprocessing import encode_for_ml, CLASSIFICATION_FEATURES, REGRESSION_FEATURES

    clf_proba    = predict_new_classification(df, clf_meta)
    pred_spend   = predict_spend(df, reg_meta)

    df_c['signup_probability']  = clf_proba
    df_c['predicted_spend']     = pred_spend
    df_c['signup_band']         = df_c['signup_probability'].apply(signup_band)
    df_c['spend_tier']          = df_c['predicted_spend'].apply(spend_tier)

    # ── Action matrix ─────────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Prescriptive Action Matrix")
    matrix_rows = []
    for cid in sorted(df_c['cluster'].unique()):
        for band in ['High', 'Medium', 'Low']:
            key = (cid, band)
            info = PRESCRIPTIVE_MATRIX.get(key, {})
            seg  = df_c[(df_c['cluster'] == cid) & (df_c['signup_band'] == band)]
            matrix_rows.append({
                'Cluster':            CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
                'Signup Band':        band,
                'Segment Size':       len(seg),
                'Avg Pred Spend (₹)': f"₹{seg['predicted_spend'].mean():,.0f}" if len(seg) > 0 else '—',
                'Urgency':            info.get('urgency', '—'),
                'Recommended Offer':  info.get('offer', '—'),
                'Discount Depth':     info.get('discount', '—'),
                'Marketing Action':   info.get('action', '—'),
                'Channel':            CLUSTER_CHANNELS.get(cid, '—')
            })
    matrix_df = pd.DataFrame(matrix_rows)
    st.dataframe(matrix_df, use_container_width=True, height=480)

    st.divider()

    # ── Segment size × signup band heatmap ────────────────────────────────────
    st.markdown("#### 📊 Segment Size by Cluster & Signup Likelihood")
    heat_data = df_c.groupby(['cluster_name', 'signup_band']).size().reset_index(name='Count')
    heat_data['signup_band'] = pd.Categorical(heat_data['signup_band'],
                                               categories=['High', 'Medium', 'Low'], ordered=True)
    heat_data = heat_data.sort_values('signup_band')

    fig = px.density_heatmap(heat_data, x='signup_band', y='cluster_name',
                              z='Count', color_continuous_scale='YlOrRd',
                              text_auto=True,
                              labels={'signup_band': 'Signup Probability Band',
                                      'cluster_name': 'Customer Segment',
                                      'Count': 'Respondents'})
    fig.update_layout(margin=dict(t=10, b=10), height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Spend tier distribution by cluster ────────────────────────────────────
    st.markdown("#### 💰 Predicted Spend Tier by Cluster")
    spend_tier_data = df_c.groupby(['cluster_name', 'spend_tier']).size().reset_index(name='Count')
    fig = px.bar(spend_tier_data, x='cluster_name', y='Count', color='spend_tier',
                 barmode='stack',
                 color_discrete_sequence=['#E63946', '#457B9D', '#2A9D8F'],
                 labels={'cluster_name': 'Cluster', 'spend_tier': 'Spend Tier'})
    fig.update_layout(xaxis_tickangle=-20, legend_title='Spend Tier',
                      margin=dict(t=10, b=60))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Revenue opportunity ────────────────────────────────────────────────────
    st.markdown("#### 🏦 Estimated Monthly Revenue Opportunity")
    rev_data = df_c[df_c['signup_band'].isin(['High', 'Medium'])].copy()
    rev_data['expected_revenue'] = (
        rev_data['predicted_spend'] *
        rev_data['signup_probability']
    )
    rev_by_cluster = rev_data.groupby('cluster_name')['expected_revenue'].sum().reset_index()
    rev_by_cluster.columns = ['Cluster', 'Expected Revenue (₹)']
    rev_by_cluster = rev_by_cluster.sort_values('Expected Revenue (₹)', ascending=False)
    rev_by_cluster['Revenue Label'] = rev_by_cluster['Expected Revenue (₹)'].apply(
        lambda x: f"₹{x:,.0f}"
    )

    fig = px.bar(rev_by_cluster, x='Cluster', y='Expected Revenue (₹)',
                 color='Cluster',
                 color_discrete_sequence=CLUSTER_COLORS,
                 text='Revenue Label')
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, xaxis_tickangle=-20,
                      margin=dict(t=10, b=60))
    st.plotly_chart(fig, use_container_width=True)

    total_rev = rev_by_cluster['Expected Revenue (₹)'].sum()
    st.success(
        f"💰 **Total Estimated Monthly Revenue Opportunity** across high + medium probability leads: "
        f"**₹{total_rev:,.0f}**  \n"
        f"This is your immediate addressable market — customers already in your dataset with measurable signup intent."
    )

    st.divider()

    # ── Priority action summary ────────────────────────────────────────────────
    st.markdown("#### 🚦 Priority Action Summary — High Urgency Leads First")
    high_urgency = df_c[df_c['signup_band'] == 'High'].groupby('cluster_name').agg(
        Count=('respondent_id', 'count'),
        Avg_Spend=('predicted_spend', 'mean'),
        Avg_Signup_Prob=('signup_probability', 'mean')
    ).reset_index()
    high_urgency.columns = ['Segment', 'High-Intent Leads', 'Avg Pred Spend (₹)', 'Avg Signup Prob']
    high_urgency['Avg Pred Spend (₹)'] = high_urgency['Avg Pred Spend (₹)'].round(0).astype(int)
    high_urgency['Avg Signup Prob']    = (high_urgency['Avg Signup Prob'] * 100).round(1).astype(str) + '%'
    high_urgency = high_urgency.sort_values('High-Intent Leads', ascending=False)

    st.dataframe(high_urgency, use_container_width=True)
    st.info(
        "🎯 **Focus your marketing budget here first.** "
        "High-intent leads across all clusters represent your lowest cost-per-acquisition opportunity. "
        "Allocate 60% of your initial marketing spend to this group."
    )
