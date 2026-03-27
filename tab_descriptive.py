"""
tab_descriptive.py — Descriptive Analysis Tab
Havisha's Skincare Recommendation Platform
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


CONCERN_LABELS = {
    'concern_acne_breakouts':        'Acne / Breakouts',
    'concern_pigmentation_dark_spots': 'Pigmentation & Dark Spots',
    'concern_open_pores':            'Open Pores',
    'concern_dullness':              'Dullness',
    'concern_dark_circles':          'Dark Circles',
    'concern_uneven_skin_tone':      'Uneven Skin Tone',
    'concern_dryness_dehydration':   'Dryness / Dehydration',
    'concern_anti_ageing_wrinkles':  'Anti-Ageing / Wrinkles',
    'concern_sensitivity_redness':   'Sensitivity / Redness',
    'concern_tan_removal':           'Tan Removal'
}

PRODUCT_LABELS = {
    'uses_cleanser':        'Cleanser',
    'uses_moisturiser':     'Moisturiser',
    'uses_sunscreen':       'Sunscreen',
    'uses_serum':           'Serum',
    'uses_acne_treatment':  'Acne Treatment',
    'uses_brightening':     'Brightening Product',
    'uses_under_eye_cream': 'Under-Eye Cream',
    'uses_ayurvedic':       'Ayurvedic Product'
}

PALETTE = px.colors.qualitative.Pastel


def render(df: pd.DataFrame):
    st.markdown("## 📊 Descriptive Analysis — Market Overview")
    st.markdown("Understanding who your respondents are and what their skincare life looks like today.")

    # ── KPI row ───────────────────────────────────────────────────────────────
    total        = len(df)
    avg_spend    = df['monthly_skincare_spend'].mean()
    signup_rate  = df['will_signup'].mean() * 100
    avg_concerns = df['total_concerns'].mean()
    past_fail    = df['past_product_failure'].mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Respondents",    f"{total:,}")
    k2.metric("Avg Monthly Spend",    f"₹{avg_spend:,.0f}")
    k3.metric("Platform Signup Rate", f"{signup_rate:.1f}%")
    k4.metric("Avg Skin Concerns",    f"{avg_concerns:.1f}")
    k5.metric("Had Product Failure",  f"{past_fail:.1f}%")

    st.divider()

    # ── Row 1: Skin type + Gender ─────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Skin Type Distribution")
        skin_counts = df['skin_type'].value_counts().reset_index()
        skin_counts.columns = ['Skin Type', 'Count']
        fig = px.pie(skin_counts, names='Skin Type', values='Count',
                     color_discrete_sequence=PALETTE, hole=0.4)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Gender Distribution")
        gen_counts = df['gender'].value_counts().reset_index()
        gen_counts.columns = ['Gender', 'Count']
        fig = px.bar(gen_counts, x='Gender', y='Count',
                     color='Gender', color_discrete_sequence=PALETTE,
                     text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='Respondents',
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Age distribution + City tier ───────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Age Distribution")
        fig = px.histogram(df, x='age', nbins=20,
                           color_discrete_sequence=['#7FCDBB'],
                           labels={'age': 'Age', 'count': 'Count'})
        fig.update_layout(bargap=0.05, xaxis_title='Age', yaxis_title='Count',
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("#### City Tier Distribution")
        tier_counts = df['city_tier'].value_counts().reset_index()
        tier_counts.columns = ['City Tier', 'Count']
        fig = px.bar(tier_counts, x='City Tier', y='Count',
                     color='City Tier',
                     color_discrete_sequence=['#FEB24C', '#FD8D3C', '#FC4E2A'],
                     text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 3: Concern frequency ──────────────────────────────────────────────
    st.markdown("#### 🔴 Skin Concern Frequency Across All Respondents")
    concern_cols = list(CONCERN_LABELS.keys())
    concern_pct  = df[concern_cols].mean() * 100
    concern_df   = pd.DataFrame({
        'Concern':    [CONCERN_LABELS[c] for c in concern_cols],
        'Percentage': concern_pct.values
    }).sort_values('Percentage', ascending=True)

    fig = px.bar(concern_df, x='Percentage', y='Concern', orientation='h',
                 color='Percentage',
                 color_continuous_scale='RdYlGn_r',
                 text=concern_df['Percentage'].apply(lambda x: f"{x:.1f}%"))
    fig.update_traces(textposition='outside')
    fig.update_layout(coloraxis_showscale=False,
                      xaxis_title='% of Respondents', yaxis_title='',
                      margin=dict(t=10, b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Concern heatmap by skin type ───────────────────────────────────
    st.markdown("#### 🌡️ Skin Concern Heatmap by Skin Type")
    pivot = df.groupby('skin_type')[concern_cols].mean() * 100
    pivot.columns = [CONCERN_LABELS[c] for c in concern_cols]
    pivot = pivot.round(1)

    fig = px.imshow(pivot, aspect='auto',
                    color_continuous_scale='YlOrRd',
                    labels=dict(color="% with concern"),
                    text_auto=True)
    fig.update_layout(xaxis_title='', yaxis_title='Skin Type',
                      margin=dict(t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 5: Product usage + Psychographic ──────────────────────────────────
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("#### 🧴 Current Product Usage")
        prod_cols = list(PRODUCT_LABELS.keys())
        prod_pct  = df[prod_cols].mean() * 100
        prod_df   = pd.DataFrame({
            'Product':    [PRODUCT_LABELS[p] for p in prod_cols],
            'Usage (%)':  prod_pct.values
        }).sort_values('Usage (%)', ascending=True)
        fig = px.bar(prod_df, x='Usage (%)', y='Product', orientation='h',
                     color='Usage (%)', color_continuous_scale='Blues',
                     text=prod_df['Usage (%)'].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False,
                          xaxis_title='% using product', yaxis_title='',
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown("#### 🧠 Psychographic Segments")
        psy_counts = df['psychographic_type'].value_counts().reset_index()
        psy_counts.columns = ['Type', 'Count']
        fig = px.pie(psy_counts, names='Type', values='Count',
                     color_discrete_sequence=px.colors.qualitative.Set2, hole=0.35)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 6: Spend distribution + Occupation ────────────────────────────────
    c7, c8 = st.columns(2)

    with c7:
        st.markdown("#### 💰 Monthly Skincare Spend Distribution")
        fig = px.histogram(df, x='monthly_skincare_spend', nbins=50,
                           color_discrete_sequence=['#9ECAE1'],
                           labels={'monthly_skincare_spend': 'Monthly Spend (₹)'})
        fig.add_vline(x=df['monthly_skincare_spend'].mean(),
                      line_dash='dash', line_color='crimson',
                      annotation_text=f"Mean ₹{df['monthly_skincare_spend'].mean():,.0f}")
        fig.add_vline(x=df['monthly_skincare_spend'].median(),
                      line_dash='dot', line_color='darkgreen',
                      annotation_text=f"Median ₹{df['monthly_skincare_spend'].median():,.0f}")
        fig.update_layout(xaxis_title='Monthly Spend (₹)', yaxis_title='Count',
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        st.markdown("#### 💼 Occupation Distribution")
        occ_counts = df['occupation'].value_counts().reset_index()
        occ_counts.columns = ['Occupation', 'Count']
        fig = px.bar(occ_counts, x='Count', y='Occupation', orientation='h',
                     color='Count', color_continuous_scale='Purples',
                     text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False,
                          xaxis_title='Count', yaxis_title='',
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 7: Spend by skin type + city tier ─────────────────────────────────
    st.markdown("#### 💸 Average Monthly Spend by Skin Type & City Tier")
    spend_pivot = df.groupby(['skin_type', 'city_tier'])['monthly_skincare_spend'].mean().reset_index()
    spend_pivot.columns = ['Skin Type', 'City Tier', 'Avg Spend (₹)']
    fig = px.bar(spend_pivot, x='Skin Type', y='Avg Spend (₹)', color='City Tier',
                 barmode='group',
                 color_discrete_sequence=['#2C7BB6', '#ABD9E9', '#FDAE61'],
                 text=spend_pivot['Avg Spend (₹)'].apply(lambda x: f"₹{x:,.0f}"))
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title='', yaxis_title='Avg Monthly Spend (₹)',
                      margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 8: Routine steps + ingredient awareness ────────────────────────────
    c9, c10 = st.columns(2)

    with c9:
        st.markdown("#### 🔢 Routine Steps Distribution")
        step_counts = df['routine_steps'].value_counts().sort_index().reset_index()
        step_counts.columns = ['Steps', 'Count']
        fig = px.bar(step_counts, x='Steps', y='Count',
                     color='Count', color_continuous_scale='Teal',
                     text='Count', labels={'Steps': 'Steps in Daily Routine'})
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c10:
        st.markdown("#### 🧪 Ingredient Awareness Score")
        fig = px.histogram(df, x='ingredient_awareness_score', nbins=20,
                           color_discrete_sequence=['#FC8D59'],
                           labels={'ingredient_awareness_score': 'Score (0–10)'})
        fig.update_layout(xaxis_title='Awareness Score (0–10)', yaxis_title='Count',
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Data snapshot ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📋 Data Snapshot (First 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)
