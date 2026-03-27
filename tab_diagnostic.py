"""
tab_diagnostic.py — Diagnostic Analysis Tab
Havisha's Skincare Recommendation Platform
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats


CONCERN_COLS = [
    'concern_acne_breakouts', 'concern_pigmentation_dark_spots',
    'concern_open_pores', 'concern_dullness', 'concern_dark_circles',
    'concern_uneven_skin_tone', 'concern_dryness_dehydration',
    'concern_anti_ageing_wrinkles', 'concern_sensitivity_redness', 'concern_tan_removal'
]
CONCERN_SHORT = [
    'Acne', 'Pigmentation', 'Open Pores', 'Dullness', 'Dark Circles',
    'Uneven Tone', 'Dryness', 'Anti-Ageing', 'Sensitivity', 'Tan Removal'
]


def render(df: pd.DataFrame):
    st.markdown("## 🔍 Diagnostic Analysis — Why Is the Market Behaving This Way?")
    st.markdown("Root-cause exploration of spend drivers, satisfaction gaps, and platform readiness signals.")

    # ── 1. Correlation matrix ─────────────────────────────────────────────────
    st.markdown("#### 📐 Correlation Matrix — Key Numerical Variables")
    num_cols = [
        'age', 'routine_steps', 'ingredient_awareness_score',
        'digital_content_hrs_week', 'current_satisfaction_score',
        'platform_appeal_score', 'total_concerns',
        'monthly_skincare_spend', 'past_product_failure', 'will_signup'
    ]
    corr = df[num_cols].corr().round(2)
    readable = {
        'age': 'Age', 'routine_steps': 'Routine Steps',
        'ingredient_awareness_score': 'Ingredient Awareness',
        'digital_content_hrs_week': 'Digital Hrs/Week',
        'current_satisfaction_score': 'Satisfaction Score',
        'platform_appeal_score': 'Platform Appeal',
        'total_concerns': 'Total Concerns',
        'monthly_skincare_spend': 'Monthly Spend (₹)',
        'past_product_failure': 'Past Failure',
        'will_signup': 'Will Signup'
    }
    corr.index   = [readable[c] for c in corr.index]
    corr.columns = [readable[c] for c in corr.columns]

    fig = px.imshow(corr, text_auto=True, aspect='auto',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    labels=dict(color="Correlation"))
    fig.update_layout(margin=dict(t=10, b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 2. Spend drivers ──────────────────────────────────────────────────────
    st.markdown("#### 💸 What Drives Monthly Skincare Spend?")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Spend by Psychographic Type**")
        spend_psy = df.groupby('psychographic_type')['monthly_skincare_spend'].mean().reset_index()
        spend_psy.columns = ['Psychographic', 'Avg Spend (₹)']
        spend_psy = spend_psy.sort_values('Avg Spend (₹)', ascending=False)
        fig = px.bar(spend_psy, x='Psychographic', y='Avg Spend (₹)',
                     color='Avg Spend (₹)', color_continuous_scale='Oranges',
                     text=spend_psy['Avg Spend (₹)'].apply(lambda x: f"₹{x:,.0f}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30,
                          margin=dict(t=10, b=60))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Spend vs Ingredient Awareness Score**")
        fig = px.scatter(df.sample(500, random_state=42),
                         x='ingredient_awareness_score',
                         y='monthly_skincare_spend',
                         color='skin_type', opacity=0.6,
                         trendline='ols',
                         labels={'ingredient_awareness_score': 'Ingredient Awareness (0–10)',
                                 'monthly_skincare_spend': 'Monthly Spend (₹)'},
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Spend by Income Band**")
        income_order = ['Below ₹20,000', '₹20,001–₹40,000', '₹40,001–₹70,000',
                        '₹70,001–₹1,00,000', 'Above ₹1,00,000']
        spend_inc = df.groupby('monthly_income_band')['monthly_skincare_spend'].mean().reset_index()
        spend_inc.columns = ['Income Band', 'Avg Spend (₹)']
        spend_inc['Income Band'] = pd.Categorical(spend_inc['Income Band'],
                                                   categories=income_order, ordered=True)
        spend_inc = spend_inc.sort_values('Income Band')
        fig = px.line(spend_inc, x='Income Band', y='Avg Spend (₹)',
                      markers=True, line_shape='spline',
                      color_discrete_sequence=['#E31A1C'])
        fig.update_traces(marker=dict(size=10))
        fig.update_layout(xaxis_tickangle=-30, margin=dict(t=10, b=60))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("**Spend vs Number of Concerns**")
        spend_conc = df.groupby('total_concerns')['monthly_skincare_spend'].mean().reset_index()
        spend_conc.columns = ['Total Concerns', 'Avg Spend (₹)']
        fig = px.bar(spend_conc, x='Total Concerns', y='Avg Spend (₹)',
                     color='Avg Spend (₹)', color_continuous_scale='Reds',
                     text=spend_conc['Avg Spend (₹)'].apply(lambda x: f"₹{x:,.0f}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 3. Satisfaction gap ───────────────────────────────────────────────────
    st.markdown("#### 😤 Satisfaction Gap Analysis")
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("**Current Satisfaction Score by Skin Type**")
        fig = px.box(df, x='skin_type', y='current_satisfaction_score',
                     color='skin_type',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     labels={'skin_type': 'Skin Type',
                             'current_satisfaction_score': 'Satisfaction Score (1–10)'})
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown("**Platform Appeal: Failure vs No Failure**")
        fig = px.histogram(df, x='platform_appeal_score',
                           color=df['past_product_failure'].map({1: 'Had Failure', 0: 'No Failure'}),
                           barmode='overlay', opacity=0.7, nbins=25,
                           color_discrete_sequence=['#E31A1C', '#1F78B4'],
                           labels={'platform_appeal_score': 'Platform Appeal Score',
                                   'color': 'Product Failure'})
        fig.update_layout(legend_title='', margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── 4. Why products fail ──────────────────────────────────────────────────
    st.markdown("#### ❌ Why Did Products Fail?")
    fail_df = df[df['past_product_failure'] == 1]
    fail_counts = fail_df['failure_reason'].value_counts().reset_index()
    fail_counts.columns = ['Failure Reason', 'Count']
    fail_counts['Percentage'] = (fail_counts['Count'] / len(fail_df) * 100).round(1)

    fig = px.bar(fail_counts, x='Percentage', y='Failure Reason', orientation='h',
                 color='Percentage', color_continuous_scale='Reds',
                 text=fail_counts['Percentage'].apply(lambda x: f"{x}%"))
    fig.update_traces(textposition='outside')
    fig.update_layout(coloraxis_showscale=False,
                      xaxis_title='% of Respondents with Failures', yaxis_title='',
                      margin=dict(t=10, b=10), height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 5. Purchase decision drivers ──────────────────────────────────────────
    st.markdown("#### 🛍️ What Drives Purchase Decisions?")
    c7, c8 = st.columns(2)

    with c7:
        driver_counts = df['purchase_decision_driver'].value_counts().reset_index()
        driver_counts.columns = ['Driver', 'Count']
        fig = px.pie(driver_counts, names='Driver', values='Count',
                     color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        st.markdown("**Signup Rate by Purchase Driver**")
        signup_driver = df.groupby('purchase_decision_driver')['will_signup'].mean().reset_index()
        signup_driver.columns = ['Driver', 'Signup Rate']
        signup_driver['Signup Rate (%)'] = (signup_driver['Signup Rate'] * 100).round(1)
        signup_driver = signup_driver.sort_values('Signup Rate (%)', ascending=False)
        fig = px.bar(signup_driver, x='Signup Rate (%)', y='Driver', orientation='h',
                     color='Signup Rate (%)', color_continuous_scale='Greens',
                     text=signup_driver['Signup Rate (%)'].apply(lambda x: f"{x}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 6. Signup rate diagnostic ─────────────────────────────────────────────
    st.markdown("#### 🎯 Who Is Most Likely to Sign Up?")
    c9, c10 = st.columns(2)

    with c9:
        st.markdown("**Signup Rate by Psychographic**")
        psy_signup = df.groupby('psychographic_type')['will_signup'].mean().reset_index()
        psy_signup.columns = ['Psychographic', 'Signup Rate']
        psy_signup['Signup Rate (%)'] = (psy_signup['Signup Rate'] * 100).round(1)
        psy_signup = psy_signup.sort_values('Signup Rate (%)', ascending=False)
        fig = px.bar(psy_signup, x='Signup Rate (%)', y='Psychographic', orientation='h',
                     color='Signup Rate (%)', color_continuous_scale='Blues',
                     text=psy_signup['Signup Rate (%)'].apply(lambda x: f"{x}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c10:
        st.markdown("**Signup Rate by City Tier & Skin Type**")
        signup_grid = df.groupby(['city_tier', 'skin_type'])['will_signup'].mean().reset_index()
        signup_grid['Signup Rate (%)'] = (signup_grid['will_signup'] * 100).round(1)
        fig = px.density_heatmap(signup_grid, x='skin_type', y='city_tier',
                                  z='Signup Rate (%)', color_continuous_scale='YlGn',
                                  text_auto=True)
        fig.update_layout(xaxis_title='Skin Type', yaxis_title='City Tier',
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── 7. Digital behaviour vs platform appeal ───────────────────────────────
    st.markdown("#### 📱 Digital Content Consumption vs Platform Appeal Score")
    fig = px.scatter(df.sample(600, random_state=1),
                     x='digital_content_hrs_week',
                     y='platform_appeal_score',
                     color='will_signup',
                     color_discrete_map={1: '#2CA02C', 0: '#D62728'},
                     opacity=0.55, trendline='ols',
                     labels={
                         'digital_content_hrs_week': 'Digital Skincare Content (hrs/week)',
                         'platform_appeal_score': 'Platform Appeal Score',
                         'will_signup': 'Will Sign Up'
                     },
                     category_orders={'will_signup': [1, 0]})
    fig.update_layout(margin=dict(t=10, b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Statistical insight summary ───────────────────────────────────────────
    st.markdown("#### 📌 Key Diagnostic Insights")
    corr_spend_concern = df['total_concerns'].corr(df['monthly_skincare_spend'])
    corr_aware_spend   = df['ingredient_awareness_score'].corr(df['monthly_skincare_spend'])
    corr_fail_appeal   = df['past_product_failure'].corr(df['platform_appeal_score'])
    corr_digital_signup = df['digital_content_hrs_week'].corr(df['will_signup'])

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Concern Count → Spend Correlation",    f"{corr_spend_concern:.2f}")
    i2.metric("Ingredient Awareness → Spend",          f"{corr_aware_spend:.2f}")
    i3.metric("Product Failure → Platform Appeal",     f"{corr_fail_appeal:.2f}")
    i4.metric("Digital Hrs → Signup Correlation",      f"{corr_digital_signup:.2f}")

    st.info(
        f"📊 **Key finding:** Customers with higher concern counts spend "
        f"significantly more (r={corr_spend_concern:.2f}). "
        f"Past product failure strongly predicts platform appeal (r={corr_fail_appeal:.2f}), "
        f"meaning **your unhappiest customers are your most convertible leads.** "
        f"Digital content consumption is a reliable signup predictor (r={corr_digital_signup:.2f})."
    )
