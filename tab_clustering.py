"""
tab_clustering.py — Customer Segmentation (K-Means Clustering) Tab
Havisha's Skincare Recommendation Platform
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

CLUSTER_COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']

PERSONA_ICONS = {
    "Urban Skincare Enthusiast":    "🌆",
    "Ayurveda & Wellness Loyalist": "🌿",
    "Budget-Conscious Explorer":    "🎓",
    "Mature & Premium Seeker":      "💎",
    "Minimalist Basics User":       "✨"
}


def render(df: pd.DataFrame, cluster_meta: dict):
    st.markdown("## 🎯 Customer Segmentation — K-Means Clustering")
    st.markdown(
        "K-Means clustering groups your respondents into distinct personas "
        "based on skin profile, behaviour, spending, and digital habits."
    )

    df_c   = cluster_meta['df_clustered'].copy()
    X_pca  = cluster_meta['X_pca']
    names  = cluster_meta['cluster_names']
    actions = cluster_meta['cluster_actions']
    channels = cluster_meta['cluster_channels']
    k_range  = cluster_meta['k_range']
    inertias = cluster_meta['inertias']
    sil_scores = cluster_meta['sil_scores']

    # ── Elbow & Silhouette ────────────────────────────────────────────────────
    st.markdown("#### 📐 Choosing the Optimal Number of Clusters")
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=inertias,
                                  mode='lines+markers',
                                  marker=dict(size=8, color='#E63946'),
                                  name='Inertia'))
        fig.update_layout(title='Elbow Method',
                           xaxis_title='Number of Clusters (k)',
                           yaxis_title='Inertia',
                           margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=sil_scores,
                                  mode='lines+markers',
                                  marker=dict(size=8, color='#2A9D8F'),
                                  name='Silhouette'))
        fig.update_layout(title='Silhouette Score',
                           xaxis_title='Number of Clusters (k)',
                           yaxis_title='Silhouette Score',
                           margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── PCA Scatter ───────────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Customer Segments — 2D PCA Projection")
    pca_df = pd.DataFrame({
        'PC1':    X_pca[:, 0],
        'PC2':    X_pca[:, 1],
        'Cluster': df_c['cluster_name']
    })
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                     color_discrete_sequence=CLUSTER_COLORS,
                     opacity=0.65,
                     labels={'PC1': 'Principal Component 1',
                             'PC2': 'Principal Component 2'})
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend_title='Cluster', height=450, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Persona cards ─────────────────────────────────────────────────────────
    st.markdown("#### 👤 Cluster Persona Cards")

    cluster_ids = sorted(df_c['cluster'].unique())
    cols = st.columns(len(cluster_ids))

    for idx, cid in enumerate(cluster_ids):
        cdf   = df_c[df_c['cluster'] == cid]
        cname = names.get(cid, f"Cluster {cid}")
        icon  = PERSONA_ICONS.get(cname, "👤")

        top_concern_col = None
        concern_cols = [c for c in df_c.columns if c.startswith('concern_')]
        if concern_cols:
            concern_means = cdf[concern_cols].mean()
            top_concern_col = concern_means.idxmax().replace('concern_', '').replace('_', ' ').title()

        with cols[idx]:
            st.markdown(
                f"""
                <div style="background:#f8f9fa;border-radius:12px;padding:14px;
                            border-left:5px solid {CLUSTER_COLORS[idx]};
                            min-height:260px;font-size:0.85rem">
                <h4 style="margin:0 0 6px 0">{icon} {cname}</h4>
                <b>Size:</b> {len(cdf):,} respondents ({len(cdf)/len(df_c)*100:.1f}%)<br>
                <b>Avg Spend:</b> ₹{cdf['monthly_skincare_spend'].mean():,.0f}/mo<br>
                <b>Signup Rate:</b> {cdf['will_signup'].mean()*100:.1f}%<br>
                <b>Top Concern:</b> {top_concern_col or '—'}<br>
                <b>Skin Type:</b> {cdf['skin_type'].mode()[0] if not cdf['skin_type'].empty else '—'}<br>
                <b>Psychographic:</b> {cdf['psychographic_type'].mode()[0] if not cdf['psychographic_type'].empty else '—'}<br>
                <b>Channel:</b> {channels.get(cid,'—')}<br><br>
                <i style="color:#555">🎯 {actions.get(cid,'—')}</i>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # ── Cluster profiles ──────────────────────────────────────────────────────
    st.markdown("#### 📊 Cluster Profile Comparisons")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Average Monthly Spend by Cluster**")
        spend_c = df_c.groupby('cluster_name')['monthly_skincare_spend'].mean().reset_index()
        spend_c.columns = ['Cluster', 'Avg Spend (₹)']
        fig = px.bar(spend_c, x='Cluster', y='Avg Spend (₹)',
                     color='Cluster',
                     color_discrete_sequence=CLUSTER_COLORS,
                     text=spend_c['Avg Spend (₹)'].apply(lambda x: f"₹{x:,.0f}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, xaxis_tickangle=-20, margin=dict(t=10, b=60))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("**Signup Rate by Cluster**")
        sign_c = df_c.groupby('cluster_name')['will_signup'].mean().reset_index()
        sign_c.columns = ['Cluster', 'Signup Rate']
        sign_c['Signup Rate (%)'] = (sign_c['Signup Rate'] * 100).round(1)
        fig = px.bar(sign_c, x='Cluster', y='Signup Rate (%)',
                     color='Cluster',
                     color_discrete_sequence=CLUSTER_COLORS,
                     text=sign_c['Signup Rate (%)'].apply(lambda x: f"{x}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, xaxis_tickangle=-20, margin=dict(t=10, b=60))
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        st.markdown("**Skin Type Distribution by Cluster**")
        skin_cluster = df_c.groupby(['cluster_name', 'skin_type']).size().reset_index(name='Count')
        fig = px.bar(skin_cluster, x='cluster_name', y='Count', color='skin_type',
                     barmode='stack',
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     labels={'cluster_name': 'Cluster', 'skin_type': 'Skin Type'})
        fig.update_layout(xaxis_tickangle=-20, margin=dict(t=10, b=60),
                          legend_title='Skin Type')
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown("**City Tier Distribution by Cluster**")
        city_cluster = df_c.groupby(['cluster_name', 'city_tier']).size().reset_index(name='Count')
        fig = px.bar(city_cluster, x='cluster_name', y='Count', color='city_tier',
                     barmode='stack',
                     color_discrete_sequence=['#2C7BB6', '#ABD9E9', '#FDAE61'],
                     labels={'cluster_name': 'Cluster', 'city_tier': 'City Tier'})
        fig.update_layout(xaxis_tickangle=-20, margin=dict(t=10, b=60),
                          legend_title='City Tier')
        st.plotly_chart(fig, use_container_width=True)

    # ── Concern radar per cluster ─────────────────────────────────────────────
    st.markdown("#### 🕸️ Skin Concern Radar by Cluster")
    concern_cols  = [c for c in df_c.columns if c.startswith('concern_')]
    concern_short = [c.replace('concern_', '').replace('_', ' ').title() for c in concern_cols]

    radar_data = df_c.groupby('cluster_name')[concern_cols].mean() * 100

    fig = go.Figure()
    for i, cname in enumerate(radar_data.index):
        values = list(radar_data.loc[cname].values) + [radar_data.loc[cname].values[0]]
        cats   = concern_short + [concern_short[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=cats, fill='toself',
            name=cname,
            line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
            opacity=0.6
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, height=480, margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw cluster table ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📋 Cluster Summary Table")
    summary_rows = []
    for cid in cluster_ids:
        cdf  = df_c[df_c['cluster'] == cid]
        cname = names.get(cid, f"Cluster {cid}")
        summary_rows.append({
            'Cluster Name':          cname,
            'Size':                  len(cdf),
            'Avg Age':               round(cdf['age'].mean(), 1),
            'Top Skin Type':         cdf['skin_type'].mode()[0] if not cdf['skin_type'].empty else '—',
            'Top Psychographic':     cdf['psychographic_type'].mode()[0] if not cdf['psychographic_type'].empty else '—',
            'Avg Spend (₹)':         int(cdf['monthly_skincare_spend'].mean()),
            'Signup Rate (%)':       round(cdf['will_signup'].mean() * 100, 1),
            'Avg Concerns':          round(cdf['total_concerns'].mean(), 1),
            'Recommended Action':    actions.get(cid, '—'),
            'Best Channel':          channels.get(cid, '—')
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)
