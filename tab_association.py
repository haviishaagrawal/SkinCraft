"""
tab_association.py — Association Rule Mining (Apriori) Tab
Havisha's Skincare Recommendation Platform
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


PRODUCT_LABELS = {
    'uses_cleanser':        'Cleanser',
    'uses_moisturiser':     'Moisturiser',
    'uses_sunscreen':       'Sunscreen',
    'uses_serum':           'Serum',
    'uses_acne_treatment':  'Acne Treatment',
    'uses_brightening':     'Brightening',
    'uses_under_eye_cream': 'Under-Eye Cream',
    'uses_ayurvedic':       'Ayurvedic'
}

PRODUCT_COLS = list(PRODUCT_LABELS.keys())


def build_rules(df: pd.DataFrame, min_support: float, min_confidence: float) -> pd.DataFrame:
    """Run Apriori on binary product basket and return filtered rules."""
    basket = df[PRODUCT_COLS].rename(columns=PRODUCT_LABELS).astype(bool)
    freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame()
    rules = association_rules(freq_items, metric='confidence', min_threshold=min_confidence)
    rules = rules[rules['lift'] > 1.0].copy()
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ' + '.join(sorted(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ' + '.join(sorted(x)))
    rules['support_pct']    = (rules['support'] * 100).round(2)
    rules['confidence_pct'] = (rules['confidence'] * 100).round(2)
    rules['lift']           = rules['lift'].round(3)
    return rules.sort_values('lift', ascending=False).reset_index(drop=True)


def render(df: pd.DataFrame):
    st.markdown("## 🛒 Association Rule Mining — Product Affinity & Bundle Insights")
    st.markdown(
        "Apriori algorithm discovers which products are bought **together**, "
        "enabling cross-sell recommendations, bundle design, and combo discounting."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Algorithm Parameters", expanded=True):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            min_support    = st.slider("Minimum Support (%)", 2, 30, 5, 1) / 100
        with cc2:
            min_confidence = st.slider("Minimum Confidence (%)", 30, 90, 55, 5) / 100
        with cc3:
            min_lift       = st.slider("Minimum Lift", 1.0, 3.0, 1.2, 0.1)

    with st.spinner("Mining association rules..."):
        rules = build_rules(df, min_support, min_confidence)

    if rules.empty:
        st.warning("No rules found with current thresholds. Try lowering support or confidence.")
        return

    rules = rules[rules['lift'] >= min_lift].reset_index(drop=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rules Found",         f"{len(rules):,}")
    k2.metric("Max Lift",                  f"{rules['lift'].max():.3f}")
    k3.metric("Avg Confidence",            f"{rules['confidence_pct'].mean():.1f}%")
    k4.metric("Avg Support",               f"{rules['support_pct'].mean():.2f}%")

    st.divider()

    # ── Product co-occurrence ──────────────────────────────────────────────────
    st.markdown("#### 🔗 Product Co-Occurrence Heatmap")
    basket = df[PRODUCT_COLS].rename(columns=PRODUCT_LABELS)
    cooc   = basket.T.dot(basket)
    np.fill_diagonal(cooc.values, 0)  # remove self-counts

    fig = px.imshow(cooc,
                    color_continuous_scale='Blues',
                    text_auto=True,
                    labels=dict(color='Co-occurrence Count'),
                    aspect='auto')
    fig.update_layout(margin=dict(t=10, b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Top rules table ────────────────────────────────────────────────────────
    st.markdown("#### 🏆 Top Association Rules (by Lift)")
    display_cols = ['antecedents_str', 'consequents_str',
                    'support_pct', 'confidence_pct', 'lift']
    col_rename   = {
        'antecedents_str': 'If Customer Uses',
        'consequents_str': 'They Also Use',
        'support_pct':     'Support (%)',
        'confidence_pct':  'Confidence (%)',
        'lift':            'Lift'
    }
    top_rules = rules[display_cols].rename(columns=col_rename).head(25)
    st.dataframe(top_rules, use_container_width=True)

    st.divider()

    # ── Confidence vs Lift scatter ─────────────────────────────────────────────
    st.markdown("#### 📈 Confidence vs Lift (Bubble = Support)")
    fig = px.scatter(rules.head(40),
                     x='confidence_pct', y='lift',
                     size='support_pct',
                     color='lift',
                     color_continuous_scale='YlOrRd',
                     hover_data={
                         'antecedents_str': True,
                         'consequents_str': True,
                         'support_pct':     True,
                         'confidence_pct':  True,
                         'lift':            True
                     },
                     labels={
                         'confidence_pct': 'Confidence (%)',
                         'lift':           'Lift',
                         'support_pct':    'Support (%)',
                         'antecedents_str': 'If Uses',
                         'consequents_str': 'Then Uses'
                     })
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Support vs Confidence scatter ─────────────────────────────────────────
    st.markdown("#### 📉 Support vs Confidence Trade-off")
    fig = px.scatter(rules.head(40),
                     x='support_pct', y='confidence_pct',
                     color='lift', size='lift',
                     color_continuous_scale='Viridis',
                     hover_data={
                         'antecedents_str': True,
                         'consequents_str': True
                     },
                     labels={
                         'support_pct':     'Support (%)',
                         'confidence_pct':  'Confidence (%)',
                         'antecedents_str': 'If Uses',
                         'consequents_str': 'Then Uses'
                     })
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Product usage frequency ────────────────────────────────────────────────
    st.markdown("#### 🧴 Individual Product Usage Frequency")
    prod_freq = basket.mean() * 100
    prod_df   = pd.DataFrame({
        'Product':    prod_freq.index.tolist(),
        'Usage (%)':  prod_freq.values
    }).sort_values('Usage (%)', ascending=True)
    fig = px.bar(prod_df, x='Usage (%)', y='Product', orientation='h',
                 color='Usage (%)', color_continuous_scale='Teal',
                 text=prod_df['Usage (%)'].apply(lambda x: f"{x:.1f}%"))
    fig.update_traces(textposition='outside')
    fig.update_layout(coloraxis_showscale=False,
                      xaxis_title='% of Respondents using product', yaxis_title='',
                      margin=dict(t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Bundle recommendations ─────────────────────────────────────────────────
    st.markdown("#### 💡 Recommended Product Bundles (from Rules)")
    top5 = rules.head(5)
    bundle_rows = []
    for _, row in top5.iterrows():
        bundle_rows.append({
            'Bundle':         f"{row['antecedents_str']} → {row['consequents_str']}",
            'Confidence (%)': row['confidence_pct'],
            'Lift':           row['lift'],
            'Support (%)':    row['support_pct'],
            'Business Action': (
                "Offer as a combo kit with 10–15% discount" if row['lift'] > 2.0
                else "Cross-sell recommendation on product page"
                if row['lift'] > 1.5
                else "Email newsletter pairing suggestion"
            )
        })
    bundle_df = pd.DataFrame(bundle_rows)
    st.dataframe(bundle_df, use_container_width=True)

    st.info(
        "💡 **How to use this:** Rules with **Lift > 2** are your strongest bundle candidates — "
        "customers using the antecedent product are 2x more likely than average to also use the consequent. "
        "These should be your first combo-discount offers."
    )
