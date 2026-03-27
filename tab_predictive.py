"""
tab_predictive.py — Predictive Analysis Tab (Classification + Regression)
Havisha's Skincare Recommendation Platform
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def render(df: pd.DataFrame, clf_meta: dict, reg_meta: dict):
    st.markdown("## 🤖 Predictive Analysis")
    st.markdown(
        "Two predictive models: **Classification** (will a customer sign up?) "
        "and **Regression** (how much will they spend monthly?)."
    )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION A — CLASSIFICATION
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🎯 Part A — Classification: Will This Customer Sign Up?")
    st.markdown("**Model: Random Forest Classifier** | Target: `will_signup` (Binary: 0 / 1)")

    # ── Performance KPIs ─────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy",  f"{clf_meta['accuracy']*100:.1f}%")
    k2.metric("Precision", f"{clf_meta['precision']*100:.1f}%")
    k3.metric("Recall",    f"{clf_meta['recall']*100:.1f}%")
    k4.metric("F1-Score",  f"{clf_meta['f1']*100:.1f}%")
    k5.metric("ROC-AUC",   f"{clf_meta['roc_auc']:.3f}")

    st.divider()

    c1, c2 = st.columns(2)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    with c1:
        st.markdown("#### ROC Curve")
        fpr = clf_meta['fpr']
        tpr = clf_meta['tpr']
        auc = clf_meta['roc_auc']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'ROC Curve (AUC = {auc:.3f})',
            line=dict(color='#E63946', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random Classifier',
            line=dict(color='grey', dash='dash')
        ))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.5, y=0.1),
            height=380, margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    with c2:
        st.markdown("#### Confusion Matrix")
        y_test = clf_meta['y_test']
        y_pred = clf_meta['y_pred']
        cm     = confusion_matrix(y_test, y_pred)

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x='Predicted', y='Actual', color='Count'),
            x=['Predicted: No Signup (0)', 'Predicted: Signup (1)'],
            y=['Actual: No Signup (0)', 'Actual: Signup (1)'],
            aspect='auto'
        )
        fig.update_layout(height=380, margin=dict(t=20, b=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("#### 🏆 Top Features Driving Signup Prediction")
    fi = clf_meta['feature_importance'].copy()
    fi['feature_clean'] = fi['feature'].str.replace('_', ' ').str.title()

    fig = px.bar(fi, x='importance', y='feature_clean', orientation='h',
                 color='importance', color_continuous_scale='Reds',
                 text=fi['importance'].apply(lambda x: f"{x:.3f}"),
                 labels={'importance': 'Importance Score', 'feature_clean': 'Feature'})
    fig.update_traces(textposition='outside')
    fig.update_layout(coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'},
                      margin=dict(t=10, b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Classification Report ─────────────────────────────────────────────────
    st.markdown("#### 📋 Full Classification Report")
    report = clf_meta['class_report']
    report_rows = []
    for label in ['0', '1']:
        label_name = 'Will NOT Signup (0)' if label == '0' else 'Will Signup (1)'
        report_rows.append({
            'Class':      label_name,
            'Precision':  f"{report[label]['precision']*100:.1f}%",
            'Recall':     f"{report[label]['recall']*100:.1f}%",
            'F1-Score':   f"{report[label]['f1-score']*100:.1f}%",
            'Support':    int(report[label]['support'])
        })
    report_rows.append({
        'Class':    'Macro Average',
        'Precision': f"{report['macro avg']['precision']*100:.1f}%",
        'Recall':    f"{report['macro avg']['recall']*100:.1f}%",
        'F1-Score':  f"{report['macro avg']['f1-score']*100:.1f}%",
        'Support':   int(report['macro avg']['support'])
    })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

    # ── Probability Distribution ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Predicted Signup Probability Distribution")
    proba_df = pd.DataFrame({
        'Signup Probability': clf_meta['y_proba'],
        'Actual':             clf_meta['y_test'].astype(str).tolist()
    })
    proba_df['Actual'] = proba_df['Actual'].map({'1': 'Actually Signed Up', '0': 'Did Not Sign Up'})
    fig = px.histogram(proba_df, x='Signup Probability', color='Actual',
                       barmode='overlay', nbins=30, opacity=0.7,
                       color_discrete_map={
                           'Actually Signed Up': '#2CA02C',
                           'Did Not Sign Up':    '#D62728'
                       })
    fig.update_layout(xaxis_title='Predicted Signup Probability',
                      yaxis_title='Count', margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION B — REGRESSION
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 💰 Part B — Regression: Predicted Monthly Skincare Spend")
    st.markdown("**Model: Gradient Boosting Regressor** | Target: `monthly_skincare_spend` (₹)")

    # ── Regression KPIs ───────────────────────────────────────────────────────
    r1, r2, r3 = st.columns(3)
    r1.metric("MAE (Mean Abs Error)",   f"₹{reg_meta['mae']:,.0f}")
    r2.metric("RMSE",                   f"₹{reg_meta['rmse']:,.0f}")
    r3.metric("R² Score",               f"{reg_meta['r2']:.3f}")

    st.divider()

    c3, c4 = st.columns(2)

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    with c3:
        st.markdown("#### Actual vs Predicted Spend")
        scatter_df = pd.DataFrame({
            'Actual (₹)':    reg_meta['y_test'],
            'Predicted (₹)': reg_meta['y_pred']
        })
        fig = px.scatter(scatter_df, x='Actual (₹)', y='Predicted (₹)',
                         opacity=0.45,
                         color_discrete_sequence=['#457B9D'])
        # Perfect prediction line
        min_val = scatter_df.min().min()
        max_val = scatter_df.max().max()
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Residuals ─────────────────────────────────────────────────────────────
    with c4:
        st.markdown("#### Residual Distribution")
        residuals = reg_meta['y_test'] - reg_meta['y_pred']
        fig = px.histogram(residuals, nbins=40,
                           color_discrete_sequence=['#2A9D8F'],
                           labels={'value': 'Residual (Actual − Predicted ₹)'})
        fig.add_vline(x=0, line_dash='dash', line_color='red',
                      annotation_text='Zero Error')
        fig.update_layout(xaxis_title='Residual (₹)', yaxis_title='Count',
                          showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Regression Feature Importance ─────────────────────────────────────────
    st.markdown("#### 🏆 Top Features Driving Spend Prediction")
    rfi = reg_meta['feature_importance'].copy()
    rfi['feature_clean'] = rfi['feature'].str.replace('_', ' ').str.title()

    fig = px.bar(rfi, x='importance', y='feature_clean', orientation='h',
                 color='importance', color_continuous_scale='Blues',
                 text=rfi['importance'].apply(lambda x: f"{x:.3f}"),
                 labels={'importance': 'Importance Score', 'feature_clean': 'Feature'})
    fig.update_traces(textposition='outside')
    fig.update_layout(coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'},
                      margin=dict(t=10, b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Predicted spend distribution ──────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Predicted Spend Distribution on Test Set")
    pred_df = pd.DataFrame({
        'Predicted Spend (₹)': reg_meta['y_pred']
    })
    fig = px.histogram(pred_df, x='Predicted Spend (₹)', nbins=40,
                       color_discrete_sequence=['#E9C46A'])
    fig.add_vline(x=reg_meta['y_pred'].mean(), line_dash='dash', line_color='red',
                  annotation_text=f"Mean ₹{reg_meta['y_pred'].mean():,.0f}")
    fig.update_layout(xaxis_title='Predicted Monthly Spend (₹)',
                      yaxis_title='Count', margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
