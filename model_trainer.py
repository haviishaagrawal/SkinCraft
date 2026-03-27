"""
model_trainer.py — Trains Classification, Clustering & Regression models
Havisha's Skincare Recommendation Platform
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    classification_report, mean_absolute_error, r2_score, mean_squared_error
)

from preprocessing import (
    load_data, encode_for_ml, scale_for_clustering,
    CLASSIFICATION_FEATURES, REGRESSION_FEATURES, CLUSTERING_FEATURES
)

MODEL_DIR = "models"


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


# ── Classification ─────────────────────────────────────────────────────────────

def train_classification(df: pd.DataFrame, force: bool = False) -> dict:
    ensure_model_dir()
    clf_path  = os.path.join(MODEL_DIR, "classifier.pkl")
    meta_path = os.path.join(MODEL_DIR, "clf_meta.pkl")

    if not force and os.path.exists(clf_path) and os.path.exists(meta_path):
        return joblib.load(meta_path)

    X = encode_for_ml(df, CLASSIFICATION_FEATURES)
    y = df['will_signup'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred      = rf.predict(X_test)
    y_proba     = rf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    feature_importance = pd.DataFrame({
        'feature': X.columns.tolist(),
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    meta = {
        'model':               rf,
        'feature_cols':        X.columns.tolist(),
        'X_test':              X_test,
        'y_test':              y_test,
        'y_pred':              y_pred,
        'y_proba':             y_proba,
        'accuracy':            accuracy_score(y_test, y_pred),
        'precision':           precision_score(y_test, y_pred),
        'recall':              recall_score(y_test, y_pred),
        'f1':                  f1_score(y_test, y_pred),
        'roc_auc':             roc_auc_score(y_test, y_proba),
        'fpr':                 fpr,
        'tpr':                 tpr,
        'feature_importance':  feature_importance,
        'class_report':        classification_report(y_test, y_pred, output_dict=True)
    }

    joblib.dump(rf, clf_path)
    joblib.dump(meta, meta_path)
    return meta


def predict_new_classification(df_new: pd.DataFrame, meta: dict) -> pd.Series:
    """Predict signup probability for new respondents."""
    model       = meta['model']
    feat_cols   = meta['feature_cols']
    X           = encode_for_ml(df_new, feat_cols)
    # Align columns
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_cols]
    return model.predict_proba(X)[:, 1]


# ── Clustering ─────────────────────────────────────────────────────────────────

CLUSTER_NAMES = {
    0: "Urban Skincare Enthusiast",
    1: "Ayurveda & Wellness Loyalist",
    2: "Budget-Conscious Explorer",
    3: "Mature & Premium Seeker",
    4: "Minimalist Basics User"
}

CLUSTER_ACTIONS = {
    0: "Premium bundle offer + personalised skin report + early access",
    1: "Ayurvedic ingredient spotlight + trust-building content series",
    2: "₹99 starter kit + referral discount + student offer",
    3: "Anti-ageing serum bundle + dermatologist endorsement campaign",
    4: "2-step routine kit + simplicity messaging + subscription plan"
}

CLUSTER_CHANNELS = {
    0: "Instagram / YouTube Reels",
    1: "WhatsApp Broadcast + Blog Content",
    2: "Referral Programme + College Campaigns",
    3: "Email Newsletter + Dermat Partnerships",
    4: "Push Notification + Email"
}


def train_clustering(df: pd.DataFrame, n_clusters: int = 5, force: bool = False) -> dict:
    ensure_model_dir()
    km_path   = os.path.join(MODEL_DIR, "kmeans.pkl")
    sc_path   = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path = os.path.join(MODEL_DIR, "cluster_meta.pkl")

    if not force and os.path.exists(meta_path):
        return joblib.load(meta_path)

    X_raw = encode_for_ml(df, CLUSTERING_FEATURES)
    X_scaled, scaler = scale_for_clustering(X_raw)

    # Elbow & silhouette pre-computed for display
    from sklearn.metrics import silhouette_score
    inertias    = []
    sil_scores  = []
    k_range     = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels   = km_final.fit_predict(X_scaled)
    df_out   = df.copy()
    df_out['cluster'] = labels
    df_out['cluster_name'] = df_out['cluster'].map(CLUSTER_NAMES)

    # PCA for 2D visualisation
    from sklearn.decomposition import PCA
    pca       = PCA(n_components=2, random_state=42)
    X_pca     = pca.fit_transform(X_scaled)

    meta = {
        'model':         km_final,
        'scaler':        scaler,
        'feature_cols':  X_raw.columns.tolist(),
        'labels':        labels,
        'df_clustered':  df_out,
        'X_pca':         X_pca,
        'k_range':       list(k_range),
        'inertias':      inertias,
        'sil_scores':    sil_scores,
        'n_clusters':    n_clusters,
        'cluster_names': CLUSTER_NAMES,
        'cluster_actions': CLUSTER_ACTIONS,
        'cluster_channels': CLUSTER_CHANNELS,
        'pca':           pca
    }

    joblib.dump(km_final, km_path)
    joblib.dump(scaler,   sc_path)
    joblib.dump(meta,     meta_path)
    return meta


def predict_cluster(df_new: pd.DataFrame, meta: dict) -> np.ndarray:
    model      = meta['model']
    scaler     = meta['scaler']
    feat_cols  = meta['feature_cols']
    X          = encode_for_ml(df_new, feat_cols)
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_cols]
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


# ── Regression ─────────────────────────────────────────────────────────────────

def train_regression(df: pd.DataFrame, force: bool = False) -> dict:
    ensure_model_dir()
    reg_path  = os.path.join(MODEL_DIR, "regressor.pkl")
    meta_path = os.path.join(MODEL_DIR, "reg_meta.pkl")

    if not force and os.path.exists(reg_path) and os.path.exists(meta_path):
        return joblib.load(meta_path)

    X = encode_for_ml(df, REGRESSION_FEATURES)
    y = df['monthly_skincare_spend'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, random_state=42
    )
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)

    feature_importance = pd.DataFrame({
        'feature':    X.columns.tolist(),
        'importance': gbr.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    meta = {
        'model':              gbr,
        'feature_cols':       X.columns.tolist(),
        'y_test':             y_test,
        'y_pred':             y_pred,
        'mae':                mean_absolute_error(y_test, y_pred),
        'rmse':               np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2':                 r2_score(y_test, y_pred),
        'feature_importance': feature_importance
    }

    joblib.dump(gbr,  reg_path)
    joblib.dump(meta, meta_path)
    return meta


def predict_spend(df_new: pd.DataFrame, meta: dict) -> np.ndarray:
    model     = meta['model']
    feat_cols = meta['feature_cols']
    X         = encode_for_ml(df_new, feat_cols)
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_cols]
    return model.predict(X)


# ── Master trainer ─────────────────────────────────────────────────────────────

def train_all_models(df: pd.DataFrame, force: bool = False) -> dict:
    clf_meta     = train_classification(df, force=force)
    cluster_meta = train_clustering(df, force=force)
    reg_meta     = train_regression(df, force=force)
    return {
        'classification': clf_meta,
        'clustering':     cluster_meta,
        'regression':     reg_meta
    }
