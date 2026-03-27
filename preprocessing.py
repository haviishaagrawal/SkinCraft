"""
preprocessing.py — Shared data cleaning & encoding pipeline
Havisha's Skincare Recommendation Platform
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Column definitions ────────────────────────────────────────────────────────

BINARY_PRODUCT_COLS = [
    'uses_cleanser', 'uses_moisturiser', 'uses_sunscreen', 'uses_serum',
    'uses_acne_treatment', 'uses_brightening', 'uses_under_eye_cream', 'uses_ayurvedic'
]

CONCERN_COLS = [
    'concern_acne_breakouts', 'concern_pigmentation_dark_spots',
    'concern_open_pores', 'concern_dullness', 'concern_dark_circles',
    'concern_uneven_skin_tone', 'concern_dryness_dehydration',
    'concern_anti_ageing_wrinkles', 'concern_sensitivity_redness', 'concern_tan_removal'
]

CATEGORICAL_COLS = [
    'gender', 'city_tier', 'occupation', 'monthly_income_band',
    'skin_type', 'skin_tone', 'climate_zone', 'water_hardness',
    'known_allergies', 'psychographic_type', 'brand_openness',
    'online_shopping_freq', 'failure_reason', 'preferred_format',
    'preferred_brand_origin', 'importance_natural_ingredients',
    'purchase_decision_driver', 'premium_willingness'
]

NUMERICAL_COLS = [
    'age', 'routine_steps', 'ingredient_awareness_score',
    'digital_content_hrs_week', 'current_satisfaction_score',
    'platform_appeal_score', 'total_concerns', 'monthly_skincare_spend'
]

CLASSIFICATION_FEATURES = [
    'age', 'city_tier', 'occupation', 'monthly_income_band',
    'skin_type', 'total_concerns', 'routine_steps',
    'ingredient_awareness_score', 'psychographic_type',
    'digital_content_hrs_week', 'online_shopping_freq',
    'past_product_failure', 'current_satisfaction_score',
    'platform_appeal_score', 'preferred_brand_origin',
    'purchase_decision_driver', 'premium_willingness'
] + BINARY_PRODUCT_COLS + CONCERN_COLS

REGRESSION_FEATURES = [
    'age', 'city_tier', 'occupation', 'monthly_income_band',
    'skin_type', 'total_concerns', 'routine_steps',
    'ingredient_awareness_score', 'psychographic_type',
    'digital_content_hrs_week', 'online_shopping_freq',
    'past_product_failure', 'current_satisfaction_score',
    'preferred_brand_origin', 'importance_natural_ingredients'
] + BINARY_PRODUCT_COLS + CONCERN_COLS

CLUSTERING_FEATURES = [
    'age', 'city_tier', 'skin_type', 'total_concerns',
    'routine_steps', 'ingredient_awareness_score',
    'psychographic_type', 'digital_content_hrs_week',
    'monthly_skincare_spend', 'past_product_failure',
    'current_satisfaction_score', 'platform_appeal_score'
] + BINARY_PRODUCT_COLS


# ── Income ordinal mapping ────────────────────────────────────────────────────

INCOME_ORDER = {
    'Below ₹20,000': 1,
    '₹20,001–₹40,000': 2,
    '₹40,001–₹70,000': 3,
    '₹70,001–₹1,00,000': 4,
    'Above ₹1,00,000': 5
}

ONLINE_SHOP_ORDER = {
    'Never': 1,
    'Rarely (once in 3–6 months)': 2,
    'Occasionally (once a month)': 3,
    'Regularly (2–3 times a month)': 4,
    'Very Frequently (weekly)': 5
}

CITY_TIER_ORDER = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}

PREMIUM_ORDER = {
    'Yes, definitely': 4,
    'Yes, if proven effective': 3,
    'Maybe': 2,
    'No': 1
}


def load_data(path: str = "skincare_survey_data.csv") -> pd.DataFrame:
    """Load and basic-clean the raw survey CSV."""
    df = pd.read_csv(path)

    # Fill known missing categoricals with mode
    for col in ['water_hardness', 'known_allergies', 'monthly_income_band']:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    return df


def encode_for_ml(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Encode a subset of columns for ML:
    - Ordinal columns → integer mapping
    - Other categoricals → LabelEncoder
    - Numericals → pass through
    Returns encoded DataFrame with only feature_list columns.
    """
    df = df.copy()

    # Fill missing
    for col in ['water_hardness', 'known_allergies', 'monthly_income_band']:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    # Ordinal mappings
    if 'monthly_income_band' in df.columns:
        df['monthly_income_band'] = df['monthly_income_band'].map(INCOME_ORDER).fillna(2)
    if 'online_shopping_freq' in df.columns:
        df['online_shopping_freq'] = df['online_shopping_freq'].map(ONLINE_SHOP_ORDER).fillna(3)
    if 'city_tier' in df.columns:
        df['city_tier'] = df['city_tier'].map(CITY_TIER_ORDER).fillna(2)
    if 'premium_willingness' in df.columns:
        df['premium_willingness'] = df['premium_willingness'].map(PREMIUM_ORDER).fillna(2)

    # Label-encode remaining categoricals
    cat_cols_to_encode = [
        c for c in CATEGORICAL_COLS
        if c in df.columns
        and c not in ['monthly_income_band', 'online_shopping_freq', 'city_tier', 'premium_willingness']
    ]
    le = LabelEncoder()
    for col in cat_cols_to_encode:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    # Keep only requested features (that exist)
    available = [f for f in feature_list if f in df.columns]
    return df[available].copy()


def scale_for_clustering(X: pd.DataFrame) -> np.ndarray:
    """StandardScaler for clustering."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler
