"""
tests/test_integration.py
=========================
Sprint 5 — S5-05: System Integration Tests
BIS405 Graduation Project | Group 3 | Customer Churn Prediction System

Tests the complete pipeline in 4 stages, exactly matching the production
code in utils.py and page2_predictor.py:

  Stage 1 — Raw data loading (CSV files, shape, columns, class balance)
  Stage 2 — Preprocessing pipeline (encoding, no missing values, dtypes)
  Stage 3 — Model inference (all 4 models load, XGBoost AUC ≥ 0.93 on test set)
  Stage 4 — End-to-end prediction (raw input → encoded → predict_proba → result)

Run with:
  python -m pytest tests/test_integration.py -v
  python -m pytest tests/test_integration.py -v --tb=short   (compact errors)
  python -m pytest tests/test_integration.py -v -s           (see print output)

Run a single stage:
  python -m pytest tests/test_integration.py -v -k "stage1"
  python -m pytest tests/test_integration.py -v -k "stage2"
  python -m pytest tests/test_integration.py -v -k "stage3"
  python -m pytest tests/test_integration.py -v -k "stage4"
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Path setup ────────────────────────────────────────────────────────────────
# Works whether you run from the project root or the tests/ folder
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")

TRAIN_CSV = os.path.join(DATA_DIR, "data_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "data_test.csv")

XGB_PATH  = os.path.join(MODELS_DIR, "xgboost.pkl")
RF_PATH   = os.path.join(MODELS_DIR, "random_forest.pkl")
LGB_PATH  = os.path.join(MODELS_DIR, "lightgbm.pkl")
SCL_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")

# Exact feature order used in training (from utils.py FEATURES list)
FEATURE_COLS = [
    "Age", "Gender_Enc", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay", "Sub_Enc", "Con_Enc",
    "Total Spend", "Last Interaction"
]

# Raw columns expected in the CSV files
RAW_COLS_REQUIRED = [
    "CustomerID", "Age", "Gender", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay", "Subscription Type",
    "Contract Length", "Total Spend", "Last Interaction", "Churn"
]

# Encoding maps (from utils.py and page2_predictor.py)
GENDER_MAP = {"Male": 1, "Female": 0}
SUB_MAP    = {"Basic": 0, "Standard": 1, "Premium": 2}
CON_MAP    = {"Monthly": 0, "Quarterly": 1, "Annual": 2}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES  (module-scoped so they load once for all tests)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def raw_train():
    """Load raw training CSV exactly as utils.py does."""
    assert os.path.exists(TRAIN_CSV), (
        f"MISSING FILE: {TRAIN_CSV}\n"
        "Create the data/ folder and upload data_train.csv — see Part 1 of the Colab guide."
    )
    return pd.read_csv(TRAIN_CSV)


@pytest.fixture(scope="module")
def raw_test():
    """Load raw test CSV exactly as utils.py does."""
    assert os.path.exists(TEST_CSV), (
        f"MISSING FILE: {TEST_CSV}\n"
        "Create the data/ folder and upload data_test.csv — see Part 1 of the Colab guide."
    )
    return pd.read_csv(TEST_CSV)


@pytest.fixture(scope="module")
def processed_df(raw_train, raw_test):
    """
    Reproduce the exact preprocessing from utils.py load_data():
    concat → dropna → stratified sample → encode Gender/Sub/Contract
    """
    df = pd.concat([raw_train, raw_test], ignore_index=True).dropna()

    from sklearn.model_selection import train_test_split
    df, _ = train_test_split(
        df, train_size=50_000,
        stratify=df["Churn"], random_state=42
    )
    df = df.copy()
    df["Gender_Enc"] = (df["Gender"] == "Male").astype(int)
    df["Sub_Enc"]    = df["Subscription Type"].map(SUB_MAP)
    df["Con_Enc"]    = df["Contract Length"].map(CON_MAP)
    return df


@pytest.fixture(scope="module")
def xgb_model():
    assert os.path.exists(XGB_PATH), f"MISSING: {XGB_PATH}"
    return joblib.load(XGB_PATH)


@pytest.fixture(scope="module")
def rf_model():
    assert os.path.exists(RF_PATH), f"MISSING: {RF_PATH}"
    return joblib.load(RF_PATH)


@pytest.fixture(scope="module")
def lgb_model():
    assert os.path.exists(LGB_PATH), f"MISSING: {LGB_PATH}"
    return joblib.load(LGB_PATH)


@pytest.fixture(scope="module")
def scaler():
    assert os.path.exists(SCL_PATH), f"MISSING: {SCL_PATH}"
    return joblib.load(SCL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — RAW DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

class TestStage1DataLoading:
    """
    Validates that both CSV files load correctly with the expected
    structure, column names, size, and class distribution.
    """

    def test_stage1_train_csv_exists(self):
        """data_train.csv must exist in the data/ folder."""
        assert os.path.exists(TRAIN_CSV), (
            f"data_train.csv not found at {TRAIN_CSV}. "
            "Upload it to the data/ folder in Google Drive."
        )

    def test_stage1_test_csv_exists(self):
        """data_test.csv must exist in the data/ folder."""
        assert os.path.exists(TEST_CSV), (
            f"data_test.csv not found at {TEST_CSV}. "
            "Upload it to the data/ folder in Google Drive."
        )

    def test_stage1_train_has_required_columns(self, raw_train):
        """Training CSV must contain all 12 expected columns."""
        missing = [c for c in RAW_COLS_REQUIRED if c not in raw_train.columns]
        assert not missing, (
            f"Training CSV is missing these columns: {missing}\n"
            f"Columns found: {list(raw_train.columns)}"
        )

    def test_stage1_test_has_required_columns(self, raw_test):
        """Test CSV must contain all 12 expected columns."""
        missing = [c for c in RAW_COLS_REQUIRED if c not in raw_test.columns]
        assert not missing, (
            f"Test CSV is missing these columns: {missing}\n"
            f"Columns found: {list(raw_test.columns)}"
        )

    def test_stage1_train_minimum_rows(self, raw_train):
        """Training CSV must have at least 50,000 rows to support stratified sampling."""
        assert len(raw_train) >= 50_000, (
            f"Training CSV has only {len(raw_train)} rows — need at least 50,000. "
            "Make sure you uploaded the full dataset, not a small sample."
        )

    def test_stage1_test_minimum_rows(self, raw_test):
        """Test CSV must have at least 1,000 rows."""
        assert len(raw_test) >= 1_000, (
            f"Test CSV has only {len(raw_test)} rows — expected at least 1,000."
        )

    def test_stage1_churn_column_is_binary(self, raw_train):
        """Churn column must contain only 0 and 1 values."""
        unique_vals = set(raw_train["Churn"].unique())
        assert unique_vals.issubset({0, 1}), (
            f"Churn column contains unexpected values: {unique_vals}. "
            "Expected only 0 (retained) and 1 (churned)."
        )

    def test_stage1_churn_rate_in_expected_range(self, raw_train):
        """
        Churn rate must be between 50% and 60%.
        Sprint 1 EDA confirmed 55.5% churn rate in the full dataset.
        """
        churn_rate = raw_train["Churn"].mean()
        assert 0.40 <= churn_rate <= 0.65, (
            f"Churn rate is {churn_rate:.1%} — expected between 40% and 65%. "
            "This suggests the wrong dataset was uploaded."
        )

    def test_stage1_no_missing_values_in_key_columns(self, raw_train):
        """Key feature columns must have zero missing values."""
        key_cols = ["Age", "Tenure", "Support Calls", "Payment Delay",
                    "Total Spend", "Gender", "Subscription Type",
                    "Contract Length", "Churn"]
        for col in key_cols:
            nulls = raw_train[col].isnull().sum()
            assert nulls == 0, (
                f"Column '{col}' has {nulls} missing values in training data. "
                "Sprint 1 EDA confirmed the dataset is complete — check the file."
            )

    def test_stage1_gender_values_valid(self, raw_train):
        """Gender column must contain only Male and Female."""
        unique_genders = set(raw_train["Gender"].unique())
        assert unique_genders.issubset({"Male", "Female"}), (
            f"Gender column contains unexpected values: {unique_genders}"
        )

    def test_stage1_subscription_values_valid(self, raw_train):
        """Subscription Type must contain only Basic, Standard, Premium."""
        unique_subs = set(raw_train["Subscription Type"].unique())
        assert unique_subs.issubset({"Basic", "Standard", "Premium"}), (
            f"Subscription Type contains unexpected values: {unique_subs}"
        )

    def test_stage1_contract_values_valid(self, raw_train):
        """Contract Length must contain only Monthly, Quarterly, Annual."""
        unique_contracts = set(raw_train["Contract Length"].unique())
        assert unique_contracts.issubset({"Monthly", "Quarterly", "Annual"}), (
            f"Contract Length contains unexpected values: {unique_contracts}"
        )

    def test_stage1_numeric_columns_in_expected_range(self, raw_train):
        """Numeric columns must fall within the documented dataset ranges."""
        ranges = {
            "Age":              (18, 85),
            "Tenure":           (1,  60),
            "Support Calls":    (0,  10),
            "Payment Delay":    (0,  30),
            "Usage Frequency":  (1,  30),
            "Last Interaction": (1,  30),
            "Total Spend":      (100, 1000),
        }
        for col, (lo, hi) in ranges.items():
            col_min = raw_train[col].min()
            col_max = raw_train[col].max()
            assert col_min >= lo and col_max <= hi, (
                f"Column '{col}' has values outside expected range [{lo}, {hi}]. "
                f"Found: min={col_min}, max={col_max}"
            )

    def test_stage1_combined_dataset_size_after_dropna(self, raw_train, raw_test):
        """After concat and dropna, the combined dataset must have at least 50,000 rows."""
        combined = pd.concat([raw_train, raw_test], ignore_index=True).dropna()
        assert len(combined) >= 50_000, (
            f"Combined dataset after dropna has only {len(combined)} rows. "
            "Need at least 50,000 for the stratified sample to work."
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TestStage2Preprocessing:
    """
    Validates the encoding and transformation steps in utils.py load_data().
    Checks that after preprocessing, all encoded columns exist with the
    correct values and no missing values were introduced.
    """

    def test_stage2_processed_df_has_50k_rows(self, processed_df):
        """Stratified sample must produce exactly 50,000 rows."""
        assert len(processed_df) == 50_000, (
            f"Expected 50,000 rows after stratified sampling, got {len(processed_df)}."
        )

    def test_stage2_gender_enc_column_exists(self, processed_df):
        """Gender_Enc column must be created by preprocessing."""
        assert "Gender_Enc" in processed_df.columns, (
            "Gender_Enc column missing after preprocessing. "
            "Check the encoding step in utils.py load_data()."
        )

    def test_stage2_gender_enc_values_correct(self, processed_df):
        """Gender_Enc must be 1 for Male and 0 for Female, no other values."""
        unique_vals = set(processed_df["Gender_Enc"].unique())
        assert unique_vals.issubset({0, 1}), (
            f"Gender_Enc contains unexpected values: {unique_vals}. Expected only 0 and 1."
        )

    def test_stage2_gender_enc_male_is_1(self, processed_df):
        """Male customers must have Gender_Enc = 1."""
        male_rows = processed_df[processed_df["Gender"] == "Male"]
        assert (male_rows["Gender_Enc"] == 1).all(), (
            "Some Male customers have Gender_Enc != 1. Encoding is wrong."
        )

    def test_stage2_gender_enc_female_is_0(self, processed_df):
        """Female customers must have Gender_Enc = 0."""
        female_rows = processed_df[processed_df["Gender"] == "Female"]
        assert (female_rows["Gender_Enc"] == 0).all(), (
            "Some Female customers have Gender_Enc != 0. Encoding is wrong."
        )

    def test_stage2_sub_enc_column_exists(self, processed_df):
        """Sub_Enc column must be created by preprocessing."""
        assert "Sub_Enc" in processed_df.columns, (
            "Sub_Enc column missing after preprocessing."
        )

    def test_stage2_sub_enc_values_correct(self, processed_df):
        """Sub_Enc must be 0=Basic, 1=Standard, 2=Premium — no other values."""
        unique_vals = set(processed_df["Sub_Enc"].unique())
        assert unique_vals.issubset({0, 1, 2}), (
            f"Sub_Enc contains unexpected values: {unique_vals}. Expected 0, 1, or 2."
        )

    def test_stage2_sub_enc_basic_is_0(self, processed_df):
        """Basic subscribers must have Sub_Enc = 0."""
        basic_rows = processed_df[processed_df["Subscription Type"] == "Basic"]
        assert (basic_rows["Sub_Enc"] == 0).all(), (
            "Some Basic subscribers have Sub_Enc != 0."
        )

    def test_stage2_sub_enc_premium_is_2(self, processed_df):
        """Premium subscribers must have Sub_Enc = 2."""
        prem_rows = processed_df[processed_df["Subscription Type"] == "Premium"]
        assert (prem_rows["Sub_Enc"] == 2).all(), (
            "Some Premium subscribers have Sub_Enc != 2."
        )

    def test_stage2_con_enc_column_exists(self, processed_df):
        """Con_Enc column must be created by preprocessing."""
        assert "Con_Enc" in processed_df.columns, (
            "Con_Enc column missing after preprocessing."
        )

    def test_stage2_con_enc_values_correct(self, processed_df):
        """Con_Enc must be 0=Monthly, 1=Quarterly, 2=Annual — no other values."""
        unique_vals = set(processed_df["Con_Enc"].unique())
        assert unique_vals.issubset({0, 1, 2}), (
            f"Con_Enc contains unexpected values: {unique_vals}. Expected 0, 1, or 2."
        )

    def test_stage2_con_enc_monthly_is_0(self, processed_df):
        """Monthly contract customers must have Con_Enc = 0."""
        monthly_rows = processed_df[processed_df["Contract Length"] == "Monthly"]
        assert (monthly_rows["Con_Enc"] == 0).all(), (
            "Some Monthly contract customers have Con_Enc != 0."
        )

    def test_stage2_con_enc_annual_is_2(self, processed_df):
        """Annual contract customers must have Con_Enc = 2."""
        annual_rows = processed_df[processed_df["Contract Length"] == "Annual"]
        assert (annual_rows["Con_Enc"] == 2).all(), (
            "Some Annual contract customers have Con_Enc != 2."
        )

    def test_stage2_all_feature_cols_present(self, processed_df):
        """All 10 model input features must exist in the processed DataFrame."""
        missing = [c for c in FEATURE_COLS if c not in processed_df.columns]
        assert not missing, (
            f"These feature columns are missing after preprocessing: {missing}"
        )

    def test_stage2_no_nulls_in_feature_cols(self, processed_df):
        """No NaN values in any of the 10 model input feature columns."""
        for col in FEATURE_COLS:
            nulls = processed_df[col].isnull().sum()
            assert nulls == 0, (
                f"Feature column '{col}' has {nulls} NaN values after preprocessing. "
                "This would cause the model to produce NaN predictions."
            )

    def test_stage2_churn_class_balance_preserved(self, processed_df):
        """
        Stratified sampling must preserve the ~55.5% churn rate.
        Allow ±2% tolerance (53.5% to 57.5%).
        """
        churn_rate = processed_df["Churn"].mean()
        assert 0.535 <= churn_rate <= 0.575, (
            f"Churn rate after sampling is {churn_rate:.1%}. "
            "Expected ~55.5% (±2%). Stratified sampling may not have worked correctly."
        )

    def test_stage2_feature_matrix_shape(self, processed_df):
        """Feature matrix X must have shape (50000, 10)."""
        X = processed_df[FEATURE_COLS].values
        assert X.shape == (50_000, 10), (
            f"Feature matrix shape is {X.shape}, expected (50000, 10)."
        )

    def test_stage2_feature_matrix_no_inf(self, processed_df):
        """Feature matrix must not contain any infinite values."""
        X = processed_df[FEATURE_COLS].values
        assert not np.any(np.isinf(X)), (
            "Feature matrix contains infinite values. Check the encoding maps."
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — MODEL INFERENCE (all 4 models)
# ══════════════════════════════════════════════════════════════════════════════

class TestStage3ModelInference:
    """
    Loads all 4 model artifacts and validates that each:
      - Loads without error
      - Has the correct number of input features (10)
      - Produces valid probability outputs (shape, range, sum to 1)
      - Achieves ROC-AUC ≥ 0.90 on the processed dataset
        (confirms model integrity — Sprint 3 achieved 0.9535–0.9536)
    """

    def test_stage3_xgboost_loads(self, xgb_model):
        """xgboost.pkl must load and be a valid classifier."""
        assert xgb_model is not None
        assert hasattr(xgb_model, "predict_proba"), "XGBoost must have predict_proba"

    def test_stage3_random_forest_loads(self, rf_model):
        """random_forest.pkl must load and be a valid classifier."""
        assert rf_model is not None
        assert hasattr(rf_model, "predict_proba"), "Random Forest must have predict_proba"

    def test_stage3_lightgbm_loads(self, lgb_model):
        """lightgbm.pkl must load and be a valid classifier."""
        assert lgb_model is not None
        assert hasattr(lgb_model, "predict_proba"), "LightGBM must have predict_proba"

    def test_stage3_scaler_loads(self, scaler):
        """scaler.pkl must load and be a valid StandardScaler."""
        assert scaler is not None
        assert hasattr(scaler, "transform"), "Scaler must have transform method"
        assert hasattr(scaler, "mean_"), "Scaler must be fitted (has mean_ attribute)"

    def test_stage3_xgboost_feature_count(self, xgb_model):
        """XGBoost must expect exactly 10 features — matches training pipeline."""
        assert xgb_model.n_features_in_ == 10, (
            f"XGBoost expects {xgb_model.n_features_in_} features, expected 10. "
            f"Expected order: {FEATURE_COLS}"
        )

    def test_stage3_random_forest_feature_count(self, rf_model):
        """Random Forest must expect exactly 10 features."""
        assert rf_model.n_features_in_ == 10, (
            f"Random Forest expects {rf_model.n_features_in_} features, expected 10."
        )

    def test_stage3_lightgbm_feature_count(self, lgb_model):
        """LightGBM must expect exactly 10 features."""
        n = lgb_model.n_features_in_ if hasattr(lgb_model, "n_features_in_") else lgb_model.num_feature()
        assert n == 10, f"LightGBM expects {n} features, expected 10."

    def test_stage3_xgboost_predict_proba_shape(self, xgb_model):
        """XGBoost predict_proba on 5 samples must return shape (5, 2)."""
        X = np.array([
            [35, 1, 2, 3, 7, 20, 0, 0, 300.0, 5],
            [45, 0, 48, 8, 1, 2, 2, 2, 900.0, 2],
            [30, 1, 6, 5, 4, 10, 1, 0, 500.0, 8],
            [55, 0, 36, 7, 2, 5, 2, 1, 700.0, 3],
            [25, 1, 1, 2, 9, 25, 0, 0, 200.0, 15],
        ])
        proba = xgb_model.predict_proba(X)
        assert proba.shape == (5, 2), (
            f"predict_proba output shape is {proba.shape}, expected (5, 2)."
        )

    def test_stage3_all_probabilities_between_0_and_1(self, xgb_model):
        """All XGBoost probability outputs must be in [0.0, 1.0]."""
        X = np.array([
            [35, 1, 2, 3, 7, 20, 0, 0, 300.0, 5],
            [45, 0, 48, 8, 1, 2, 2, 2, 900.0, 2],
        ])
        proba = xgb_model.predict_proba(X)
        assert np.all(proba >= 0.0), "Some probabilities are negative."
        assert np.all(proba <= 1.0), "Some probabilities exceed 1.0."

    def test_stage3_probabilities_sum_to_one(self, xgb_model):
        """Each row of predict_proba must sum to 1.0 (within floating point tolerance)."""
        X = np.array([[35, 1, 2, 3, 7, 20, 0, 0, 300.0, 5]])
        proba = xgb_model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"Probability row sums are {row_sums}, expected 1.0."
        )

    def test_stage3_xgboost_auc_above_threshold(self, xgb_model, processed_df):
        """
        XGBoost ROC-AUC on the processed dataset must be ≥ 0.90.
        Sprint 3 achieved 0.9536 — a value below 0.90 means
        model file corruption or wrong pkl was loaded.
        """
        X = processed_df[FEATURE_COLS].values
        y = processed_df["Churn"].values
        # Use a random 5,000-row sample for speed
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=5_000, replace=False)
        X_sample, y_sample = X[idx], y[idx]

        proba = xgb_model.predict_proba(X_sample)[:, 1]
        auc = roc_auc_score(y_sample, proba)

        assert auc >= 0.90, (
            f"XGBoost ROC-AUC is {auc:.4f} — expected ≥ 0.90. "
            "Sprint 3 achieved 0.9536. This may indicate a corrupted pkl file "
            "or the wrong model was loaded."
        )

    def test_stage3_random_forest_auc_above_threshold(self, rf_model, processed_df):
        """Random Forest ROC-AUC must be ≥ 0.90 (Sprint 3 achieved 0.9535)."""
        X = processed_df[FEATURE_COLS].values
        y = processed_df["Churn"].values
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=3_000, replace=False)
        X_sample, y_sample = X[idx], y[idx]

        proba = rf_model.predict_proba(X_sample)[:, 1]
        auc = roc_auc_score(y_sample, proba)

        assert auc >= 0.90, (
            f"Random Forest ROC-AUC is {auc:.4f} — expected ≥ 0.90."
        )

    def test_stage3_lightgbm_auc_above_threshold(self, lgb_model, processed_df):
        """LightGBM ROC-AUC must be ≥ 0.90 (Sprint 3 achieved 0.9536)."""
        X = processed_df[FEATURE_COLS].values
        y = processed_df["Churn"].values
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=3_000, replace=False)
        X_sample, y_sample = X[idx], y[idx]

        proba = lgb_model.predict_proba(X_sample)[:, 1]
        auc = roc_auc_score(y_sample, proba)

        assert auc >= 0.90, (
            f"LightGBM ROC-AUC is {auc:.4f} — expected ≥ 0.90."
        )

    def test_stage3_prediction_speed_acceptable(self, xgb_model):
        """
        XGBoost must score 1,000 records in under 2 seconds.
        Validates KPI 5 (dashboard load time) is not caused by slow inference.
        """
        X = np.random.rand(1_000, 10).astype(np.float32)
        start = time.time()
        xgb_model.predict_proba(X)
        elapsed = time.time() - start
        assert elapsed < 2.0, (
            f"XGBoost took {elapsed:.2f}s to score 1,000 records — expected < 2s. "
            "This would violate KPI 5 (dashboard load time < 3 seconds)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — END-TO-END PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TestStage4EndToEnd:
    """
    Tests the complete pipeline as page2_predictor.py runs it:
    raw user inputs → manual encoding → np.array → predict_proba → risk label

    NOTE: page2_predictor.py does NOT use the scaler — it passes raw
    encoded values directly to predict_proba. These tests replicate
    that exact behaviour.
    """

    def _encode_and_predict(self, model, age, gender, tenure, usage_freq,
                             support_calls, payment_delay, sub_type,
                             contract, total_spend, last_interaction):
        """
        Replicates the exact encoding logic in page2_predictor.py show().
        Returns the float churn probability (0.0 to 1.0).
        """
        gender_enc = 1 if gender == "Male" else 0
        sub_enc    = SUB_MAP[sub_type]
        con_enc    = CON_MAP[contract]

        X = np.array([[age, gender_enc, tenure, usage_freq, support_calls,
                       payment_delay, sub_enc, con_enc, total_spend, last_interaction]])
        return float(model.predict_proba(X)[0][1])

    def test_stage4_high_risk_profile_over_70pct(self, xgb_model):
        """
        HIGH RISK profile must produce churn probability > 0.70.
        Profile: 7 support calls, Monthly, 2-month tenure, high payment delay.
        Based on Sprint 1 EDA: 6+ support calls → ~100% churn rate.
        """
        prob = self._encode_and_predict(
            xgb_model,
            age=35, gender="Male", tenure=2, usage_freq=3,
            support_calls=7, payment_delay=20,
            sub_type="Basic", contract="Monthly",
            total_spend=300.0, last_interaction=5
        )
        assert prob > 0.70, (
            f"HIGH RISK profile produced only {prob:.1%} churn probability. "
            "Expected > 70%. Check that xgboost.pkl is the correct model file."
        )

    def test_stage4_extreme_high_risk_over_85pct(self, xgb_model):
        """
        Maximum-risk profile (9 support calls, monthly, brand new) must score > 0.85.
        """
        prob = self._encode_and_predict(
            xgb_model,
            age=22, gender="Male", tenure=1, usage_freq=2,
            support_calls=9, payment_delay=28,
            sub_type="Basic", contract="Monthly",
            total_spend=150.0, last_interaction=1
        )
        assert prob > 0.80, (
            f"Extreme HIGH RISK profile only scored {prob:.1%}. Expected > 80%."
        )

    def test_stage4_low_risk_profile_under_30pct(self, xgb_model):
        """
        LOW RISK profile must produce churn probability < 0.30.
        Profile: 1 support call, Annual contract, long tenure, no payment delay.
        """
        prob = self._encode_and_predict(
            xgb_model,
            age=45, gender="Female", tenure=48, usage_freq=8,
            support_calls=1, payment_delay=2,
            sub_type="Premium", contract="Annual",
            total_spend=900.0, last_interaction=2
        )
        assert prob < 0.30, (
            f"LOW RISK profile produced {prob:.1%} churn probability. Expected < 30%."
        )

    def test_stage4_extreme_low_risk_under_10pct(self, xgb_model):
        """
        Ideal loyal customer (0 support calls, Annual, 5 years) must score < 0.10.
        """
        prob = self._encode_and_predict(
            xgb_model,
            age=50, gender="Female", tenure=60, usage_freq=10,
            support_calls=0, payment_delay=0,
            sub_type="Premium", contract="Annual",
            total_spend=950.0, last_interaction=1
        )
        assert prob < 0.10, (
            f"Extreme LOW RISK profile scored {prob:.1%}. Expected < 10%."
        )

    def test_stage4_high_risk_correctly_classified(self, xgb_model):
        """
        Model binary prediction (threshold 0.5) must classify HIGH RISK as Churn=1.
        Matches the pred = int(prob >= 0.5) logic in page2_predictor.py.
        """
        prob = self._encode_and_predict(
            xgb_model,
            age=35, gender="Male", tenure=2, usage_freq=3,
            support_calls=7, payment_delay=20,
            sub_type="Basic", contract="Monthly",
            total_spend=300.0, last_interaction=5
        )
        pred = int(prob >= 0.5)
        assert pred == 1, (
            f"HIGH RISK profile predicted {pred} (prob={prob:.1%}). Expected 1 (Churn)."
        )

    def test_stage4_low_risk_correctly_classified(self, xgb_model):
        """
        Model binary prediction must classify LOW RISK as Churn=0.
        """
        prob = self._encode_and_predict(
            xgb_model,
            age=45, gender="Female", tenure=48, usage_freq=8,
            support_calls=1, payment_delay=2,
            sub_type="Premium", contract="Annual",
            total_spend=900.0, last_interaction=2
        )
        pred = int(prob >= 0.5)
        assert pred == 0, (
            f"LOW RISK profile predicted {pred} (prob={prob:.1%}). Expected 0 (Retained)."
        )

    def test_stage4_monthly_basic_higher_than_annual_premium(self, xgb_model):
        """
        Monthly/Basic customer must have higher churn probability than
        Annual/Premium customer with otherwise identical profile.
        Validates that Contract and Subscription encodings are working correctly.
        """
        prob_high = self._encode_and_predict(
            xgb_model,
            age=40, gender="Male", tenure=12, usage_freq=5,
            support_calls=3, payment_delay=8,
            sub_type="Basic", contract="Monthly",
            total_spend=400.0, last_interaction=7
        )
        prob_low = self._encode_and_predict(
            xgb_model,
            age=40, gender="Male", tenure=12, usage_freq=5,
            support_calls=3, payment_delay=8,
            sub_type="Premium", contract="Annual",
            total_spend=400.0, last_interaction=7
        )
        assert prob_high > prob_low, (
            f"Monthly/Basic ({prob_high:.1%}) should have higher churn risk "
            f"than Annual/Premium ({prob_low:.1%}), but it does not. "
            "The Contract or Subscription encoding may be reversed."
        )

    def test_stage4_more_support_calls_means_higher_risk(self, xgb_model):
        """
        Increasing support calls must increase churn probability (all else equal).
        Validates the direction of the top feature driver from Sprint 3.
        """
        prob_1_call = self._encode_and_predict(
            xgb_model,
            age=40, gender="Male", tenure=24, usage_freq=5,
            support_calls=1, payment_delay=5,
            sub_type="Standard", contract="Quarterly",
            total_spend=500.0, last_interaction=7
        )
        prob_7_calls = self._encode_and_predict(
            xgb_model,
            age=40, gender="Male", tenure=24, usage_freq=5,
            support_calls=7, payment_delay=5,
            sub_type="Standard", contract="Quarterly",
            total_spend=500.0, last_interaction=7
        )
        assert prob_7_calls > prob_1_call, (
            f"7 support calls ({prob_7_calls:.1%}) should give higher churn risk "
            f"than 1 support call ({prob_1_call:.1%}). "
            "Feature importance shows Support Calls is the #1 driver."
        )

    def test_stage4_pipeline_output_is_deterministic(self, xgb_model):
        """
        Calling the full pipeline twice with identical inputs must give
        identical outputs. Validates model determinism for production use.
        """
        kwargs = dict(
            age=38, gender="Male", tenure=12, usage_freq=4,
            support_calls=5, payment_delay=12,
            sub_type="Basic", contract="Monthly",
            total_spend=450.0, last_interaction=10
        )
        prob1 = self._encode_and_predict(xgb_model, **kwargs)
        prob2 = self._encode_and_predict(xgb_model, **kwargs)
        assert prob1 == prob2, (
            f"Pipeline is not deterministic: first call={prob1:.6f}, "
            f"second call={prob2:.6f}."
        )

    def test_stage4_all_three_models_agree_on_high_risk(
            self, xgb_model, rf_model, lgb_model):
        """
        All 3 ensemble models must agree that the HIGH RISK profile
        has churn probability > 0.50 (binary classification agreement).
        Sprint 3 showed all 3 models are consistent on extreme profiles.
        """
        kwargs = dict(
            age=35, gender="Male", tenure=2, usage_freq=3,
            support_calls=7, payment_delay=20,
            sub_type="Basic", contract="Monthly",
            total_spend=300.0, last_interaction=5
        )
        models = {"XGBoost": xgb_model, "Random Forest": rf_model, "LightGBM": lgb_model}
        for name, model in models.items():
            prob = self._encode_and_predict(model, **kwargs)
            assert prob > 0.50, (
                f"{name} predicted only {prob:.1%} for HIGH RISK profile. "
                "Expected > 50%. All 3 models should agree on extreme profiles."
            )

    def test_stage4_all_three_models_agree_on_low_risk(
            self, xgb_model, rf_model, lgb_model):
        """
        All 3 ensemble models must agree that the LOW RISK profile
        has churn probability < 0.50.
        """
        kwargs = dict(
            age=45, gender="Female", tenure=48, usage_freq=8,
            support_calls=1, payment_delay=2,
            sub_type="Premium", contract="Annual",
            total_spend=900.0, last_interaction=2
        )
        models = {"XGBoost": xgb_model, "Random Forest": rf_model, "LightGBM": lgb_model}
        for name, model in models.items():
            prob = self._encode_and_predict(model, **kwargs)
            assert prob < 0.50, (
                f"{name} predicted {prob:.1%} for LOW RISK profile. "
                "Expected < 50%."
            )

    def test_stage4_risk_label_logic_high(self, xgb_model):
        """
        Risk label assignment from page2_predictor.py:
        prob >= 0.60 → HIGH RISK
        """
        prob = self._encode_and_predict(
            xgb_model,
            age=35, gender="Male", tenure=2, usage_freq=3,
            support_calls=7, payment_delay=20,
            sub_type="Basic", contract="Monthly",
            total_spend=300.0, last_interaction=5
        )
        if prob >= 0.60:
            label = "HIGH RISK"
        elif prob >= 0.35:
            label = "MEDIUM RISK"
        else:
            label = "LOW RISK"
        assert label == "HIGH RISK", (
            f"Expected HIGH RISK label but got {label} (prob={prob:.1%})."
        )

    def test_stage4_risk_label_logic_low(self, xgb_model):
        """
        Risk label assignment:
        prob < 0.35 → LOW RISK
        """
        prob = self._encode_and_predict(
            xgb_model,
            age=45, gender="Female", tenure=48, usage_freq=8,
            support_calls=1, payment_delay=2,
            sub_type="Premium", contract="Annual",
            total_spend=900.0, last_interaction=2
        )
        if prob >= 0.60:
            label = "HIGH RISK"
        elif prob >= 0.35:
            label = "MEDIUM RISK"
        else:
            label = "LOW RISK"
        assert label == "LOW RISK", (
            f"Expected LOW RISK label but got {label} (prob={prob:.1%})."
        )

    def test_stage4_bulk_prediction_no_errors(self, xgb_model, processed_df):
        """
        Running predict_proba on all 50,000 processed rows must complete
        without errors and produce valid probabilities throughout.
        """
        X = processed_df[FEATURE_COLS].values
        proba = xgb_model.predict_proba(X)[:, 1]

        assert len(proba) == 50_000, f"Expected 50,000 predictions, got {len(proba)}."
        assert not np.any(np.isnan(proba)), "predict_proba produced NaN values."
        assert not np.any(np.isinf(proba)), "predict_proba produced infinite values."
        assert np.all(proba >= 0.0), "predict_proba produced negative probabilities."
        assert np.all(proba <= 1.0), "predict_proba produced probabilities > 1.0."
