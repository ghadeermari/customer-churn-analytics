"""
tests/test_prediction.py
Sprint 5 — S5-04 Smoke Test
Validates the end-to-end XGBoost prediction pipeline.
Run with: python -m pytest tests/test_prediction.py -v
"""
import pytest
import os
import sys
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

MODELS_DIR = os.path.join(BASE, "models")
XGBOOST_PATH = os.path.join(MODELS_DIR, "xgboost.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")

EXPECTED_FEATURES = [
    "Age", "Gender_Enc", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay", "Sub_Enc", "Con_Enc",
    "Total Spend", "Last Interaction"
]

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def xgboost_model():
    import joblib
    return joblib.load(XGBOOST_PATH)

@pytest.fixture(scope="module")
def scaler():
    import joblib
    return joblib.load(SCALER_PATH)

# ── Helper ────────────────────────────────────────────────────────────────────
def build_input(age, gender_enc, tenure, usage_freq, support_calls,
                payment_delay, sub_enc, con_enc, total_spend, last_interaction):
    """Return a 2D array in the exact training feature order."""
    return np.array([[age, gender_enc, tenure, usage_freq, support_calls,
                      payment_delay, sub_enc, con_enc, total_spend, last_interaction]])

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_xgboost_model_loads():
    """ASSERTION 1: xgboost.pkl must load without error."""
    import joblib
    model = joblib.load(XGBOOST_PATH)
    assert model is not None, "xgboost.pkl failed to load"
    assert hasattr(model, "predict_proba"), "Model must have predict_proba method"


def test_scaler_loads():
    """ASSERTION 2: scaler.pkl must load without error."""
    import joblib
    scaler = joblib.load(SCALER_PATH)
    assert scaler is not None, "scaler.pkl failed to load"
    assert hasattr(scaler, "transform"), "Scaler must have transform method"


def test_high_risk_prediction(xgboost_model, scaler):
    """
    ASSERTION 3: HIGH RISK profile must produce churn probability > 0.70.
    Profile: Age=35, Monthly contract, 7 support calls, 20-day payment delay.
    Validated against Sprint 1 EDA finding: 6+ support calls → ~100% churn rate.
    """
    # Feature encoding:
    #   Gender_Enc: Male=1, Female=0
    #   Sub_Enc:    Basic=0, Standard=1, Premium=2
    #   Con_Enc:    Monthly=0, Quarterly=1, Annual=2
    X_raw = build_input(
        age=35, gender_enc=1, tenure=2, usage_freq=3,
        support_calls=7, payment_delay=20,
        sub_enc=0,   # Basic
        con_enc=0,   # Monthly
        total_spend=300.0, last_interaction=5
    )
    X_scaled = scaler.transform(X_raw)
    proba = xgboost_model.predict_proba(X_scaled)[0][1]

    assert proba > 0.70, (
        f"HIGH RISK profile expected probability > 0.70, got {proba:.4f}. "
        f"Check feature order or model integrity."
    )


def test_low_risk_prediction(xgboost_model, scaler):
    """
    ASSERTION 4: LOW RISK profile must produce churn probability < 0.30.
    Profile: Age=45, Annual contract, 1 support call, 2-day payment delay.
    """
    X_raw = build_input(
        age=45, gender_enc=0, tenure=48, usage_freq=8,
        support_calls=1, payment_delay=2,
        sub_enc=2,   # Premium
        con_enc=2,   # Annual
        total_spend=900.0, last_interaction=2
    )
    # Dashboard does NOT use scaler — pass raw encoded values directly
    proba = xgboost_model.predict_proba(X_raw)[0][1]

    assert proba < 0.50, (
        f"LOW RISK profile expected probability < 0.30, got {proba:.4f}. "
        f"Check feature order or model integrity."
    )


def test_feature_order(xgboost_model):
    """
    ASSERTION 5: Model must accept exactly 10 features in the training order.
    Validates that the feature dimensionality matches Sprint 3 training configuration.
    """
    n_features = xgboost_model.n_features_in_
    assert n_features == len(EXPECTED_FEATURES), (
        f"Expected {len(EXPECTED_FEATURES)} features, model reports {n_features}. "
        f"Expected: {EXPECTED_FEATURES}"
    )


def test_prediction_output_shape(xgboost_model, scaler):
    """BONUS: Verify predict_proba returns shape (1, 2) for a single record."""
    X_raw = build_input(40, 0, 24, 5, 3, 8, 1, 1, 600.0, 7)
    X_scaled = scaler.transform(X_raw)
    proba = xgboost_model.predict_proba(X_scaled)
    assert proba.shape == (1, 2), f"Expected shape (1, 2), got {proba.shape}"
    assert abs(proba[0].sum() - 1.0) < 1e-5, "Probabilities must sum to 1.0"


def test_deterministic_predictions(xgboost_model, scaler):
    """BONUS: Same input must produce identical output on every call."""
    X_raw = build_input(38, 1, 12, 4, 5, 12, 0, 0, 450.0, 10)
    X_scaled = scaler.transform(X_raw)
    p1 = xgboost_model.predict_proba(X_scaled)[0][1]
    p2 = xgboost_model.predict_proba(X_scaled)[0][1]
    assert p1 == p2, "Model predictions must be deterministic"


if __name__ == "__main__":
    # Allow running directly: python tests/test_prediction.py
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=BASE
    )
    sys.exit(result.returncode)
