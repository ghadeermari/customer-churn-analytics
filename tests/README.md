# Sprint 5 — Test Suite
## BIS405 Group 3 | Customer Churn Prediction System

---

## What is included

| File | Type | Tests | What it validates |
|------|------|-------|-------------------|
| `test_prediction.py` | Smoke test | 7 | Model and scaler load, HIGH/LOW RISK predictions, feature order, output shape, determinism |
| `test_integration.py` | Integration test | 60 | Full 4-stage pipeline: data loading → preprocessing → model inference → end-to-end prediction |

---

## How to run

### Install test dependencies first
```
pip install pytest scikit-learn xgboost lightgbm joblib numpy pandas
```

### Run the smoke tests only (fast — no CSV data needed)
```
python -m pytest tests/test_prediction.py -v
```

### Run the full integration test suite (requires CSV files in data/)
```
python -m pytest tests/test_integration.py -v
```

### Run everything
```
python -m pytest tests/ -v
```

### Run only one stage
```
python -m pytest tests/test_integration.py -v -k "Stage1"
python -m pytest tests/test_integration.py -v -k "Stage2"
python -m pytest tests/test_integration.py -v -k "Stage3"
python -m pytest tests/test_integration.py -v -k "Stage4"
```

### Run on Google Colab (after Cell 3 — path fix)
```python
!pip install pytest -q
!python -m pytest tests/test_prediction.py -v
!python -m pytest tests/test_integration.py -v
```

---

## Required folder structure

```
sprint4-deliverables/
├── data/
│   ├── data_train.csv   ← needed for test_integration.py
│   └── data_test.csv    ← needed for test_integration.py
├── models/
│   ├── xgboost.pkl      ← needed for both test files
│   ├── random_forest.pkl
│   ├── lightgbm.pkl
│   └── scaler.pkl
├── src/
│   └── (dashboard files)
└── tests/
    ├── conftest.py
    ├── test_prediction.py
    └── test_integration.py
```

---

## What the 4 integration test stages cover

### Stage 1 — Raw Data Loading (14 tests)
- Both CSV files exist in the data/ folder
- All 12 expected columns are present
- Minimum row counts (train ≥ 50,000, test ≥ 1,000)
- Churn column is binary (0/1 only)
- Churn rate is between 50%–60% (confirms correct dataset)
- No missing values in key columns
- Valid values in Gender, Subscription Type, Contract Length
- All numeric columns are within documented dataset ranges
- Combined dataset ≥ 50,000 rows after dropna

### Stage 2 — Preprocessing Pipeline (18 tests)
- Stratified sample produces exactly 50,000 rows
- Gender_Enc column created (Male=1, Female=0)
- Sub_Enc column created (Basic=0, Standard=1, Premium=2)
- Con_Enc column created (Monthly=0, Quarterly=1, Annual=2)
- All 10 model feature columns exist after encoding
- No NaN values introduced by preprocessing
- Churn class balance preserved (±2% tolerance)
- Feature matrix is shape (50000, 10)
- No infinite values in feature matrix

### Stage 3 — Model Inference (14 tests)
- All 4 pkl files load without errors
- XGBoost, Random Forest, LightGBM each expect exactly 10 features
- Scaler is fitted (has mean_ attribute)
- predict_proba returns shape (n, 2)
- All probabilities are between 0.0 and 1.0
- Probabilities sum to 1.0 per row
- XGBoost ROC-AUC ≥ 0.90 on 5,000-row sample
- Random Forest ROC-AUC ≥ 0.90 on 3,000-row sample
- LightGBM ROC-AUC ≥ 0.90 on 3,000-row sample
- XGBoost scores 1,000 records in < 2 seconds (KPI 5 validation)

### Stage 4 — End-to-End Prediction (14 tests)
- HIGH RISK profile (7 calls, Monthly) → probability > 70%
- Extreme HIGH RISK (9 calls, Monthly, new) → probability > 85%
- LOW RISK profile (1 call, Annual) → probability < 30%
- Extreme LOW RISK (0 calls, Annual, loyal) → probability < 10%
- Binary classification: HIGH RISK predicts 1, LOW RISK predicts 0
- Monthly/Basic always scores higher than Annual/Premium (same profile)
- More support calls → higher churn risk (feature direction validation)
- Pipeline is deterministic (same input = same output every time)
- All 3 models agree on HIGH RISK classification
- All 3 models agree on LOW RISK classification
- Risk label logic: HIGH (≥60%), MEDIUM (35–60%), LOW (<35%)
- Bulk prediction on 50,000 rows produces no NaN/Inf values
