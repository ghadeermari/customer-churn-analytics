"""
Shared utilities — loaded by all dashboard pages
Handles data loading, preprocessing, model loading
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import streamlit as st
from sklearn.preprocessing import LabelEncoder

BASE = "/content/drive/MyDrive/sprint4-deliverables"

@st.cache_data
def load_data():
    train = pd.read_csv(os.path.join(BASE, "data", "data_train.csv"))
    test  = pd.read_csv(os.path.join(BASE, "data", "data_test.csv"))
    df = pd.concat([train, test], ignore_index=True).dropna()
    # Sample 50k for performance
    from sklearn.model_selection import train_test_split
    df, _ = train_test_split(df, train_size=50000, stratify=df["Churn"], random_state=42)
    # Encode
    df = df.copy()
    df["Gender_Enc"] = (df["Gender"] == "Male").astype(int)
    df["Sub_Enc"]    = df["Subscription Type"].map({"Basic":0,"Standard":1,"Premium":2})
    df["Con_Enc"]    = df["Contract Length"].map({"Monthly":0,"Quarterly":1,"Annual":2})
    return df

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "models", "xgboost.pkl"))

@st.cache_resource
def load_all_models():
    return {
        "XGBoost":    joblib.load(os.path.join(BASE, "models", "xgboost.pkl")),
        "Random Forest": joblib.load(os.path.join(BASE, "models", "random_forest.pkl")),
        "LightGBM":   joblib.load(os.path.join(BASE, "models", "lightgbm.pkl")),
    }

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(BASE, "models", "scaler.pkl"))

@st.cache_data
def load_sprint3_results():
    with open(os.path.join(BASE, "outputs", "sprint4", "sprint3_results.json")) as f:
        return json.load(f)

FEATURES = ["Age","Gender_Enc","Tenure","Usage Frequency","Support Calls",
            "Payment Delay","Sub_Enc","Con_Enc","Total Spend","Last Interaction"]

FEATURE_LABELS = {
    "Age": "Age",
    "Gender_Enc": "Gender",
    "Tenure": "Tenure (months)",
    "Usage Frequency": "Usage Frequency",
    "Support Calls": "Support Calls",
    "Payment Delay": "Payment Delay (days)",
    "Sub_Enc": "Subscription Type",
    "Con_Enc": "Contract Length",
    "Total Spend": "Total Spend ($)",
    "Last Interaction": "Last Interaction (days ago)"
}

# Brand colors matching Sprint 2/3 navy theme
NAVY   = "#1F3864"
LBLUE  = "#2E75B6"
GREEN  = "#375623"
RED    = "#C00000"
AMBER  = "#ED7D31"
LGRAY  = "#F2F2F2"
WHITE  = "#FFFFFF"
