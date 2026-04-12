"""
Data Preprocessing Module
Author: Student 3 (Data Engineer / Scrum Master)
Sprint 1: Initial setup (full pipeline used in Sprint 2)

Handles encoding, scaling, and train/test splitting for the Kaggle churn dataset.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    """Prepares customer churn data for machine learning models."""

    CATEGORICAL_COLS = ["Gender", "Subscription Type", "Contract Length"]
    DROP_COLS = ["CustomerID"]

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def preprocess(self, df, fit=True):
        """
        Encode categorical features, scale numeric features.

        Args:
            df: Raw DataFrame with all columns.
            fit: If True, fit encoders/scaler. If False, use previously fitted.

        Returns:
            X (np.array): Preprocessed feature matrix.
            y (pd.Series or None): Binary target (1=Churn, 0=Retained).
        """
        df = df.copy()

        # Extract target
        y = None
        if "Churn" in df.columns:
            y = df["Churn"].astype(int)
            df = df.drop("Churn", axis=1)

        # Drop ID column
        for col in self.DROP_COLS:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Encode categorical variables
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))

        # Scale features
        self.feature_names = df.columns.tolist()
        if fit:
            X = self.scaler.fit_transform(df)
            self.is_fitted = True
        else:
            X = self.scaler.transform(df)

        return X, y

    def split_data(self, X, y, test_size=0.3, random_state=42):
        """
        Split data into training and test sets (stratified).

        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Fraction for test set (default 0.3 = 70/30 split).
            random_state: Seed for reproducibility.

        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"[Preprocessor] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
        return X_train, X_test, y_train, y_test

    def save_artifacts(self, output_dir="models"):
        """Save scaler and encoders for later use."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
        print(f"[Preprocessor] Artifacts saved to {output_dir}/")

    def load_artifacts(self, output_dir="models"):
        """Load previously saved scaler and encoders."""
        self.scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))
        self.is_fitted = True
        print(f"[Preprocessor] Artifacts loaded from {output_dir}/")


if __name__ == "__main__":
    df = pd.read_csv("data/customer_churn_dataset-training-master.csv").dropna()
    print(f"Loaded {len(df):,} records")

    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)
    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]:,}")
    print(f"Feature names: {preprocessor.feature_names}")

    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    print(f"Train churn rate: {y_train.mean()*100:.1f}%")
    print(f"Test churn rate:  {y_test.mean()*100:.1f}%")

    preprocessor.save_artifacts()
    print("✅ Preprocessing pipeline ready for Sprint 2")
