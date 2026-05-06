"""
Data Loader Module
Author: Student 1 (Data Scientist)
Sprint 1: Data Foundation & EDA

Loads, validates, and provides basic statistics for the Kaggle customer churn dataset.
Source: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
"""
import pandas as pd
import numpy as np


class DataLoader:
    """Handles loading and validation of customer churn data."""

    EXPECTED_COLUMNS = [
        "CustomerID", "Age", "Gender", "Tenure", "Usage Frequency",
        "Support Calls", "Payment Delay", "Subscription Type",
        "Contract Length", "Total Spend", "Last Interaction", "Churn"
    ]

    def __init__(self, filepath="data/customer_churn_dataset-training-master.csv"):
        self.filepath = filepath
        self.df = None
        self.validation_results = {}

    def load(self):
        """Load CSV dataset into DataFrame."""
        self.df = pd.read_csv(self.filepath)
        print(f"[DataLoader] Loaded {len(self.df):,} records, {self.df.shape[1]} columns")
        return self.df

    def clean(self):
        """Drop rows with missing values (dataset has ~1 null row)."""
        if self.df is None:
            self.load()
        before = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        dropped = before - len(self.df)
        if dropped > 0:
            print(f"[DataLoader] Dropped {dropped} rows with missing values")
        # Ensure integer types for numeric columns
        int_cols = ["CustomerID", "Age", "Tenure", "Usage Frequency",
                    "Support Calls", "Payment Delay", "Last Interaction", "Churn"]
        for col in int_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)
        return self.df

    def validate(self):
        """Run data quality checks and return validation report."""
        if self.df is None:
            self.load()

        results = {
            "total_records": len(self.df),
            "total_features": self.df.shape[1],
            "missing_values": int(self.df.isnull().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "columns": list(self.df.columns),
            "churn_counts": self.df["Churn"].value_counts().to_dict(),
            "churn_rate": round(self.df["Churn"].mean() * 100, 2),
        }
        self.validation_results = results
        return results

    def get_summary_stats(self):
        """Return descriptive statistics for numeric columns."""
        if self.df is None:
            self.load()
        return self.df.describe()

    def get_categorical_summary(self):
        """Return unique value counts for categorical columns."""
        if self.df is None:
            self.load()
        summary = {}
        for col in ["Gender", "Subscription Type", "Contract Length"]:
            if col in self.df.columns:
                summary[col] = self.df[col].value_counts().to_dict()
        return summary

    def print_report(self):
        """Print a formatted validation report."""
        v = self.validate()
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"Total Records:   {v['total_records']:,}")
        print(f"Total Features:  {v['total_features']}")
        print(f"Missing Values:  {v['missing_values']}")
        print(f"Duplicate Rows:  {v['duplicate_rows']}")
        print(f"Churn Rate:      {v['churn_rate']}%")
        print(f"\nChurn Distribution:")
        for k, val in v["churn_counts"].items():
            label = "Churned" if k == 1 else "Retained"
            print(f"  {label} ({k}): {val:,}")
        print(f"\nColumns: {v['columns']}")
        print("=" * 60)


if __name__ == "__main__":
    loader = DataLoader("data/customer_churn_dataset-training-master.csv")
    loader.load()
    loader.clean()
    loader.print_report()

    print("\nNumeric Summary:")
    print(loader.get_summary_stats())

    print("\nCategorical Summary:")
    for col, vals in loader.get_categorical_summary().items():
        print(f"\n{col}:")
        for k, v in vals.items():
            print(f"  {k}: {v:,}")
