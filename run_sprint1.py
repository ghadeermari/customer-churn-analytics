"""
Sprint 1 Main Runner
Customer Churn Analytics Project - BIS405 Graduation Project

Executes the full Sprint 1 pipeline:
  1. Data loading and validation   (Student 1)
  2. EDA visualizations            (Student 1)
  3. Business insights report      (Student 2)
  4. Preprocessing pipeline test   (Student 3)
"""
import os
os.makedirs("outputs/eda_plots", exist_ok=True)

from src.data_loader import DataLoader
from src.eda_analysis import EDAAnalyzer
from src.business_insights import BusinessInsights
from src.data_preprocessing import DataPreprocessor


def main():
    print("=" * 60)
    print("SPRINT 1: DATA FOUNDATION & EDA")
    print("Customer Churn Analytics Project")
    print("=" * 60)

    # Step 1: Load and validate data (Student 1)
    print("\n📂 STEP 1: Loading & Validating Data...")
    loader = DataLoader("data/customer_churn_dataset-training-master.csv")
    df = loader.load()
    df = loader.clean()
    loader.print_report()

    # Step 2: Run EDA visualizations (Student 1)
    print("\n📊 STEP 2: Generating EDA Visualizations...")
    eda = EDAAnalyzer(df, output_dir="outputs/eda_plots")
    eda.run_all()

    # Step 3: Generate business insights (Student 2)
    print("\n📋 STEP 3: Generating Business Insights...")
    bi = BusinessInsights(df)
    bi.generate_report("outputs/business_insights_report.txt")
    bi.export_json("outputs/eda_summary.json")

    # Step 4: Test preprocessing pipeline (Student 3)
    print("\n⚙️  STEP 4: Testing Preprocessing Pipeline...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    preprocessor.save_artifacts()

    # Summary
    print("\n" + "=" * 60)
    print("✅ SPRINT 1 COMPLETE")
    print("=" * 60)
    print(f"  📊 11 EDA plots       → outputs/eda_plots/")
    print(f"  📄 Business report    → outputs/business_insights_report.txt")
    print(f"  📄 JSON summary       → outputs/eda_summary.json")
    print(f"  ⚙️  Preprocessing      → models/scaler.pkl, label_encoders.pkl")
    print(f"  📦 Dataset validated  → {len(df):,} records")
    print("=" * 60)


if __name__ == "__main__":
    main()
