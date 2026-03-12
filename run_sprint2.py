"""
Sprint 2 Main Runner - BIS405 Graduation Project
Baseline Modeling: Preprocess → Train LR → Cross-Validate → Plots → Document → Compare
"""
import os
os.makedirs("outputs/sprint2", exist_ok=True)

from src.data_loader import DataLoader
from src.data_preprocessing import DataPreprocessor
from src.baseline_model import BaselineModelTrainer
from src.model_documentation import ModelDocumentor
from src.model_comparison import ModelComparisonFramework


def main():
    print("=" * 60)
    print("SPRINT 2: BASELINE MODELING")
    print("=" * 60)

    # Step 1: Load & preprocess (Student 3)
    print("\n⚙️  STEP 1: Loading & Preprocessing...")
    loader = DataLoader("data/customer_churn_dataset-training-master.csv")
    df = loader.load()
    df = loader.clean()
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    preprocessor.save_artifacts()

    # Step 2: Train baseline (Student 1)
    print("\n🤖 STEP 2: Training Logistic Regression...")
    trainer = BaselineModelTrainer(output_dir="outputs/sprint2")
    trainer.train(X_train, y_train)
    results = trainer.evaluate(X_test, y_test, feature_names=preprocessor.feature_names)

    # Step 3: Cross-validation (Student 1)
    print("\n🔄 STEP 3: Cross-Validation...")
    trainer.cross_validate(X_train, y_train, cv=5)

    # Step 4: Evaluation plots (Student 1)
    print("\n📊 STEP 4: Generating Evaluation Plots...")
    trainer.plot_all()

    # Step 5: Save model + results (Student 1)
    print("\n💾 STEP 5: Saving Artifacts...")
    trainer.save_model()
    trainer.save_results()

    # Step 6: Document (Student 2)
    print("\n📋 STEP 6: Documenting Performance...")
    info = {"train_size": X_train.shape[0], "test_size": X_test.shape[0]}
    ModelDocumentor(results, info).generate_report("outputs/sprint2/sprint2_model_report.txt")

    # Step 7: Comparison framework (Student 3)
    print("\n📊 STEP 7: Model Comparison Framework...")
    fw = ModelComparisonFramework(output_path="outputs/sprint2/model_comparison.json")
    fw.add_model("Logistic Regression (Baseline)", results)
    fw.print_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("✅ SPRINT 2 COMPLETE")
    print("=" * 60)
    print(f"  🤖 Model      → models/logistic_regression_baseline.pkl")
    print(f"  📊 5 plots    → outputs/sprint2/")
    print(f"  📄 Report     → outputs/sprint2/sprint2_model_report.txt")
    print(f"  📄 Results    → outputs/sprint2/baseline_results.json")
    cv_mean = results.get("cv_mean", "N/A")
    print(f"  KPI: ROC-AUC = {results['roc_auc']:.4f} | CV Mean = {cv_mean}")
    print("=" * 60)


if __name__ == "__main__":
    main()
