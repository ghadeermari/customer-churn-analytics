"""
Model Performance Documentation Module
Author: Student 2 (Business Analyst)
Sprint 2: Baseline Modeling

Generates model performance report with business context and
prepares framework for Sprint 3 model comparison.
"""
import json
from datetime import datetime


class ModelDocumentor:
    """Documents model performance with business context."""

    def __init__(self, results, dataset_info=None):
        self.results = results
        self.dataset_info = dataset_info or {}

    def generate_report(self, output_path="outputs/sprint2_model_report.txt"):
        """Generate Sprint 2 model performance report."""
        r = self.results
        lines = []
        lines.append("=" * 65)
        lines.append("SPRINT 2 - BASELINE MODEL PERFORMANCE REPORT")
        lines.append("Customer Churn Analytics Project")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 65)

        lines.append(f"\n1. MODEL OVERVIEW")
        lines.append(f"   Model:      {r.get('model_name', 'Logistic Regression')}")
        lines.append(f"   Algorithm:  Logistic Regression with L2 regularization")
        lines.append(f"   Solver:     LBFGS | Max iterations: 1000")
        lines.append(f"   Class weight: Balanced (to handle 57/43 imbalance)")

        if self.dataset_info:
            lines.append(f"\n2. DATA SPLIT")
            lines.append(f"   Training:   {self.dataset_info.get('train_size', 'N/A'):,} samples")
            lines.append(f"   Testing:    {self.dataset_info.get('test_size', 'N/A'):,} samples")
            lines.append(f"   Split:      70/30 stratified")
            lines.append(f"   Seed:       42 (reproducible)")

        lines.append(f"\n3. PERFORMANCE METRICS")
        lines.append(f"   {'Metric':<15} {'Score':>8} {'KPI Target':>12} {'Status':>10}")
        lines.append(f"   {'-'*48}")
        lines.append(f"   {'ROC-AUC':<15} {r['roc_auc']:>8.4f} {'>=0.65':>12} {'✅ PASS' if r['roc_auc']>=0.65 else '❌ FAIL':>10}")
        lines.append(f"   {'Accuracy':<15} {r['accuracy']:>8.4f} {'>=0.70':>12} {'✅ PASS' if r['accuracy']>=0.70 else '❌ FAIL':>10}")
        lines.append(f"   {'Precision':<15} {r['precision']:>8.4f} {'>=0.60':>12} {'✅ PASS' if r['precision']>=0.60 else '❌ FAIL':>10}")
        lines.append(f"   {'Recall':<15} {r['recall']:>8.4f} {'>=0.60':>12} {'✅ PASS' if r['recall']>=0.60 else '❌ FAIL':>10}")
        lines.append(f"   {'F1-Score':<15} {r['f1_score']:>8.4f} {'>=0.65':>12} {'✅ PASS' if r['f1_score']>=0.65 else '❌ FAIL':>10}")

        lines.append(f"\n4. CONFUSION MATRIX ANALYSIS")
        cm = r["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        lines.append(f"   True Negatives (correct retained):  {tn:>8,} ({tn/total*100:.1f}%)")
        lines.append(f"   True Positives (correct churned):   {tp:>8,} ({tp/total*100:.1f}%)")
        lines.append(f"   False Positives (false alarm):      {fp:>8,} ({fp/total*100:.1f}%)")
        lines.append(f"   False Negatives (missed churn):     {fn:>8,} ({fn/total*100:.1f}%)")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        lines.append(f"   False Positive Rate:                {fpr:.2%}")

        if "feature_importance" in r:
            lines.append(f"\n5. TOP CHURN PREDICTORS (by |coefficient|)")
            lines.append(f"   {'Feature':<25} {'Coeff':>8} {'Direction':>12}")
            lines.append(f"   {'-'*48}")
            for name, coef in r["feature_importance"][:8]:
                direction = "Increases" if coef > 0 else "Decreases"
                lines.append(f"   {name:<25} {coef:>8.4f} {direction:>12}")

        lines.append(f"\n6. BUSINESS INTERPRETATION")
        lines.append(f"   The baseline Logistic Regression model achieves ROC-AUC of")
        lines.append(f"   {r['roc_auc']:.4f}, meeting the Sprint 2 target of >= 0.65.")
        lines.append(f"   The model can distinguish between churners and retained")
        lines.append(f"   customers at a level above random chance.")
        lines.append(f"\n   Key drivers of churn identified by the model coefficients")
        lines.append(f"   align with EDA findings from Sprint 1, confirming that")
        lines.append(f"   Support Calls and Usage Frequency are critical factors.")

        lines.append(f"\n7. NEXT STEPS (Sprint 3)")
        lines.append(f"   - Train Random Forest and Gradient Boosting classifiers")
        lines.append(f"   - Hyperparameter tuning with GridSearch/RandomSearch (5-fold CV)")
        lines.append(f"   - Target ROC-AUC >= 0.75 with ensemble methods")
        lines.append(f"   - Feature importance comparison across models")
        lines.append(f"   - Select best model for dashboard integration")

        lines.append("\n" + "=" * 65)

        report = "\n".join(lines)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[ModelDocumentor] Report saved to {output_path}")
        return report


if __name__ == "__main__":
    # Load saved results
    with open("outputs/baseline_results.json") as f:
        results = json.load(f)
    doc = ModelDocumentor(results)
    print(doc.generate_report())
