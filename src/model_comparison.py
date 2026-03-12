"""
Model Comparison Framework
Author: Student 3 (Data Engineer)
Sprint 2: Baseline Modeling

Provides a reusable framework for comparing multiple models.
Stores results for side-by-side comparison in Sprint 3.
"""
import json
import os
from datetime import datetime


class ModelComparisonFramework:
    """Stores and compares model results across sprints."""

    def __init__(self, output_path="outputs/model_comparison.json"):
        self.output_path = output_path
        self.models = {}
        self._load_existing()

    def _load_existing(self):
        """Load previously saved model results."""
        if os.path.exists(self.output_path):
            with open(self.output_path) as f:
                self.models = json.load(f)

    def add_model(self, name, results):
        """Register a model's evaluation results."""
        self.models[name] = {
            "accuracy": results.get("accuracy"),
            "precision": results.get("precision"),
            "recall": results.get("recall"),
            "f1_score": results.get("f1_score"),
            "roc_auc": results.get("roc_auc"),
            "confusion_matrix": results.get("confusion_matrix"),
            "added_at": datetime.now().isoformat(),
        }
        self._save()
        print(f"[Comparison] Added '{name}' (ROC-AUC: {results.get('roc_auc', 'N/A')})")

    def _save(self):
        """Save comparison data to JSON."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.models, f, indent=2, default=str)

    def get_best_model(self, metric="roc_auc"):
        """Return the model with the highest score for given metric."""
        if not self.models:
            return None, 0
        best = max(self.models.items(), key=lambda x: x[1].get(metric, 0))
        return best[0], best[1].get(metric, 0)

    def print_comparison(self):
        """Print all model results in a table."""
        if not self.models:
            print("[Comparison] No models registered yet.")
            return

        print("=" * 75)
        print("MODEL COMPARISON TABLE")
        print("=" * 75)
        print(f"  {'Model':<30} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>7} {'AUC':>7}")
        print(f"  {'-'*70}")
        for name, m in self.models.items():
            print(f"  {name:<30} {m['accuracy']:>8.4f} {m['precision']:>9.4f} "
                  f"{m['recall']:>7.4f} {m['f1_score']:>7.4f} {m['roc_auc']:>7.4f}")

        best_name, best_score = self.get_best_model()
        print(f"\n  🏆 Best Model: {best_name} (ROC-AUC: {best_score:.4f})")
        print("=" * 75)


if __name__ == "__main__":
    framework = ModelComparisonFramework()
    framework.print_comparison()
