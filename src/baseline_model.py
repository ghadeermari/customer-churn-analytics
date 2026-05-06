"""
Baseline Model Training Module
Author: Student 1 (Data Scientist)
Sprint 2: Baseline Modeling

Trains Logistic Regression baseline, evaluates with all metrics, generates plots.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib, json, os, warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12, "figure.facecolor": "white"})


class BaselineModelTrainer:
    def __init__(self, output_dir="outputs/sprint2"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = None
        self.results = {}

    def train(self, X_train, y_train):
        print("[Trainer] Training Logistic Regression baseline...")
        self.model = LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs",
            C=1.0, class_weight="balanced")
        self.model.fit(X_train, y_train)
        print("[Trainer] Model trained")
        return self.model

    def evaluate(self, X_test, y_test, feature_names=None):
        print("[Trainer] Evaluating...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.results = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "avg_precision": round(average_precision_score(y_test, y_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        self._y_test, self._y_pred, self._y_proba = y_test, y_pred, y_proba
        self._feature_names = feature_names
        self._print_results()
        return self.results

    def cross_validate(self, X_train, y_train, cv=5):
        print(f"[Trainer] {cv}-fold CV...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_train, y_train, cv=skf, scoring="roc_auc")
        self.results["cv_scores"] = scores.tolist()
        self.results["cv_mean"] = round(scores.mean(), 4)
        self.results["cv_std"] = round(scores.std(), 4)
        print(f"[Trainer] CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    def plot_all(self):
        self._plot_confusion_matrix()
        self._plot_roc_curve()
        self._plot_precision_recall()
        self._plot_feature_importance()
        self._plot_summary()
        print(f"[Trainer] All plots saved to {self.output_dir}/")

    def _plot_confusion_matrix(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(self._y_test, self._y_pred)
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
        ax.set_title("Confusion Matrix - Logistic Regression", fontweight="bold", fontsize=15)
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.7, f"({cm[i][j]/total*100:.1f}%)", ha="center", fontsize=10, color="gray")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/01_confusion_matrix.png", bbox_inches="tight"); plt.close()
        print("  Plot: Confusion Matrix")

    def _plot_roc_curve(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self._y_test, self._y_proba)
        auc = self.results["roc_auc"]
        ax.plot(fpr, tpr, color="#e74c3c", lw=2.5, label=f"Logistic Regression (AUC={auc:.4f})")
        ax.plot([0,1],[0,1], "k--", lw=1, label="Random (AUC=0.50)")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Baseline Model", fontweight="bold", fontsize=15)
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/02_roc_curve.png", bbox_inches="tight"); plt.close()
        print("  Plot: ROC Curve")

    def _plot_precision_recall(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        prec, rec, _ = precision_recall_curve(self._y_test, self._y_proba)
        ap = self.results["avg_precision"]
        ax.plot(rec, prec, color="#3498db", lw=2.5, label=f"LR (AP={ap:.4f})")
        ax.fill_between(rec, prec, alpha=0.1, color="#3498db")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=15)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/03_precision_recall.png", bbox_inches="tight"); plt.close()
        print("  Plot: Precision-Recall")

    def _plot_feature_importance(self):
        if not self._feature_names: return
        fig, ax = plt.subplots(figsize=(10, 7))
        coefs = self.model.coef_[0]
        imp = pd.DataFrame({"Feature": self._feature_names, "Coef": coefs}).sort_values("Coef")
        colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in imp["Coef"]]
        ax.barh(imp["Feature"], imp["Coef"], color=colors, edgecolor="gray", height=0.6)
        ax.set_title("Feature Coefficients (Logistic Regression)", fontweight="bold", fontsize=15)
        ax.set_xlabel("Coefficient"); ax.axvline(x=0, color="black", lw=0.8); ax.grid(axis="x", alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#e74c3c", label="Increases Churn"),
                           Patch(facecolor="#2ecc71", label="Decreases Churn")], loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_feature_importance.png", bbox_inches="tight"); plt.close()
        print("  Plot: Feature Importance")

    def _plot_summary(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics = ["ROC-AUC", "F1", "Precision", "Recall", "Accuracy"]
        vals = [self.results["roc_auc"], self.results["f1_score"], self.results["precision"],
                self.results["recall"], self.results["accuracy"]]
        colors = ["#2ecc71" if v >= 0.65 else "#f39c12" for v in vals]
        bars = ax.bar(metrics, vals, color=colors, edgecolor="gray", width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.4f}",
                    ha="center", fontweight="bold", fontsize=11)
        ax.axhline(y=0.65, color="blue", ls="--", lw=1.5, label="Target (0.65)")
        ax.set_title("Baseline Performance Summary", fontweight="bold", fontsize=15)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/05_performance_summary.png", bbox_inches="tight"); plt.close()
        print("  Plot: Performance Summary")

    def save_model(self, path="models"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, f"{path}/logistic_regression_baseline.pkl")
        print(f"[Trainer] Model saved to {path}/")

    def save_results(self):
        r = {k: v for k, v in self.results.items()}
        r["model"] = "Logistic Regression"; r["class_weight"] = "balanced"
        with open(f"{self.output_dir}/baseline_results.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"[Trainer] Results saved")

    def _print_results(self):
        r = self.results
        print("\n" + "=" * 55)
        print("BASELINE MODEL - Logistic Regression")
        print("=" * 55)
        for m in ["accuracy","roc_auc","f1_score","precision","recall"]:
            print(f"  {m:<18} {r[m]:.4f}")
        status = "PASSED" if r["roc_auc"] >= 0.65 else "BELOW TARGET"
        print(f"\n  Target ROC-AUC >= 0.65: {status}")
        print("=" * 55)
