"""
Exploratory Data Analysis (EDA) Module
Author: Student 1 (Data Scientist)
Sprint 1: Data Foundation & EDA

Generates 11 visualizations for the Kaggle Customer Churn dataset.
Features: Age, Gender, Tenure, Usage Frequency, Support Calls,
          Payment Delay, Subscription Type, Contract Length,
          Total Spend, Last Interaction, Churn (0/1)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12, "figure.facecolor": "white",
})
COLORS = {0: "#2ecc71", 1: "#e74c3c"}
COLORS_STR = {"0": "#2ecc71", "1": "#e74c3c"}
LABELS = {0: "Retained", 1: "Churned"}


class EDAAnalyzer:
    """Generates all EDA visualizations for Sprint 1."""

    def __init__(self, df, output_dir="outputs/eda_plots"):
        self.df = df.copy()
        self.df["Churn_Label"] = self.df["Churn"].map(LABELS)
        self.output_dir = output_dir

    def run_all(self):
        """Generate all 11 plots."""
        self.plot_01_churn_distribution()
        self.plot_02_churn_by_contract()
        self.plot_03_tenure_distribution()
        self.plot_04_support_calls()
        self.plot_05_churn_by_subscription()
        self.plot_06_payment_delay()
        self.plot_07_age_distribution()
        self.plot_08_correlation_heatmap()
        self.plot_09_scatter_spend_tenure()
        self.plot_10_gender_analysis()
        self.plot_11_summary_dashboard()
        print(f"\n✅ All 11 plots saved to {self.output_dir}/")

    # ---- Plot 1: Churn Distribution (Pie + Bar) ----
    def plot_01_churn_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        counts = self.df["Churn"].value_counts().sort_index()
        colors = [COLORS[0], COLORS[1]]

        ax1.pie(counts, labels=["Retained", "Churned"], autopct="%1.1f%%",
                colors=colors, startangle=90, explode=(0, 0.05),
                textprops={"fontsize": 13, "fontweight": "bold"})
        ax1.set_title("Overall Churn Distribution", fontweight="bold")

        bars = ax2.bar(["Retained (0)", "Churned (1)"], counts.values, color=colors, edgecolor="gray")
        for bar, val in zip(bars, counts.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                     f"{val:,}", ha="center", fontweight="bold", fontsize=12)
        ax2.set_ylabel("Number of Customers")
        ax2.set_title("Customer Count by Churn Status", fontweight="bold")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/01_churn_distribution.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 01: Churn Distribution")

    # ---- Plot 2: Churn Rate by Contract Length ----
    def plot_02_churn_by_contract(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        order = ["Monthly", "Quarterly", "Annual"]
        ct = self.df.groupby("Contract Length")["Churn"].mean().reindex(order) * 100
        bars = ax.bar(ct.index, ct.values, color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="gray", width=0.6)
        for bar, val in zip(bars, ct.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12)
        ax.set_title("Churn Rate by Contract Length", fontweight="bold", fontsize=15)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Contract Length")
        ax.set_ylim(0, max(ct.values) * 1.15)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/02_churn_by_contract.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 02: Churn by Contract Length")

    # ---- Plot 3: Tenure Distribution by Churn ----
    def plot_03_tenure_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for label, color in COLORS.items():
            subset = self.df[self.df["Churn"] == label]["Tenure"]
            ax1.hist(subset, bins=30, alpha=0.6, color=color,
                     label=LABELS[label], edgecolor="gray")
        ax1.set_title("Tenure Distribution by Churn Status", fontweight="bold")
        ax1.set_xlabel("Tenure (months)")
        ax1.set_ylabel("Count")
        ax1.legend()

        sns.boxplot(data=self.df, x="Churn_Label", y="Tenure",
                    palette={"Retained": COLORS[0], "Churned": COLORS[1]}, ax=ax2)
        ax2.set_xticklabels(["Retained", "Churned"])
        ax2.set_title("Tenure Boxplot by Churn Status", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/03_tenure_distribution.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 03: Tenure Distribution")

    # ---- Plot 4: Support Calls vs Churn ----
    def plot_04_support_calls(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Churn rate by support calls
        sc_churn = self.df.groupby("Support Calls")["Churn"].mean() * 100
        ax1.bar(sc_churn.index, sc_churn.values, color="#e74c3c", edgecolor="gray", alpha=0.8)
        ax1.set_title("Churn Rate by Number of Support Calls", fontweight="bold")
        ax1.set_xlabel("Support Calls")
        ax1.set_ylabel("Churn Rate (%)")

        # Distribution
        sns.boxplot(data=self.df, x="Churn_Label", y="Support Calls",
                    palette={"Retained": COLORS[0], "Churned": COLORS[1]}, ax=ax2)
        ax2.set_xticklabels(["Retained", "Churned"])
        ax2.set_title("Support Calls Distribution by Churn", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_support_calls.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 04: Support Calls vs Churn")

    # ---- Plot 5: Churn by Subscription Type ----
    def plot_05_churn_by_subscription(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        order = ["Basic", "Standard", "Premium"]

        ct = self.df.groupby("Subscription Type")["Churn"].mean().reindex(order) * 100
        bars = ax1.bar(ct.index, ct.values, color=["#3498db", "#f39c12", "#9b59b6"],
                       edgecolor="gray", width=0.6)
        for bar, val in zip(bars, ct.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
        ax1.set_title("Churn Rate by Subscription Type", fontweight="bold")
        ax1.set_ylabel("Churn Rate (%)")

        # Count
        counts = self.df.groupby(["Subscription Type", "Churn"]).size().unstack(fill_value=0).reindex(order)
        counts.plot(kind="bar", stacked=True, ax=ax2, color=[COLORS[0], COLORS[1]], edgecolor="gray")
        ax2.set_title("Customer Count by Subscription & Churn", fontweight="bold")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.legend(["Retained", "Churned"])
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/05_churn_by_subscription.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 05: Churn by Subscription Type")

    # ---- Plot 6: Payment Delay Analysis ----
    def plot_06_payment_delay(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        sns.kdeplot(data=self.df, x="Payment Delay", hue="Churn_Label",
                    palette={"Retained": COLORS[0], "Churned": COLORS[1]},
                    fill=True, alpha=0.4, ax=ax1)
        ax1.set_title("Payment Delay Distribution (KDE)", fontweight="bold")
        ax1.set_xlabel("Payment Delay (days)")

        sns.violinplot(data=self.df, x="Churn_Label", y="Payment Delay",
                      palette={"Retained": COLORS[0], "Churned": COLORS[1]}, ax=ax2, inner="box")
        ax2.set_xticklabels(["Retained", "Churned"])
        ax2.set_title("Payment Delay Violin Plot", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/06_payment_delay.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 06: Payment Delay Analysis")

    # ---- Plot 7: Age Distribution by Churn ----
    def plot_07_age_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(data=self.df, x="Age", hue="Churn_Label",
                     palette={"Retained": COLORS[0], "Churned": COLORS[1]},
                     bins=25, alpha=0.6, ax=ax1)
        ax1.set_title("Age Distribution by Churn", fontweight="bold")

        df_temp = self.df.copy()
        df_temp["AgeGroup"] = pd.cut(self.df["Age"], bins=[17, 25, 35, 45, 55, 66],
                                     labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
        ct = df_temp.groupby("AgeGroup", observed=True)["Churn"].mean() * 100
        ct.plot(kind="bar", ax=ax2, color="#e74c3c", edgecolor="gray")
        ax2.set_title("Churn Rate by Age Group", fontweight="bold")
        ax2.set_ylabel("Churn Rate (%)")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        for i, v in enumerate(ct):
            ax2.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/07_age_distribution.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 07: Age Distribution")

    # ---- Plot 8: Correlation Heatmap ----
    def plot_08_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        df_num = self.df.copy()
        df_num["Gender_Num"] = (df_num["Gender"] == "Male").astype(int)
        df_num["Subscription_Num"] = df_num["Subscription Type"].map(
            {"Basic": 0, "Standard": 1, "Premium": 2})
        df_num["Contract_Num"] = df_num["Contract Length"].map(
            {"Monthly": 0, "Quarterly": 1, "Annual": 2})

        cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay",
                "Total Spend", "Last Interaction", "Gender_Num", "Subscription_Num",
                "Contract_Num", "Churn"]
        labels = ["Age", "Tenure", "Usage Freq", "Support Calls", "Pay Delay",
                  "Total Spend", "Last Interact", "Gender", "Subscription", "Contract", "Churn"]
        corr = df_num[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=15, pad=15)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/08_correlation_heatmap.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 08: Correlation Heatmap")

    # ---- Plot 9: Total Spend vs Tenure Scatter ----
    def plot_09_scatter_spend_tenure(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sample = self.df.sample(n=min(10000, len(self.df)), random_state=42)
        for label, color in COLORS.items():
            subset = sample[sample["Churn"] == label]
            ax.scatter(subset["Tenure"], subset["Total Spend"], c=color,
                       alpha=0.3, s=15, label=LABELS[label], edgecolors="none")
        ax.set_title("Total Spend vs Tenure by Churn Status", fontweight="bold", fontsize=15)
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Total Spend ($)")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/09_scatter_spend_tenure.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 09: Scatter - Spend vs Tenure")

    # ---- Plot 10: Gender Analysis ----
    def plot_10_gender_analysis(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ct = self.df.groupby("Gender")["Churn"].mean() * 100
        bars = ax1.bar(ct.index, ct.values, color=["#3498db", "#e91e63"], edgecolor="gray", width=0.5)
        for bar, val in zip(bars, ct.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
        ax1.set_title("Churn Rate by Gender", fontweight="bold")
        ax1.set_ylabel("Churn Rate (%)")
        ax1.set_ylim(0, max(ct.values) * 1.15)

        gender_counts = self.df["Gender"].value_counts()
        ax2.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
                colors=["#e91e63", "#3498db"], startangle=90,
                textprops={"fontsize": 12, "fontweight": "bold"})
        ax2.set_title("Gender Distribution", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/10_gender_analysis.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 10: Gender Analysis")

    # ---- Plot 11: Summary Dashboard ----
    def plot_11_summary_dashboard(self):
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Churn by contract
        ax1 = fig.add_subplot(gs[0, 0])
        order = ["Monthly", "Quarterly", "Annual"]
        ct = self.df.groupby("Contract Length")["Churn"].mean().reindex(order) * 100
        ct.plot(kind="bar", ax=ax1, color="#e74c3c", edgecolor="gray")
        ax1.set_title("Churn Rate by Contract", fontweight="bold")
        ax1.set_ylabel("Churn %")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

        # Avg Support Calls
        ax2 = fig.add_subplot(gs[0, 1])
        avg_sc = self.df.groupby("Churn")["Support Calls"].mean()
        bars = ax2.bar(["Retained", "Churned"], avg_sc.values,
                       color=[COLORS[0], COLORS[1]], edgecolor="gray")
        for bar, val in zip(bars, avg_sc.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", fontweight="bold")
        ax2.set_title("Avg Support Calls by Churn", fontweight="bold")

        # Avg Tenure
        ax3 = fig.add_subplot(gs[0, 2])
        avg_t = self.df.groupby("Churn")["Tenure"].mean()
        bars = ax3.bar(["Retained", "Churned"], avg_t.values,
                       color=[COLORS[0], COLORS[1]], edgecolor="gray")
        for bar, val in zip(bars, avg_t.values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{val:.1f} mo", ha="center", fontweight="bold")
        ax3.set_title("Avg Tenure by Churn", fontweight="bold")

        # Subscription
        ax4 = fig.add_subplot(gs[1, 0])
        sub_order = ["Basic", "Standard", "Premium"]
        sub_ct = self.df.groupby("Subscription Type")["Churn"].mean().reindex(sub_order) * 100
        sub_ct.plot(kind="bar", ax=ax4, color="#e74c3c", edgecolor="gray")
        ax4.set_title("Churn by Subscription", fontweight="bold")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

        # Usage Frequency
        ax5 = fig.add_subplot(gs[1, 1])
        avg_uf = self.df.groupby("Churn")["Usage Frequency"].mean()
        bars = ax5.bar(["Retained", "Churned"], avg_uf.values,
                       color=[COLORS[0], COLORS[1]], edgecolor="gray")
        for bar, val in zip(bars, avg_uf.values):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", fontweight="bold")
        ax5.set_title("Avg Usage Frequency by Churn", fontweight="bold")

        # Key Metrics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        churn_rate = self.df["Churn"].mean() * 100
        text = (f"KEY METRICS\n\n"
                f"Total Customers: {len(self.df):,}\n"
                f"Churn Rate: {churn_rate:.1f}%\n"
                f"Churned: {self.df['Churn'].sum():,}\n"
                f"Retained: {(self.df['Churn']==0).sum():,}\n\n"
                f"Avg Tenure: {self.df['Tenure'].mean():.1f} months\n"
                f"Avg Total Spend: ${self.df['Total Spend'].mean():,.0f}\n"
                f"Avg Support Calls: {self.df['Support Calls'].mean():.1f}")
        ax6.text(0.1, 0.95, text, transform=ax6.transAxes, fontsize=12,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

        fig.suptitle("Customer Churn Analytics - EDA Summary", fontsize=18, fontweight="bold", y=1.01)
        plt.savefig(f"{self.output_dir}/11_summary_dashboard.png", bbox_inches="tight")
        plt.close()
        print("  ✓ Plot 11: Summary Dashboard")


if __name__ == "__main__":
    df = pd.read_csv("data/customer_churn_dataset-training-master.csv").dropna()
    print(f"\n📊 Running EDA on {len(df):,} records...\n")
    eda = EDAAnalyzer(df)
    eda.run_all()
