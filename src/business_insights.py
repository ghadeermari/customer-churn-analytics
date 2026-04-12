"""
Business Insights Module
Author: Student 2 (Business Analyst)
Sprint 1: Data Foundation & EDA

Generates business-focused statistical summaries, churn driver analysis,
and key insights for stakeholder reporting.
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime


class BusinessInsights:
    """Analyzes churn patterns and generates business-focused insights."""

    def __init__(self, df):
        self.df = df
        self.insights = {}

    def analyze_churn_drivers(self):
        """Identify top churn drivers across all categorical features."""
        drivers = {}
        for col in ["Contract Length", "Subscription Type", "Gender"]:
            drivers[col] = self.df.groupby(col)["Churn"].apply(
                lambda x: round(x.mean() * 100, 2)
            ).to_dict()
        self.insights["churn_drivers"] = drivers
        return drivers

    def analyze_numeric_patterns(self):
        """Compare numeric features between churners and non-churners."""
        numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                        "Payment Delay", "Total Spend", "Last Interaction"]
        patterns = {}
        for col in numeric_cols:
            churned = self.df[self.df["Churn"] == 1][col]
            retained = self.df[self.df["Churn"] == 0][col]
            patterns[col] = {
                "churned_mean": round(churned.mean(), 2),
                "retained_mean": round(retained.mean(), 2),
                "difference": round(churned.mean() - retained.mean(), 2),
            }
        self.insights["numeric_patterns"] = patterns
        return patterns

    def segment_risk_by_support_calls(self):
        """Segment customers by support calls — the strongest churn predictor."""
        df_temp = self.df.copy()
        df_temp["SupportSegment"] = pd.cut(
            df_temp["Support Calls"],
            bins=[-1, 2, 5, 7, 11],
            labels=["Low (0-2)", "Medium (3-5)", "High (6-7)", "Critical (8-10)"],
        )
        segment_stats = df_temp.groupby("SupportSegment", observed=True).agg(
            total=("Churn", "count"),
            churned=("Churn", "sum"),
            churn_rate=("Churn", lambda x: round(x.mean() * 100, 2)),
            avg_spend=("Total Spend", lambda x: round(x.mean(), 2)),
        ).to_dict(orient="index")
        self.insights["support_segments"] = segment_stats
        return segment_stats

    def generate_report(self, output_path="outputs/business_insights_report.txt"):
        """Generate full business insights report."""
        self.analyze_churn_drivers()
        self.analyze_numeric_patterns()
        self.segment_risk_by_support_calls()

        lines = []
        lines.append("=" * 65)
        lines.append("BUSINESS INSIGHTS REPORT - SPRINT 1")
        lines.append("Customer Churn Analytics Project")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 65)

        total = len(self.df)
        churned = int(self.df["Churn"].sum())
        lines.append(f"\n1. OVERVIEW")
        lines.append(f"   Total Customers: {total:,}")
        lines.append(f"   Churned:         {churned:,} ({churned/total*100:.1f}%)")
        lines.append(f"   Retained:        {total-churned:,} ({(total-churned)/total*100:.1f}%)")

        lines.append(f"\n2. CHURN DRIVERS BY CATEGORY")
        for feature, rates in self.insights["churn_drivers"].items():
            lines.append(f"\n   {feature}:")
            for cat, rate in sorted(rates.items(), key=lambda x: -x[1]):
                lines.append(f"     {cat:20s} -> {rate}% churn")

        lines.append(f"\n3. NUMERIC FEATURE COMPARISON (Churned vs Retained)")
        for col, stats in self.insights["numeric_patterns"].items():
            lines.append(f"\n   {col}:")
            lines.append(f"     Churned avg:  {stats['churned_mean']}")
            lines.append(f"     Retained avg: {stats['retained_mean']}")
            lines.append(f"     Difference:   {stats['difference']}")

        lines.append(f"\n4. RISK SEGMENTATION BY SUPPORT CALLS")
        lines.append(f"   {'Segment':<20} {'Total':>8} {'Churned':>8} {'Rate':>8} {'Avg $':>8}")
        lines.append(f"   {'-'*55}")
        for seg, stats in self.insights["support_segments"].items():
            lines.append(
                f"   {seg:<20} {stats['total']:>8,} {stats['churned']:>8,} "
                f"{stats['churn_rate']:>7}% ${stats['avg_spend']:>7}"
            )

        lines.append(f"\n5. KEY INSIGHTS FOR STAKEHOLDERS")
        lines.append(f"   - Support Calls is the strongest churn predictor")
        lines.append(f"   - Usage Frequency strongly differentiates churners from retained")
        lines.append(f"   - Contract Length and Subscription Type show similar churn rates")
        lines.append(f"   - Last Interaction recency matters: churners have fewer recent days")
        lines.append(f"   - Total Spend is lower for churners, indicating value perception")

        lines.append(f"\n6. RECOMMENDATIONS")
        lines.append(f"   - Implement proactive outreach for customers with 5+ support calls")
        lines.append(f"   - Create engagement programs to boost usage frequency")
        lines.append(f"   - Offer retention incentives to low-spend, high-support-call customers")
        lines.append(f"   - Monitor last interaction gaps and trigger re-engagement campaigns")

        lines.append("\n" + "=" * 65)

        report = "\n".join(lines)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[BusinessInsights] Report saved to {output_path}")
        return report

    def export_json(self, output_path="outputs/eda_summary.json"):
        """Export insights as JSON for dashboard consumption."""
        if not self.insights:
            self.analyze_churn_drivers()
            self.analyze_numeric_patterns()
            self.segment_risk_by_support_calls()

        with open(output_path, "w") as f:
            json.dump(self.insights, f, indent=2, default=str)
        print(f"[BusinessInsights] JSON exported to {output_path}")


if __name__ == "__main__":
    df = pd.read_csv("data/customer_churn_dataset-training-master.csv").dropna()
    bi = BusinessInsights(df)
    report = bi.generate_report()
    bi.export_json()
    print(report)
