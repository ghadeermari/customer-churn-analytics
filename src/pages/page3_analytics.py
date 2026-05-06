"""
pages/page3_analytics.py
Sprint 4 — Deliverable D (Hadeel, S4-04)
Sprint 5 — SHAP waterfall chart added (Maha, S5-04)
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_data, load_model

NAVY  = "#1F3864"
GOLD  = "#E8B04A"
RED   = "#E74C3C"
AMBER = "#F39C12"
BLUE  = "#2980B9"

FEATURE_COLS = [
    "Age", "Gender_Enc", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay", "Sub_Enc", "Con_Enc",
    "Total Spend", "Last Interaction"
]

def render():
    st.title("📈 Advanced Analytics")
    st.markdown("Deep-dive into feature importance, correlations, and data distributions.")

    df = load_data()
    model = load_model()

    # ── Feature Importance ────────────────────────────────────────────────────
    st.subheader("XGBoost Feature Importance")
    importances = model.feature_importances_
    feat_labels = ["Age","Gender","Tenure","Usage Freq","Support Calls",
                   "Payment Delay","Subscription","Contract","Total Spend","Last Interaction"]
    colors = [RED if v >= 0.20 else AMBER if v >= 0.10 else BLUE for v in importances]
    idx = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[idx], y=[feat_labels[i] for i in idx],
        orientation="h", marker_color=[colors[i] for i in idx],
        text=[f"{importances[i]*100:.1f}%" for i in idx], textposition="outside"
    ))
    fig.update_layout(
        title="Feature Importance — XGBoost (Sprint 3)",
        paper_bgcolor=NAVY, plot_bgcolor=NAVY,
        font=dict(color="white"), height=420,
        margin=dict(l=10, r=60, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.subheader("Correlation Heatmap")
    num_cols = ["Age","Tenure","Usage Frequency","Support Calls",
                "Payment Delay","Total Spend","Last Interaction","Churn"]
    corr = df[num_cols].corr()
    fig2 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Pearson Correlation Matrix"
    )
    fig2.update_layout(paper_bgcolor=NAVY, font=dict(color="white"), height=420)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Distributions ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.histogram(df, x="Age", color="Churn", barmode="overlay",
            title="Age Distribution by Churn Status",
            color_discrete_map={0:"#27AE60", 1:"#E74C3C"},
            labels={"Churn":"Churn Status"})
        fig3.update_layout(paper_bgcolor=NAVY, plot_bgcolor=NAVY, font=dict(color="white"))
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = px.box(df, x="Churn", y="Payment Delay", color="Churn",
            title="Payment Delay Distribution by Churn",
            color_discrete_map={0:"#27AE60", 1:"#E74C3C"})
        fig4.update_layout(paper_bgcolor=NAVY, plot_bgcolor=NAVY, font=dict(color="white"))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Scatter ───────────────────────────────────────────────────────────────
    st.subheader("Usage Frequency vs Support Calls")
    sample = df.sample(n=min(3000, len(df)), random_state=42)
    fig5 = px.scatter(sample, x="Usage Frequency", y="Support Calls", color="Churn",
        color_discrete_map={0:"#27AE60", 1:"#E74C3C"},
        title="High-Risk Cluster: Low Usage + High Support Calls",
        labels={"Churn":"Churn Status"}, opacity=0.6)
    fig5.update_layout(paper_bgcolor=NAVY, plot_bgcolor=NAVY, font=dict(color="white"))
    st.plotly_chart(fig5, use_container_width=True)

    # ── SHAP Waterfall (Sprint 5 — S5-04) ────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 SHAP Explainability — Waterfall Chart")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows **which features pushed the prediction "
        "higher or lower** for a specific customer. 🔴 Red bars increase churn risk; 🔵 blue bars reduce it."
    )

    try:
        import shap

        FEATURE_LABELS = [
            "Age", "Gender", "Tenure (months)", "Usage Frequency",
            "Support Calls", "Payment Delay (days)", "Subscription Type",
            "Contract Length", "Total Spend (SAR)", "Days Since Last Interaction"
        ]

        sample_df = df[FEATURE_COLS].sample(n=min(500, len(df)), random_state=42)
        X_sample = sample_df.values

        col_shap1, col_shap2 = st.columns([1, 2])
        with col_shap1:
            st.markdown("**Select Customer Profile:**")
            example_type = st.selectbox(
                "Profile type",
                ["Highest Risk Customer", "Lowest Risk Customer", "Average Customer"],
                label_visibility="collapsed"
            )

        if example_type == "Highest Risk Customer":
            X_explain = sample_df[(sample_df["Con_Enc"] == 0) & (sample_df["Support Calls"] >= 6)].head(1)
            if X_explain.empty: X_explain = sample_df.head(1)
        elif example_type == "Lowest Risk Customer":
            X_explain = sample_df[(sample_df["Con_Enc"] == 2) & (sample_df["Support Calls"] <= 1)].head(1)
            if X_explain.empty: X_explain = sample_df.tail(1)
        else:
            X_explain = sample_df.iloc[[len(sample_df)//2]]

        X_explain_vals = X_explain.values

        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model, data=X_sample)
            shap_values = explainer(X_explain_vals)

        shap_values.feature_names = FEATURE_LABELS

        fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
        fig_shap.patch.set_facecolor("#0E1117")
        ax_shap.set_facecolor("#0E1117")
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        for text in fig_shap.findobj(plt.Text):
            text.set_color("white")
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        shap_vals_1d = shap_values.values[0]
        top_idx = abs(shap_vals_1d).argsort()[::-1][:3]
        st.markdown("**Top 3 contributing features for this profile:**")
        for idx in top_idx:
            direction = "🔴 increases" if shap_vals_1d[idx] > 0 else "🔵 decreases"
            st.markdown(f"- **{FEATURE_LABELS[idx]}** → {direction} churn risk by `{abs(shap_vals_1d[idx]):.4f}`")

    except ImportError:
        st.warning("SHAP not installed. Run: `pip install shap==0.44.0`")
    except Exception as e:
        st.error(f"SHAP error: {e}")
