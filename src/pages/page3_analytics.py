"""
Page 3 — Advanced Analytics
Owner: Hadeel (Student 2) | S4-04
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, load_model, NAVY, LBLUE, GREEN, RED, AMBER

def show():
    st.markdown("""
    <div class="main-header">
        <h1>📈 Advanced Analytics</h1>
        <p>Feature importance, correlations, and churn distribution analysis</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    model = load_model()

    # ── Feature Importance ────────────────────────────────────────────────
    st.markdown('<div class="section-head">🏆 XGBoost Feature Importance (Sprint 3 Results)</div>', unsafe_allow_html=True)

    features_display = ["Age","Gender","Tenure","Usage Freq.","Support Calls",
                        "Payment Delay","Subscription","Contract Length","Total Spend","Last Interaction"]
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature":features_display,"Importance":importances})
    fi_df = fi_df.sort_values("Importance", ascending=True)

    col1, col2 = st.columns([2,1])
    with col1:
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale=[[0,LBLUE],[0.5,AMBER],[1,RED]],
                        text=fi_df["Importance"].apply(lambda x: f"{x:.3f}"))
        fig_fi.update_traces(textposition='outside')
        fig_fi.update_layout(height=350, coloraxis_showscale=False,
                             plot_bgcolor="white", paper_bgcolor="white",
                             xaxis_title="Importance Score", yaxis_title="",
                             margin=dict(t=10,b=10,l=10,r=60))
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div style="background:#F0F4FA;border-radius:8px;padding:14px;margin-top:10px">
            <b style="color:{NAVY}">Top 5 Churn Drivers</b><br><br>
            <b style="color:{RED}">🥇 Support Calls</b> — 26.3%<br>
            <small>6+ calls = ~100% churn rate</small><br><br>
            <b style="color:{RED}">🥈 Contract Length</b> — 23.0%<br>
            <small>Monthly vs Annual gap is critical</small><br><br>
            <b style="color:{AMBER}">🥉 Payment Delay</b> — 13.5%<br>
            <small>15+ days delay = high risk</small><br><br>
            <b style="color:{AMBER}">4️⃣ Total Spend</b> — 11.6%<br>
            <small>Value perception concern</small><br><br>
            <b style="color:{LBLUE}">5️⃣ Age</b> — 8.5%<br>
            <small>Younger customers churn more</small>
        </div>""", unsafe_allow_html=True)

    # ── Correlation Heatmap ───────────────────────────────────────────────
    st.markdown('<div class="section-head">🔥 Feature Correlation Heatmap</div>', unsafe_allow_html=True)

    num_cols = ["Age","Tenure","Usage Frequency","Support Calls",
                "Payment Delay","Total Spend","Last Interaction","Churn"]
    corr = df[num_cols].corr().round(2)

    fig_hm = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0,"#2E75B6"],[0.5,"white"],[1,"#C00000"]],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont={"size":11},
        hoverongaps=False
    ))
    fig_hm.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                         margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Churn distributions ───────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-head">📊 Age Distribution by Churn Status</div>', unsafe_allow_html=True)
        fig_age = px.histogram(df, x="Age", color="Churn",
                               color_discrete_map={0:LBLUE,1:RED},
                               barmode="overlay", nbins=25, opacity=0.75)
        fig_age.for_each_trace(lambda t: t.update(name="Retained" if t.name=="0" else "Churned"))
        fig_age.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_title="Age", yaxis_title="Count",
                              margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_age, use_container_width=True)

    with col4:
        st.markdown('<div class="section-head">📊 Payment Delay by Churn Status</div>', unsafe_allow_html=True)
        fig_pd = px.box(df, x=df["Churn"].map({0:"Retained",1:"Churned"}),
                        y="Payment Delay",
                        color=df["Churn"].map({0:"Retained",1:"Churned"}),
                        color_discrete_map={"Retained":LBLUE,"Churned":RED})
        fig_pd.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                             xaxis_title="", yaxis_title="Payment Delay (days)",
                             showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_pd, use_container_width=True)

    # ── Usage Frequency vs Support Calls scatter ──────────────────────────
    st.markdown('<div class="section-head">🔍 Usage Frequency vs Support Calls (Churn Highlighted)</div>', unsafe_allow_html=True)
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig_sc = px.scatter(sample, x="Usage Frequency", y="Support Calls",
                        color=sample["Churn"].map({0:"Retained",1:"Churned"}),
                        color_discrete_map={"Retained":LBLUE,"Churned":RED},
                        opacity=0.5, size_max=6)
    fig_sc.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                         xaxis_title="Usage Frequency", yaxis_title="Support Calls",
                         margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Insight box ───────────────────────────────────────────────────────
    churn_corr = df[num_cols].corr()["Churn"].drop("Churn").abs().sort_values(ascending=False)
    top_corr_feat = churn_corr.index[0]
    top_corr_val  = churn_corr.iloc[0]
    st.markdown(f"""
    <div style="background:#F0F4FA;border-left:4px solid {NAVY};padding:14px 18px;border-radius:6px;margin-top:10px">
        <b>📌 Analytics Insight:</b> The strongest numerical correlation with churn is
        <b>{top_corr_feat}</b> (r = {top_corr_val:.2f}). Support Calls and Contract Length
        together explain the majority of churn variance — confirmed consistently across
        Logistic Regression (Sprint 2), Random Forest, XGBoost, and LightGBM (Sprint 3).
        Gender and Last Interaction show the weakest predictive power.
    </div>""", unsafe_allow_html=True)
