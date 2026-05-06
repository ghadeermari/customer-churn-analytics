"""
Page 1 — Executive Dashboard
Owner: Maha (Student 1) | S4-02
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, NAVY, LBLUE, GREEN, RED, AMBER

def show():
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Executive Dashboard</h1>
        <p>BIS405 Graduation Project | Customer Churn Prediction System | Analytics Team Alpha</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    # ── KPI Cards ────────────────────────────────────────────────────────
    total       = len(df)
    churned     = int(df["Churn"].sum())
    retained    = total - churned
    churn_rate  = churned / total * 100
    avg_spend   = df["Total Spend"].mean()
    avg_tenure  = df["Tenure"].mean()
    high_risk   = int((df["Support Calls"] >= 5).sum())

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    cards = [
        (c1, f"{total:,}",        "Total Customers",     NAVY),
        (c2, f"{churned:,}",      "Churned Customers",   RED),
        (c3, f"{retained:,}",     "Retained Customers",  GREEN),
        (c4, f"{churn_rate:.1f}%","Overall Churn Rate",  AMBER),
        (c5, f"${avg_spend:.0f}", "Avg Customer Spend",  LBLUE),
        (c6, f"{high_risk:,}",    "High-Risk Customers", "#7030A0"),
    ]
    for col, val, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{color}">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Churn by Contract + Churn by Subscription ─────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-head">📋 Churn Rate by Contract Length</div>', unsafe_allow_html=True)
        con_data = df.groupby("Contract Length").agg(
            Total=("Churn","count"), Churned=("Churn","sum")
        ).reset_index()
        con_data["Churn Rate (%)"] = (con_data["Churned"] / con_data["Total"] * 100).round(1)
        order = ["Monthly","Quarterly","Annual"]
        con_data["Contract Length"] = pd.Categorical(con_data["Contract Length"], categories=order, ordered=True)
        con_data = con_data.sort_values("Contract Length")
        fig = px.bar(con_data, x="Contract Length", y="Churn Rate (%)",
                     color="Churn Rate (%)", color_continuous_scale=["#375623","#ED7D31","#C00000"],
                     text="Churn Rate (%)", title="")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=320, showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", coloraxis_showscale=False,
                          xaxis_title="", yaxis_title="Churn Rate (%)",
                          margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">💎 Churn Rate by Subscription Tier</div>', unsafe_allow_html=True)
        sub_data = df.groupby("Subscription Type").agg(
            Total=("Churn","count"), Churned=("Churn","sum")
        ).reset_index()
        sub_data["Churn Rate (%)"] = (sub_data["Churned"] / sub_data["Total"] * 100).round(1)
        fig2 = px.pie(sub_data, names="Subscription Type", values="Churned",
                      color_discrete_sequence=[NAVY, LBLUE, "#BDD7EE"],
                      hole=0.45, title="")
        fig2.update_traces(textposition='inside', textinfo='percent+label',
                           textfont_size=13)
        fig2.update_layout(height=320, showlegend=True,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Support Calls distribution + Spend vs Churn ───────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-head">📞 Churn Rate by Support Calls</div>', unsafe_allow_html=True)
        sc_data = df.groupby("Support Calls").agg(
            Total=("Churn","count"), Churned=("Churn","sum")
        ).reset_index()
        sc_data["Churn Rate (%)"] = (sc_data["Churned"] / sc_data["Total"] * 100).round(1)
        fig3 = px.bar(sc_data, x="Support Calls", y="Churn Rate (%)",
                      color="Churn Rate (%)", color_continuous_scale=["#375623","#ED7D31","#C00000"],
                      text="Churn Rate (%)", title="")
        fig3.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig3.update_layout(height=320, showlegend=False, plot_bgcolor="white",
                           paper_bgcolor="white", coloraxis_showscale=False,
                           xaxis_title="Number of Support Calls", yaxis_title="Churn Rate (%)",
                           margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-head">💰 Avg Total Spend: Churned vs Retained</div>', unsafe_allow_html=True)
        spend_data = df.groupby("Churn")["Total Spend"].mean().reset_index()
        spend_data["Status"] = spend_data["Churn"].map({0:"Retained",1:"Churned"})
        spend_data["Avg Spend"] = spend_data["Total Spend"].round(2)
        fig4 = px.bar(spend_data, x="Status", y="Avg Spend",
                      color="Status", color_discrete_map={"Retained":GREEN,"Churned":RED},
                      text="Avg Spend", title="")
        fig4.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
        fig4.update_layout(height=320, showlegend=False, plot_bgcolor="white",
                           paper_bgcolor="white", xaxis_title="",
                           yaxis_title="Average Total Spend ($)",
                           margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Tenure distribution ────────────────────────────────────────
    st.markdown('<div class="section-head">📅 Customer Tenure Distribution by Churn Status</div>', unsafe_allow_html=True)
    fig5 = px.histogram(df, x="Tenure", color="Churn",
                        color_discrete_map={0:"#2E75B6",1:"#C00000"},
                        barmode="overlay", nbins=30,
                        labels={"Churn":"Status","Tenure":"Tenure (months)"},
                        title="")
    fig5.update_traces(opacity=0.75)
    newnames = {0:"Retained",1:"Churned"}
    fig5.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))
    fig5.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Tenure (months)", yaxis_title="Customer Count",
                       margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig5, use_container_width=True)

    # ── Summary box ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#F0F4FA;border-left:4px solid {NAVY};padding:14px 18px;border-radius:6px;margin-top:10px">
        <b>📌 Executive Summary:</b> Of {total:,} customers, <b style="color:{RED}">{churned:,} ({churn_rate:.1f}%)</b>
        have churned. Monthly contract customers show the highest churn risk, while Annual subscribers
        demonstrate strong retention. <b>Support Calls</b> is the strongest single churn predictor —
        customers making 5+ calls churn at near 100%. Average spend of churned customers (${df[df['Churn']==1]['Total Spend'].mean():.0f})
        vs retained (${df[df['Churn']==0]['Total Spend'].mean():.0f}) indicates price sensitivity as a secondary driver.
    </div>
    """, unsafe_allow_html=True)
