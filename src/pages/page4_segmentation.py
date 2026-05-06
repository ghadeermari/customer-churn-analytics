"""
Page 4 — Customer Segmentation
Owner: Ghadeer (Student 3) | S4-05
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
        <h1>👥 Customer Segmentation</h1>
        <p>Tenure/spend clusters, risk heatmaps, and churn patterns by segment</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    # ── Segment definition ────────────────────────────────────────────────
    def assign_segment(row):
        if row["Tenure"] <= 12 and row["Total Spend"] < 400:
            return "New Low-Value"
        elif row["Tenure"] <= 12 and row["Total Spend"] >= 400:
            return "New High-Value"
        elif row["Tenure"] <= 36 and row["Total Spend"] < 400:
            return "Mid Low-Value"
        elif row["Tenure"] <= 36 and row["Total Spend"] >= 400:
            return "Mid High-Value"
        elif row["Total Spend"] < 400:
            return "Loyal Low-Value"
        else:
            return "Loyal High-Value"

    df["Segment"] = df.apply(assign_segment, axis=1)

    # ── KPI row ───────────────────────────────────────────────────────────
    seg_stats = df.groupby("Segment").agg(
        Count=("Churn","count"),
        Churned=("Churn","sum"),
        AvgSpend=("Total Spend","mean"),
        AvgTenure=("Tenure","mean")
    ).reset_index()
    seg_stats["Churn Rate"] = (seg_stats["Churned"]/seg_stats["Count"]*100).round(1)
    highest_risk = seg_stats.loc[seg_stats["Churn Rate"].idxmax(), "Segment"]
    lowest_risk  = seg_stats.loc[seg_stats["Churn Rate"].idxmin(), "Segment"]

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(seg_stats)}</div><div class="kpi-label">Segments Identified</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{RED}">{highest_risk}</div><div class="kpi-label">Highest Risk Segment</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{GREEN}">{lowest_risk}</div><div class="kpi-label">Lowest Risk Segment</div></div>', unsafe_allow_html=True)
    with c4:
        monthly_churn = df[df["Contract Length"]=="Monthly"]["Churn"].mean()*100
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{RED}">{monthly_churn:.1f}%</div><div class="kpi-label">Monthly Contract Churn</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tenure vs Spend scatter ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-head">🗺️ Tenure vs Total Spend Segmentation</div>', unsafe_allow_html=True)
        sample = df.sample(min(4000,len(df)), random_state=42)
        fig_seg = px.scatter(sample, x="Tenure", y="Total Spend",
                             color="Segment",
                             symbol=sample["Churn"].map({0:"circle",1:"x"}),
                             opacity=0.6, size_max=8,
                             color_discrete_sequence=px.colors.qualitative.Set2)
        fig_seg.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_title="Tenure (months)", yaxis_title="Total Spend ($)",
                              margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_seg, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">📊 Churn Rate by Segment</div>', unsafe_allow_html=True)
        seg_sorted = seg_stats.sort_values("Churn Rate", ascending=True)
        fig_seg2 = px.bar(seg_sorted, x="Churn Rate", y="Segment", orientation="h",
                          color="Churn Rate",
                          color_continuous_scale=["#375623","#ED7D31","#C00000"],
                          text="Churn Rate")
        fig_seg2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_seg2.update_layout(height=360, coloraxis_showscale=False,
                               plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_title="Churn Rate (%)", yaxis_title="",
                               margin=dict(t=10,b=10,l=10,r=60))
        st.plotly_chart(fig_seg2, use_container_width=True)

    # ── Risk Heatmap (Contract × Subscription) ────────────────────────────
    st.markdown('<div class="section-head">🔥 Risk Heatmap — Contract Length × Subscription Type</div>', unsafe_allow_html=True)

    heat_df = df.groupby(["Contract Length","Subscription Type"])["Churn"].mean().reset_index()
    heat_df["Churn Rate (%)"] = (heat_df["Churn"]*100).round(1)
    heat_pivot = heat_df.pivot(index="Contract Length", columns="Subscription Type", values="Churn Rate (%)")
    row_order = ["Monthly","Quarterly","Annual"]
    heat_pivot = heat_pivot.reindex([r for r in row_order if r in heat_pivot.index])

    fig_hm = go.Figure(data=go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=heat_pivot.index.tolist(),
        colorscale=[[0,"#375623"],[0.5,"#FFF2CC"],[1,"#C00000"]],
        text=heat_pivot.values.round(1),
        texttemplate="%{text}%",
        textfont={"size":14, "color":"black"},
        hoverongaps=False,
        colorbar=dict(title="Churn %")
    ))
    fig_hm.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                         xaxis_title="Subscription Type", yaxis_title="Contract Length",
                         margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Segment detail table ──────────────────────────────────────────────
    st.markdown('<div class="section-head">📋 Segment Summary Table</div>', unsafe_allow_html=True)
    display_df = seg_stats[["Segment","Count","Churned","Churn Rate","AvgSpend","AvgTenure"]].copy()
    display_df.columns = ["Segment","Total","Churned","Churn Rate (%)","Avg Spend ($)","Avg Tenure (mo)"]
    display_df["Avg Spend ($)"] = display_df["Avg Spend ($)"].round(0).astype(int)
    display_df["Avg Tenure (mo)"] = display_df["Avg Tenure (mo)"].round(1)
    display_df = display_df.sort_values("Churn Rate (%)", ascending=False)

    def color_churn(val):
        if val >= 65: return f"background-color:#FCE4D6;color:{RED};font-weight:bold"
        elif val >= 45: return f"background-color:#FFF2CC"
        else: return f"background-color:#E2EFDA;color:{GREEN}"

    styled = display_df.style.applymap(color_churn, subset=["Churn Rate (%)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Churn by support calls per contract ──────────────────────────────
    st.markdown('<div class="section-head">📞 Support Calls Distribution by Contract Type</div>', unsafe_allow_html=True)
    fig_box = px.box(df, x="Contract Length", y="Support Calls",
                     color="Contract Length",
                     color_discrete_sequence=[RED, AMBER, GREEN],
                     category_orders={"Contract Length":["Monthly","Quarterly","Annual"]})
    fig_box.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="", yaxis_title="Support Calls",
                          showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_box, use_container_width=True)
