"""
Page 5 — Model Performance
Owner: Hadeel (Student 2) | S4-06
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, load_all_models, load_sprint3_results, NAVY, LBLUE, GREEN, RED, AMBER

def show():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Model Performance</h1>
        <p>Sprint 3 model comparison, ROC curves, confusion matrices, and business impact calculator</p>
    </div>
    """, unsafe_allow_html=True)

    df      = load_data()
    models  = load_all_models()
    results = load_sprint3_results()

    # ── Results table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-head">📊 All Models — KPI Achievement Summary</div>', unsafe_allow_html=True)

    res_data = {
        "Model":       ["Logistic Regression","Random Forest","XGBoost ★","LightGBM"],
        "ROC-AUC":     [0.9006, 0.9535, 0.9536, 0.9536],
        "Accuracy":    [0.8263, 0.9293, 0.9255, 0.9245],
        "F1-Score":    [0.8363, 0.9394, 0.9357, 0.9348],
        "Precision":   [0.8767, 0.8971, 0.8983, 0.8984],
        "Recall":      [0.7995, 0.9858, 0.9762, 0.9743],
        "FPR":         ["20.0%","10.1%","10.2%","10.3%"],
        "Status":      ["Baseline","✅ KPI Met","✅ Selected","✅ KPI Met"],
    }
    res_df = pd.DataFrame(res_data)

    # Color the AUC column
    def color_auc(val):
        try:
            v = float(val)
            if v >= 0.95: return f"background:#E2EFDA;color:{GREEN};font-weight:bold"
            elif v >= 0.75: return f"background:#FFF2CC"
            return ""
        except: return ""

    styled = res_df.style.applymap(color_auc, subset=["ROC-AUC","Accuracy","F1-Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style="background:#E2EFDA;border-left:4px solid {GREEN};padding:10px 16px;
                border-radius:6px;margin:8px 0 16px 0;font-size:0.9rem">
        ✅ <b>KPI 1 EXCEEDED:</b> Target ROC-AUC ≥ 0.75 — All advanced models achieved 0.95+ &nbsp;|&nbsp;
        ✅ <b>KPI 2 EXCEEDED:</b> Precision ≥ 70% — Achieved 89.83% &nbsp;|&nbsp;
        ✅ <b>KPI 3 EXCEEDED:</b> FPR &lt; 30% — Achieved 10.2%
    </div>""", unsafe_allow_html=True)

    # ── ROC Curves ────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">📉 ROC Curve Comparison — All Models</div>', unsafe_allow_html=True)

    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    FEATURES = ["Age","Gender_Enc","Tenure","Usage Frequency","Support Calls",
                "Payment Delay","Sub_Enc","Con_Enc","Total Spend","Last Interaction"]
    X = df[FEATURES].values
    y = df["Churn"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    lr.fit(X_tr_sc, y_tr)

    model_list = [
        ("Logistic Regression", lr.predict_proba(X_te_sc)[:,1], "#595959"),
        ("Random Forest",       models["Random Forest"].predict_proba(X_te)[:,1], LBLUE),
        ("XGBoost ★",           models["XGBoost"].predict_proba(X_te)[:,1], RED),
        ("LightGBM",            models["LightGBM"].predict_proba(X_te)[:,1], AMBER),
    ]

    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
                      line=dict(dash="dash", color="#AAAAAA", width=1))
    for name, proba, color in model_list:
        fpr, tpr, _ = roc_curve(y_te, proba)
        auc = roc_auc_score(y_te, proba)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.4f})",
                                     line=dict(color=color, width=2.5)))
    fig_roc.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.01]),
                          legend=dict(x=0.45, y=0.15, bgcolor="rgba(255,255,255,0.9)"),
                          margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-head">🟦 XGBoost Confusion Matrix</div>', unsafe_allow_html=True)
        from sklearn.metrics import confusion_matrix
        xgb_pred = models["XGBoost"].predict(X_te)
        cm = confusion_matrix(y_te, xgb_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted: Retained","Predicted: Churned"],
            y=["Actual: Retained","Actual: Churned"],
            colorscale=[[0,"white"],[1,NAVY]],
            text=cm, texttemplate="<b>%{text}</b>", textfont={"size":20},
            showscale=False
        ))
        fig_cm.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                             margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_cm, use_container_width=True)

        tn,fp,fn,tp = cm.ravel()
        st.markdown(f"""
        <div style="background:#F0F4FA;border-radius:8px;padding:10px 14px;font-size:0.85rem">
            ✅ True Positives: <b>{tp:,}</b> &nbsp;|&nbsp; ✅ True Negatives: <b>{tn:,}</b><br>
            ❌ False Positives: <b style="color:{AMBER}">{fp:,}</b> &nbsp;|&nbsp;
            ❌ False Negatives: <b style="color:{RED}">{fn:,}</b>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-head">📊 Metric Comparison — All Models vs KPI</div>', unsafe_allow_html=True)
        metrics = ["ROC-AUC","Accuracy","F1-Score","Precision","Recall"]
        vals = {
            "Logistic Regression": [0.9006,0.8263,0.8363,0.8767,0.7995],
            "Random Forest":       [0.9535,0.9293,0.9394,0.8971,0.9858],
            "XGBoost":             [0.9536,0.9255,0.9357,0.8983,0.9762],
            "LightGBM":            [0.9536,0.9245,0.9348,0.8984,0.9743],
        }
        colors_m = ["#595959", LBLUE, RED, AMBER]
        fig_bar = go.Figure()
        for (name, v), color in zip(vals.items(), colors_m):
            fig_bar.add_trace(go.Bar(name=name, x=metrics, y=v, marker_color=color))
        fig_bar.add_hline(y=0.75, line_dash="dash", line_color="#C00000",
                          annotation_text="KPI Target 0.75", annotation_position="top right")
        fig_bar.update_layout(height=280, barmode="group", plot_bgcolor="white",
                              paper_bgcolor="white", yaxis=dict(range=[0.5,1.05]),
                              legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                              margin=dict(t=10,b=60,l=10,r=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Business Impact Calculator ─────────────────────────────────────────
    st.markdown('<div class="section-head">💰 Business Impact Calculator</div>', unsafe_allow_html=True)
    st.markdown("*Estimate the financial value of churn prevention using the XGBoost model*")

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        total_customers = st.number_input("Total Customer Base", value=50000, step=1000, min_value=1000)
    with bc2:
        avg_clv = st.number_input("Avg Customer Lifetime Value ($)", value=650, step=50, min_value=100)
    with bc3:
        retention_rate = st.slider("% of At-Risk Customers Retained", 10, 80, 40)

    churn_rt   = 0.555
    precision  = 0.8983
    recall     = 0.9762

    predicted_churners  = int(total_customers * churn_rt * recall)
    true_positives_pct  = precision
    actionable          = int(predicted_churners * true_positives_pct)
    saved               = int(actionable * (retention_rate/100))
    revenue_saved       = saved * avg_clv
    false_alarms        = predicted_churners - actionable
    cost_false_alarms   = false_alarms * 15  # ~$15 per outreach

    rc1, rc2, rc3, rc4 = st.columns(4)
    metrics_biz = [
        (rc1, f"{predicted_churners:,}", "At-Risk Customers Flagged", RED),
        (rc2, f"{actionable:,}", "True Churn Predictions", AMBER),
        (rc3, f"{saved:,}", f"Customers Saved ({retention_rate}%)", GREEN),
        (rc4, f"${revenue_saved:,.0f}", "Revenue Retained ($)", NAVY),
    ]
    for col, val, label, color in metrics_biz:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:{color};font-size:1.6rem">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#F0F4FA;border-left:4px solid {NAVY};padding:14px 18px;
                border-radius:6px;margin-top:14px">
        <b>💡 Business Case:</b> With {total_customers:,} customers and XGBoost's
        <b>89.83% precision</b> and <b>97.62% recall</b>, the model flags {predicted_churners:,} at-risk customers.
        Of these, {actionable:,} are true churners. Retaining {retention_rate}% through targeted interventions
        saves <b style="color:{GREEN}">${revenue_saved:,.0f}</b> in customer lifetime value.
        False alarm outreach cost is approximately ${cost_false_alarms:,} — a strong positive ROI.
    </div>""", unsafe_allow_html=True)

    # ── KPI 5: Load time note ────────────────────────────────────────────
    import time
    st.markdown('<div class="section-head">⏱️ KPI 5 — Dashboard Load Time</div>', unsafe_allow_html=True)
    if st.button("▶️ Measure Page Load Time"):
        t_start = time.time()
        _ = load_data()
        _ = load_all_models()
        elapsed = round((time.time() - t_start) * 1000)
        status = "✅ KPI MET" if elapsed < 3000 else "⚠️ Review needed"
        color  = GREEN if elapsed < 3000 else RED
        st.markdown(f"""
        <div style="background:white;border:2px solid {color};border-radius:8px;
                    padding:16px;text-align:center;margin-top:10px">
            <span style="font-size:2rem;font-weight:700;color:{color}">{elapsed} ms</span><br>
            <span style="color:{color};font-weight:600">{status}</span><br>
            <small style="color:#666">Target: &lt; 3,000 ms &nbsp;|&nbsp; Achieved: {elapsed} ms
            &nbsp;|&nbsp; Margin: {3000-elapsed} ms</small>
        </div>""", unsafe_allow_html=True)
