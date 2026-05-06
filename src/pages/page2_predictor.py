"""
Page 2 — Churn Predictor
Owner: Maha (Student 1) S4-03 + Ghadeer (Student 3) S4-07
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, NAVY, LBLUE, GREEN, RED, AMBER

def get_recommendations(inputs, prob):
    recs = []
    if inputs["Support Calls"] >= 5:
        recs.append(("🚨 Critical", "Customer has made 5+ support calls — assign dedicated support agent immediately and offer service credit."))
    elif inputs["Support Calls"] >= 3:
        recs.append(("⚠️ Important", "Elevated support calls detected — proactively reach out to resolve open issues before they escalate."))
    if inputs["Contract Length"] == "Monthly":
        recs.append(("💡 Upsell", "Customer is on Monthly contract — offer 15% discount to upgrade to Annual (highest retention lever)."))
    if inputs["Payment Delay"] >= 15:
        recs.append(("💳 Financial", "Payment delay of 15+ days detected — consider flexible payment plan or payment reminder campaign."))
    if inputs["Usage Frequency"] <= 8:
        recs.append(("📲 Engagement", "Low usage frequency — send personalised re-engagement email with feature highlights and tutorials."))
    if inputs["Tenure"] <= 6:
        recs.append(("🌱 Onboarding", "Early-stage customer (≤6 months) — enrol in onboarding programme to increase product stickiness."))
    if inputs["Subscription Type"] == "Basic" and prob > 0.5:
        recs.append(("⬆️ Upgrade", "Basic plan customer at high risk — offer free 30-day Premium trial to demonstrate higher value."))
    if not recs:
        recs.append(("✅ Low Risk", "No immediate risk factors detected. Continue standard engagement and monitor quarterly."))
    return recs

def show():
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Churn Predictor</h1>
        <p>Real-time churn probability scoring powered by XGBoost (AUC = 0.9536)</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-head">📝 Enter Customer Profile</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age     = st.slider("Age", 18, 70, 35)
            tenure  = st.slider("Tenure (months)", 1, 60, 24)
            support = st.slider("Support Calls", 0, 10, 2)
            pay_del = st.slider("Payment Delay (days)", 0, 30, 5)
            last_int= st.slider("Last Interaction (days ago)", 1, 30, 10)
        with c2:
            gender  = st.selectbox("Gender", ["Male","Female"])
            sub     = st.selectbox("Subscription Type", ["Basic","Standard","Premium"])
            contract= st.selectbox("Contract Length", ["Monthly","Quarterly","Annual"])
            usage   = st.slider("Usage Frequency", 1, 30, 14)
            spend   = st.slider("Total Spend ($)", 100, 1000, 450)

        predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-head">📊 Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            # Build feature vector (must match training order)
            gender_enc = 1 if gender == "Male" else 0
            sub_enc    = {"Basic":0,"Standard":1,"Premium":2}[sub]
            con_enc    = {"Monthly":0,"Quarterly":1,"Annual":2}[contract]

            X = np.array([[age, gender_enc, tenure, usage, support,
                           pay_del, sub_enc, con_enc, spend, last_int]])
            prob = float(model.predict_proba(X)[0][1])
            pred = int(prob >= 0.5)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob*100,1),
                delta={"reference":50,"increasing":{"color":RED},"decreasing":{"color":GREEN}},
                number={"suffix":"%","font":{"size":42,"color":NAVY}},
                gauge={
                    "axis":{"range":[0,100],"tickwidth":1,"tickcolor":NAVY},
                    "bar":{"color": RED if prob>0.6 else AMBER if prob>0.35 else GREEN, "thickness":0.25},
                    "bgcolor":"white",
                    "steps":[
                        {"range":[0,35],"color":"#E2EFDA"},
                        {"range":[35,60],"color":"#FFF2CC"},
                        {"range":[60,100],"color":"#FCE4D6"},
                    ],
                    "threshold":{"line":{"color":NAVY,"width":3},"thickness":0.8,"value":50}
                },
                title={"text":"Churn Probability","font":{"size":16,"color":NAVY}}
            ))
            fig.update_layout(height=260, margin=dict(t=30,b=10,l=30,r=30),
                              paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            # Risk badge
            if prob >= 0.6:
                risk_label, risk_class = "HIGH RISK", "risk-high"
                risk_msg = "Immediate retention action required."
            elif prob >= 0.35:
                risk_label, risk_class = "MEDIUM RISK", "risk-medium"
                risk_msg = "Monitor closely and apply targeted interventions."
            else:
                risk_label, risk_class = "LOW RISK", "risk-low"
                risk_msg = "Customer is likely to stay. Continue standard engagement."

            st.markdown(f"""
            <div style="text-align:center;margin:10px 0">
                <span class="{risk_class}">{risk_label}</span>
                <p style="color:#555;margin-top:8px;font-size:0.9rem">{risk_msg}</p>
            </div>""", unsafe_allow_html=True)

            # Customer summary
            st.markdown(f"""
            <div style="background:#F0F4FA;border-radius:8px;padding:12px 16px;margin-top:8px">
                <b>Profile Summary:</b> {age}yr {gender} | {sub} | {contract} contract |
                {tenure}mo tenure | {support} support calls | ${spend} spend
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#F0F4FA;border-radius:8px;padding:30px;text-align:center;margin-top:20px">
                <span style="font-size:3rem">🔮</span>
                <p style="color:#666;margin-top:10px">Fill in the customer profile on the left<br>and click <b>Predict Churn Risk</b></p>
            </div>""", unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────
    if predict_btn:
        st.markdown('<div class="section-head">💡 Personalised Retention Recommendations</div>', unsafe_allow_html=True)
        inputs = {"Support Calls":support,"Contract Length":contract,
                  "Payment Delay":pay_del,"Usage Frequency":usage,
                  "Tenure":tenure,"Subscription Type":sub}
        recs = get_recommendations(inputs, prob)
        cols = st.columns(min(len(recs),3))
        for i, (tag, rec) in enumerate(recs):
            with cols[i % len(cols)]:
                color = RED if "Critical" in tag else AMBER if "Important" in tag or "Warning" in tag else NAVY
                st.markdown(f"""
                <div style="background:white;border:1px solid #DEE2E6;border-top:3px solid {color};
                            border-radius:8px;padding:14px;height:100%;margin-bottom:8px">
                    <b style="color:{color}">{tag}</b>
                    <p style="color:#444;margin-top:6px;font-size:0.88rem">{rec}</p>
                </div>""", unsafe_allow_html=True)

        # Feature contribution bar
        st.markdown('<div class="section-head">🔍 Key Risk Factors for This Customer</div>', unsafe_allow_html=True)
        factor_scores = {
            "Support Calls":   min(support / 10, 1.0),
            "Contract Length": {"Monthly":0.9,"Quarterly":0.5,"Annual":0.1}[contract],
            "Payment Delay":   min(pay_del / 30, 1.0),
            "Usage Frequency": max(0, 1 - usage/30),
            "Tenure":          max(0, 1 - tenure/60),
            "Total Spend":     max(0, 1 - spend/1000),
        }
        import plotly.express as px
        import pandas as pd
        df_factors = pd.DataFrame({
            "Factor": list(factor_scores.keys()),
            "Risk Score": [round(v*100,1) for v in factor_scores.values()]
        }).sort_values("Risk Score", ascending=True)
        fig_f = px.bar(df_factors, x="Risk Score", y="Factor", orientation="h",
                       color="Risk Score", color_continuous_scale=["#375623","#ED7D31","#C00000"],
                       text="Risk Score")
        fig_f.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig_f.update_layout(height=250, showlegend=False, coloraxis_showscale=False,
                            plot_bgcolor="white", paper_bgcolor="white",
                            xaxis_title="Risk Contribution (%)", yaxis_title="",
                            margin=dict(t=10,b=10,l=10,r=60))
        st.plotly_chart(fig_f, use_container_width=True)
