import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib;matplotlib.use("Agg")
import numpy as np

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)
st.markdown("""
<style>
/* ---- App background: blue/cyan telecom mesh ---- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 20% -10%, rgba(37,99,235,0.32), transparent 55%),
        radial-gradient(800px 500px at 100% 100%, rgba(6,182,212,0.22), transparent 55%),
        linear-gradient(160deg, #080C1A 0%, #0F1733 60%, #0D1F3C 100%);
}
/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: rgba(8, 12, 26, 0.60);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(37,99,235,0.20);
}
/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: rgba(37,99,235,0.10);
    border: 1px solid rgba(37,99,235,0.28);
    border-radius: 18px;
    padding: 18px 20px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(37,99,235,0.15);
}
[data-testid="stMetricLabel"] { color: #93C5FD !important; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
/* ---- Tabs ---- */
[data-testid="stTabs"] button[role="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px 10px 0 0;
    color: #93C5FD;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: rgba(37,99,235,0.28);
    color: #FFFFFF;
}
/* ---- Expanders ---- */
[data-testid="stExpander"] {
    background: rgba(37,99,235,0.07);
    border: 1px solid rgba(37,99,235,0.18);
    border-radius: 12px;
}
/* ---- Buttons ---- */
.stButton button {
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    color: #FFFFFF;
    font-weight: 700;
    border: none;
    border-radius: 10px;
}
.stButton button:hover { filter: brightness(1.15); }
/* ---- Containers ---- */
[data-testid="stDataFrame"], .stAlert {
    border-radius: 14px;
}
/* ---- Progress bar ---- */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #2563EB, #06B6D4);
}
</style>
""", unsafe_allow_html=True)


API_URL = "https://churn-5i6f.onrender.com"

st.title("📉 TelcoNova Customer Churn Prediction System")

# API health check 
try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    shap_ok = h.get("shap_available", False)
    st.success(
        f"✅ API connected  "
        f"SHAP: {'✓ enabled' if shap_ok else '✗ install shap'}"
    )
except Exception:
    st.error(
        "⚠ API not running. Start it first:\n"
    )
    st.stop()

st.divider()

#  Sidebar inputs 
st.sidebar.header("Customer Profile")

tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 2)
monthly        = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 75.5)
total          = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 150.0)
contract       = st.sidebar.selectbox("Contract",
                    ["Month-to-month", "One year", "Two year"])
payment        = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
internet       = st.sidebar.selectbox("Internet Service",
                    ["Fiber optic", "DSL", "No"])
online_sec     = st.sidebar.selectbox("Online Security", ["No", "Yes"])
tech_support   = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior         = st.sidebar.selectbox("Senior Citizen",
                    [0, 1], format_func=lambda x: "Yes" if x else "No")
partner        = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents     = st.sidebar.selectbox("Dependents", ["No", "Yes"])
online_backup  = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
device_prot    = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
streaming_tv   = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
streaming_mov  = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])

predict_btn = st.sidebar.button(
    "🔍 Predict + Explain", type="primary", use_container_width=True
)

# Predict + Explain
if predict_btn:
    payload = {
        "tenure"           : tenure,
        "MonthlyCharges"   : monthly,
        "TotalCharges"     : total,
        "Contract"         : contract,
        "PaymentMethod"    : payment,
        "InternetService"  : internet,
        "OnlineSecurity"   : online_sec,
        "TechSupport"      : tech_support,
        "PaperlessBilling" : paperless,
        "SeniorCitizen"    : senior,
        "Partner"          : partner,
        "Dependents"       : dependents,
        "OnlineBackup"     : online_backup,
        "DeviceProtection" : device_prot,
        "StreamingTV"      : streaming_tv,
        "StreamingMovies"  : streaming_mov,
        "MultipleLines"    : multiple_lines,
    }

    with st.spinner("Scoring and generating SHAP explanation..."):
        try:
            # Call /predict/explain — returns prediction + SHAP in one shot
            r = requests.post(
                f"{API_URL}/predict/explain",
                json=payload,
                timeout=15
            )
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.json()}")
                st.stop()

            data = r.json()

        except requests.exceptions.ConnectionError:
            st.error("Lost connection to API.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out — SHAP can take up to 15 seconds.")
            st.stop()

    prob  = data["churn_probability"]
    tier  = data["risk_tier"]
    churn = data["will_churn"]

    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Probability", f"{prob*100:.1f}%",
              delta=f"{(prob - 0.265)*100:+.1f}% vs baseline")
    c2.metric("Risk Tier",  tier)
    c3.metric("Decision",   "⚠️Will Churn" if churn else "✅ Will Stay")
    c4.metric("Response",   f"{data['processing_ms']}ms")

    # Alert banner
    if tier == "HIGH":
        st.error(f"⚠️ HIGH CHURN RISK — {data['recommendation']}")
    elif tier == "MEDIUM":
        st.warning(f"🟡 MEDIUM RISK — {data['recommendation']}")
    else:
        st.success(f"✅ LOW RISK — {data['recommendation']}")

    st.progress(min(prob, 1.0), text=f"Churn score: {prob*100:.1f}%")
    st.divider()

    # Two columns: SHAP chart + signals 
    col_shap, col_signals = st.columns([1.4, 1])

    with col_shap:
        st.subheader("🧠 SHAP Explanation")
        st.caption(data.get("shap_explanation", ""))

        # Build SHAP bar chart from API response
        top_drivers = data.get("shap_top_drivers", [])

        if top_drivers:
            features = [d["feature"] for d in top_drivers]
            values   = [d["shap_value"] for d in top_drivers]
            colors   = ["#EF4444" if v > 0 else "#3B82F6" for v in values]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(
                range(len(features)),
                values,
                color=colors,
                edgecolor="white",
                height=0.6,
            )
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP Value (impact on churn probability)", fontsize=9)
            ax.set_title("Top SHAP Drivers — This Customer", fontsize=11, fontweight="bold")
            ax.invert_yaxis()

            # Add value labels on bars
            for bar, val in zip(bars, values):
                xpos = val + 0.002 if val >= 0 else val - 0.002
                ax.text(xpos, bar.get_y() + bar.get_height()/2,
                        f"{val:+.3f}",
                        va="center", ha="left" if val >= 0 else "right",
                        fontsize=8, color="#1F2937")

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.caption(
                "🔴 Red bars push towards **churn**. "
                "🔵 Blue bars push towards **staying**."
            )
            st.caption(data.get("shap_note", ""))
        else:
            st.info("SHAP values not available — check API logs.")

    with col_signals:
        st.subheader("⚡ Churn Signals")
        for signal in data.get("churn_signals", []):
            st.warning(f"▸ {signal}")

        st.subheader("🛡️ Protective Factors")
        for prot in data.get("protective_signals", []):
            st.success(f"▸ {prot}")

        st.divider()
        st.subheader("📋 Recommended Action")
        if tier == "HIGH":
            st.error(
                "**Immediate retention offer:**\n"
                "- Offer contract upgrade discount\n"
                "- Call within 48 hours\n"
                "- Escalate to retention team"
            )
        elif tier == "MEDIUM":
            st.warning(
                "**Proactive check-in:**\n"
                "- Schedule customer satisfaction call\n"
                "- Review billing and service issues\n"
                "- Consider loyalty incentive"
            )
        else:
            st.success(
                "**No immediate action needed.**\n"
                "- Include in standard NPS survey\n"
                "- Maintain normal contact cadence"
            )

else:
    # ── Default state ─────────────────────────────────────
    st.info("👈 Fill in the customer profile and click **Predict + Explain**.")
    
