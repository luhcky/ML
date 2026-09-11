import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

# ── Ocean Teal Theme ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 10% 0%,   rgba(0,210,180,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%,  rgba(0,180,160,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 100% 80% at 50% 50%,  rgba(0,140,130,0.08) 0%, transparent 70%),
        linear-gradient(160deg, #060F0E 0%, #091A18 45%, #0B2220 100%);
}
[data-testid="stSidebar"] {
    background: rgba(6,15,14,0.72);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(0,210,180,0.15);
}
[data-testid="stSidebar"] * { color: #A7F3EE !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #5EEAD4 !important; }

[data-testid="stMetric"] {
    background: rgba(0,210,180,0.07);
    border: 1px solid rgba(0,210,180,0.22);
    border-radius: 20px;
    padding: 20px 22px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,210,180,0.10);
}
[data-testid="stMetricLabel"] { color: #5EEAD4 !important; font-size:13px; }
[data-testid="stMetricValue"] { color: #FFFFFF  !important; font-weight:700; font-size:26px; }
[data-testid="stMetricDelta"] { color: #99F6E4 !important; }

[data-testid="stTabs"] button[role="tab"] {
    background: rgba(0,210,180,0.06);
    border-radius: 10px 10px 0 0;
    color: #5EEAD4;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: rgba(0,210,180,0.20);
    color: #FFFFFF;
    border-bottom: 2px solid #00D2B4;
}
[data-testid="stExpander"] {
    background: rgba(0,210,180,0.05);
    border: 1px solid rgba(0,210,180,0.15);
    border-radius: 14px;
}
.stButton button {
    background: linear-gradient(135deg, #00D2B4 0%, #00897B 100%);
    color: #060F0E;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    letter-spacing: 0.3px;
}
.stButton button:hover { filter: brightness(1.12); transform: translateY(-1px); }
[data-testid="stDataFrame"], .stAlert { border-radius: 16px; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00D2B4, #00897B);
    border-radius: 999px;
}
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 { color: #FFFFFF; }
.stSuccess  { background: rgba(0,210,180,0.12) !important; border-left: 4px solid #00D2B4; border-radius: 10px; }
.stWarning  { background: rgba(251,191,36,0.10) !important; border-left: 4px solid #FBBF24; border-radius: 10px; }
.stError    { background: rgba(239,68,68,0.10)  !important; border-left: 4px solid #EF4444; border-radius: 10px; }
.stInfo     { background: rgba(0,210,180,0.07)  !important; border-left: 4px solid #5EEAD4; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

API_URL = "https://churn-5i6f.onrender.com"

# ── Header ────────────────────────────────────────────────
st.title("📉 TelcoNova Customer Churn Prediction")
st.markdown("**IBM Telco · XGBoost · AUC 0.941 · SHAP Explained · What-If Simulator · Batch Scoring**")

# ── API health check ──────────────────────────────────────
try:
    h = requests.get(f"{API_URL}/health", timeout=5).json()
    shap_ok = h.get("shap_available", False)
    st.success(
        f"✅ API connected | "
        f"Threshold: {h.get('threshold', h.get('threshold_used','0.5'))} | "
        f"SHAP: {'✓ enabled' if shap_ok else '✗ not available'}"
    )
except Exception:
    st.error("⚠ API not running.")
    st.stop()

st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["👤 Single Customer", "📂 Batch CSV Upload"])

# ══════════════════════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════
with tab1:

    # ── Sidebar inputs ────────────────────────────────────
    st.sidebar.header("Customer Profile")

    with st.sidebar.expander("📋 CONTRACT & BILLING", expanded=True):
        tenure   = st.slider("Tenure (months)", 0, 72, 2)
        monthly  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 75.5)
        total    = st.number_input("Total Charges ($)", 0.0, 10000.0, 150.0)
        contract = st.selectbox("Contract",
                    ["Month-to-month","One year","Two year"])
        payment  = st.selectbox("Payment Method",
                    ["Electronic check","Mailed check",
                     "Bank transfer (automatic)","Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    with st.sidebar.expander("🌐 INTERNET SERVICES", expanded=True):
        internet     = st.selectbox("Internet Service",
                        ["Fiber optic","DSL","No"])
        online_sec   = st.selectbox("Online Security",  ["No","Yes"])
        tech_support = st.selectbox("Tech Support",     ["No","Yes"])
        online_bk    = st.selectbox("Online Backup",    ["No","Yes"])
        device_prot  = st.selectbox("Device Protection",["No","Yes"])
        streaming_tv = st.selectbox("Streaming TV",     ["No","Yes"])
        streaming_mv = st.selectbox("Streaming Movies", ["No","Yes"])
        multi_lines  = st.selectbox("Multiple Lines",   ["No","Yes"])

    with st.sidebar.expander("👤 DEMOGRAPHICS", expanded=False):
        senior     = st.selectbox("Senior Citizen",
                        [0,1], format_func=lambda x:"Yes" if x else "No")
        partner    = st.selectbox("Partner",    ["No","Yes"])
        dependents = st.selectbox("Dependents", ["No","Yes"])

    predict_btn = st.sidebar.button(
        "🔍 Predict + Explain",
        type="primary",
        use_container_width=True
    )

    # Build payload
    payload = {
        "tenure"          : tenure,
        "MonthlyCharges"  : monthly,
        "TotalCharges"    : total,
        "Contract"        : contract,
        "PaymentMethod"   : payment,
        "InternetService" : internet,
        "OnlineSecurity"  : online_sec,
        "TechSupport"     : tech_support,
        "PaperlessBilling": paperless,
        "SeniorCitizen"   : senior,
        "Partner"         : partner,
        "Dependents"      : dependents,
        "OnlineBackup"    : online_bk,
        "DeviceProtection": device_prot,
        "StreamingTV"     : streaming_tv,
        "StreamingMovies" : streaming_mv,
        "MultipleLines"   : multi_lines,
    }

    # ── FIX: On click, score the API and STASH the result + the
    # inputs that produced it in session_state. A button's True value
    # only exists for the single rerun right after the click — every
    # other interaction on the page (like moving a What-If slider)
    # triggers a fresh rerun where predict_btn is False again. Without
    # session_state, that rerun would fall straight into the "else"
    # branch and wipe out the whole results view.
    if predict_btn:
        with st.spinner("Scoring and generating SHAP explanation..."):
            try:
                r = requests.post(f"{API_URL}/predict/explain",
                                  json=payload, timeout=15)
                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.json()}")
                    st.stop()
                st.session_state["churn_data"]    = r.json()
                st.session_state["churn_payload"] = payload
            except requests.exceptions.Timeout:
                st.error("Request timed out — try again.")
                st.stop()
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.stop()

    # ── FIX: Render off session_state instead of predict_btn, so the
    # results survive reruns caused by the What-If widgets below.
    if "churn_data" in st.session_state:
        data    = st.session_state["churn_data"]
        payload = st.session_state["churn_payload"]

        prob  = data["churn_probability"]
        tier  = data["risk_tier"]
        churn = data["will_churn"]

        # ── KPI row ───────────────────────────────────────
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Churn Probability", f"{prob*100:.1f}%",
                  delta=f"{(prob-0.265)*100:+.1f}% vs baseline")
        c2.metric("Risk Tier",  tier)
        c3.metric("Decision",   "⚠ Will Churn" if churn else "✅ Will Stay")
        c4.metric("Response",   f"{data['processing_ms']}ms")

        if tier == "HIGH":
            st.error(f"⚠ HIGH CHURN RISK — {data['recommendation']}")
        elif tier == "MEDIUM":
            st.warning(f"🟡 MEDIUM RISK — {data['recommendation']}")
        else:
            st.success(f"✅ LOW RISK — {data['recommendation']}")

        st.progress(min(prob, 1.0), text=f"Churn score: {prob*100:.1f}%")
        st.divider()

        # ── SHAP + Signals ────────────────────────────────
        col_shap, col_signals = st.columns([1.4, 1])

        with col_shap:
            st.subheader("🧠 SHAP Explanation")
            st.caption(data.get("shap_explanation",""))
            top_drivers = data.get("shap_top_drivers",[])
            if top_drivers:
                features = [d["feature"]    for d in top_drivers]
                values   = [d["shap_value"] for d in top_drivers]

                BG, AMBER, CYAN, GRID, TEXT, MUTED = (
                    "#060F0E","#00D2B4","#FBBF24",
                    "#0D2926","#E0FDF9","#4B9E96"
                )
                fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)
                y_pos  = range(len(features))
                colors = [AMBER if v > 0 else CYAN for v in values]

                for y, v, c in zip(y_pos, values, colors):
                    ax.plot([0,v],[y,y], color=c, linewidth=14,
                            solid_capstyle="round", zorder=3, alpha=0.95)

                ax.axvline(0, color=MUTED, linewidth=1,
                           linestyle=(0,(4,3)), zorder=2)
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(features, fontsize=10.5, color=TEXT)
                ax.invert_yaxis()
                ax.set_xlabel("SHAP value → impact on churn probability",
                              fontsize=9.5, color=MUTED, labelpad=10)
                ax.set_title("Top Churn Drivers", fontsize=17, color=TEXT,
                             family="serif", weight="bold", pad=16, loc="left")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.6, zorder=0)
                ax.tick_params(colors=MUTED, length=0)
                for y, v in zip(y_pos, values):
                    label_x = v + (0.018 if v >= 0 else -0.018)
                    ha = "left" if v >= 0 else "right"
                    ax.text(label_x, y, f"{v:+.3f}", va="center", ha=ha,
                            fontsize=9, color=TEXT, weight="bold", zorder=4)
                ax.set_xlim(min(values)-0.12, max(values)+0.12)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                st.caption("🟡 Amber = increases churn risk.  🩵 Teal = decreases risk.")
                st.caption(data.get("shap_note",""))

        with col_signals:
            st.subheader("⚡ Churn Signals")
            for sig in data.get("churn_signals",[]):
                st.warning(f"▸ {sig}")
            st.subheader("🛡️ Protective Factors")
            for prot in data.get("protective_signals",[]):
                st.success(f"▸ {prot}")
            st.divider()
            st.subheader("📋 Recommended Action")
            if tier == "HIGH":
                st.error("**Immediate retention offer:**\n"
                         "- Offer contract upgrade discount\n"
                         "- Call within 48 hours\n"
                         "- Escalate to retention team")
            elif tier == "MEDIUM":
                st.warning("**Proactive check-in:**\n"
                           "- Schedule satisfaction call\n"
                           "- Review billing issues\n"
                           "- Consider loyalty incentive")
            else:
                st.success("**No immediate action needed.**\n"
                           "- Include in standard NPS survey\n"
                           "- Maintain normal contact cadence")

        # ══════════════════════════════════════════════════
        # WHAT-IF SIMULATOR
        # ══════════════════════════════════════════════════
        st.divider()
        st.subheader("🔧 What-If Simulator")
        st.caption(
            "Change one factor below and instantly see how the "
            "churn probability responds — no resubmit needed."
        )

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_contract = st.selectbox(
                "What if Contract was:",
                ["Month-to-month","One year","Two year"],
                index=["Month-to-month","One year","Two year"].index(payload["Contract"]),
                key="sim_contract"
            )
        with sim_c2:
            sim_charges = st.slider(
                "What if Monthly Charges were ($):",
                0.0, 200.0, float(payload["MonthlyCharges"]), 5.0,
                key="sim_charges"
            )
        with sim_c3:
            sim_security = st.selectbox(
                "What if Online Security was:",
                ["No","Yes"],
                index=0 if payload["OnlineSecurity"] == "No" else 1,
                key="sim_security"
            )

        # Call API with modified payload
        sim_payload = {
            **payload,
            "Contract"      : sim_contract,
            "MonthlyCharges": sim_charges,
            "OnlineSecurity": sim_security,
        }

        try:
            sim_r    = requests.post(f"{API_URL}/predict",
                                     json=sim_payload, timeout=10)
            sim_data = sim_r.json()
            sim_prob = sim_data["churn_probability"]
            delta    = sim_prob - prob

            # KPI comparison
            k1, k2, k3 = st.columns(3)
            k1.metric("Original Score",   f"{prob*100:.1f}%")
            k2.metric("Simulated Score",  f"{sim_prob*100:.1f}%",
                      delta=f"{delta*100:+.1f}%",
                      delta_color="inverse")
            k3.metric("Risk Change",
                      "⬆ Higher Risk"  if delta >  0.05 else
                      "⬇ Lower Risk"   if delta < -0.05 else
                      "↔ No Change")

            # Interpretation
            if delta < -0.20:
                st.success(
                    f"✅ Strong improvement — these changes reduce churn "
                    f"by {abs(delta)*100:.1f} percentage points. "
                    f"Recommend implementing immediately."
                )
            elif delta < -0.10:
                st.info(
                    f"🟡 Moderate improvement — {abs(delta)*100:.1f}pp reduction. "
                    f"Worth offering as a retention package."
                )
            elif delta < -0.05:
                st.info(
                    f"↔ Small improvement — {abs(delta)*100:.1f}pp reduction. "
                    f"May not justify cost alone."
                )
            elif delta > 0.05:
                st.warning(
                    f"⚠ These changes would INCREASE churn risk "
                    f"by {delta*100:.1f}pp. Avoid."
                )
            else:
                st.info("↔ Minimal impact on churn probability.")

            # Visual comparison — premium pill bars
            BG2 = "#060F0E"
            fig2, ax2 = plt.subplots(figsize=(8, 1.8), dpi=180)
            fig2.patch.set_facecolor(BG2)
            ax2.set_facecolor(BG2)

            ax2.plot([0, prob],     [1,1], color="#FBBF24", linewidth=18,
                     solid_capstyle="round", alpha=0.9,
                     label=f"Original  {prob*100:.1f}%")
            ax2.plot([0, sim_prob], [0,0], color="#00D2B4", linewidth=18,
                     solid_capstyle="round", alpha=0.9,
                     label=f"Simulated {sim_prob*100:.1f}%")

            ax2.set_xlim(-0.05, 1.05)
            ax2.set_yticks([0,1])
            ax2.set_yticklabels(["Simulated","Original"],
                                fontsize=10, color="#E0FDF9")
            ax2.set_xlabel("Churn Probability",
                           fontsize=8.5, color="#4B9E96")
            ax2.axvline(0.5, color="#4B9E96", linewidth=0.8,
                        linestyle="--", alpha=0.5, label="Threshold 50%")
            ax2.tick_params(colors="#4B9E96", length=0)
            for spine in ax2.spines.values():
                spine.set_visible(False)
            ax2.legend(fontsize=8.5, labelcolor="#E0FDF9",
                       facecolor=BG2, edgecolor="#0D2926",
                       loc="lower right")
            plt.tight_layout()
            st.pyplot(fig2, clear_figure=True)

        except Exception as e:
            st.warning(f"Simulation unavailable: {e}")

        # ── Optional: let the user clear the stored result and
        # start over with a fresh profile.
        st.divider()
        if st.button("🔄 Clear result and start over"):
            del st.session_state["churn_data"]
            del st.session_state["churn_payload"]
            st.rerun()

    else:
        st.info("👈 Fill in the customer profile and click **Predict + Explain**.")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Dataset",        "7,043 customers")
        m2.metric("Baseline Churn", "26.5%")
        m3.metric("Model AUC",      "0.941")
        m4.metric("Recall",         "78.4%")
        st.markdown("""
        **Key findings:**
        - Month-to-month contracts churn at **42.7%**
        - New customers (0–6 months) churn at **61.4%**
        - Electronic check payment = **45.3%** churn vs credit card **15.2%**
        - Fiber optic + no security = **2.1x** average churn rate
        """)

# ══════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📂 Batch Score — Upload a CSV of Customers")
    st.markdown(
        "Upload a CSV with one customer per row. "
        "Returns churn probability, risk tier, and recommendation for all customers."
    )

    # Template
    st.markdown("#### Step 1 — Download the template")
    template = pd.DataFrame([
        {"tenure":2,"MonthlyCharges":85.5,"TotalCharges":171.0,
         "Contract":"Month-to-month","PaymentMethod":"Electronic check",
         "InternetService":"Fiber optic","OnlineSecurity":"No",
         "TechSupport":"No","PaperlessBilling":"Yes","SeniorCitizen":0,
         "Partner":"No","Dependents":"No","OnlineBackup":"No",
         "DeviceProtection":"No","StreamingTV":"No",
         "StreamingMovies":"No","MultipleLines":"No"},
        {"tenure":48,"MonthlyCharges":55.0,"TotalCharges":2640.0,
         "Contract":"Two year","PaymentMethod":"Credit card (automatic)",
         "InternetService":"DSL","OnlineSecurity":"Yes",
         "TechSupport":"Yes","PaperlessBilling":"No","SeniorCitizen":0,
         "Partner":"Yes","Dependents":"Yes","OnlineBackup":"Yes",
         "DeviceProtection":"Yes","StreamingTV":"No",
         "StreamingMovies":"No","MultipleLines":"No"},
    ])
    st.download_button("⬇️ Download CSV Template",
                       template.to_csv(index=False),
                       "churn_template.csv","text/csv")

    st.markdown("#### Step 2 — Upload your filled CSV")
    uploaded = st.file_uploader("Upload customer CSV",
                                type=["csv"], help="Max 500 customers")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}"); st.stop()

        st.success(f"✅ Loaded {len(df)} customers")
        required = ["tenure","MonthlyCharges","TotalCharges",
                    "Contract","PaymentMethod"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}"); st.stop()
        if len(df) > 500:
            st.warning("Scoring first 500 only.")
            df = df.head(500)

        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Showing first 5 of {len(df)} rows")

        if st.button(f"🚀 Score All {len(df)} Customers", type="primary"):
            with st.spinner(f"Scoring {len(df)} customers via API..."):
                defaults = {
                    "InternetService":"Fiber optic","OnlineSecurity":"No",
                    "TechSupport":"No","PaperlessBilling":"Yes",
                    "SeniorCitizen":0,"Partner":"No","Dependents":"No",
                    "OnlineBackup":"No","DeviceProtection":"No",
                    "StreamingTV":"No","StreamingMovies":"No","MultipleLines":"No",
                }
                for col, val in defaults.items():
                    if col not in df.columns:
                        df[col] = val

                customers, errors = [], []
                for i, row in df.iterrows():
                    try:
                        customers.append({
                            "tenure"          : int(row.get("tenure",1)),
                            "MonthlyCharges"  : float(row.get("MonthlyCharges",50)),
                            "TotalCharges"    : float(row.get("TotalCharges",50)),
                            "Contract"        : str(row.get("Contract","Month-to-month")),
                            "PaymentMethod"   : str(row.get("PaymentMethod","Electronic check")),
                            "InternetService" : str(row.get("InternetService","Fiber optic")),
                            "OnlineSecurity"  : str(row.get("OnlineSecurity","No")),
                            "TechSupport"     : str(row.get("TechSupport","No")),
                            "PaperlessBilling": str(row.get("PaperlessBilling","Yes")),
                            "SeniorCitizen"   : int(row.get("SeniorCitizen",0)),
                            "Partner"         : str(row.get("Partner","No")),
                            "Dependents"      : str(row.get("Dependents","No")),
                            "OnlineBackup"    : str(row.get("OnlineBackup","No")),
                            "DeviceProtection": str(row.get("DeviceProtection","No")),
                            "StreamingTV"     : str(row.get("StreamingTV","No")),
                            "StreamingMovies" : str(row.get("StreamingMovies","No")),
                            "MultipleLines"   : str(row.get("MultipleLines","No")),
                        })
                    except Exception as e:
                        errors.append(f"Row {i+1}: {e}")
                if errors:
                    st.warning("\n".join(errors[:5]))

                try:
                    r = requests.post(f"{API_URL}/predict/batch",
                                      json=customers, timeout=60)
                    if r.status_code != 200:
                        st.error(f"Batch API error {r.status_code}: {r.text}")
                        st.stop()
                    results = r.json()
                except Exception as e:
                    st.error(f"API failed: {e}"); st.stop()

            preds = results["predictions"]
            df_r  = df.copy()
            df_r["churn_probability"] = [
                round(p["churn_probability"]*100,1) for p in preds]
            df_r["risk_tier"]  = [p["risk_tier"]  for p in preds]
            df_r["will_churn"] = [p["will_churn"] for p in preds]
            df_r = df_r.sort_values(
                "churn_probability", ascending=False
            ).reset_index(drop=True)

            tier_counts = results.get("tier_breakdown",{})
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total Scored", results["total_customers"])
            m2.metric("🔴 HIGH",      tier_counts.get("HIGH",0))
            m3.metric("🟡 MEDIUM",    tier_counts.get("MEDIUM",0))
            m4.metric("🟢 LOW",       tier_counts.get("LOW",0))

            st.info(
                f"**{results['flagged_count']} of {results['total_customers']} "
                f"customers** flagged for retention action "
                f"({results['churn_rate_pct']}% batch churn rate). "
                f"Processed in {results['processing_ms']}ms."
            )

            # Charts
            BG3 = "#060F0E"
            fig3, axes = plt.subplots(1, 2, figsize=(11, 3.5), dpi=150)
            fig3.patch.set_facecolor(BG3)

            tiers  = ["LOW","MEDIUM","HIGH"]
            counts = [tier_counts.get(t,0) for t in tiers]
            t_colors = ["#00D2B4","#FBBF24","#EF4444"]
            y_p = range(len(tiers))

            axes[0].set_facecolor(BG3)
            for y, v, c in zip(y_p, counts, t_colors):
                axes[0].plot([0,v],[y,y], color=c, linewidth=22,
                             solid_capstyle="round", zorder=3, alpha=0.9)
                axes[0].text(v+max(counts)*0.02, y, str(v),
                             va="center", fontsize=10,
                             color="#E0FDF9", weight="bold")
            axes[0].set_yticks(list(y_p))
            axes[0].set_yticklabels(tiers, fontsize=11, color="#E0FDF9")
            axes[0].set_title("Customers by Risk Tier",
                              color="#E0FDF9", fontsize=11,
                              weight="bold", loc="left", family="serif")
            axes[0].set_xlabel("Count", color="#4B9E96", fontsize=9)
            axes[0].axvline(0, color="#4B9E96", linewidth=0.8,
                            linestyle=(0,(4,3)))
            for spine in axes[0].spines.values():
                spine.set_visible(False)
            axes[0].tick_params(colors="#4B9E96", length=0)
            axes[0].grid(axis="x", color="#0D2926",
                         linewidth=0.6, alpha=0.5)
            axes[0].set_xlim(-max(counts)*0.05, max(counts)*1.25)

            axes[1].set_facecolor(BG3)
            axes[1].hist(df_r["churn_probability"].tolist(),
                         bins=20, color="#00D2B4",
                         edgecolor="none", alpha=0.85)
            axes[1].axvline(50, color="#EF4444", linestyle="--",
                            lw=1.5, label="Threshold 50%")
            axes[1].set_title("Churn Probability Distribution",
                              color="#E0FDF9", fontsize=11,
                              weight="bold", loc="left", family="serif")
            axes[1].set_xlabel("Churn Probability (%)",
                               color="#4B9E96", fontsize=9)
            axes[1].set_ylabel("Customers", color="#4B9E96", fontsize=9)
            axes[1].tick_params(colors="#4B9E96", length=0)
            axes[1].legend(fontsize=8, labelcolor="#E0FDF9",
                           facecolor=BG3, edgecolor="#0D2926")
            for spine in axes[1].spines.values():
                spine.set_visible(False)
            axes[1].grid(axis="y", color="#0D2926",
                         linewidth=0.6, alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig3, clear_figure=True)

            # Results table
            st.markdown("#### Full Results — Sorted by Risk")

            def color_tier(val):
                return {
                    "HIGH"  :"background-color:#FEE2E2;color:#7F1D1D;font-weight:bold",
                    "MEDIUM":"background-color:#FEF3C7;color:#78350F;font-weight:bold",
                    "LOW"   :"background-color:#CCFBF1;color:#134E4A",
                }.get(val,"")

            display_cols = [c for c in
                ["tenure","Contract","PaymentMethod","InternetService",
                 "MonthlyCharges","churn_probability","risk_tier","will_churn"]
                if c in df_r.columns]

            styled = (
                df_r[display_cols].style
                .applymap(color_tier, subset=["risk_tier"])
                .format({"churn_probability":"{:.1f}%",
                         "MonthlyCharges":"${:.2f}"})
            )
            st.dataframe(styled, use_container_width=True, height=420)

            st.download_button(
                "⬇️ Download Scored Results CSV",
                df_r.to_csv(index=False),
                "churn_batch_results.csv","text/csv",
                type="primary"
            )

            st.markdown("#### 🚨 Top 10 Highest Churn Risk Customers")
            st.dataframe(df_r.head(10)[display_cols],
                         use_container_width=True)
    else:
        st.info("👆 Download the template, fill it, then upload here.")
        st.markdown("**Required columns:** `tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod`")
        st.markdown("**Optional columns** *(defaults applied if missing):* `InternetService, OnlineSecurity, TechSupport, PaperlessBilling, SeniorCitizen, Partner, Dependents`")
