import streamlit as st
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="M-PESA Fraud Detection",
    page_icon="📱",
    layout="wide",
)

API_URL = "https://m-api-q3hs.onrender.com"   # ← replace with your real deployed URL

KENYA_COUNTIES = [
    "Nairobi","Mombasa","Kisumu","Nakuru","Uasin Gishu","Meru","Kilifi",
    "Kakamega","Machakos","Garissa","Turkana","Mandera","Wajir","Marsabit",
    "Isiolo","Tana River","Kwale","Taita Taveta","Lamu","Siaya","Homa Bay",
    "Migori","Nyamira","Kericho","Bomet","Nandi","Trans Nzoia","West Pokot",
    "Elgeyo Marakwet","Baringo","Laikipia","Samburu","Tharaka Nithi","Embu",
    "Kirinyaga","Murang'a","Kiambu","Nyandarua","Nyeri","Vihiga","Bungoma",
    "Busia","Kitui","Makueni","Kajiado","Narok","Kisii",
]
CHANNELS = ["PESA", "AGENT", "TILL", "PAYBILL"]

# ── Light glassmorphic glow theme ──────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stApp {
    background:
        radial-gradient(460px 460px at 18% -8%, rgba(167,139,250,0.55), transparent 60%),
        radial-gradient(400px 400px at 100% 100%, rgba(45,212,191,0.40), transparent 60%),
        radial-gradient(320px 320px at 65% 25%, rgba(251,113,133,0.35), transparent 60%),
        linear-gradient(160deg, #1B1330 0%, #2E1E4E 50%, #452B66 100%) !important;
}
[data-testid="stSidebar"] {
    background: rgba(20,14,35,0.55) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.12);
}
[data-testid="stSidebar"] h3 {
    color: #C4A9FF !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 12.5px !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 18px;
    padding: 18px 20px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}
[data-testid="stMetricLabel"] { color: #B8AFD6 !important; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 800; }
[data-testid="stTabs"] button[role="tab"] {
    background: rgba(255,255,255,0.08);
    border-radius: 10px 10px 0 0;
    color: #B8AFD6;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: rgba(167,139,250,0.28);
    color: #FFFFFF;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 14px;
    backdrop-filter: blur(14px);
}
.stButton button {
    background: linear-gradient(90deg, #A78BFA, #EC4899);
    color: #1B1330;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    box-shadow: 0 0 26px rgba(167,139,250,0.55), 0 4px 14px rgba(236,72,153,0.35);
}
.stButton button:hover { filter: brightness(1.08); }
[data-testid="stDataFrame"] { border-radius: 14px; }
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 14px;
    backdrop-filter: blur(14px);
}
.stAlert { border-radius: 14px; }
h1, h2, h3 { color: #FFFFFF !important; }
p, span, label { color: #C6BEE0; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.title("📱 M-PESA Fraud Detection")
st.markdown("**SHAP Explained · RAG Fraud Assistant · Kenya Counties**")

try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(
        f"✅ API connected — "
        f"SHAP: {'✓' if h.get('shap_available') else '✗'} | "
        f"RAG: {'✓' if h.get('rag_available') else '✗ not built'}"
    )
except Exception:
    st.error("⚠ API not running.Please wait as the API starts then refresh.")
    st.stop()

st.divider()

# ── Sidebar inputs ─────────────────────────────────────────
with st.sidebar:
    st.header("📱 Transaction")

    amount_kes         = st.number_input("Amount (KES)", min_value=1.0, max_value=150000.0, value=9535.0, step=100.0)
    sender_account_age = st.number_input("Sender Account Age (days)", min_value=0, max_value=3650, value=268)
    sender_tx          = st.slider("Sender Transaction Count (velocity)", 0, 500, 34)

    st.markdown("### Route")
    sender_county   = st.selectbox("Sender County", sorted(KENYA_COUNTIES), index=sorted(KENYA_COUNTIES).index("Vihiga"))
    receiver_county = st.selectbox("Receiver County", sorted(KENYA_COUNTIES), index=sorted(KENYA_COUNTIES).index("Isiolo"))
    channel         = st.selectbox("Channel", CHANNELS, index=0)

    st.markdown("### Timing")
    hour        = st.slider("Hour of day (0–23)", 0, 23, 9)
    day_of_week = st.selectbox("Day of week", list(range(7)),
                                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x], index=4)

    predict_btn = st.button("🔍 Predict + Explain", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["🔍 Score Transaction", "📂 Batch CSV Upload", "💬 Fraud Assistant"])

ALERT_COLORS = {
    "BLOCK" : "#EF4444",  
    "REVIEW": "#D97706", 
    "CLEAR" : "#0F9D74",   
}

# ══════════════════════════════════════════════════════════
# TAB 1 — SCORE A TRANSACTION
# ══════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        payload = {
            "amount_kes"        : amount_kes,
            "sender_account_age": sender_account_age,
            "sender_county"     : sender_county,
            "receiver_county"   : receiver_county,
            "channel"           : channel,
            "hour"              : hour,
            "day_of_week"       : day_of_week,
            "sender_tx"         : sender_tx,
        }

        with st.spinner("Scoring transaction and generating SHAP explanation..."):
            try:
                r = requests.post(f"{API_URL}/predict/explain", json=payload, timeout=15)
                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.text}")
                    st.stop()
                data = r.json()
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

        prob  = data["fraud_probability"]
        alert = data["alert_level"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud Probability", f"{data['fraud_pct']}%")
        c2.metric("Alert Level", alert)
        c3.metric("Route", data["county_route"])
        c4.metric("Response", f"{data['processing_ms']}ms")

        if alert == "BLOCK":
            st.error(f"🚨 BLOCK — {data['action']}")
        elif alert == "REVIEW":
            st.warning(f"🟡 REVIEW — {data['action']}")
        else:
            st.success(f"✅ CLEAR — {data['action']}")

        st.progress(min(prob, 1.0), text=f"Fraud score: {data['fraud_pct']}%")
        st.divider()

        col_shap, col_signals = st.columns([1.4, 1])

        with col_shap:
            st.subheader("🧠 SHAP Explanation")
            st.caption(data.get("shap_explanation", ""))

            top_drivers = data.get("shap_top_drivers", [])
            if top_drivers:
                features = [d["feature"]    for d in top_drivers]
                values   = [d["shap_value"] for d in top_drivers]

                BG          = "#241A3D"
                PURPLE      = "#A78BFA"
                PINK        = "#EC4899"
                TEAL        = "#2DD4BF"
                GRID        = "#3A2E58"
                TEXT        = "#FFFFFF"
                MUTED       = "#C6BEE0"

                fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)

                y_pos = range(len(features))
                colors = [PURPLE if v > 0 else TEAL for v in values]

                for y, v, c in zip(y_pos, values, colors):
                    ax.plot([0, v], [y, y], color=c, linewidth=14,
                             solid_capstyle="round", zorder=3, alpha=0.95)

                ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2, alpha=0.5)
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(features, fontsize=10, color=TEXT)
                ax.invert_yaxis()

                ax.set_xlabel("SHAP value  →  impact on fraud probability",
                              fontsize=9, color=MUTED, labelpad=10)
                ax.set_title("Top Fraud Risk Drivers", fontsize=16, color=TEXT,
                              weight="bold", pad=14, loc="left")

                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(axis="x", color=GRID, linewidth=0.9, alpha=0.9, zorder=0)
                ax.tick_params(colors=MUTED, length=0)

                for y, v in zip(y_pos, values):
                    label_x = v + (0.0015 if v >= 0 else -0.0015)
                    ha = "left" if v >= 0 else "right"
                    ax.text(label_x, y, f"{v:+.4f}", va="center", ha=ha,
                             fontsize=8.5, color=TEXT, weight="bold", zorder=4)

                span = max(abs(min(values)), abs(max(values))) * 1.25
                ax.set_xlim(-span, span)

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                st.caption("🟣 Purple = increases fraud risk.  🟢 Teal = decreases fraud risk.")
                st.caption(data.get("shap_note", ""))

        with col_signals:
            st.subheader("⚡ Fraud Signals")
            for sig in data.get("fraud_signals", []):
                st.warning(f"▸ {sig}")

    else:
        st.info("👈 Set the transaction details in the sidebar, then click **Predict + Explain**.")

# ══════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📂 Batch Score — Upload a CSV of Transactions")
    st.markdown(
        "Upload a CSV with one transaction per row, matching the fields below. "
        "Scores up to 1000 transactions per upload."
    )

    template = pd.DataFrame([{
        "amount_kes": 9535, "sender_account_age": 268,
        "sender_county": "Vihiga", "receiver_county": "Isiolo",
        "channel": "PESA", "hour": 9, "day_of_week": 4, "sender_tx": 34,
    }, {
        "amount_kes": 45000, "sender_account_age": 5,
        "sender_county": "Nairobi", "receiver_county": "Turkana",
        "channel": "AGENT", "hour": 2, "day_of_week": 6, "sender_tx": 12,
    }])

    st.download_button(
        "⬇️ Download CSV Template",
        template.to_csv(index=False),
        "mpesa_template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"], help="Max 1000 transactions")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.success(f"✅ Loaded {len(df)} transactions")

        required = ["amount_kes", "sender_account_age", "sender_county",
                    "receiver_county", "channel", "hour", "day_of_week", "sender_tx"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        if len(df) > 1000:
            st.warning("More than 1000 rows — only the first 1000 will be scored.")
            df = df.head(1000)

        st.dataframe(df.head(5), use_container_width=True)

        if st.button(f"🚀 Score All {len(df)} Transactions", type="primary"):
            with st.spinner(f"Scoring {len(df)} transactions..."):
                transactions = df[required].to_dict(orient="records")
                try:
                    r = requests.post(f"{API_URL}/predict/batch", json=transactions, timeout=60)
                    if r.status_code != 200:
                        st.error(f"Batch API error {r.status_code}: {r.text}")
                        st.stop()
                    results = r.json()
                except Exception as e:
                    st.error(f"API call failed: {e}")
                    st.stop()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Scored", results["total_transactions"])
            m2.metric("Fraud Count", results["fraud_count"])
            m3.metric("Fraud Rate", f"{results['fraud_rate_pct']}%")
            m4.metric("Fraud Amount (KES)", f"{results['fraud_amount_kes']:,.0f}")

            alert_breakdown = results["alert_breakdown"]
            fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
            fig.patch.set_facecolor("#241A3D")
            ax.set_facecolor("#241A3D")
            levels = ["CLEAR", "REVIEW", "BLOCK"]
            counts = [alert_breakdown.get(l, 0) for l in levels]
            colors = [ALERT_COLORS["CLEAR"], ALERT_COLORS["REVIEW"], ALERT_COLORS["BLOCK"]]
            ax.bar(levels, counts, color=colors, edgecolor="#FFFFFF")
            ax.set_title("Alert Breakdown", color="#FFFFFF", fontsize=13, weight="bold", loc="left")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors="#C6BEE0")
            for i, c in enumerate(counts):
                ax.text(i, c + max(counts)*0.02, str(c), ha="center", color="#FFFFFF", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

            df_results = pd.DataFrame(results["predictions"])
            st.markdown("#### Full Results")
            st.dataframe(df_results, use_container_width=True, height=350)

            st.download_button(
                "⬇️ Download Scored Results CSV",
                df_results.to_csv(index=False),
                "mpesa_batch_results.csv",
                "text/csv",
                type="primary",
            )

# ══════════════════════════════════════════════════════════
# TAB 3 — RAG FRAUD ASSISTANT
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("💬 Fraud Knowledge Assistant")
    st.caption(
        "Ask about SIMSwap patterns, county risk, model performance, SHAP, or alert thresholds. "
        "Answers come from the fraud knowledge base — Claude-generated if configured, otherwise keyword-matched."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about M-PESA fraud patterns...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching fraud knowledge base..."):
                try:
                    r = requests.post(f"{API_URL}/chat", json={"question": question, "n_results": 3}, timeout=20)
                    if r.status_code != 200:
                        answer = f"⚠ API error {r.status_code}: {r.text}"
                        sources = []
                        model_used = ""
                    else:
                        resp = r.json()
                        answer = resp["answer"]
                        sources = resp.get("sources", [])
                        model_used = resp.get("model_used", "")
                except Exception as e:
                    answer = f"⚠ Could not reach the API: {e}"
                    sources = []
                    model_used = ""

            st.markdown(answer)
            if sources:
                with st.expander(f"📚 Sources ({model_used})"):
                    for s in sources:
                        st.caption(f"**{s['topic']}** — relevance {s['relevance']:.2f}")
                        st.text(s["excerpt"])

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
