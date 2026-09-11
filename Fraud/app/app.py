import streamlit as st
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

API_URL = "https://fraud-detector-zgo6.onrender.com"  

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 20% -10%, rgba(124,92,255,0.35), transparent 55%),
        radial-gradient(800px 500px at 100% 100%, rgba(94,234,212,0.25), transparent 55%),
        linear-gradient(160deg, #0D1024 0%, #171B3D 60%, #1E2350 100%);
}
[data-testid="stSidebar"] {
    background: rgba(13, 16, 36, 0.55);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.10);
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 18px 20px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
[data-testid="stMetricLabel"] { color: #C6C9F0 !important; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
[data-testid="stTabs"] button[role="tab"] {
    background: rgba(255,255,255,0.06);
    border-radius: 10px 10px 0 0;
    color: #C6C9F0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: rgba(124,92,255,0.25);
    color: #FFFFFF;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
}
.stButton button {
    background: linear-gradient(90deg, #7C5CFF, #5EEAD4);
    color: #0D1024;
    font-weight: 700;
    border: none;
    border-radius: 10px;
}
.stButton button:hover { filter: brightness(1.1); }
[data-testid="stDataFrame"] { border-radius: 14px; }
h1, h2, h3 { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.markdown("**Real-Time Scoring · Live Monitoring**")

try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(
        f"✅ API connected — "
        f"Features: {h.get('feature_count','?')}"
    )
except Exception:
    st.error("⚠ API not running.Please wait for the API to start then refresh.")
    st.stop()

st.divider()

tab1, tab2 = st.tabs(["🔍 Score Transaction", "📊 Live Monitoring"])

# ══════════════════════════════════════════════════════════
# TAB 1 — SCORE A TRANSACTION
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Score a Transaction")

    col_a, col_b = st.columns(2)
    with col_a:
        amount = st.number_input("Transaction Amount (USD)", min_value=0.0, value=149.62, step=1.0)
    with col_b:
        time_val = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=406.0, step=1.0)

    st.caption(
        "V1–V28 are PCA-anonymized features from the original dataset and have no "
        "individually meaningful real-world label. They default to 0.0 (dataset mean); "
        "adjust them if you're testing a specific known transaction pattern."
    )

    v_values = {}
    with st.expander("Advanced — PCA Components (V1–V28)"):
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v_{i}")

    predict_btn = st.button("🔍 Predict + Explain", type="primary", use_container_width=True)

    if predict_btn:
        payload = {"Time": time_val, "Amount": amount, **v_values}

        with st.spinner("Scoring and generating SHAP explanation..."):
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
        c1.metric("Fraud Probability", f"{prob*100:.2f}%")
        c2.metric("Alert Level", alert)
        c3.metric("Decision", "⚠ Fraud" if data["is_fraud"] else "✅ Legitimate")
        c4.metric("Response", f"{data['processing_ms']}ms")

        if alert == "BLOCK":
            st.error("🚨 BLOCK — high-confidence fraud. Block transaction and investigate immediately.")
        elif alert == "REVIEW":
            st.warning("🟡 REVIEW — flagged for manual review.")
        else:
            st.success("✅ CLEAR — approve transaction.")

        st.progress(min(prob, 1.0), text=f"Fraud score: {prob*100:.2f}%")
        st.divider()


        st.subheader("🧠 SHAP Explanation")
        st.caption(data.get("shap_explanation", ""))

        top_drivers = data.get("shap_top_drivers", [])
        if top_drivers:
            features = [d["feature"]    for d in top_drivers]
            values   = [d["shap_value"] for d in top_drivers]

            BG, PURPLE, TEAL, GRID, TEXT, MUTED = (
                "#171B3D", "#8B6CFF", "#5EEAD4", "#3A3F6B", "#FFFFFF", "#C6C9F0"
            )

            fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)

            y_pos = range(len(features))
            colors = [PURPLE if v > 0 else TEAL for v in values]

            for y, v, c in zip(y_pos, values, colors):
                ax.plot([0, v], [y, y], color=c, linewidth=14,
                         solid_capstyle="round", zorder=3, alpha=0.95)

            ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2, alpha=0.6)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(features, fontsize=10.5, color=TEXT)
            ax.invert_yaxis()

            ax.set_xlabel("SHAP value  →  impact on fraud probability",
                          fontsize=9.5, color=MUTED, labelpad=10)
            ax.set_title("Top Fraud Risk Drivers", fontsize=17, color=TEXT,
                          weight="bold", pad=16, loc="left")

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.6, zorder=0)
            ax.tick_params(colors=MUTED, length=0)

            for y, v in zip(y_pos, values):
                label_x = v + (0.0015 if v >= 0 else -0.0015)
                ha = "left" if v >= 0 else "right"
                ax.text(label_x, y, f"{v:+.4f}", va="center", ha=ha,
                         fontsize=9, color=TEXT, weight="bold", zorder=4)

            span = max(abs(min(values)), abs(max(values))) * 1.25
            ax.set_xlim(-span, span)

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.caption("🟣 Purple = increases fraud risk.  🟦 Teal = decreases fraud risk.")
            st.caption(f"SHAP space: {data.get('shap_space','?')}  ·  Base fraud rate: {data.get('shap_base_value',0)*100:.2f}%")
    else:
        st.info("👈 Set the transaction amount, time, and any PCA components, then click **Predict + Explain**.")

# ══════════════════════════════════════════════════════════
# TAB 2 — LIVE MONITORING
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Live Model Monitoring")
    st.caption(
        "Operational monitoring — request volume, latency, and live prediction "
        "distribution since the API last restarted. This does not track feature-level "
        "drift against the training distribution."
    )

    if st.button("🔄 Refresh Stats"):
        st.rerun()

    try:
        stats = requests.get(f"{API_URL}/monitoring/stats", timeout=5).json()
    except Exception as e:
        st.error(f"Could not reach monitoring endpoint: {e}")
        st.stop()

    if "window_size" not in stats:
        st.info(stats.get("note", "No predictions logged yet — score a transaction first."))
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Requests", stats["total_requests_since_startup"])
        m2.metric("Total Flagged", stats["total_flagged_since_startup"])
        m3.metric("Window Fraud Rate", f"{stats['window_fraud_rate_pct']}%")
        m4.metric("Avg Latency", f"{stats['window_avg_latency_ms']}ms")

        m5, m6, m7 = st.columns(3)
        m5.metric("Window Avg Probability", f"{stats['window_avg_probability']*100:.2f}%")
        m6.metric("Window Max Probability", f"{stats['window_max_probability']*100:.2f}%")
        m7.metric("Uptime", f"{stats['uptime_seconds']/60:.1f} min")

        st.divider()
        st.markdown("#### Recent Predictions")
        try:
            recent = requests.get(f"{API_URL}/monitoring/recent", params={"limit": 30}, timeout=5).json()
            df_recent = pd.DataFrame(recent["recent_predictions"])
            if not df_recent.empty:
                df_recent["probability_pct"] = (df_recent["probability"] * 100).round(2)
                st.dataframe(
                    df_recent[["timestamp", "probability_pct", "is_fraud", "processing_ms"]],
                    use_container_width=True, height=320,
                )

                fig, ax = plt.subplots(figsize=(9, 2.8), dpi=200)
                BG, PURPLE, GRID, TEXT, MUTED = "#171B3D", "#8B6CFF", "#3A3F6B", "#FFFFFF", "#C6C9F0"
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)
                ax.plot(range(len(df_recent)), df_recent["probability_pct"],
                         color=PURPLE, linewidth=2, marker="o", markersize=3)
                ax.set_title("Fraud Probability — Recent Requests", fontsize=13,
                              color=TEXT, weight="bold", loc="left")
                ax.set_ylabel("Probability %", fontsize=9, color=MUTED)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
                ax.tick_params(colors=MUTED, length=0)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No recent predictions to show yet.")
        except Exception as e:
            st.warning(f"Could not load recent predictions: {e}")
    st.info(f'Risk Level: {risk} | Off-Hours: {"Yes" if off_hours else "No"} | Amount: EUR{amount:.2f}')
