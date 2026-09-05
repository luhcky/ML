
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

st.set_page_config(
    page_title="Agricultural Yield Prediction",
    page_icon="🌽",
    layout="wide",
)
st.markdown("""
            <style>
            
            .stApp{
                background-color:green;}
                
            section[data-testid="stSidebar"]{
                background-color:green;
                border-right: 1px solid black;}
                
            div[data-testid="stMetric"],div[data-testid="stExpander"],div[data-testid="stDataFrame"]{background-color:black;
            border-radius: 10px; padding:10px;}
            .stButton>button{
                background-color:#0E9F4A;
                color: white; border-radius: 8px; border:none;
                font-weight:600;}
                
            .stButton.button:hover{background-color:#1B5E20;
            color:white}
            
            .stDownloadButton>button{background-color:#2E7D32;
            color:white;
            border-radius: 8px; border:none;
            font-weight:700;}
            
            thead tr th{background-color:#E8F5E9;color:#1B5E20;}
            </style>
            """, unsafe_allow_html=True)

API_URL = "https://churn-1-racq.onrender.com"

COUNTIES = sorted([
    "Uasin Gishu","Trans Nzoia","Nakuru","Nandi","Kericho","Bomet",
    "Kakamega","Bungoma","Busia","Nairobi","Mombasa","Kisumu","Meru",
    "Kilifi","Machakos","Garissa","Turkana","Mandera","Wajir","Marsabit",
    "Isiolo","Tana River","Kwale","Taita Taveta","Lamu","Siaya",
    "Homa Bay","Migori","Nyamira","Kisii","Vihiga","Laikipia","Nyeri",
    "Kirinyaga","Murang'a","Kiambu","Nyandarua","Embu","Tharaka Nithi",
    "Samburu","Baringo","West Pokot","Elgeyo Marakwet","Kajiado",
    "Narok","Kitui","Makueni",
])

CROPS        = ["Maize","Wheat","Beans","Potatoes","Sorghum",
                "Millet","Tea","Coffee","Rice","Barley"]
SEASONS      = ["Long Rain","Short Rain"]
SEED_VARS    = ["Hybrid","Traditional","Improved"]

# ── Header ────────────────────────────────────────────────
st.title("🌽 AgriInsight — Kenya")
st.markdown(
    "**47 Counties · XGBOOST · R² 0.80 · MAE 0.42 t/ha · SHAP Explained**"
)

# ── API health check ──────────────────────────────────────
try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    shap_ok = h.get("shap_available", False)
    base    = h.get("shap_base_yield")
    st.success(
        f"✅ API connected — R²: {h.get('r2','?')} | "
        f"MAE: {h.get('mae_tha','?')} t/ha | "
        f"SHAP: {'✓ enabled' if shap_ok else '✗ install shap'}"
        + (f" | Base yield: {base} t/ha" if base else "")
    )
except Exception:
    st.error(
        "⚠ API not running. Start it:\n"
    )
    st.stop()

st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🌾 Single Farm", "📂 Batch CSV Upload"])

# ══════════════════════════════════════════════════════════
# TAB 1 — SINGLE FARM
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict Yield for a Single Farm")

    with st.sidebar:
        st.header("Farm Profile")
        st.caption("Fields match training dataset columns exactly.")

        year    = st.number_input("year", 1990, 2030, 1990)
        county  = st.selectbox("county", COUNTIES,
                    index=COUNTIES.index("Trans Nzoia"))
        season  = st.selectbox("season", SEASONS)
        crop    = st.selectbox("crop", CROPS)
        area    = st.number_input("area_ha", 0.1, 1000.0, 14.56, 0.5)
        rain    = st.slider("rainfall_mm", 0, 3000, 1425)
        temp    = st.slider("avg_temp (°C)", 5, 40, 15)
        fert    = st.slider("fertiliser (kg/ha)", 0, 500, 97)
        seed    = st.selectbox("seed_variety", SEED_VARS)

        predict_btn = st.button(
            "🌱 Predict + Explain",
            type="primary",
            use_container_width=True
        )

    payload = {
        "year"        : int(year),
        "county"      : county,
        "season"      : season,
        "crop"        : crop,
        "area_ha"     : float(area),
        "rainfall_mm" : float(rain),
        "avg_temp"    : float(temp),
        "fertiliser"  : float(fert),
        "seed_variety": seed,
    }

    if predict_btn:
        with st.spinner("Predicting yield and generating SHAP..."):
            try:
                r    = requests.post(f"{API_URL}/predict/explain",
                                     json=payload, timeout=15)
                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.json()}")
                    st.stop()
                data = r.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        yield_tha = data["predicted_yield_tha"]
        total_t   = data["total_yield_tonnes"]
        vs_nat    = data["vs_national_pct"]

        # Metrics
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Predicted Yield",    f"{yield_tha} t/ha",
                  delta=f"{vs_nat:+.1f}% vs national")
        c2.metric("Total Production",   f"{total_t} tonnes")
        c3.metric("Total (kg)",         f"{data['total_yield_kg']:,.0f} kg")
        c4.metric("National Avg",       f"{data['national_avg_tha']} t/ha")
        c5.metric("Response",           f"{data['processing_ms']}ms")

        # Banner
        if vs_nat >= 30:
            st.success(
                f"🌟 Excellent — {vs_nat:+.1f}% above national average. "
                f"Confidence: {data['confidence_range_tha'][0]}–"
                f"{data['confidence_range_tha'][1]} t/ha"
            )
        elif vs_nat >= 0:
            st.info(
                f"✅ Above average — {vs_nat:+.1f}% above national. "
                f"Confidence: {data['confidence_range_tha'][0]}–"
                f"{data['confidence_range_tha'][1]} t/ha"
            )
        elif vs_nat >= -25:
            st.warning(f"⚠ Below average — {vs_nat:.1f}% below national average.")
        else:
            st.error(f"🔴 Significantly below average — {vs_nat:.1f}%. Review inputs.")

        st.progress(
            min(yield_tha / 7.0, 1.0),
            text=f"Yield: {yield_tha:.2f} t/ha"
        )
        st.divider()

        # Two columns
        col_shap, col_insights = st.columns([1.4, 1])

        with col_shap:
            st.subheader("🧠 SHAP Explanation")
            st.caption(data.get("shap_explanation", ""))

            top_drivers = data.get("shap_top_drivers", [])
            if top_drivers:
                features = [d["feature"]    for d in top_drivers]
                values   = [d["shap_value"] for d in top_drivers]
                colors   = ["#10B981" if v > 0 else "#EF4444" for v in values]

                fig, ax = plt.subplots(figsize=(8, 4))
                bars    = ax.barh(range(len(features)), values,
                                  color=colors, edgecolor="white", height=0.6)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features, fontsize=9)
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_xlabel("SHAP Value (t/ha impact on yield)", fontsize=9)
                ax.set_title(f"Top Yield Drivers — {county}, {crop}, {season}",
                             fontsize=11, fontweight="bold")
                ax.invert_yaxis()
                for bar, val in zip(bars, values):
                    xpos = val + 0.005 if val >= 0 else val - 0.005
                    ax.text(xpos, bar.get_y() + bar.get_height()/2,
                            f"{val:+.3f} t/ha", va="center",
                            ha="left" if val >= 0 else "right",
                            fontsize=8, color="#1F2937")
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                st.caption(
                    "🟢 Green = increases yield above base. "
                    "🔴 Red = reduces yield below base."
                )
                st.caption(data.get("shap_note", ""))

        with col_insights:
            st.subheader("🌾 Agronomic Insights")
            for insight in data.get("insights", []):
                if any(w in insight.lower() for w in
                       ["below","low","traditional","high temp","frost","short rain"]):
                    st.warning(f"▸ {insight}")
                else:
                    st.success(f"▸ {insight}")

            st.divider()
            st.subheader("📍 Farm Details")
            st.info(
                f"**County:** {county} | **Year:** {year}\n\n"
                f"**Crop:** {crop} ({seed} seed)\n\n"
                f"**Season:** {season} | **Area:** {area} ha\n\n"
                f"**Rainfall:** {rain} mm | **Temp:** {temp}°C\n\n"
                f"**Fertiliser:** {fert} kg/ha"
            )
    else:
        st.info("👈 Fill in the farm profile and click **Predict + Explain**.")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Counties",     "47 Kenya")
        c2.metric("Model R²",     "0.88")
        c3.metric("MAE",          "0.19 t/ha")
        c4.metric("National Avg", "1.84 t/ha")

# ══════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📂 Batch Score — Upload a CSV of Farms")
    st.markdown(
        "Upload a CSV with one farm per row matching training columns. "
        "Download results with predicted yields, totals, and national comparison."
    )

    # Template
    st.markdown("#### Step 1 — Download the template")
    template = pd.DataFrame([
        {"year":1990,"county":"Trans Nzoia","season":"Long Rain",
         "crop":"Maize","area_ha":14.56,"rainfall_mm":1425.8,
         "avg_temp":15.3,"fertiliser":96.7,"seed_variety":"Hybrid"},
        {"year":1990,"county":"Uasin Gishu","season":"Long Rain",
         "crop":"Wheat","area_ha":5.02,"rainfall_mm":915.5,
         "avg_temp":17.9,"fertiliser":96.7,"seed_variety":"Hybrid"},
        {"year":1990,"county":"Nakuru","season":"Short Rain",
         "crop":"Beans","area_ha":2.23,"rainfall_mm":827.6,
         "avg_temp":20.3,"fertiliser":87.5,"seed_variety":"Traditional"},
        {"year":1990,"county":"Nakuru","season":"Long Rain",
         "crop":"Potatoes","area_ha":6.45,"rainfall_mm":864.6,
         "avg_temp":19.6,"fertiliser":87.5,"seed_variety":"Improved"},
    ])

    st.download_button(
        "⬇️ Download CSV Template",
        template.to_csv(index=False),
        "agri_template.csv",
        "text/csv",
    )

    st.markdown("#### Step 2 — Upload your filled CSV")
    st.markdown("**Required columns:** `year, county, season, crop, area_ha, rainfall_mm, avg_temp, fertiliser, seed_variety`")

    uploaded = st.file_uploader("Upload farm CSV", type=["csv"],
                                 help="Max 200 farms per upload")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.success(f"✅ Loaded {len(df)} farms")

        required = ["year","county","season","crop","area_ha",
                    "rainfall_mm","avg_temp","fertiliser","seed_variety"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            st.error(
                f"Missing columns: {missing_cols}\n"
                "Download the template above to see the correct format."
            )
            st.stop()

        if len(df) > 200:
            st.warning("More than 200 rows — scoring first 200 only.")
            df = df.head(200)

        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Showing first 5 of {len(df)} rows")

        score_btn = st.button(
            f"🚀 Score All {len(df)} Farms",
            type="primary"
        )

        if score_btn:
            with st.spinner(f"Scoring {len(df)} farms via API..."):
                farms   = []
                errors  = []
                for i, row in df.iterrows():
                    try:
                        farms.append({
                            "year"        : int(row.get("year", 2024)),
                            "county"      : str(row.get("county","Trans Nzoia")),
                            "season"      : str(row.get("season","Long Rain")),
                            "crop"        : str(row.get("crop","Maize")),
                            "area_ha"     : float(row.get("area_ha", 1.0)),
                            "rainfall_mm" : float(row.get("rainfall_mm", 800)),
                            "avg_temp"    : float(row.get("avg_temp", 20)),
                            "fertiliser"  : float(row.get("fertiliser", 80)),
                            "seed_variety": str(row.get("seed_variety","Hybrid")),
                        })
                    except Exception as e:
                        errors.append(f"Row {i+1}: {e}")

                if errors:
                    st.warning(f"{len(errors)} rows had errors:\n"
                               + "\n".join(errors[:5]))

                try:
                    r = requests.post(
                        f"{API_URL}/predict/batch",
                        json=farms,
                        timeout=60
                    )
                    if r.status_code != 200:
                        st.error(f"Batch API error {r.status_code}: {r.text}")
                        st.stop()
                    results = r.json()
                except Exception as e:
                    st.error(f"API call failed: {e}")
                    st.stop()

            preds      = results["predictions"]
            df_results = df.copy()
            df_results["yield_tha"]    = [p["yield_tha"]    for p in preds]
            df_results["total_tonnes"] = [p["total_tonnes"] for p in preds]
            df_results["vs_national"]  = [p["vs_national"]  for p in preds]
            df_results = df_results.sort_values(
                "yield_tha", ascending=False
            ).reset_index(drop=True)

            # Summary metrics
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Farms Scored",     results["total_farms"])
            m2.metric("Avg Yield",        f"{results['avg_yield_tha']} t/ha")
            m3.metric("Total Production", f"{results['total_production_t']} t")
            m4.metric("vs National Avg",  f"{results['vs_national_pct']:+.1f}%")

            st.info(
                f"Average yield: **{results['avg_yield_tha']} t/ha** vs "
                f"national average of 1.84 t/ha. "
                f"Total production: **{results['total_production_t']} tonnes**. "
                f"Processed in {results['processing_ms']}ms."
            )

            # Charts
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))

            # Yield by crop
            crop_avg = df_results.groupby("crop")["yield_tha"].mean().sort_values()
            axes[0].barh(crop_avg.index, crop_avg.values,
                         color="#10B981", edgecolor="white")
            axes[0].axvline(1.84, color="#EF4444", linestyle="--",
                            lw=1.5, label="National avg")
            axes[0].set_title("Average Yield by Crop")
            axes[0].set_xlabel("Yield (t/ha)")
            axes[0].legend(fontsize=8)

            # Yield distribution
            axes[1].hist(df_results["yield_tha"].tolist(), bins=15,
                         color="#3B82F6", edgecolor="white")
            axes[1].axvline(1.84, color="#EF4444", linestyle="--",
                            lw=1.5, label="National avg (1.84)")
            axes[1].set_title("Yield Distribution")
            axes[1].set_xlabel("Yield (t/ha)")
            axes[1].set_ylabel("Farms")
            axes[1].legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

            # Results table
            st.markdown("#### Full Results — Sorted by Yield (Highest First)")

            def color_yield(val):
                if val >= 3.0:
                    return "background-color:#DCFCE7;color:#166534;font-weight:bold"
                elif val >= 1.84:
                    return "background-color:#FEF9C3;color:#713F12"
                else:
                    return "background-color:#FEE2E2;color:#7F1D1D"

            display_cols = [c for c in
                ["year","county","season","crop","area_ha",
                 "seed_variety","yield_tha","total_tonnes","vs_national"]
                if c in df_results.columns]

            styled = (
                df_results[display_cols]
                .style
                .map(color_yield, subset=["yield_tha"])
                .format({
                    "yield_tha"    : "{:.3f}",
                    "total_tonnes" : "{:.2f}",
                    "vs_national"  : "{:+.1f}%",
                })
            )
            st.dataframe(styled, use_container_width=True, height=400)

            # Download
            st.download_button(
                "⬇️ Download Scored Results CSV",
                df_results.to_csv(index=False),
                "agri_batch_results.csv",
                "text/csv",
                type="primary"
            )

            # Top 10
            st.markdown("#### 🏆 Top 10 Highest Yielding Farms")
            st.dataframe(
                df_results.head(10)[display_cols],
                use_container_width=True
            )

    else:
        st.info(
            "👆 Download the template, fill it with your farm data, "
            "then upload it here."
        )
       
