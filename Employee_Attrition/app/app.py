
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
)

API_URL = "https://employee-attrition-fvgg.onrender.com"

# ── Header ────────────────────────────────────────────────
st.title("👥 Employee Attrition Prediction")
st.markdown("**IBM HR Analytics · XGBoost · SHAP Explained · Batch Scoring**")

# ── API health check ──────────────────────────────────────
try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    shap_ok = h.get("shap_available", False)
    st.success(
        f"✅ API connected — "
    
        f"SHAP: {'✓ enabled' if shap_ok else '✗ install shap'}"
    )
except Exception:
    st.error("⚠ API not running.")
    st.stop()

st.divider()

# ── Tabs — Single vs Batch ────────────────────────────────
tab1, tab2 = st.tabs(["👤 Single Employee", "📂 Batch CSV Upload"])

# ══════════════════════════════════════════════════════════
# TAB 1 — SINGLE EMPLOYEE
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Score a Single Employee")

    # Sidebar inputs
    with st.sidebar:
        st.header("Employee Profile")

        with st.expander("Personal", expanded=True):
            age         = st.slider("Age", 18, 60, 28)
            gender      = st.selectbox("Gender", ["Male", "Female"])
            marital     = st.selectbox("Marital Status",
                            ["Single", "Married", "Divorced"])
            distance    = st.slider("Distance From Home (miles)", 1, 29, 10)

        with st.expander("Job", expanded=True):
            dept        = st.selectbox("Department",
                            ["Research & Development","Sales","Human Resources"])
            job_role    = st.selectbox("Job Role", [
                "Sales Executive","Research Scientist","Laboratory Technician",
                "Manufacturing Director","Healthcare Representative","Manager",
                "Sales Representative","Research Director","Human Resources"])
            job_level   = st.selectbox("Job Level (1=Entry, 5=Senior)",
                            [1,2,3,4,5])
            job_sat     = st.selectbox("Job Satisfaction",
                            [1,2,3,4], index=1,
                            format_func=lambda x:{1:"Low",2:"Medium",
                                                  3:"High",4:"Very High"}[x])
            overtime    = st.selectbox("Works Overtime?", ["No","Yes"])
            travel      = st.selectbox("Business Travel",
                            ["Non-Travel","Travel_Rarely","Travel_Frequently"])

        with st.expander("Compensation", expanded=True):
            income      = st.number_input("Monthly Income ($)",
                            1000, 20000, 3500, 500)
            stock       = st.selectbox("Stock Option Level", [0,1,2,3])
            hike        = st.slider("Last Salary Hike (%)", 11, 25, 13)

        with st.expander("Experience", expanded=True):
            total_yrs   = st.slider("Total Working Years", 0, 40, 5)
            yrs_company = st.slider("Years at Company", 0, 40, 3)
            yrs_role    = st.slider("Years in Current Role", 0, 18, 2)
            yrs_promo   = st.slider("Years Since Last Promotion", 0, 15, 1)
            yrs_mgr     = st.slider("Years With Manager", 0, 17, 2)
            num_co      = st.slider("Companies Worked At", 0, 9, 2)
            training    = st.slider("Training Sessions Last Year", 0, 6, 3)
            env_sat     = st.selectbox("Environment Satisfaction",
                            [1,2,3,4], index=2,
                            format_func=lambda x:{1:"Low",2:"Medium",
                                                  3:"High",4:"Very High"}[x])
            rel_sat     = st.selectbox("Relationship Satisfaction",
                            [1,2,3,4], index=2,
                            format_func=lambda x:{1:"Low",2:"Medium",
                                                  3:"High",4:"Very High"}[x])
            wlb         = st.selectbox("Work-Life Balance",
                            [1,2,3,4], index=2,
                            format_func=lambda x:{1:"Bad",2:"Good",
                                                  3:"Better",4:"Best"}[x])

        predict_btn = st.button(
            "🔍 Predict + Explain",
            type="primary",
            use_container_width=True
        )

    # Build payload
    payload = {
        "Age": age, "MonthlyIncome": income, "OverTime": overtime,
        "JobSatisfaction": job_sat, "StockOptionLevel": stock,
        "JobLevel": job_level, "TotalWorkingYears": total_yrs,
        "YearsAtCompany": yrs_company, "YearsInCurrentRole": yrs_role,
        "YearsSinceLastPromotion": yrs_promo, "YearsWithCurrManager": yrs_mgr,
        "NumCompaniesWorked": num_co, "BusinessTravel": travel,
        "MaritalStatus": marital, "Department": dept, "JobRole": job_role,
        "Gender": gender, "EnvironmentSatisfaction": env_sat,
        "RelationshipSatisfaction": rel_sat, "WorkLifeBalance": wlb,
        "Education": 3, "JobInvolvement": 3, "PerformanceRating": 3,
        "PercentSalaryHike": hike, "TrainingTimesLastYear": training,
        "DistanceFromHome": distance, "DailyRate": 800,
        "HourlyRate": 65, "MonthlyRate": 14000,
    }

    if predict_btn:
        with st.spinner("Scoring and generating SHAP explanation..."):
            try:
                r = requests.post(f"{API_URL}/predict/explain",
                                    json=payload, timeout=15)
                data = r.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        prob = data["attrition_probability"]
        tier = data["risk_tier"]

        # Metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Attrition Probability", f"{prob*100:.1f}%", delta=f"{(prob-0.161)*100:+.1f}% vs baseline")
        c2.metric("Risk Tier",  tier)
        c3.metric("Decision",   "⚠ Will Leave" if data["will_leave"] else "✅ Will Stay")
        c4.metric("Response",   f"{data['processing_ms']}ms")

        if tier == "CRITICAL":
            st.error(f"🚨 CRITICAL — {data['recommendation']}")
        elif tier == "HIGH":
            st.error(f"⚠ HIGH — {data['recommendation']}")
        elif tier == "MEDIUM":
            st.warning(f"🟡 MEDIUM — {data['recommendation']}")
        else:
            st.success(f"✅ LOW — {data['recommendation']}")

        st.progress(min(prob, 1.0),
                    text=f"Attrition score: {prob*100:.1f}%")
        st.divider()

        col_shap, col_signals = st.columns([1.4, 1])
        with col_shap:
            st.subheader("🧠 SHAP Explanation")
            st.caption(data.get("shap_explanation", ""))
            top_drivers = data.get("shap_top_drivers", [])
            if top_drivers:
                features = [d["feature"]    for d in top_drivers]
                values   = [d["shap_value"] for d in top_drivers]
                NAVY, GOLD, TEAL, GRID, TEXT, MUTED = (
                    "#0B1F3A", "#F0A868", "#5EEAD4", "#24406B", "#F5F1E8", "#9FB0C9"
                )
                fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
                fig.patch.set_facecolor(NAVY)
                ax.set_facecolor(NAVY)

                y_pos = range(len(features))
                colors = [GOLD if v > 0 else TEAL for v in values]

               # rounded "pill" bars — thick round-capped lines instead of plain barh
                for y, v, c in zip(y_pos, values, colors):
                    ax.plot([0, v], [y, y], color=c, linewidth=14,
                            solid_capstyle="round", zorder=3, alpha=0.95)

                ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)

                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(features, fontsize=10.5, color=TEXT)
                ax.invert_yaxis()

                ax.set_xlabel("SHAP value  →  impact on attrition probability",
                              fontsize=9.5, color=MUTED, labelpad=10)
                ax.set_title("Top SHAP Drivers", fontsize=17, color=TEXT,
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

                xmin, xmax = min(values) - 0.12, max(values) + 0.12
                ax.set_xlim(xmin, xmax)

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                st.caption("🟠 Gold = increases attrition risk.  🟦 Teal = decreases risk.")
                st.caption(data.get("shap_note", ""))
 
        with col_signals:
            st.subheader("⚡ Risk Signals")
            for sig in data.get("risk_factors", []):
                st.warning(f"▸ {sig}")
            st.subheader("🛡️ Protective Factors")
            for prot in data.get("protective_factors", []):
                st.success(f"▸ {prot}")
            st.divider()
            st.subheader("📋 Recommended Action")
            if tier in ("CRITICAL","HIGH"):
                st.error("- Schedule 1:1 this week\n"
                         "- Review compensation vs market\n"
                         "- Explore internal mobility options")
            elif tier == "MEDIUM":
                st.warning("- Quarterly career development conversation\n"
                           "- Review workload and overtime\n"
                           "- Discuss promotion timeline")
            else:
                st.success("- Routine engagement\n"
                           "- Include in standard pulse survey")
    else:
        st.info("👈 Fill in the employee profile and click **Predict + Explain**.")

# ══════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📂 Batch Score — Upload a CSV of Employees")
    st.markdown(
        "Upload a CSV with one employee per row. "
        "The API scores all employees and returns risk tiers, "
        "probabilities, and recommendations. Download results as CSV."
    )

    # ── Template download ─────────────────────────────────
    st.markdown("#### Step 1 — Download the template")
    template = pd.DataFrame([{
        "Age": 28, "MonthlyIncome": 3500, "OverTime": "Yes",
        "JobSatisfaction": 2, "StockOptionLevel": 0,
        "JobLevel": 1, "TotalWorkingYears": 3,
        "YearsAtCompany": 1, "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 0,
        "NumCompaniesWorked": 3, "BusinessTravel": "Travel_Frequently",
        "MaritalStatus": "Single", "Department": "Sales",
        "JobRole": "Sales Representative", "Gender": "Male",
        "EnvironmentSatisfaction": 2, "RelationshipSatisfaction": 2,
        "WorkLifeBalance": 2, "Education": 3, "JobInvolvement": 2,
        "PerformanceRating": 3, "PercentSalaryHike": 11,
        "TrainingTimesLastYear": 1, "DistanceFromHome": 15,
        "DailyRate": 500, "HourlyRate": 45, "MonthlyRate": 9000,
    },{
        "Age": 45, "MonthlyIncome": 12000, "OverTime": "No",
        "JobSatisfaction": 4, "StockOptionLevel": 3,
        "JobLevel": 4, "TotalWorkingYears": 20,
        "YearsAtCompany": 15, "YearsInCurrentRole": 7,
        "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 8,
        "NumCompaniesWorked": 2, "BusinessTravel": "Non-Travel",
        "MaritalStatus": "Married", "Department": "Research & Development",
        "JobRole": "Research Director", "Gender": "Female",
        "EnvironmentSatisfaction": 4, "RelationshipSatisfaction": 4,
        "WorkLifeBalance": 4, "Education": 5, "JobInvolvement": 4,
        "PerformanceRating": 3, "PercentSalaryHike": 20,
        "TrainingTimesLastYear": 5, "DistanceFromHome": 2,
        "DailyRate": 1200, "HourlyRate": 90, "MonthlyRate": 24000,
    }])

    st.download_button(
        "⬇️ Download CSV Template",
        template.to_csv(index=False),
        "attrition_template.csv",
        "text/csv",
        help="Fill this template and upload it below"
    )

    # ── Upload and score ──────────────────────────────────
    st.markdown("#### Step 2 — Upload your filled CSV")
    uploaded = st.file_uploader(
        "Upload employee CSV", type=["csv"],
        help="Max 500 employees per upload"
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.success(f"✅ Loaded {len(df)} employees")

        # Validate required columns
        required = ["Age", "MonthlyIncome", "OverTime",
                    "JobSatisfaction", "StockOptionLevel"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            st.error(
                f"Missing required columns: {missing_cols}\n"
                "Download the template above to see the correct format."
            )
            st.stop()

        if len(df) > 500:
            st.warning("More than 500 rows — only the first 500 will be scored.")
            df = df.head(500)

        st.markdown("#### Step 3 — Score all employees")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Showing first 5 of {len(df)} rows")

        score_btn = st.button(
            f"🚀 Score All {len(df)} Employees",
            type="primary"
        )

        if score_btn:
            with st.spinner(f"Scoring {len(df)} employees via API..."):

                # Fill missing optional columns with defaults
                defaults = {
                    "JobLevel": 2, "TotalWorkingYears": 5,
                    "YearsAtCompany": 3, "YearsInCurrentRole": 2,
                    "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2,
                    "NumCompaniesWorked": 2, "BusinessTravel": "Travel_Rarely",
                    "MaritalStatus": "Single",
                    "Department": "Research & Development",
                    "JobRole": "Research Scientist", "Gender": "Male",
                    "EnvironmentSatisfaction": 3, "RelationshipSatisfaction": 3,
                    "WorkLifeBalance": 3, "Education": 3, "JobInvolvement": 3,
                    "PerformanceRating": 3, "PercentSalaryHike": 13,
                    "TrainingTimesLastYear": 3, "DistanceFromHome": 5,
                    "DailyRate": 800, "HourlyRate": 65, "MonthlyRate": 14000,
                }
                for col, default in defaults.items():
                    if col not in df.columns:
                        df[col] = default

                # Build employees list for batch API
                employees = []
                errors    = []
                for i, row in df.iterrows():
                    try:
                        employees.append({
                            "Age"                      : int(row.get("Age", 30)),
                            "MonthlyIncome"            : float(row.get("MonthlyIncome", 5000)),
                            "OverTime"                 : str(row.get("OverTime", "No")),
                            "JobSatisfaction"          : int(row.get("JobSatisfaction", 3)),
                            "StockOptionLevel"         : int(row.get("StockOptionLevel", 0)),
                            "JobLevel"                 : int(row.get("JobLevel", 2)),
                            "TotalWorkingYears"        : int(row.get("TotalWorkingYears", 5)),
                            "YearsAtCompany"           : int(row.get("YearsAtCompany", 3)),
                            "YearsInCurrentRole"       : int(row.get("YearsInCurrentRole", 2)),
                            "YearsSinceLastPromotion"  : int(row.get("YearsSinceLastPromotion", 1)),
                            "YearsWithCurrManager"     : int(row.get("YearsWithCurrManager", 2)),
                            "NumCompaniesWorked"       : int(row.get("NumCompaniesWorked", 2)),
                            "BusinessTravel"           : str(row.get("BusinessTravel","Travel_Rarely")),
                            "MaritalStatus"            : str(row.get("MaritalStatus", "Single")),
                            "Department"               : str(row.get("Department","Research & Development")),
                            "JobRole"                  : str(row.get("JobRole","Research Scientist")),
                            "Gender"                   : str(row.get("Gender", "Male")),
                            "EnvironmentSatisfaction"  : int(row.get("EnvironmentSatisfaction", 3)),
                            "RelationshipSatisfaction" : int(row.get("RelationshipSatisfaction", 3)),
                            "WorkLifeBalance"          : int(row.get("WorkLifeBalance", 3)),
                            "Education"                : int(row.get("Education", 3)),
                            "JobInvolvement"           : int(row.get("JobInvolvement", 3)),
                            "PerformanceRating"        : int(row.get("PerformanceRating", 3)),
                            "PercentSalaryHike"        : int(row.get("PercentSalaryHike", 13)),
                            "TrainingTimesLastYear"    : int(row.get("TrainingTimesLastYear", 3)),
                            "DistanceFromHome"         : int(row.get("DistanceFromHome", 5)),
                            "DailyRate"                : int(row.get("DailyRate", 800)),
                            "HourlyRate"               : int(row.get("HourlyRate", 65)),
                            "MonthlyRate"              : int(row.get("MonthlyRate", 14000)),
                        })
                    except Exception as e:
                        errors.append(f"Row {i+1}: {e}")

                if errors:
                    st.warning(f"{len(errors)} rows had errors and were skipped:\n"
                               + "\n".join(errors[:5]))

                try:
                    r = requests.post(
                        f"{API_URL}/predict/batch",
                        json=employees,
                        timeout=60
                    )
                    if r.status_code != 200:
                        st.error(f"Batch API error {r.status_code}: {r.text}")
                        st.stop()
                    results = r.json()
                except Exception as e:
                    st.error(f"API call failed: {e}")
                    st.stop()

            # ── Results summary ───────────────────────────
            preds = results["predictions"]
            df_results = df.copy()
            df_results["attrition_probability"] = [
                round(p["attrition_probability"]*100, 1) for p in preds]
            df_results["risk_tier"]  = [p["risk_tier"]  for p in preds]
            df_results["will_leave"] = [p["will_leave"]  for p in preds]
            df_results["recommendation"] = [
                p["recommendation"] for p in preds]
            df_results = df_results.sort_values(
                "attrition_probability", ascending=False
            ).reset_index(drop=True)

            # Summary metrics
            tier_counts = results.get("tier_breakdown", {})
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Total Scored",  results["total_employees"])
            m2.metric("🔴 CRITICAL",   tier_counts.get("CRITICAL", 0))
            m3.metric("🟠 HIGH",       tier_counts.get("HIGH", 0))
            m4.metric("🟡 MEDIUM",     tier_counts.get("MEDIUM", 0))
            m5.metric("🟢 LOW",        tier_counts.get("LOW", 0))

            flagged = results["flagged_count"]
            total   = results["total_employees"]
            st.info(
                f"**{flagged} of {total} employees** flagged for HR review "
                f"({results['attrition_rate_pct']}% batch attrition rate). "
                f"Processed in {results['processing_ms']}ms."
            )

            # ── Risk tier chart ───────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))

            tiers  = ["LOW","MEDIUM","HIGH","CRITICAL"]
            counts = [tier_counts.get(t, 0) for t in tiers]
            colors = ["#10B981","#F59E0B","#EF4444","#7F1D1D"]
            axes[0].bar(tiers, counts, color=colors, edgecolor="white")
            axes[0].set_title("Employees by Risk Tier")
            axes[0].set_ylabel("Count")
            for i, (t, c) in enumerate(zip(tiers, counts)):
                if c > 0:
                    axes[0].text(i, c + 0.3, str(c),
                                 ha="center", fontweight="bold")

            probs_list = df_results["attrition_probability"].tolist()
            axes[1].hist(probs_list, bins=20,
                         color="#3B82F6", edgecolor="white")
            axes[1].axvline(
                results.get("attrition_rate_pct", 16.1),
                color="#EF4444", linestyle="--", lw=1.5,
                label="Batch avg"
            )
            axes[1].set_title("Attrition Probability Distribution")
            axes[1].set_xlabel("Attrition Probability (%)")
            axes[1].set_ylabel("Employees")
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

            # ── Full results table ────────────────────────
            st.markdown("#### Full Results — Sorted by Risk (Highest First)")

            def color_tier(val):
                colors_map = {
                    "CRITICAL": "background-color: #FEE2E2; color: #7F1D1D; font-weight: bold",
                    "HIGH"    : "background-color: #FEF3C7; color: #78350F; font-weight: bold",
                    "MEDIUM"  : "background-color: #FEF9C3; color: #713F12",
                    "LOW"     : "background-color: #DCFCE7; color: #166534",
                }
                return colors_map.get(val, "")

            display_cols = (
                [c for c in ["Age","Department","JobRole","JobLevel",
                              "MonthlyIncome","OverTime","MaritalStatus"]
                 if c in df_results.columns] +
                ["attrition_probability","risk_tier","recommendation"]
            )

            styled = (
                df_results[display_cols]
                .style
                .map(color_tier, subset=["risk_tier"])
                .format({"attrition_probability": "{:.1f}%"})
            )
            st.dataframe(styled, use_container_width=True, height=400)

            # ── Download results ──────────────────────────
            st.markdown("#### Download Results")
            csv_out = df_results.to_csv(index=False)
            st.download_button(
                "⬇️ Download Scored Results CSV",
                csv_out,
                "attrition_batch_results.csv",
                "text/csv",
                type="primary"
            )

            # ── Top 10 highest risk ───────────────────────
            st.markdown("#### 🚨 Top 10 Highest Risk Employees")
            top10 = df_results.head(10)[display_cols]
            st.dataframe(top10, use_container_width=True)

    else:
        # Template preview
        st.info(
            "👆 Download the template, fill it with your employee data, "
            "then upload it here. Missing optional columns will use "
            "sensible defaults automatically."
        )
        st.markdown("**Required columns:**")
        st.code(
            "Age, MonthlyIncome, OverTime (Yes/No), "
            "JobSatisfaction (1-4), StockOptionLevel (0-3)"
        )
        st.markdown("**Optional columns** *(defaults applied if missing):*")
        st.code(
            "JobLevel, TotalWorkingYears, YearsAtCompany, BusinessTravel,\n"
            "MaritalStatus, Department, JobRole, Gender,\n"
            "EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance"
        )
