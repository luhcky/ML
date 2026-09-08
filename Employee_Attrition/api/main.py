from fastapi import FastAPI, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,validator
from typing import List, Dict, Optional
import numpy as np 
import time 
import pandas as pd 
import joblib 
import os 
import logging
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR,"models")

model = joblib.load(f'{MODEL_DIR}/attrition_pipe.pkl')
FEATURES = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
THRESHOLD = 0.5
try:
    import shap
    model = model.named_steps['xgb']
    EXPLAINER =shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
    ev = EXPLAINER.expected_value
    if hasattr(ev,"__len__") and len(ev) > 1:
            EXPECTED_VALUE =1 - float(ev[1]) 
    elif hasattr(ev,"__len__"):
            EXPECTED_VALUE =1 - float (ev[0])
    else: EXPECTED_VALUE = float(ev)
    logger.info(f'SHAP explainer loaded.Base value={EXPECTED_VALUE:.4f}')
except ImportError:
    SHAP_AVAILABLE =False
    logger.warning("SHAP not installed - /predict/explain will be unavailable")
logger.info(f'Attrition model loaded.Features={len(FEATURES)}')


app = FastAPI(title='Employee Attrition Prediction Model',description=('XGBoost Model with SHAP explainability'),
version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_methods=['*'],    
)

class EmployeeInput(BaseModel):
        Age               : int   = Field(..., ge=18, le=60,      example=28)
        MonthlyIncome     : float = Field(..., ge=1000, le=20000, example=3500)
        OverTime          : str   = Field(...,                    example="Yes")
        JobSatisfaction   : int   = Field(..., ge=1, le=4,        example=1)
        StockOptionLevel  : int   = Field(..., ge=0, le=3,        example=0)
        JobLevel          : int   = Field(2,  ge=1, le=5,         example=1)
        TotalWorkingYears : int   = Field(5,  ge=0, le=40,        example=3)
        YearsAtCompany    : int   = Field(3,  ge=0, le=40,        example=1)
        YearsInCurrentRole: int   = Field(2,  ge=0, le=18,        example=1)
        YearsSinceLastPromotion: int = Field(1, ge=0, le=15,      example=1)
        YearsWithCurrManager: int = Field(2,  ge=0, le=17,        example=0)
        NumCompaniesWorked: int   = Field(2,  ge=0, le=9,         example=3)
        BusinessTravel    : str   = Field("Travel_Rarely",        example="Travel_Frequently")
        MaritalStatus     : str   = Field("Single",               example="Single")
        Department        : str   = Field("Research & Development", example="Sales")
        JobRole           : str   = Field("Research Scientist",   example="Sales Representative")
        Gender            : str   = Field("Male",                 example="Male")
        EnvironmentSatisfaction : int = Field(3, ge=1, le=4,      example=1)
        RelationshipSatisfaction: int = Field(3, ge=1, le=4,      example=2)
        WorkLifeBalance   : int   = Field(3, ge=1, le=4,          example=1)
        Education         : int   = Field(3, ge=1, le=5,          example=3)
        JobInvolvement    : int   = Field(3, ge=1, le=4,          example=2)
        PerformanceRating : int   = Field(3, ge=1, le=4,          example=3)
        PercentSalaryHike : int   = Field(13, ge=11, le=25,       example=11)
        TrainingTimesLastYear: int = Field(3, ge=0, le=6,         example=1)
        DistanceFromHome  : int   = Field(5,  ge=1, le=29,        example=20)
        DailyRate         : int   = Field(800, ge=102, le=1499,   example=400)
        HourlyRate        : int   = Field(65,  ge=30, le=100,     example=40)
        MonthlyRate       : int   = Field(14000, ge=2094, le=26999, example=8000)
    
        @validator("OverTime")
        def v_overtime(cls, v):
            if v not in ("Yes", "No"):
                raise ValueError("OverTime must be 'Yes' or 'No'")
            return v
    
        @validator("BusinessTravel")
        def v_travel(cls, v):
            valid = ("Non-Travel", "Travel_Rarely", "Travel_Frequently")
            if v not in valid: raise ValueError(f"BusinessTravel must be one of {valid}")
            return v
    
        @validator("MaritalStatus")
        def v_marital(cls, v):
            if v not in ("Single", "Married", "Divorced"):
                raise ValueError("MaritalStatus must be Single, Married, or Divorced")
            return v
    
        @validator("Department")
        def v_dept(cls, v):
            valid = ("Research & Development", "Sales", "Human Resources")
            if v not in valid: raise ValueError(f"Department must be one of {valid}")
            return v
    
        @validator("Gender")
        def v_gender(cls, v):
            if v not in ("Male", "Female"):
                raise ValueError("Gender must be Male or Female")
            return v
    
def build_features(emp:EmployeeInput) -> pd.DataFrame:
    travel_enc ={"Non-Travel":0,"Travel_Rarely":1,"Travel_Frequently":2}
    row = {f: 0 for f in FEATURES}
    row.update({
             "Age"                      : emp.Age,
                "BusinessTravel"           : travel_enc[emp.BusinessTravel],
                "DailyRate"                : emp.DailyRate,
                "DistanceFromHome"         : emp.DistanceFromHome,
                "Education"                : emp.Education,
                "EnvironmentSatisfaction"  : emp.EnvironmentSatisfaction,
                "Gender"                   : 1 if emp.Gender == "Male" else 0,
                "HourlyRate"               : emp.HourlyRate,
                "JobInvolvement"           : emp.JobInvolvement,
                "JobLevel"                 : emp.JobLevel,
                "JobSatisfaction"          : emp.JobSatisfaction,
                "MonthlyIncome"            : emp.MonthlyIncome,
                "MonthlyRate"              : emp.MonthlyRate,
                "NumCompaniesWorked"       : emp.NumCompaniesWorked,
                "OverTime"                 : 1 if emp.OverTime == "Yes" else 0,
                "PercentSalaryHike"        : emp.PercentSalaryHike,
                "PerformanceRating"        : emp.PerformanceRating,
                "RelationshipSatisfaction" : emp.RelationshipSatisfaction,
                "StockOptionLevel"         : emp.StockOptionLevel,
                "TotalWorkingYears"        : emp.TotalWorkingYears,
                "TrainingTimesLastYear"    : emp.TrainingTimesLastYear,
                "WorkLifeBalance"          : emp.WorkLifeBalance,
                "YearsAtCompany"           : emp.YearsAtCompany,
                "YearsInCurrentRole"       : emp.YearsInCurrentRole,
                "YearsSinceLastPromotion"  : emp.YearsSinceLastPromotion,
                "YearsWithCurrManager"     : emp.YearsWithCurrManager,
            
                "LogMonthlyIncome"  : np.log1p(emp.MonthlyIncome),
                "YearsPerCompany"   : emp.TotalWorkingYears / max(emp.NumCompaniesWorked, 1),
                "PromotionLag"      : emp.YearsSinceLastPromotion - emp.YearsInCurrentRole,
                "IncomePerYear"     : emp.MonthlyIncome / (emp.TotalWorkingYears + 1),
                "HighRiskProfile"   : int(emp.Age < 32 and emp.MaritalStatus == "Single"
                                          and emp.BusinessTravel == "Travel_Frequently"),
     })
    for col, val in [
        ("Department_Research & Development", emp.Department == "Research & Development"),
        ("Department_Sales",                  emp.Department == "Sales"),
        ("MaritalStatus_Married",             emp.MaritalStatus == "Married"),
        ("MaritalStatus_Single",              emp.MaritalStatus == "Single"),
    ]:
        if col in row: row[col] = int(val)
    role_key = f"JobRole_{emp.JobRole}"
    if role_key in row: row[role_key] = 1
    return pd.DataFrame([row])[FEATURES]

def get_risk_tier(prob: float) -> str:
    if   prob >= 0.70:        return "CRITICAL"
    elif prob >= THRESHOLD:   return "HIGH"
    elif prob >= 0.4: return "MEDIUM"
    else:                     return "LOW"
def get_recommendation(tier: str) -> str:
    return {
        "CRITICAL": "Immediate action. Schedule 1:1 this week. Review compensation and explore internal mobility.",
        "HIGH"    : "Proactive check-in. Review workload, overtime hours, and promotion timeline.",
        "MEDIUM"  : "Include in quarterly career development conversation. Monitor over 60 days.",
        "LOW"     : "Routine engagement. Include in standard pulse survey.",
    }[tier]

def get_risk_factors(emp: EmployeeInput) -> List[str]:
    factors = []
    if emp.OverTime == "Yes":                       factors.append("Works overtime — 3x higher attrition risk")
    if emp.MaritalStatus == "Single":               factors.append("Single — more mobile, higher exit rate")
    if emp.BusinessTravel == "Travel_Frequently":   factors.append("Frequent travel — burnout risk")
    if emp.JobSatisfaction <= 2:                    factors.append(f"Low job satisfaction ({emp.JobSatisfaction}/4)")
    if emp.JobLevel == 1:                           factors.append("Entry-level — highest attrition tier (26%)")
    if emp.StockOptionLevel == 0:                   factors.append("No stock options — less financial retention")
    if emp.Age < 28:                                factors.append("Age under 28 — high career exploration phase")
    if emp.YearsSinceLastPromotion >= 3:            factors.append("No promotion in 3+ years — career stagnation")
    if emp.MonthlyIncome < 3000:                    factors.append(f"Below-median income (${emp.MonthlyIncome:,}/month)")
    if emp.NumCompaniesWorked >= 4:                 factors.append(f"Worked {emp.NumCompaniesWorked} companies — job-hopping history")
    if emp.EnvironmentSatisfaction <= 2:            factors.append(f"Low environment satisfaction ({emp.EnvironmentSatisfaction}/4)")
    if emp.WorkLifeBalance <= 2:                    factors.append(f"Poor work-life balance ({emp.WorkLifeBalance}/4)")
    return factors or ["No strong risk signals detected"]

def get_protective_factors(emp: EmployeeInput) -> List[str]:
    factors = []
    if emp.StockOptionLevel >= 2:                   factors.append(f"Stock options level {emp.StockOptionLevel} — strong retention")
    if emp.YearsAtCompany >= 8:                     factors.append(f"{emp.YearsAtCompany} years tenure — high loyalty")
    if emp.JobSatisfaction >= 3:                    factors.append(f"Good job satisfaction ({emp.JobSatisfaction}/4)")
    if emp.WorkLifeBalance >= 3:                    factors.append(f"Good work-life balance ({emp.WorkLifeBalance}/4)")
    if emp.MaritalStatus == "Married":              factors.append("Married — lower mobility")
    if emp.OverTime == "No" and emp.JobLevel >= 3:  factors.append("Senior + no overtime — low burnout risk")
    return factors or ["No strong protective factors detected"]

#endpoints
@app.get('/')
def root():
    return{
        'API': "Employee Attrition Prediction API",
        'Version': '1.0.0',
        'SHAP':'SHAP_AVAILABLE',
        'docs':'/docs',}
    
@app.get('/health')
def health():
    return{
        'status':'OK',
        'Model':'XGBoost',
        'Feature_count':len(FEATURES),
            }
@app.post('/predict')
def predict(employee: EmployeeInput):
    start = time.time()
    try:
        X = build_features(employee)
        prob = float(model.predict_proba(X)[0][1])
        tier = get_risk_tier(prob)
        return{
            "attrition_probability" : round(prob, 4),
                        "risk_score_pct"        : round(prob * 100, 2),
                        "risk_tier"             : tier,
                        "will_leave"            : prob >= THRESHOLD,
                        "threshold_used"        : THRESHOLD,
                        "recommendation"        : get_recommendation(tier),
                        "risk_factors"          : get_risk_factors(employee),
                        "protective_factors"    : get_protective_factors(employee),
                        "processing_ms"         : round((time.time() - start) * 1000, 2),
                    }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/predict/explain')
def predict_explain(employee: EmployeeInput):
    if not SHAP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='SHAP not available.Run:pip install shap'
        )         
    start = time.time()
    try:
        X = build_features(employee)
        X_df =pd.DataFrame(X, columns=FEATURES)
        
        prob = 1 - float(model.predict_proba(X)[0][1])
        tier = get_risk_tier(prob)
        
        shap_vals = EXPLAINER.shap_values(X_df)
        sv = shap_vals[1] if isinstance(shap_vals,list) else shap_vals
        sv_row = -sv[0]
        
        shap_dict = {
            feat: round(float(val),6)
            for feat,val in zip(FEATURES,sv_row)
        }
        sorted_shap = sorted(shap_dict.items(),key=lambda x:abs(x[1]),reverse=True)
        top_drivers = [
            {
                'feature':feat,
                'shap_value':val,
                'direction':'increases attrition risk' if val > 0 else 'decrease attrition risk',
                'magnitude':'high' if abs(val) > 0.1 else 'medium' if abs(val) > 0.3 else 'low',               
            }
            for feat, val in sorted_shap[:10]
        ]
        top3 = sorted_shap[:3]
        explanation_parts = []
        for feat, val in top3:
                    direction = "increased" if val > 0 else "decreased"
                    explanation_parts.append(
                        f"{feat} {direction} attrition probability by {abs(val)*100:.1f} percentage points"
                    )
        shap_explanation = ". ".join(explanation_parts) + "."
        
        return {
                    # Standard prediction fields
                    "attrition_probability" : round(prob, 4),
                    "risk_score_pct"        : round(prob * 100, 2),
                    "risk_tier"             : tier,
                    "will_leave"            : prob >= THRESHOLD,
                    "threshold_used"        : THRESHOLD,
                    "recommendation"        : get_recommendation(tier),
                    "risk_factors"          : get_risk_factors(employee),
                    "protective_factors"    : get_protective_factors(employee),
                    # SHAP explanation fields
                    "shap_base_value"       : round(EXPECTED_VALUE, 4),
                    "shap_values"           : shap_dict,
                    "shap_top_drivers"      : top_drivers,
                    "shap_explanation"      : shap_explanation,
                    "shap_note"             : (
                        f"Base probability across all employees: {EXPECTED_VALUE*100:.1f}%. "
                        f"SHAP values show how much each feature moved this employee's "
                        f"probability from that base to {prob*100:.1f}%."
                    ),
                    "processing_ms"         : round((time.time() - start) * 1000, 2),
                }
    except Exception as e:
        logger.error(f"SHAP explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(employees: List[EmployeeInput]):
            if len(employees) > 1000:
                raise HTTPException(status_code=422, detail="Max 1000 employees per batch")
            start = time.time()
            results, flagged = [], 0
            for emp in employees:
                X_s  = scaler.transform(build_features(emp))
                prob = 1 - float(model.predict_proba(X_s)[0][1])
                tier = get_risk_tier(prob)
                if prob >= THRESHOLD: flagged += 1
                results.append({
                    "attrition_probability": round(prob, 4),
                    "risk_tier"            : tier,
                    "will_leave"           : prob >= THRESHOLD,
                    "recommendation"       : get_recommendation(tier),
                })
            return {
                "total_employees"   : len(results),
                "flagged_count"     : flagged,
                "attrition_rate_pct": round(flagged / len(results) * 100, 1),
                "tier_breakdown"    : {
                    tier: sum(1 for r in results if r["risk_tier"] == tier)
                    for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                },
                "predictions"       : results,
                "processing_ms"     : round((time.time() - start) * 1000, 2),
            }
        
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled error on {request.url}: {exc}")
            return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})
        
