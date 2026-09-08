from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List
import numpy as np
import pandas as pd
import joblib
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── LOAD ARTIFACTS AT STARTUP ─────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

pipeline  = joblib.load(f"{MODEL_DIR}/mpesa_pipeline.pkl")
FEATURES  = joblib.load(f"{MODEL_DIR}/mpesa_feature_names.pkl")
scaler = pipeline.named_steps['scaler']

# Extract XGBoost model from pipeline for SHAP
try:
    xgb_model = pipeline.named_steps["model"]
except Exception:
    xgb_model = pipeline.steps[-1][1]

# SHAP explainer loaded once at startup
try:
    import shap
    EXPLAINER      = shap.TreeExplainer(xgb_model)
    SHAP_AVAILABLE = True
    ev = EXPLAINER.expected_value
    if hasattr(ev,"__len__") and len(ev) > 1:
        EXPECTED_VALUE = float(ev[1]) 
    elif hasattr(ev,"__len__"):
        EXPECTED_VALUE = float (ev[0])
    else:
        EXPECTED_VALUE = float(ev)
    logger.info(f"SHAP loaded. Base value={EXPECTED_VALUE:.6f}")
except ImportError:
    SHAP_AVAILABLE = False
    EXPECTED_VALUE = None
    logger.warning("shap not installed — pip install shap")

logger.info(f"M-PESA pipeline loaded. Threshold=0.5, Features={len(FEATURES)}")
logger.info(f"Feature names: {FEATURES}")

KENYA_COUNTIES = [
    "Nairobi","Mombasa","Kisumu","Nakuru","Uasin Gishu","Meru","Kilifi",
    "Kakamega","Machakos","Garissa","Turkana","Mandera","Wajir","Marsabit",
    "Isiolo","Tana River","Kwale","Taita Taveta","Lamu","Siaya","Homa Bay",
    "Migori","Nyamira","Kericho","Bomet","Nandi","Trans Nzoia","West Pokot",
    "Elgeyo Marakwet","Baringo","Laikipia","Samburu","Tharaka Nithi","Embu",
    "Kirinyaga","Murang'a","Kiambu","Nyandarua","Nyeri","Vihiga","Bungoma",
    "Busia","Kitui","Makueni","Kajiado","Narok","Kisii",
]

CHANNELS = ("PESA", "AGENT", "TILL", "PAYBILL")

# ── APP ───────────────────────────────────────────────────
app = FastAPI(
    title="M-PESA Fraud Detection API",
    description=(
        "XGBoost M-PESA fraud model with SHAP explainability.\n\n"
        "Input fields match the actual training dataset columns.\n\n"
        "**Endpoints:**\n"
        "- `POST /predict` — fraud score + alert level + signals\n"
        "- `POST /predict/explain` — full SHAP values per feature\n"
        "- `POST /predict/batch` — score up to 500 transactions\n"
        "- `GET /features` — see exact feature names from training\n"
        "- `GET /health` — model status"
    ),
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── INPUT SCHEMA — matches training columns exactly ───────
class MpesaTransaction(BaseModel):
    """
    Fields match the actual training dataset columns.
    Engineered features (log_amount, is_round, new_sim, high_velocity)
    are calculated automatically from raw inputs.
    """
    # Raw transaction fields — user provides these
    amount_kes          : float = Field(..., ge=1, le=150000,  example=9535.0,
                                        description="Transaction amount in KES")
    sender_account_age  : int   = Field(..., ge=0, le=3650,    example=268,
                                        description="Sender account age in days")
    sender_county       : str   = Field(...,                   example="Vihiga",
                                        description="Sender's county")
    receiver_county     : str   = Field(...,                   example="Isiolo",
                                        description="Receiver's county")
    channel             : str   = Field(...,                   example="PESA",
                                        description="Transaction channel")
    hour                : int   = Field(..., ge=0, le=23,      example=9,
                                        description="Hour of transaction (0-23)")
    day_of_week         : int   = Field(..., ge=0, le=6,       example=4,
                                        description="Day of week (0=Mon, 6=Sun)")
    sender_tx           : int   = Field(1, ge=0, le=500,       example=34,
                                        description="Sender transaction count (velocity)")

    @validator("channel")
    def v_channel(cls, v):
        if v not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        return v

    @validator("sender_county", "receiver_county")
    def v_county(cls, v):
        if v not in KENYA_COUNTIES:
            raise ValueError(f"'{v}' is not a valid Kenya county. "
                             f"Check /counties endpoint for full list.")
        return v

    class Config:
        schema_extra = {"example": {
            "amount_kes"        : 9535,
            "sender_account_age": 268,
            "sender_county"     : "Vihiga",
            "receiver_county"   : "Isiolo",
            "channel"           : "PESA",
            "hour"              : 9,
            "day_of_week"       : 4,
            "sender_tx"         : 34,
        }}

# ── FEATURE BUILDER — mirrors training pipeline ───────────
def build_row(tx: MpesaTransaction) -> pd.DataFrame:
    """
    Reconstruct the feature vector exactly as it was during training.
    Engineered features are calculated here the same way clean.py did.
    """
    # Auto-derive flags from raw inputs
    is_weekend    = 1 if tx.day_of_week >= 5 else 0
    is_night      = 1 if tx.hour >= 22 or tx.hour <= 5 else 0
    is_cross      = 1 if tx.sender_county != tx.receiver_county else 0
    log_amount    = np.log1p(tx.amount_kes)
    # Round number flag — amounts ending in 000 are suspicious
    is_round      = 1 if tx.amount_kes % 1000 == 0 else 0
    # New SIM flag — account younger than 30 days
    new_sim       = 1 if tx.sender_account_age <= 30 else 0
    # High velocity — more than 5 transactions from this sender
    high_velocity = 1 if tx.sender_tx > 5 else 0

    # ASAL counties — known high-risk destinations
    asal = ["Turkana","Mandera","Wajir","Marsabit","Isiolo",
            "Tana River","Garissa","Lamu","West Pokot","Samburu"]
    is_asal = 1 if tx.receiver_county in asal else 0

    # Start with all zeros — handles one-hot encoded county/channel cols
    row = {f: 0 for f in FEATURES}

    # Fill in the columns we know from the training data
    for col, val in [
        ("amount_kes",         tx.amount_kes),
        ("sender_account_age", tx.sender_account_age),
        ("sender_tx",          tx.sender_tx),
        ("hour",               tx.hour),
        ("day_of_week",        tx.day_of_week),
        ("is_weekend",         is_weekend),
        ("is_night",           is_night),
        ("is_cross_county",    is_cross),
        ("log_amount",         log_amount),
        ("is_round",           is_round),
        ("new_sim",            new_sim),
        ("high_velocity",      high_velocity),
        ("is_asal",            is_asal),
    ]:
        if col in row:
            row[col] = val

    # One-hot encoded channel
    ch_col = f"channel_{tx.channel}"
    if ch_col in row:
        row[ch_col] = 1

    # One-hot encoded counties — try both underscore and space formats
    for fmt in [tx.sender_county, tx.sender_county.replace(" ", "_")]:
        col = f"sender_county_{fmt}"
        if col in row:
            row[col] = 1
            break

    for fmt in [tx.receiver_county, tx.receiver_county.replace(" ", "_")]:
        col = f"receiver_county_{fmt}"
        if col in row:
            row[col] = 1
            break

    return pd.DataFrame([row])[FEATURES]

def get_fraud_signals(tx: MpesaTransaction) -> List[str]:
    signals = []
    is_night = tx.hour >= 22 or tx.hour <= 5
    is_cross = tx.sender_county != tx.receiver_county
    asal     = ["Turkana","Mandera","Wajir","Marsabit","Isiolo",
                "Tana River","Garissa","Lamu","West Pokot","Samburu"]

    if tx.sender_account_age <= 30:
        signals.append(
            f"New account ({tx.sender_account_age} days old) — SIMSwap risk")
    if is_night:
        signals.append("Off-hours transaction — elevated fraud rate at night")
    if is_cross:
        signals.append(
            f"Cross-county: {tx.sender_county} → {tx.receiver_county}")
    if tx.receiver_county in asal:
        signals.append("ASAL region destination — known high-risk zone")
    if tx.sender_tx > 5:
        signals.append(
            f"High transaction velocity — {tx.sender_tx} txns from this sender")
    if tx.amount_kes > 50000:
        signals.append(f"Large amount: KES {tx.amount_kes:,.0f}")
    if tx.amount_kes % 1000 == 0:
        signals.append(
            f"Round number amount (KES {tx.amount_kes:,.0f}) — fraud pattern")
    if tx.channel == "AGENT":
        signals.append("AGENT channel — highest fraud rate channel")
    if tx.sender_account_age <= 30 and tx.amount_kes > 20000:
        signals.append(
            "New account + large amount — highest risk combination")
    return signals or ["No specific fraud signals detected"]

# ── ENDPOINTS ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "api"      : "M-PESA Fraud Detection API",
        "version"  : "2.0.0",
        "counties" : len(KENYA_COUNTIES),
        "threshold": 0.5,
        "shap"     : SHAP_AVAILABLE,
        "docs"     : "/docs",
        "note"     : "Input fields match actual training dataset columns",
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status"         : "healthy",
        "model"          : "XGBoost Pipeline",
        "auc"            : 0.960,
        "threshold"      : 0.5,
        "feature_count"  : len(FEATURES),
        "shap_available" : SHAP_AVAILABLE,
        "shap_base_value": round(EXPECTED_VALUE, 6) if EXPECTED_VALUE else None,
    }

@app.get("/features", tags=["Info"])
def get_features():
    """Returns exact feature names the model was trained on."""
    return {
        "feature_count" : len(FEATURES),
        "features"      : FEATURES,
        "raw_inputs"    : [
            "amount_kes", "sender_account_age", "sender_tx",
            "hour", "day_of_week", "sender_county",
            "receiver_county", "channel"
        ],
        "auto_engineered": [
            "is_weekend", "is_night", "is_cross_county",
            "log_amount", "is_round", "new_sim", "high_velocity"
        ],
    }

@app.get("/counties", tags=["Info"])
def get_counties():
    """Returns all valid Kenya county names."""
    return {"counties": sorted(KENYA_COUNTIES), "count": len(KENYA_COUNTIES)}

@app.post("/predict", tags=["Prediction"])
def predict(tx: MpesaTransaction):
    """Score one transaction — fraud probability + alert level + signals."""
    start = time.time()
    try:
        X = build_row(tx)
        X_s = scaler.transform(X)
        prob  = float(pipeline.predict_proba(X_s)[0][1])
        fraud = prob >=0.5
        alert = "BLOCK" if prob >= 0.70 else "REVIEW" if fraud else "CLEAR"
        return {
            "fraud_probability" : round(prob, 4),
            "fraud_pct"         : round(prob * 100, 2),
            "is_fraud"          : fraud,
            "threshold_used"    : 0.5,
            "alert_level"       : alert,
            "fraud_signals"     : get_fraud_signals(tx),
            "action"            : (
                "Block transaction and investigate immediately"
                if alert == "BLOCK" else
                "Hold for manual review"
                if alert == "REVIEW" else
                "Approve transaction"
            ),
            "amount_kes"        : tx.amount_kes,
            "county_route"      : f"{tx.sender_county} → {tx.receiver_county}",
            "processing_ms"     : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(tx: MpesaTransaction):
    if not SHAP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SHAP not installed. Run: pip install shap"
        )
    start = time.time()
    try:
        X  = build_row(tx)
        X_s = scaler.transform(X)
        prob  = float(pipeline.predict_proba(X_s)[0][1])
        fraud = prob >= 0.5
        alert = "BLOCK" if prob >= 0.70 else "REVIEW" if fraud else "CLEAR"

        X_df      = pd.DataFrame(X_s, columns=FEATURES)
        shap_vals = EXPLAINER.shap_values(X_df)
        sv        = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        sv_row    = sv[0]

        # Full SHAP dict
        shap_dict = {
            feat: round(float(val), 6)
            for feat, val in zip(FEATURES, sv_row)
        }

        # Top 10 drivers
        sorted_shap = sorted(
            shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_drivers = [
            {
                "feature"  : feat,
                "shap_value": val,
                "direction": "increases fraud risk" if val > 0
                             else "decreases fraud risk",
                "magnitude": (
                    "high"   if abs(val) > 0.05 else
                    "medium" if abs(val) > 0.01 else "low"
                ),
            }
            for feat, val in sorted_shap[:10]
        ]

        # Plain English
        top3  = sorted_shap[:3]
        parts = []
        for feat, val in top3:
            direction = "increased" if val > 0 else "decreased"
            parts.append(
                f"{feat} {direction} fraud probability "
                f"by {abs(val)*100:.1f} percentage points"
            )
        explanation = ". ".join(parts) + "."

        return {
            "fraud_probability" : round(prob, 4),
            "fraud_pct"         : round(prob * 100, 2),
            "is_fraud"          : fraud,
            "threshold_used"    : 0.5,
            "alert_level"       : alert,
            "fraud_signals"     : get_fraud_signals(tx),
            "action"            : (
                "Block transaction and investigate immediately"
                if alert == "BLOCK" else
                "Hold for manual review"
                if alert == "REVIEW" else
                "Approve transaction"
            ),
            "amount_kes"        : tx.amount_kes,
            "county_route"      : f"{tx.sender_county} → {tx.receiver_county}",
            "shap_base_value"   : round(EXPECTED_VALUE, 6),
            "shap_values"       : shap_dict,
            "shap_top_drivers"  : top_drivers,
            "shap_explanation"  : explanation,
            "shap_note"         : (
                f"Base fraud rate: {EXPECTED_VALUE*100:.3f}%. "
                f"SHAP values show how each feature moved this transaction "
                f"from that base to {prob*100:.2f}%."
            ),
            "processing_ms"     : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: List[MpesaTransaction]):

    if len(transactions) > 1000:
        raise HTTPException(status_code=422,
                            detail="Max 500 transactions per batch")
    start = time.time()
    results, fraud_count = [], 0
    for tx in transactions:
        prob  = float(pipeline.predict_proba(build_row(tx))[0][1])
        is_f  = prob >= 0.5
        alert = "BLOCK" if prob >= 0.70 else "REVIEW" if is_f else "CLEAR"
        if is_f: fraud_count += 1
        results.append({
            "fraud_probability": round(prob, 4),
            "is_fraud"         : is_f,
            "alert_level"      : alert,
            "amount_kes"       : tx.amount_kes,
            "county_route"     : f"{tx.sender_county} → {tx.receiver_county}",
        })
    return {
        "total_transactions": len(results),
        "fraud_count"       : fraud_count,
        "fraud_rate_pct"    : round(fraud_count / len(results) * 100, 3),
        "total_amount_kes"  : round(sum(t.amount_kes for t in transactions), 2),
        "fraud_amount_kes"  : round(
            sum(r["amount_kes"] for r in results if r["is_fraud"]), 2),
        "alert_breakdown"   : {
            level: sum(1 for r in results if r["alert_level"] == level)
            for level in ["CLEAR", "REVIEW", "BLOCK"]
        },
        "predictions"       : results,
        "processing_ms"     : round((time.time() - start) * 1000, 2),
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": str(exc)}
    )
