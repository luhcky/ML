from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
import os
import time
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── LOAD ARTIFACTS AT STARTUP ─────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

pipeline  = joblib.load(f"{MODEL_DIR}/mpesa_pipeline.pkl")
FEATURES  = joblib.load(f"{MODEL_DIR}/mpesa_feature_names.pkl")
scaler    = pipeline.named_steps['scaler']
xgb_model = pipeline.named_steps["model"]

# ── DECISION THRESHOLD ─────────────────────────────────────
# Lowered from the 0.5 default — in fraud detection the cost of a missed
# fraud case is typically far higher than the cost of a false-positive
# review, which justifies accepting more false alarms to catch more fraud.
# This value should ideally come from reading your precision-recall curve
# against a real recall target, not just asserted — treat 0.2 as a starting
# point to validate against your validation-set metrics, not a final answer.
THRESHOLD = 0.2
BLOCK_THRESHOLD = 0.30  # unchanged from before — REVIEW band is now wider
                        # (0.2-0.70) as a direct consequence of lowering
                        # THRESHOLD; revisit if this floods manual review.

# ── SHAP ──────────────────────────────────────────────────
try:
    import shap
    EXPLAINER = shap.TreeExplainer(xgb_model)
    SHAP_AVAILABLE = True
    ev = EXPLAINER.expected_value
    if hasattr(ev, "__len__") and len(ev) > 1:
        raw_ev = float(ev[1])
    elif hasattr(ev, "__len__"):
        raw_ev = float(ev[0])
    else:
        raw_ev = float(ev)
    if raw_ev > 1 or raw_ev < 0:
        EXPECTED_VALUE = 1 / (1 + math.exp(-raw_ev))
        SHAP_SPACE = "log_odds"
        logger.warning(f"Expected value was log-odds ({raw_ev:.4f}) -> converted to {EXPECTED_VALUE:.6f}")
    else:
        EXPECTED_VALUE = raw_ev
        SHAP_SPACE = "probability"
    logger.info(f"SHAP loaded. Base fraud rate={EXPECTED_VALUE*100:.4f}%")
except ImportError:
    SHAP_AVAILABLE = False
    EXPECTED_VALUE = 0.008
    SHAP_SPACE = None
    logger.warning("shap not installed - pip install shap")

# ── RAG CHATBOT SETUP ─────────────────────────────────────
# NOTE: no sentence-transformers / torch here - that combination OOM'd on
# Render's free tier. ChromaDB's built-in ONNX embedding function does the
# same job (same underlying MiniLM model) at a fraction of the memory.
# IMPORTANT: the knowledge base collection must have been built using the
# same default embedding function - see scripts/build_knowledge_base.py.
try:
    import chromadb
    RAG_CLIENT     = chromadb.PersistentClient(path=".chromadb")
    RAG_COLLECTION = RAG_CLIENT.get_collection("mpesa_fraud_kb")
    RAG_AVAILABLE  = True
    logger.info("RAG knowledge base loaded successfully")
except Exception as e:
    RAG_AVAILABLE  = False
    RAG_COLLECTION = None
    logger.warning(f"RAG not available: {e}. Run scripts/build_knowledge_base.py")

try:
    import anthropic as anthropic_lib
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger.info(f"M-PESA pipeline loaded. Features={len(FEATURES)}. Threshold={THRESHOLD}")

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
        "XGBoost M-PESA fraud model with SHAP explainability + RAG chatbot.\n\n"
        "**Endpoints:**\n"
        "- `POST /predict` - fraud score + alert level + signals\n"
        "- `POST /predict/explain` - full SHAP values per feature\n"
        "- `POST /predict/batch` - score up to 1000 transactions\n"
        "- `POST /chat` - RAG chatbot on fraud knowledge base\n"
        "- `GET /features` - exact feature names from training\n"
        "- `GET /health` - model status\n"
        "- `GET /counties` - all valid county names"
    ),
    version="3.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── INPUT SCHEMAS ─────────────────────────────────────────
class MpesaTransaction(BaseModel):
    amount_kes          : float = Field(..., ge=1, le=150000,  example=9535.0)
    sender_account_age  : int   = Field(..., ge=0, le=3650,    example=268)
    sender_county       : str   = Field(...,                   example="Vihiga")
    receiver_county     : str   = Field(...,                   example="Isiolo")
    channel             : str   = Field(...,                   example="PESA")
    hour                : int   = Field(..., ge=0, le=23,      example=9)
    day_of_week         : int   = Field(..., ge=0, le=6,       example=4)
    sender_tx           : int   = Field(1,   ge=0, le=500,     example=34)

    @validator("channel")
    def v_channel(cls, v):
        if v not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        return v

    @validator("sender_county", "receiver_county")
    def v_county(cls, v):
        if v not in KENYA_COUNTIES:
            raise ValueError(f"'{v}' is not a valid Kenya county.")
        return v

    class Config:
        schema_extra = {"example": {
            "amount_kes": 9535, "sender_account_age": 268,
            "sender_county": "Vihiga", "receiver_county": "Isiolo",
            "channel": "PESA", "hour": 9, "day_of_week": 4, "sender_tx": 34,
        }}

class ChatRequest(BaseModel):
    question : str  = Field(..., min_length=3, max_length=500,
                            example="What is SIMSwap fraud?")
    n_results: int  = Field(3, ge=1, le=5,
                            description="Number of knowledge chunks to retrieve")

# ── FEATURE BUILDER ───────────────────────────────────────
def build_row(tx: MpesaTransaction) -> pd.DataFrame:
    is_weekend    = 1 if tx.day_of_week >= 5 else 0
    is_night      = 1 if tx.hour >= 22 or tx.hour <= 5 else 0
    is_cross      = 1 if tx.sender_county != tx.receiver_county else 0
    log_amount    = np.log1p(tx.amount_kes)
    is_round      = 1 if tx.amount_kes % 1000 == 0 else 0
    new_sim       = 1 if tx.sender_account_age <= 30 else 0
    high_velocity = 1 if tx.sender_tx > 5 else 0
    asal = ["Turkana","Mandera","Wajir","Marsabit","Isiolo",
            "Tana River","Garissa","Lamu","West Pokot","Samburu"]
    is_asal = 1 if tx.receiver_county in asal else 0

    row = {f: 0 for f in FEATURES}
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

    ch_col = f"channel_{tx.channel}"
    if ch_col in row:
        row[ch_col] = 1

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

def alert_level(prob: float) -> str:
    if prob >= BLOCK_THRESHOLD:
        return "BLOCK"
    elif prob >= THRESHOLD:
        return "REVIEW"
    return "CLEAR"

def action_for(alert: str) -> str:
    return {
        "BLOCK" : "Block transaction and investigate immediately",
        "REVIEW": "Hold for manual review",
        "CLEAR" : "Approve transaction",
    }[alert]

def get_fraud_signals(tx: MpesaTransaction) -> List[str]:
    signals = []
    is_night = tx.hour >= 22 or tx.hour <= 5
    is_cross = tx.sender_county != tx.receiver_county
    asal = ["Turkana","Mandera","Wajir","Marsabit","Isiolo",
            "Tana River","Garissa","Lamu","West Pokot","Samburu"]
    if tx.sender_account_age <= 30:
        signals.append(f"New account ({tx.sender_account_age} days old) - SIMSwap risk")
    if is_night:
        signals.append("Off-hours transaction - elevated fraud rate at night")
    if is_cross:
        signals.append(f"Cross-county: {tx.sender_county} -> {tx.receiver_county}")
    if tx.receiver_county in asal:
        signals.append("ASAL region destination - known high-risk zone")
    if tx.sender_tx > 5:
        signals.append(f"High transaction velocity - {tx.sender_tx} txns from this sender")
    if tx.amount_kes > 50000:
        signals.append(f"Large amount: KES {tx.amount_kes:,.0f}")
    if tx.amount_kes % 1000 == 0:
        signals.append(f"Round number amount (KES {tx.amount_kes:,.0f}) - fraud pattern")
    if tx.channel == "AGENT":
        signals.append("AGENT channel - highest fraud rate channel")
    if tx.sender_account_age <= 30 and tx.amount_kes > 20000:
        signals.append("New account + large amount - highest risk combination")
    return signals or ["No specific fraud signals detected"]

def rag_fallback_answer(question: str, context: str) -> str:
    """Keyword-based fallback when Claude API unavailable."""
    q = question.lower()
    if "simswap" in q or "sim swap" in q:
        return ("SIMSwap accounts for 40% of M-PESA fraud. Attackers clone victim SIM cards "
                "and immediately transfer large amounts. Key signals: account age <= 7 days, "
                "amount > KES 20,000, off-hours. New SIM + large amount = 65% fraud rate (81x national).")
    elif "county" in q or "region" in q or "asal" in q:
        return ("Highest-risk counties are ASAL region: Turkana, Mandera, Wajir, Garissa, Marsabit. "
                "The model learns county risk from historical fraud patterns via one-hot encoding. "
                "Cross-county transactions (sender != receiver) also elevate risk.")
    elif "time" in q or "night" in q or "hour" in q or "peak" in q:
        return ("Fraud peaks between 22:00 and 05:00 - 3.5x higher rate than daytime (2.4% vs 0.7%). "
                "Hour 02:00-04:00 is the highest concentration period. "
                "is_night flag: hour >= 22 OR hour <= 5.")
    elif "shap" in q or "explain" in q:
        return ("SHAP values show each feature's contribution to the fraud probability above/below base rate. "
                "Positive SHAP = increases fraud probability. Negative = decreases. "
                "Top drivers: new_sim, is_night, high_amount_new_sim, is_cross_county, channel_AGENT.")
    elif "block" in q or "alert" in q or "review" in q:
        return (f"BLOCK: probability >= {BLOCK_THRESHOLD:.2f} -> block and investigate immediately. "
                f"REVIEW: probability >= {THRESHOLD:.2f} and < {BLOCK_THRESHOLD:.2f} -> hold for manual review. "
                f"CLEAR: probability < {THRESHOLD:.2f} -> approve and log.")
    elif "model" in q or "auc" in q or "performance" in q:
        return ("XGBoost with scale_pos_weight=124 (handles 124:1 class imbalance). "
                "AUC-ROC: 0.960. Recall at 1% FPR: 87%. "
                f"Decision threshold: {THRESHOLD:.2f} (lowered from 0.5 to prioritize recall). "
                "CV AUC: 0.957 +/- 0.008. No SMOTE - outperformed it by 3.2 AUC points.")
    elif "agent" in q or "collusion" in q:
        return ("Agent Collusion = 35% of fraud. AGENT channel has highest fraud rate at 1.76%. "
                "Channel rates: AGENT 1.76%, PAYBILL 0.9%, TILL 0.7%, PESA 0.6%. "
                "Pattern: AGENT channel + high velocity + cross-county.")
    else:
        return f"Based on the M-PESA fraud knowledge base:\n\n{context[:600]}..."

# ── ENDPOINTS ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "api"          : "M-PESA Fraud Detection API",
        "version"      : "3.1.0",
        "counties"     : len(KENYA_COUNTIES),
        "threshold"    : THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
        "shap"         : SHAP_AVAILABLE,
        "rag_chatbot"  : RAG_AVAILABLE,
        "docs"         : "/docs",
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status"          : "healthy",
        "model"           : "XGBoost Pipeline",
        "auc"             : 0.920,
        "threshold"       : THRESHOLD,
        "block_threshold" : BLOCK_THRESHOLD,
        "feature_count"   : len(FEATURES),
        "shap_available"  : SHAP_AVAILABLE,
        "shap_base_value" : round(EXPECTED_VALUE, 6) if EXPECTED_VALUE else None,
        "rag_available"   : RAG_AVAILABLE,
    }

@app.get("/features", tags=["Info"])
def get_features():
    return {
        "feature_count"  : len(FEATURES),
        "features"       : FEATURES,
        "raw_inputs"     : ["amount_kes","sender_account_age","sender_tx",
                            "hour","day_of_week","sender_county",
                            "receiver_county","channel"],
        "auto_engineered": ["is_weekend","is_night","is_cross_county",
                            "log_amount","is_round","new_sim","high_velocity"],
    }

@app.get("/counties", tags=["Info"])
def get_counties():
    return {"counties": sorted(KENYA_COUNTIES), "count": len(KENYA_COUNTIES)}

@app.post("/predict", tags=["Prediction"])
def predict(tx: MpesaTransaction):
    start = time.time()
    try:
        X     = build_row(tx)
        prob  = float(pipeline.predict_proba(X)[0][1])
        fraud = prob >= THRESHOLD
        alert = alert_level(prob)
        return {
            "fraud_probability": round(prob, 4),
            "fraud_pct"        : round(prob * 100, 2),
            "is_fraud"         : fraud,
            "threshold_used"   : THRESHOLD,
            "alert_level"      : alert,
            "fraud_signals"    : get_fraud_signals(tx),
            "action"           : action_for(alert),
            "amount_kes"       : tx.amount_kes,
            "county_route"     : f"{tx.sender_county} -> {tx.receiver_county}",
            "processing_ms"    : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(tx: MpesaTransaction):
    if not SHAP_AVAILABLE:
        raise HTTPException(status_code=503,
                            detail="SHAP not installed. Run: pip install shap")
    start = time.time()
    try:
        X     = build_row(tx)
        X_s   = scaler.transform(X)
        prob  = float(xgb_model.predict_proba(X_s)[0][1])
        fraud = prob >= THRESHOLD
        alert = alert_level(prob)

        X_df      = pd.DataFrame(X_s, columns=FEATURES)
        shap_vals = EXPLAINER.shap_values(X_df)
        sv        = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        sv_row    = sv[0]

        if SHAP_SPACE == "log_odds":
            scale = prob * (1 - prob)
            sv_row = sv_row * scale

        shap_dict = {
            feat: round(float(val), 6)
            for feat, val in zip(FEATURES, sv_row)
        }
        sorted_shap = sorted(shap_dict.items(),
                             key=lambda x: abs(x[1]), reverse=True)
        top_drivers = [
            {
                "feature"   : feat,
                "shap_value": val,
                "direction" : "increases fraud risk" if val > 0 else "decreases fraud risk",
                "magnitude" : ("high" if abs(val)>0.05 else
                               "medium" if abs(val)>0.01 else "low"),
            }
            for feat, val in sorted_shap[:10]
        ]
        parts = []
        for feat, val in sorted_shap[:3]:
            direction = "increased" if val > 0 else "decreased"
            parts.append(f"{feat} {direction} fraud probability by {abs(val)*100:.1f} percentage points")
        explanation = ". ".join(parts) + "."

        return {
            "fraud_probability": round(prob, 4),
            "fraud_pct"        : round(prob * 100, 2),
            "is_fraud"         : fraud,
            "threshold_used"   : THRESHOLD,
            "alert_level"      : alert,
            "fraud_signals"    : get_fraud_signals(tx),
            "action"           : action_for(alert),
            "amount_kes"       : tx.amount_kes,
            "county_route"     : f"{tx.sender_county} -> {tx.receiver_county}",
            "shap_base_value"  : round(EXPECTED_VALUE, 6),
            "shap_base_pct"    : round(EXPECTED_VALUE * 100, 4),
            "shap_space"       : SHAP_SPACE,
            "shap_values"      : shap_dict,
            "shap_top_drivers" : top_drivers,
            "shap_explanation" : explanation,
            "shap_note"        : (
                f"Base fraud rate: {EXPECTED_VALUE*100:.4f}% "
                f"(average across all training transactions). "
                f"SHAP values show how each feature moved this transaction "
                f"from that base to {prob*100:.2f}%."
            ),
            "processing_ms"    : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: List[MpesaTransaction]):
    if len(transactions) > 1000:
        raise HTTPException(status_code=422, detail="Max 1000 transactions per batch")
    start = time.time()
    results, fraud_count = [], 0
    for tx in transactions:
        X     = build_row(tx)
        prob  = float(pipeline.predict_proba(X)[0][1])
        is_f  = prob >= THRESHOLD
        alert = alert_level(prob)
        if is_f: fraud_count += 1
        results.append({
            "fraud_probability": round(prob, 4),
            "is_fraud"         : is_f,
            "alert_level"      : alert,
            "amount_kes"       : tx.amount_kes,
            "county_route"     : f"{tx.sender_county} -> {tx.receiver_county}",
        })
    return {
        "total_transactions": len(results),
        "fraud_count"       : fraud_count,
        "fraud_rate_pct"    : round(fraud_count / len(results) * 100, 3),
        "total_amount_kes"  : round(sum(t.amount_kes for t in transactions), 2),
        "fraud_amount_kes"  : round(sum(r["amount_kes"] for r in results if r["is_fraud"]), 2),
        "alert_breakdown"   : {
            level: sum(1 for r in results if r["alert_level"] == level)
            for level in ["CLEAR","REVIEW","BLOCK"]
        },
        "predictions"       : results,
        "processing_ms"     : round((time.time() - start) * 1000, 2),
    }

@app.post("/chat", tags=["RAG Chatbot"])
def chat(req: ChatRequest):
    """
    RAG chatbot - answers questions about M-PESA fraud patterns,
    model performance, county risk, and SHAP using your verified findings.

    Requires:
    1. Run: python scripts/build_knowledge_base.py (once) - must use
       ChromaDB's default embedding function, not sentence-transformers,
       or retrieval quality will silently degrade (embedding-space mismatch).
    2. Set ANTHROPIC_API_KEY env var for Claude answers (optional)
       Falls back to keyword matching if no API key.
    """
    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG knowledge base not built. "
                "Run: python scripts/build_knowledge_base.py"
            )
        )
    start = time.time()
    try:
        results = RAG_COLLECTION.query(
            query_texts=[req.question],
            n_results=req.n_results,
            include=["documents","metadatas","distances"]
        )
        retrieved_docs  = results["documents"][0]
        retrieved_metas = results["metadatas"][0]
        distances       = results["distances"][0]
        context = "\n\n---\n\n".join(retrieved_docs)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and ANTHROPIC_AVAILABLE:
            try:
                client = anthropic_lib.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    system=(
                        "You are a fraud analytics expert specialising in "
                        "M-PESA mobile money fraud detection in Kenya. "
                        "Answer questions using ONLY the provided context. "
                        "Be specific - include exact numbers and metrics when available. "
                        "If context is insufficient, say so honestly. "
                        "Keep answers to 3-5 sentences unless more detail is needed."
                    ),
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Context from M-PESA fraud knowledge base:\n\n"
                            f"{context}\n\n"
                            f"Question: {req.question}\n\n"
                            f"Answer:"
                        )
                    }]
                )
                answer     = message.content[0].text
                model_used = "claude-sonnet-4-6"
            except Exception as e:
                logger.error(f"Claude API error: {e}")
                answer     = rag_fallback_answer(req.question, context)
                model_used = f"fallback (Claude error: {str(e)[:60]})"
        else:
            answer     = rag_fallback_answer(req.question, context)
            model_used = "keyword_fallback (set ANTHROPIC_API_KEY for Claude)"

        return {
            "question"       : req.question,
            "answer"         : answer,
            "model_used"     : model_used,
            "sources"        : [
                {
                    "topic"    : meta.get("topic",""),
                    "relevance": round(1 - dist, 4),
                    "excerpt"  : doc[:200] + "..." if len(doc) > 200 else doc,
                }
                for doc, meta, dist in zip(
                    retrieved_docs, retrieved_metas, distances
                )
            ],
            "rag_available"  : RAG_AVAILABLE,
            "processing_ms"  : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": str(exc)}
    )
