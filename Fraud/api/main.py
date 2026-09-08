from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import joblib
import os
import time
import logging
from collections import deque
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
model    = joblib.load(f'{MODEL_DIR}/fraud_model.pkl')
scaler   = joblib.load(f'{MODEL_DIR}/fraud_scaler.pkl')
FEATURES = joblib.load(f'{MODEL_DIR}/fraud_feature_names.pkl')

THRESHOLD = 0.5   # was missing entirely — used everywhere but never defined

# ── SHAP explainer, loaded once at startup ────────────────
# ── SHAP explainer, loaded once at startup ────────────────
try:
    import shap
    EXPLAINER = shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
    SHAP_SPACE = "log_odds"   # XGBoost TreeExplainer default output space

    ev = EXPLAINER.expected_value
    raw_ev = float(ev[1]) if hasattr(ev, "__len__") and len(ev) > 1 else float(ev[0] if hasattr(ev, "__len__") else ev)
    EXPECTED_VALUE = 1 / (1 + np.exp(-raw_ev))   # sigmoid: log-odds -> probability

    logger.info(f"SHAP loaded. base_value(log-odds)={raw_ev:.4f} -> base_value(prob)={EXPECTED_VALUE:.4f}")
except ImportError:
    SHAP_AVAILABLE = False
    SHAP_SPACE = None
    EXPECTED_VALUE = None
    logger.warning("shap not installed — pip install shap")

# ── Lightweight in-memory model monitoring ────────────────
# NOTE: this resets on every restart/deploy — it's operational monitoring
# (volume, latency, prediction distribution), not persistent production
# monitoring. For that you'd log to a database or logging service instead.
MONITOR_WINDOW = 500   # keep the last N predictions in memory
_prediction_log = deque(maxlen=MONITOR_WINDOW)
_start_time = time.time()
_total_requests = 0
_total_flagged = 0

def log_prediction(prob: float, is_fraud: bool, processing_ms: float):
    global _total_requests, _total_flagged
    _total_requests += 1
    if is_fraud:
        _total_flagged += 1
    _prediction_log.append({
        "timestamp"    : datetime.utcnow().isoformat(),
        "probability"  : round(prob, 4),
        "is_fraud"     : is_fraud,
        "processing_ms": processing_ms,
    })

app = FastAPI(title='Credit Card Fraud Detection API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class TransactionInput(BaseModel):
    Time  : float = Field(..., description='Seconds from first transaction')
    Amount: float = Field(..., ge=0, description='Transaction amount in USD')
    V1 : float = Field(0.0); V2 : float = Field(0.0); V3 : float = Field(0.0)
    V4 : float = Field(0.0); V5 : float = Field(0.0); V6 : float = Field(0.0)
    V7 : float = Field(0.0); V8 : float = Field(0.0); V9 : float = Field(0.0)
    V10: float = Field(0.0); V11: float = Field(0.0); V12: float = Field(0.0)
    V13: float = Field(0.0); V14: float = Field(0.0); V15: float = Field(0.0)
    V16: float = Field(0.0); V17: float = Field(0.0); V18: float = Field(0.0)
    V19: float = Field(0.0); V20: float = Field(0.0); V21: float = Field(0.0)
    V22: float = Field(0.0); V23: float = Field(0.0); V24: float = Field(0.0)
    V25: float = Field(0.0); V26: float = Field(0.0); V27: float = Field(0.0)
    V28: float = Field(0.0)


def build_row(tx: TransactionInput) -> pd.DataFrame:
    row = {f: getattr(tx, f, 0.0) for f in FEATURES}
    return pd.DataFrame([row])[FEATURES]


@app.get('/')
def root():
    return {'api': 'Credit Card Fraud Detection', 'shap_available': SHAP_AVAILABLE}

@app.get('/health')
def health():
    return {'status': 'OK', 'feature_count': len(FEATURES), 'threshold': THRESHOLD}

@app.post('/predict')
def predict(tx: TransactionInput):
    start = time.time()
    try:
        X   = build_row(tx)
        X_s = scaler.transform(X)
        prob  = float(model.predict_proba(X_s)[0][1])
        fraud = prob >= THRESHOLD
        processing_ms = round((time.time() - start) * 1000, 2)

        log_prediction(prob, fraud, processing_ms)

        return {
            'fraud_probability': round(prob, 4),
            'is_fraud'         : fraud,
            'threshold_used'   : THRESHOLD,
            'alert_level'      : 'BLOCK' if prob >= 0.7 else 'REVIEW' if fraud else 'CLEAR',
            'processing_ms'    : processing_ms,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/explain')
def predict_explain(tx: TransactionInput):
    if not SHAP_AVAILABLE:
        raise HTTPException(status_code=503, detail='SHAP not installed. Run: pip install shap')

    start = time.time()
    try:
        X   = build_row(tx)
        X_s = scaler.transform(X)
        prob  = float(model.predict_proba(X_s)[0][1])
        fraud = prob >= THRESHOLD

        X_df      = pd.DataFrame(X_s, columns=FEATURES)
        shap_vals = EXPLAINER.shap_values(X_df)
        sv        = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        sv_row    = sv[0]

        if SHAP_SPACE == "log_odds":
            # Convert log-odds SHAP values to approximate probability-scale
            # using the sigmoid derivative at this prediction (first-order
            # local approximation — exact at this point, not across the
            # whole path from base value to final prediction).
            scale = prob * (1 - prob)
            sv_row = sv_row * scale

        shap_dict = {feat: round(float(val), 6) for feat, val in zip(FEATURES, sv_row)}
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        top_drivers = [
            {
                'feature'   : feat,
                'shap_value': val,
                'direction' : 'increases fraud risk' if val > 0 else 'decreases fraud risk',
                'magnitude' : 'high' if abs(val) > 0.05 else 'medium' if abs(val) > 0.01 else 'low',
            }
            for feat, val in sorted_shap[:10]
        ]

        top3 = sorted_shap[:3]
        parts = [
            f"{feat} {'increased' if val > 0 else 'decreased'} fraud probability by {abs(val)*100:.1f} percentage points"
            for feat, val in top3
        ]
        explanation = ". ".join(parts) + "."
        processing_ms = round((time.time() - start) * 1000, 2)

        log_prediction(prob, fraud, processing_ms)

        return {
            'fraud_probability': round(prob, 4),
            'is_fraud'         : fraud,
            'threshold_used'   : THRESHOLD,
            'alert_level'      : 'BLOCK' if prob >= 0.7 else 'REVIEW' if fraud else 'CLEAR',
            'shap_base_value'  : round(EXPECTED_VALUE, 4),
            'shap_space'       : SHAP_SPACE,   # honest about which conversion path was used
            'shap_values'      : shap_dict,
            'shap_top_drivers' : top_drivers,
            'shap_explanation' : explanation,
            'processing_ms'    : processing_ms,
        }
    except Exception as e:
        logger.error(f"SHAP explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/batch')
def predict_batch(transactions: List[TransactionInput]):   # was missing this parameter entirely
    if len(transactions) > 1000:
        raise HTTPException(status_code=422, detail='Max 1000 transactions')

    start = time.time()
    results, flagged = [], 0
    for tx in transactions:
        X_s  = scaler.transform(build_row(tx))
        prob = float(model.predict_proba(X_s)[0][1])   # was never computed before
        is_f = prob >= THRESHOLD
        if is_f:
            flagged += 1
        results.append({'fraud_probability': round(prob, 4), 'is_fraud': is_f})

    processing_ms = round((time.time() - start) * 1000, 2)
    for r in results:
        log_prediction(r['fraud_probability'], r['is_fraud'], processing_ms / len(results))

    return {
        'total'          : len(results),
        'fraud_count'    : flagged,
        'fraud_rate_pct' : round(flagged / len(results) * 100, 4),
        'predictions'    : results,
        'processing_ms'  : processing_ms,
    }


# ── Model monitoring endpoints ────────────────────────────
@app.get('/monitoring/stats')
def monitoring_stats():
    if not _prediction_log:
        return {
            'total_requests_since_startup': _total_requests,
            'note': 'No predictions logged yet this session.',
        }

    probs   = [p['probability']   for p in _prediction_log]
    latency = [p['processing_ms'] for p in _prediction_log]
    flagged_in_window = sum(1 for p in _prediction_log if p['is_fraud'])

    return {
        'uptime_seconds'               : round(time.time() - _start_time, 1),
        'total_requests_since_startup' : _total_requests,
        'total_flagged_since_startup'  : _total_flagged,
        'window_size'                  : len(_prediction_log),
        'window_avg_probability'       : round(sum(probs) / len(probs), 4),
        'window_max_probability'       : round(max(probs), 4),
        'window_fraud_rate_pct'        : round(flagged_in_window / len(_prediction_log) * 100, 2),
        'window_avg_latency_ms'        : round(sum(latency) / len(latency), 2),
        'window_max_latency_ms'        : round(max(latency), 2),
    }

@app.get('/monitoring/recent')
def monitoring_recent(limit: int = 20):
    return {'recent_predictions': list(_prediction_log)[-limit:]}
