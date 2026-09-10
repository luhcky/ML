
from fastapi import FastAPI, HTTPException, Request

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

model    = joblib.load(f"{MODEL_DIR}/agri_model.pkl")
scaler   = joblib.load(f"{MODEL_DIR}/agri_scaler.pkl")
FEATURES = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

# Load SHAP explainer once at startup
try:
    import shap
    EXPLAINER      = shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
    ev = EXPLAINER.expected_value
    if hasattr(ev,"__len__") and len(ev) > 1:
        EXPECTED_VALUE = float(ev[1]) 
    elif hasattr(ev,"__len__"):
        EXPECTED_VALUE = float (ev[0])
    else:
        EXPECTED_VALUE = float(ev)
    logger.info(f"SHAP loaded. Base yield={EXPECTED_VALUE:.4f} t/ha")
except ImportError:
    SHAP_AVAILABLE = False
    EXPECTED_VALUE = None
    logger.warning("shap not installed — pip install shap")

logger.info(f"Agri model loaded. Features={len(FEATURES)}")
logger.info(f"Feature names: {FEATURES}")

# Valid values from training data
COUNTIES = [
    "Uasin Gishu","Trans Nzoia","Nakuru","Nandi","Kericho","Bomet",
    "Kakamega","Bungoma","Busia","Trans Nzo",  # from screenshot
    "Nairobi","Mombasa","Kisumu","Meru","Kilifi","Machakos","Garissa",
    "Turkana","Mandera","Wajir","Marsabit","Isiolo","Tana River","Kwale",
    "Taita Taveta","Lamu","Siaya","Homa Bay","Migori","Nyamira",
    "Kisii","Vihiga","Laikipia","Nyeri","Kirinyaga","Murang'a","Kiambu",
    "Nyandarua","Embu","Tharaka Nithi","Samburu","Baringo","West Pokot",
    "Elgeyo Marakwet","Kajiado","Narok","Kitui","Makueni",
]

CROPS = [
    "Maize","Wheat","Beans","Potatoes","Sorghum",
    "Millet","Tea","Coffee","Rice","Barley",
]

SEASONS = ["Long Rain", "Short Rain"]

SEED_VARIETIES = ["Hybrid", "Traditional", "Improved"]

# ── APP ───────────────────────────────────────────────────
app = FastAPI(
    title="Agricultural Yield Prediction API",
    description=(
        "Random Forest yield model with SHAP explainability.\n\n"
        "Input fields match actual training dataset columns:\n"
        "year, county, season, crop, area_ha, rainfall_mm,\n"
        "avg_temp, fertiliser, seed_variety\n\n"
        "**Endpoints:**\n"
        "- `POST /predict` — yield prediction + insights\n"
        "- `POST /predict/explain` — full SHAP values\n"
        "- `POST /predict/batch` — score up to 200 farms\n"
        "- `GET /features` — exact feature names from training\n"
        "- `GET /health` — model status"
    ),
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── INPUT SCHEMA — matches training columns exactly ───────
class FarmInput(BaseModel):
    """
    Fields match the actual training dataset columns exactly.
    """
    year         : int   = Field(..., ge=1990, le=2030, example=1990,
                                 description="Season year")
    county       : str   = Field(...,                   example="Trans Nzoia",
                                 description="Kenya county name")
    season       : str   = Field(...,                   example="Long Rain",
                                 description="Long Rain or Short Rain")
    crop         : str   = Field(...,                   example="Maize",
                                 description="Crop type")
    area_ha      : float = Field(..., ge=0.1, le=1000,  example=14.56,
                                 description="Area planted in hectares")
    rainfall_mm  : float = Field(..., ge=0,   le=3000,  example=1425.8,
                                 description="Seasonal rainfall in mm")
    avg_temp     : float = Field(..., ge=5,   le=40,    example=15.3,
                                 description="Average temperature in Celsius")
    fertiliser   : float = Field(0.0, ge=0,  le=500,   example=96.7,
                                 description="Fertiliser applied in kg/ha")
    seed_variety : str   = Field(...,                   example="Hybrid",
                                 description="Hybrid, Traditional, or Improved")

    @validator("season")
    def v_season(cls, v):
        if v not in SEASONS:
            raise ValueError(f"season must be one of {SEASONS}")
        return v

    @validator("crop")
    def v_crop(cls, v):
        if v not in CROPS:
            raise ValueError(f"crop must be one of {CROPS}")
        return v

    @validator("seed_variety")
    def v_seed(cls, v):
        if v not in SEED_VARIETIES:
            raise ValueError(f"seed_variety must be one of {SEED_VARIETIES}")
        return v

    @validator("county")
    def v_county(cls, v):
        # Flexible match — allow partial match for counties like "Trans Nzo"
        if v not in COUNTIES:
            # try prefix match
            matches = [c for c in COUNTIES if c.startswith(v[:6])]
            if matches:
                return matches[0]
            raise ValueError(
                f"'{v}' is not a valid Kenya county. "
                f"Check /counties for full list."
            )
        return v

    class Config:
        schema_extra = {"example": {
            "year"        : 1990,
            "county"      : "Trans Nzoia",
            "season"      : "Long Rain",
            "crop"        : "Maize",
            "area_ha"     : 14.56,
            "rainfall_mm" : 1425.8,
            "avg_temp"    : 15.3,
            "fertiliser"  : 96.7,
            "seed_variety": "Hybrid",
        }}

# ── FEATURE BUILDER — mirrors training pipeline ───────────
def build_row(f: FarmInput) -> pd.DataFrame:
    """
    Reconstruct feature vector exactly as during training.
    Engineered features calculated the same way as clean.py.
    """
    row = {feat: 0 for feat in FEATURES}

    # Direct columns from training data
    for col, val in [
        ("year",         f.year),
        ("area_ha",      f.area_ha),
        ("rainfall_mm",  f.rainfall_mm),
        ("avg_temp",     f.avg_temp),
        ("fertiliser",   f.fertiliser),
        # Engineered features
        ("rainfall_temp_ratio", f.rainfall_mm / max(f.avg_temp, 1)),
        ("fertiliser_per_ha",   f.fertiliser  / max(f.area_ha,  0.1)),
        ("is_long_rain",        1 if f.season == "Long Rain" else 0),
        ("log_rainfall",        np.log1p(f.rainfall_mm)),
        ("log_area",            np.log1p(f.area_ha)),
    ]:
        if col in row:
            row[col] = val

    # One-hot: county
    for fmt in [f.county, f.county.replace(" ", "_"),
                f.county.replace(" ", "")]:
        col = f"county_{fmt}"
        if col in row:
            row[col] = 1
            break

    # One-hot: crop
    for fmt in [f.crop, f.crop.replace(" ", "_")]:
        col = f"crop_{fmt}"
        if col in row:
            row[col] = 1
            break

    # One-hot: seed_variety
    for fmt in [f.seed_variety, f.seed_variety.replace(" ", "_")]:
        col = f"seed_{fmt}"
        if col in row:
            row[col] = 1
            break

    # One-hot: season
    col = f"season_{f.season.replace(' ', '_')}"
    if col in row:
        row[col] = 1

    return pd.DataFrame([row])[FEATURES]

def get_insights(f: FarmInput, yield_tha: float) -> List[str]:
    insights = []
    national_avg = 1.84  # maize national average

    if f.seed_variety == "Traditional":
        insights.append("Traditional seed — switch to Hybrid for ~35% yield uplift")
    if f.seed_variety == "Hybrid":
        insights.append("Hybrid seed — typically 35% higher yield than traditional")
    if f.rainfall_mm < 400:
        insights.append("Low rainfall (<400mm) — irrigation strongly recommended")
    if f.rainfall_mm > 2000:
        insights.append("Very high rainfall — monitor for waterlogging")
    if f.fertiliser < 30:
        insights.append("Low fertiliser — increasing to 80-100 kg/ha improves yield")
    if f.avg_temp > 30:
        insights.append("High temperature — heat stress may reduce yield")
    if f.avg_temp < 12:
        insights.append("Low temperature — frost risk for some crops")
    if f.season == "Short Rain":
        insights.append("Short Rain season — typically lower yields than Long Rain")
    if yield_tha > national_avg * 1.5:
        insights.append(f"Above-average yield ({yield_tha:.2f} vs {national_avg} national avg)")
    if yield_tha < national_avg * 0.5:
        insights.append("Significantly below national average — review inputs")
    return insights or ["No specific agronomic flags for this profile"]

# ── ENDPOINTS ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "api"     : "Agricultural Yield Prediction API",
        "version" : "2.0.0",
        "counties": len(COUNTIES),
        "crops"   : CROPS,
        "seasons" : SEASONS,
        "shap"    : SHAP_AVAILABLE,
        "docs"    : "/docs",
        "note"    : "Input fields match actual training dataset columns",
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status"          : "healthy",
        "model"           : "RandomForestRegressor",
        "r2"              : 0.80,
        "mae_tha"         : 0.42,
        "feature_count"   : len(FEATURES),
        "shap_available"  : SHAP_AVAILABLE,
        "shap_base_yield" : round(EXPECTED_VALUE, 4) if EXPECTED_VALUE else None,
    }

@app.get("/features", tags=["Info"])
def get_features():
    """Returns exact feature names the model was trained on."""
    return {
        "feature_count"   : len(FEATURES),
        "features"        : FEATURES,
        "raw_inputs"      : ["year","county","season","crop",
                             "area_ha","rainfall_mm","avg_temp",
                             "fertiliser","seed_variety"],
        "auto_engineered" : ["rainfall_temp_ratio","fertiliser_per_ha",
                             "is_long_rain","log_rainfall","log_area"],
    }

@app.get("/counties", tags=["Info"])
def get_counties():
    return {"counties": sorted(COUNTIES), "count": len(COUNTIES)}

@app.get("/crops", tags=["Info"])
def get_crops():
    return {"crops": CROPS, "seasons": SEASONS, "seed_varieties": SEED_VARIETIES}

@app.post("/predict", tags=["Prediction"])
def predict(farm: FarmInput):
    """Predict crop yield — returns t/ha, total tonnes, insights."""
    start = time.time()
    try:
        X      = build_row(farm)
        X_s    = scaler.transform(X)
        yield_ = max(0.0, float(model.predict(X_s)[0]))
        total  = round(yield_ * farm.area_ha, 2)
        nat    = 1.84
        return {
            "predicted_yield_tha"  : round(yield_, 3),
            "predicted_yield_kgha" : round(yield_ * 1000, 0),
            "total_yield_tonnes"   : total,
            "total_yield_kg"       : round(total * 1000, 0),
            "area_ha"              : farm.area_ha,
            "county"               : farm.county,
            "crop"                 : farm.crop,
            "season"               : farm.season,
            "seed_variety"         : farm.seed_variety,
            "confidence_range_tha" : [round(yield_ - 0.19, 3),
                                      round(yield_ + 0.19, 3)],
            "model_mae_tha"        : 0.19,
            "national_avg_tha"     : nat,
            "vs_national_pct"      : round((yield_ - nat) / nat * 100, 1),
            "insights"             : get_insights(farm, yield_),
            "processing_ms"        : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(farm: FarmInput):
    """
    Predict yield AND return full SHAP explanation.

    SHAP values are in t/ha — how much each feature
    added or removed from the base yield prediction.

    Example:
    base_value = 2.1 t/ha (average yield in training)
    rainfall_mm shap = +0.72 → good rainfall added 0.72 t/ha
    avg_temp shap = -0.15 → high temp removed 0.15 t/ha
    """
    if not SHAP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SHAP not installed. Run: pip install shap"
        )
    start = time.time()
    try:
        X      = build_row(farm)
        X_s    = scaler.transform(X)
        X_df   = pd.DataFrame(X_s, columns=FEATURES)
        yield_ = max(0.0, float(model.predict(X_s)[0]))
        total  = round(yield_ * farm.area_ha, 2)
        nat    = 1.84

        # SHAP — regression returns single array
        shap_vals = EXPLAINER.shap_values(X_df)
        sv_row    = shap_vals[0]

        shap_dict = {
            feat: round(float(val), 6)
            for feat, val in zip(FEATURES, sv_row)
        }

        sorted_shap = sorted(
            shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_drivers = [
            {
                "feature"   : feat,
                "shap_value": val,
                "shap_tha"  : round(val, 4),
                "direction" : "increases yield" if val > 0 else "reduces yield",
                "magnitude" : (
                    "high"   if abs(val) > 0.3 else
                    "medium" if abs(val) > 0.1 else "low"
                ),
            }
            for feat, val in sorted_shap[:10]
        ]

        top3  = sorted_shap[:3]
        parts = []
        for feat, val in top3:
            direction = "added" if val > 0 else "removed"
            parts.append(
                f"{feat} {direction} {abs(val):.2f} t/ha "
                f"{'above' if val > 0 else 'below'} base yield"
            )
        explanation = ". ".join(parts) + "."

        return {
            "predicted_yield_tha"  : round(yield_, 3),
            "predicted_yield_kgha" : round(yield_ * 1000, 0),
            "total_yield_tonnes"   : total,
            "total_yield_kg"       : round(total * 1000, 0),
            "area_ha"              : farm.area_ha,
            "county"               : farm.county,
            "crop"                 : farm.crop,
            "season"               : farm.season,
            "seed_variety"         : farm.seed_variety,
            "confidence_range_tha" : [round(yield_ - 0.19, 3),
                                      round(yield_ + 0.19, 3)],
            "model_mae_tha"        : 0.19,
            "national_avg_tha"     : nat,
            "vs_national_pct"      : round((yield_ - nat) / nat * 100, 1),
            "insights"             : get_insights(farm, yield_),
            "shap_base_value"      : round(EXPECTED_VALUE, 4),
            "shap_values"          : shap_dict,
            "shap_top_drivers"     : top_drivers,
            "shap_explanation"     : explanation,
            "shap_note"            : (
                f"Base yield across all training farms: {EXPECTED_VALUE:.2f} t/ha. "
                f"SHAP values show how each factor moved this farm's "
                f"yield from that base to {yield_:.3f} t/ha. "
                f"Values are in tonnes per hectare."
            ),
            "processing_ms"        : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(farms: List[FarmInput]):

    if len(farms) > 200:
        raise HTTPException(status_code=422, detail="Max 200 farms per batch")
    start   = time.time()
    results = []
    for farm in farms:
        X_s    = scaler.transform(build_row(farm))
        yield_ = max(0.0, round(float(model.predict(X_s)[0]), 3))
        results.append({
            "county"       : farm.county,
            "crop"         : farm.crop,
            "season"       : farm.season,
            "area_ha"      : farm.area_ha,
            "yield_tha"    : yield_,
            "total_tonnes" : round(yield_ * farm.area_ha, 2),
            "vs_national"  : round((yield_ - 1.84) / 1.84 * 100, 1),
        })
    avg_yield  = round(sum(r["yield_tha"]    for r in results) / len(results), 3)
    total_prod = round(sum(r["total_tonnes"] for r in results), 2)
    return {
        "total_farms"        : len(results),
        "avg_yield_tha"      : avg_yield,
        "total_production_t" : total_prod,
        "national_avg_tha"   : 1.84,
        "vs_national_pct"    : round((avg_yield - 1.84) / 1.84 * 100, 1),
        "predictions"        : results,
        "processing_ms"      : round((time.time() - start) * 1000, 2),
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": str(exc)}
    )
