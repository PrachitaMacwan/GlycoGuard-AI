import os
import json
import joblib
import io
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Import transformers so joblib can unpickle the pipeline seamlessly
from src.transformers import ZeroToNaNConverter, DiabetesFeatureEngineer
from backend.schemas import (
    PatientInput, PredictionResponse, FeatureComparison,
    BatchPredictionResponse, WhatIfRequest, WhatIfResponse, CurvePoint, ModelInfoResponse
)

app = FastAPI(
    title="GlycoGuard AI - Diabetes Risk Prediction API",
    description="Production Machine Learning API for clinical diabetes screening, risk quantification, XAI local explanations, and what-if sensitivity simulations.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
PIPELINE_PATH = os.path.join(MODELS_DIR, "glycoguard_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
FEATURE_IMP_PATH = os.path.join(MODELS_DIR, "feature_importance.json")

# Global state for loaded model and metadata
model_pipeline = None
model_metadata = {}
feature_importances = []

def load_artifacts():
    global model_pipeline, model_metadata, feature_importances
    if os.path.exists(PIPELINE_PATH):
        model_pipeline = joblib.load(PIPELINE_PATH)
        print("[+] Model pipeline loaded successfully.")
    else:
        print("[!] Warning: Model pipeline file not found. Please run training first.")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)

    if os.path.exists(FEATURE_IMP_PATH):
        with open(FEATURE_IMP_PATH, "r", encoding="utf-8") as f:
            feature_importances = json.load(f)

load_artifacts()

def get_risk_tier(prob: float):
    if prob < 0.30:
        return "Low Risk", "#10B981"      # Emerald Green
    elif prob < 0.50:
        return "Moderate Risk", "#F59E0B" # Amber / Orange
    elif prob < 0.75:
        return "High Risk", "#EF4444"     # Red
    else:
        return "Critical Risk", "#7F1D1D" # Dark Crimson

def generate_recommendations(patient_dict: dict, prob: float) -> List[str]:
    recs = []
    if patient_dict.get("Glucose", 0) >= 140:
        recs.append("Elevated blood glucose detected: Schedule a confirmatory Fasting Plasma Glucose (FPG) or HbA1c laboratory diagnostic panel.")
    elif patient_dict.get("Glucose", 0) >= 100:
        recs.append("Borderline prediabetic glucose range: Implement dietary carbohydrate management and increase daily physical activity.")

    if patient_dict.get("BMI", 0) >= 30:
        recs.append("Classified as Obese (BMI >= 30): A structured 5-7% intentional body weight reduction program is strongly indicated to improve insulin sensitivity.")
    elif patient_dict.get("BMI", 0) >= 25:
        recs.append("Classified as Overweight: Maintain active caloric balance through aerobic and resistance exercises.")

    if patient_dict.get("BloodPressure", 0) >= 80:
        recs.append("Diastolic blood pressure elevated (>= 80 mm Hg): Monitor cardiovascular health and reduce sodium intake.")

    if patient_dict.get("DiabetesPedigreeFunction", 0) > 0.60:
        recs.append("Elevated genetic predisposition score: Routine annual screening is recommended regardless of immediate symptom absence.")

    if prob >= 0.50:
        recs.append("Model confidence indicates high probability of diabetes: Immediate consultation with a certified endocrinologist or primary physician is recommended.")
    else:
        recs.append("General wellness advice: Maintain balanced nutrition, adequate sleep hygiene, and routine preventative health checkups.")

    return recs

def compute_local_explanations(patient_dict: dict) -> List[FeatureComparison]:
    explanations = []
    healthy_baselines = model_metadata.get("healthy_baseline", {})
    
    for feat, patient_val in patient_dict.items():
        if feat in healthy_baselines:
            baseline = healthy_baselines[feat]
            diff = patient_val - baseline
            dev_pct = (diff / baseline * 100) if baseline != 0 else 0.0

            if diff > 0.15 * baseline:
                status = "elevated"
                risk_level = "High Risk Factor"
            elif diff < -0.15 * baseline:
                status = "below"
                risk_level = "Lower than Healthy Average"
            else:
                status = "optimal"
                risk_level = "Within Optimal Healthy Range"

            explanations.append(FeatureComparison(
                feature=feat,
                patient_value=float(np.round(patient_val, 2)),
                healthy_baseline=float(np.round(baseline, 2)),
                deviation_pct=float(np.round(dev_pct, 1)),
                status=status,
                risk_level=risk_level
            ))

    return explanations

# --- API Endpoints ---

@app.api_route("/api/health", methods=["GET", "HEAD"])
def health_check():
    return {
        "status": "healthy",
        "service": "GlycoGuard AI API",
        "model_loaded": model_pipeline is not None,
        "version": "1.0.0"
    }

@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info():
    if not model_metadata:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable.")
    return ModelInfoResponse(
        model_name=model_metadata.get("model_name", "GlycoGuard AI"),
        algorithm=model_metadata.get("algorithm", "RandomForestClassifier"),
        best_params=model_metadata.get("best_params", {}),
        cv_roc_auc=model_metadata.get("cv_roc_auc", 0.0),
        test_metrics=model_metadata.get("test_metrics", {}),
        benchmarks=model_metadata.get("benchmarks", []),
        feature_importances=feature_importances,
        healthy_baseline=model_metadata.get("healthy_baseline", {}),
        diabetic_baseline=model_metadata.get("diabetic_baseline", {}),
        feature_ranges=model_metadata.get("ranges", {})
    )

@app.post("/api/predict", response_model=PredictionResponse)
def predict_patient(patient: PatientInput):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    input_df = pd.DataFrame([patient.model_dump()])
    
    # Predict probabilities
    prob = float(model_pipeline.predict_proba(input_df)[0, 1])
    pred = int(model_pipeline.predict(input_df)[0])

    risk_tier, risk_color = get_risk_tier(prob)
    confidence = float(np.round((prob if pred == 1 else (1.0 - prob)) * 100, 1))

    explanations = compute_local_explanations(patient.model_dump())
    recommendations = generate_recommendations(patient.model_dump(), prob)

    return PredictionResponse(
        prediction=pred,
        outcome_label="Diabetic Risk Indicated" if pred == 1 else "Non-Diabetic / Low Risk",
        risk_probability=float(np.round(prob, 4)),
        risk_percentage=float(np.round(prob * 100, 1)),
        risk_tier=risk_tier,
        risk_color=risk_color,
        confidence_score=confidence,
        local_explanations=explanations,
        clinical_recommendations=recommendations
    )

@app.post("/api/what-if", response_model=WhatIfResponse)
def what_if_simulation(req: WhatIfRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    base_dict = req.base_patient.model_dump()
    var = req.variable_to_sweep

    if var not in base_dict:
        raise HTTPException(status_code=400, detail=f"Invalid variable '{var}' for simulation.")

    ranges = model_metadata.get("ranges", {})
    var_range = ranges.get(var, {})

    min_val = req.min_value if req.min_value is not None else var_range.get("min", 0.0)
    max_val = req.max_value if req.max_value is not None else var_range.get("max", 200.0)

    # Generate sweep values
    sweep_values = np.linspace(min_val, max_val, req.steps)
    sim_rows = []
    for val in sweep_values:
        row = base_dict.copy()
        row[var] = float(val)
        sim_rows.append(row)

    sim_df = pd.DataFrame(sim_rows)
    probs = model_pipeline.predict_proba(sim_df)[:, 1]

    # Baseline probability
    baseline_df = pd.DataFrame([base_dict])
    base_prob = float(model_pipeline.predict_proba(baseline_df)[0, 1])

    curve = []
    inflection = None
    for val, p in zip(sweep_values, probs):
        tier, _ = get_risk_tier(p)
        curve.append(CurvePoint(
            value=float(np.round(val, 2)),
            risk_percentage=float(np.round(p * 100, 1)),
            risk_tier=tier
        ))
        if inflection is None and p >= 0.50:
            inflection = {
                "threshold_value": float(np.round(val, 2)),
                "description": f"Risk shifts into High Risk (>= 50%) when {var} exceeds {val:.1f}"
            }

    summary = (
        f"Sweeping {var} from {min_val:.1f} to {max_val:.1f} shifted predicted risk from "
        f"{probs[0]*100:.1f}% to {probs[-1]*100:.1f}%. Baseline risk is {base_prob*100:.1f}%."
    )

    return WhatIfResponse(
        variable=var,
        baseline_value=float(base_dict[var]),
        baseline_risk_percentage=float(np.round(base_prob * 100, 1)),
        curve=curve,
        inflection_point=inflection,
        summary=summary
    )

@app.post("/api/predict/batch")
async def batch_predict_csv(file: UploadFile = File(...)):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV format.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    required_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"CSV missing required columns: {missing_cols}")

    probs = model_pipeline.predict_proba(df[required_cols])[:, 1]
    preds = model_pipeline.predict(df[required_cols])

    results = []
    diabetic_count = int(np.sum(preds == 1))
    non_diabetic_count = int(np.sum(preds == 0))

    for idx, (p, pred) in enumerate(zip(probs, preds)):
        tier, color = get_risk_tier(p)
        record = df.iloc[idx].to_dict()
        record["Risk_Probability"] = float(np.round(p, 4))
        record["Risk_Percentage"] = float(np.round(p * 100, 1))
        record["Prediction"] = int(pred)
        record["Risk_Tier"] = tier
        results.append(record)

    return BatchPredictionResponse(
        total_processed=len(df),
        diabetic_count=diabetic_count,
        non_diabetic_count=non_diabetic_count,
        average_risk_percentage=float(np.round(np.mean(probs) * 100, 1)),
        results=results
    )

# Mount frontend directory for static serving
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.api_route("/", methods=["GET", "HEAD"])
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
