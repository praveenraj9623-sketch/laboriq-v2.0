from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ROLE_MODEL_PATH = MODELS_DIR / "role_classifier.joblib"
SALARY_MODEL_PATH = MODELS_DIR / "salary_model.joblib"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.json"
TOP_SKILLS_PATH = REPORTS_DIR / "top_skills.csv"

KNOWN_ROLE_FAMILIES = {
    "Data Scientist",
    "Data Analyst",
    "Data Engineer",
    "AI Engineer",
    "ML Engineer",
    "NLP Engineer",
    "Business Analyst",
    "Research Analyst",
}


class HealthResponse(BaseModel):
    status: str
    role_classifier_loaded: bool
    salary_model_loaded: bool
    mlflow_tracking: str


class RolePredictionRequest(BaseModel):
    job_description: str = Field(..., min_length=1)


class RoleProbability(BaseModel):
    role: str
    probability: float


class RolePredictionResponse(BaseModel):
    predicted_role: str
    confidence: float
    top_3_roles: list[RoleProbability]


class SalaryPredictionRequest(BaseModel):
    text_for_model: str = Field(..., min_length=1)
    role_label: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    experience_level: str = Field(..., min_length=1)
    skill_count: int = Field(..., ge=0)


class SalaryPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_salary_lpa: float
    model_mae: float


class SkillRecord(BaseModel):
    skill: str
    mentions: int | None = None
    unique_postings: int | None = None


class TopSkillsResponse(BaseModel):
    skills: list[SkillRecord]


app = FastAPI(title="LaborIQ API", version="1.0.0")


@lru_cache(maxsize=1)
def load_role_classifier():
    return joblib.load(ROLE_MODEL_PATH)


@lru_cache(maxsize=1)
def load_salary_model():
    return joblib.load(SALARY_MODEL_PATH)


@lru_cache(maxsize=1)
def load_model_metrics() -> dict[str, Any]:
    if not MODEL_METRICS_PATH.exists():
        return {}
    with MODEL_METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _can_load_model(loader) -> bool:
    try:
        loader()
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        role_classifier_loaded=_can_load_model(load_role_classifier),
        salary_model_loaded=_can_load_model(load_salary_model),
        mlflow_tracking="local",
    )


@app.post("/predict/role", response_model=RolePredictionResponse)
def predict_role(payload: RolePredictionRequest) -> RolePredictionResponse:
    try:
        model = load_role_classifier()
        predicted_role = str(model.predict([payload.job_description])[0])
        top_roles: list[RoleProbability]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([payload.job_description])[0]
            classes = list(model.classes_)
            order = np.argsort(probabilities)[::-1][:3]
            top_roles = [
                RoleProbability(role=str(classes[index]), probability=round(float(probabilities[index]), 4))
                for index in order
            ]
            confidence = top_roles[0].probability if top_roles else 0.0
        else:
            top_roles = [RoleProbability(role=predicted_role, probability=1.0)]
            confidence = 1.0
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Role classifier unavailable: {exc}") from exc

    return RolePredictionResponse(
        predicted_role=predicted_role,
        confidence=confidence,
        top_3_roles=top_roles,
    )


@app.post("/predict/salary", response_model=SalaryPredictionResponse)
def predict_salary(payload: SalaryPredictionRequest) -> SalaryPredictionResponse:
    try:
        model = load_salary_model()
        features = pd.DataFrame([payload.dict()])
        prediction = float(model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Salary model unavailable: {exc}") from exc

    metrics = load_model_metrics()
    model_mae = metrics.get("salary_model", {}).get("mae_lpa", 4.54)

    return SalaryPredictionResponse(
        predicted_salary_lpa=round(prediction, 2),
        model_mae=round(float(model_mae), 2),
    )


@app.get("/skills/top", response_model=TopSkillsResponse)
def top_skills() -> TopSkillsResponse:
    if not TOP_SKILLS_PATH.exists():
        return TopSkillsResponse(skills=[])

    records = pd.read_csv(TOP_SKILLS_PATH).head(20).replace({np.nan: None}).to_dict(orient="records")
    return TopSkillsResponse(skills=[SkillRecord(**record) for record in records])
