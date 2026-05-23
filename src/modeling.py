from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import RANDOM_STATE


@dataclass
class ModelArtifacts:
    role_model: Pipeline
    salary_model: Pipeline
    metrics: dict


def train_role_classifier(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    data = df.dropna(subset=["role_label", "text_for_model"]).copy()
    counts = data["role_label"].value_counts()
    stratify = data["role_label"] if counts.min() >= 2 and len(counts) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        data["text_for_model"], data["role_label"], test_size=0.25, random_state=RANDOM_STATE, stratify=stratify
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, preds, average="weighted")), 4),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }
    return pipe, metrics


def train_salary_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    data = df.dropna(subset=["salary_mid_lpa", "text_for_model"]).copy()
    feature_cols = ["text_for_model", "role_label", "location", "experience_level", "skill_count"]
    X = data[feature_cols]
    y = data["salary_mid_lpa"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english"), "text_for_model"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["role_label", "location", "experience_level"]),
        ("num", "passthrough", ["skill_count"]),
    ])
    pipe = Pipeline([
        ("prep", preprocessor),
        ("reg", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = {
        "mae_lpa": round(float(mean_absolute_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "prediction_note": "Salary model is a portfolio baseline. Use larger real data before business deployment."
    }
    return pipe, metrics


def train_models(df: pd.DataFrame, model_dir: Path) -> ModelArtifacts:
    model_dir.mkdir(parents=True, exist_ok=True)
    role_model, role_metrics = train_role_classifier(df)
    salary_model, salary_metrics = train_salary_model(df)
    joblib.dump(role_model, model_dir / "role_classifier.joblib")
    joblib.dump(salary_model, model_dir / "salary_model.joblib")
    return ModelArtifacts(role_model, salary_model, {"role_classifier": role_metrics, "salary_model": salary_metrics})


def predict_role(model: Pipeline, text: str) -> dict:
    pred = model.predict([text])[0]
    result = {"predicted_role": pred}
    if hasattr(model.named_steps.get("clf"), "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        order = np.argsort(probs)[::-1][:3]
        result["top_roles"] = [{"role": classes[i], "probability": round(float(probs[i]), 4)} for i in order]
    return result
