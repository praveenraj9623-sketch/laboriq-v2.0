from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import warnings

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "LaborIQ"


@dataclass
class ModelArtifacts:
    role_model: Pipeline
    salary_model: Pipeline
    metrics: dict


def _configure_mlflow():
    try:
        import mlflow
        import mlflow.sklearn  # noqa: F401
    except ImportError:
        warnings.warn(
            "MLflow is not installed; continuing without experiment tracking.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_DIR.as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME))
    return mlflow


def _log_and_register_sklearn_model(mlflow, model: Pipeline, artifact_path: str, registered_model_name: str) -> None:
    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        _transition_model_to_staging(registered_model_name, model_info)
    except Exception as exc:
        warnings.warn(
            f"MLflow model registration failed for {registered_model_name}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _transition_model_to_staging(model_name: str, model_info) -> None:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        version = getattr(model_info, "registered_model_version", None)
        if version is None:
            versions = client.search_model_versions(f"name = '{model_name}'")
            if versions:
                version = max(int(model.version) for model in versions)

        if version is None:
            return

        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Staging",
            archive_existing_versions=False,
        )
        client.set_model_version_tag(
            name=model_name,
            version=str(version),
            key="deployment_stage",
            value="Staging",
        )
    except Exception as exc:
        warnings.warn(
            f"Could not move {model_name} to Staging in MLflow registry: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def train_role_classifier(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    mlflow = _configure_mlflow()
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

    def fit_and_score() -> dict:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        return {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_test, preds, average="weighted")), 4),
            "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        }

    if mlflow is None:
        metrics = fit_and_score()
    else:
        with mlflow.start_run(run_name="role_classifier_training", nested=mlflow.active_run() is not None):
            mlflow.log_params({
                "model_type": "TfidfVectorizer + LogisticRegression",
                "test_size": 0.25,
                "random_state": RANDOM_STATE,
                "tfidf_max_features": 6000,
                "tfidf_ngram_range": "(1, 2)",
                "tfidf_min_df": 1,
                "tfidf_stop_words": "english",
                "logreg_C": pipe.named_steps["clf"].C,
                "logreg_max_iter": pipe.named_steps["clf"].max_iter,
                "logreg_class_weight": str(pipe.named_steps["clf"].class_weight),
            })
            metrics = fit_and_score()
            mlflow.log_metrics({
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            })
            mlflow.log_dict(metrics, "role_classifier_metrics.json")
            _log_and_register_sklearn_model(mlflow, pipe, "role_classifier", "LaborIQRoleClassifier")

    return pipe, metrics


def train_salary_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    mlflow = _configure_mlflow()
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

    def fit_and_score() -> dict:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        return {
            "mae_lpa": round(float(mean_absolute_error(y_test, preds)), 4),
            "r2": round(float(r2_score(y_test, preds)), 4),
            "prediction_note": "Salary model is a portfolio baseline. Use larger real data before business deployment."
        }

    if mlflow is None:
        metrics = fit_and_score()
    else:
        text_vectorizer = preprocessor.transformers[0][1]
        categorical_encoder = preprocessor.transformers[1][1]
        with mlflow.start_run(run_name="salary_model_training", nested=mlflow.active_run() is not None):
            mlflow.log_params({
                "model_type": "ColumnTransformer + Ridge",
                "feature_cols": ",".join(feature_cols),
                "test_size": 0.25,
                "random_state": RANDOM_STATE,
                "tfidf_max_features": text_vectorizer.max_features,
                "tfidf_ngram_range": "(1, 2)",
                "tfidf_stop_words": text_vectorizer.stop_words,
                "onehot_handle_unknown": categorical_encoder.handle_unknown,
                "ridge_alpha": pipe.named_steps["reg"].alpha,
            })
            metrics = fit_and_score()
            mlflow.log_metrics({
                "mae_lpa": metrics["mae_lpa"],
                "r2": metrics["r2"],
            })
            mlflow.log_dict(metrics, "salary_model_metrics.json")
            _log_and_register_sklearn_model(mlflow, pipe, "salary_model", "LaborIQSalaryModel")

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
