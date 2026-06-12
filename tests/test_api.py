from fastapi.testclient import TestClient

from api.main import KNOWN_ROLE_FAMILIES, app


client = TestClient(app)


def test_health_returns_status_ok():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["role_classifier_loaded"], bool)
    assert isinstance(payload["salary_model_loaded"], bool)
    assert payload["mlflow_tracking"] == "local"


def test_predict_role_returns_known_role_family():
    response = client.post(
        "/predict/role",
        json={
            "job_description": (
                "Build machine learning models with Python, SQL, feature engineering, "
                "model evaluation, dashboards, and stakeholder communication."
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_role"] in KNOWN_ROLE_FAMILIES
    assert isinstance(payload["confidence"], float)
    assert 0 <= payload["confidence"] <= 1
    assert 1 <= len(payload["top_3_roles"]) <= 3
    assert all(item["role"] in KNOWN_ROLE_FAMILIES for item in payload["top_3_roles"])
