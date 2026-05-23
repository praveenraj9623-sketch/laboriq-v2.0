import pandas as pd
from src.forecasting import build_workforce_forecast


def test_forecasting_outputs_tables():
    df = pd.DataFrame({
        "month": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
        "skill": ["RAG"] * 6,
        "mentions": [1, 2, 3, 5, 8, 13],
        "postings": [1, 2, 3, 5, 8, 13],
    })
    out = build_workforce_forecast(df, periods=3)
    assert not out["skill_trends"].empty
    assert len(out["skill_forecasts"]) == 3
