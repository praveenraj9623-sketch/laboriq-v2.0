from src.adzuna_client import AdzunaSearchSpec, build_cache_key, normalize_adzuna_results


def test_normalize_adzuna_results_converts_salary_to_lpa():
    payload = {
        "results": [
            {
                "id": "123",
                "created": "2026-05-21T10:00:00Z",
                "company": {"display_name": "Demo Company"},
                "title": "Data Scientist",
                "location": {"area": ["India", "Tamil Nadu", "Chennai"]},
                "contract_time": "full_time",
                "salary_min": 800000,
                "salary_max": 1200000,
                "description": "Python SQL NLP machine learning",
                "redirect_url": "https://example.com/job/123",
                "category": {"label": "IT Jobs"},
            }
        ]
    }
    df = normalize_adzuna_results(payload, query="data scientist", location="Chennai")
    assert len(df) == 1
    assert df.loc[0, "job_id"] == "ADZUNA_123"
    assert df.loc[0, "salary_min_lpa"] == 8.0
    assert df.loc[0, "salary_max_lpa"] == 12.0
    assert df.loc[0, "source"] == "adzuna_api"
    assert "Chennai" in df.loc[0, "location"]


def test_cache_key_is_stable():
    specs = [AdzunaSearchSpec(query="data scientist", location="India", pages=1)]
    key1 = build_cache_key(specs, country="in", results_per_page=25)
    key2 = build_cache_key(specs, country="in", results_per_page=25)
    assert key1 == key2
    assert len(key1) == 16
