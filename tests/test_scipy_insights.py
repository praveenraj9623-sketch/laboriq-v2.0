import pandas as pd

from src.scipy_insights import run_statistical_tests, build_role_skill_similarity


def test_run_statistical_tests_outputs_named_scipy_functions():
    jobs = pd.DataFrame({
        "job_id": [f"j{i}" for i in range(12)],
        "role_label": ["Data Scientist"] * 6 + ["Data Analyst"] * 6,
        "experience_level": ["Entry", "Mid", "Senior"] * 4,
        "skill_count": [3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7],
        "salary_mid_lpa": [7, 8, 9, 10, 11, 12, 5, 6, 7, 8, 9, 10],
    })
    skill_fact = pd.DataFrame({
        "job_id": ["j0", "j1", "j2", "j6", "j7", "j8"],
        "skill": ["Python", "Python", "SQL", "Excel", "SQL", "Excel"],
        "role_label": ["Data Scientist", "Data Scientist", "Data Scientist", "Data Analyst", "Data Analyst", "Data Analyst"],
    })
    out = run_statistical_tests(jobs, skill_fact)
    assert not out.empty
    assert out["scipy_function"].str.contains("scipy.stats").any()


def test_role_skill_similarity_uses_cosine_distance():
    skill_fact = pd.DataFrame({
        "role_label": ["Data Scientist", "Data Scientist", "ML Engineer", "ML Engineer"],
        "skill": ["Python", "SQL", "Python", "Machine Learning"],
    })
    out = build_role_skill_similarity(skill_fact)
    assert not out.empty
    assert "cosine_similarity" in out.columns
    assert out.iloc[0]["scipy_function"] == "scipy.spatial.distance.cosine"
