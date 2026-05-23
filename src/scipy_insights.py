from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine


def _safe_round(value: float | int | None, digits: int = 6):
    if value is None:
        return None
    try:
        if np.isnan(value) or np.isinf(value):
            return None
    except TypeError:
        pass
    return round(float(value), digits)


def _interpret_p_value(p_value: float | None, alpha: float = 0.05) -> str:
    if p_value is None or pd.isna(p_value):
        return "insufficient_data"
    return "statistically_significant" if p_value < alpha else "not_statistically_significant"


def run_statistical_tests(jobs: pd.DataFrame, skill_fact: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create SciPy-powered statistical checks for labor-market insights.

    These tests are not used as final hiring decisions; they help validate whether
    observed salary, role, experience, and skill patterns are likely to be signal
    rather than random noise in the available sample.
    """
    rows: list[dict] = []
    df = jobs.copy()

    if {"skill_count", "salary_mid_lpa"}.issubset(df.columns):
        corr_df = df[["skill_count", "salary_mid_lpa"]].dropna()
        if len(corr_df) >= 3 and corr_df["skill_count"].nunique() > 1 and corr_df["salary_mid_lpa"].nunique() > 1:
            rho, p_value = stats.spearmanr(corr_df["skill_count"], corr_df["salary_mid_lpa"])
            rows.append({
                "test_name": "Spearman correlation",
                "business_question": "Do postings mentioning more skills tend to have higher salary midpoints?",
                "variables": "skill_count vs salary_mid_lpa",
                "statistic": _safe_round(rho),
                "p_value": _safe_round(p_value),
                "result": _interpret_p_value(p_value),
                "interpretation": "Positive statistic means higher skill-count postings are associated with higher salary midpoints.",
                "scipy_function": "scipy.stats.spearmanr",
            })

    if {"role_label", "salary_mid_lpa"}.issubset(df.columns):
        groups = [g["salary_mid_lpa"].dropna().astype(float).values for _, g in df.groupby("role_label")]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) >= 2:
            stat, p_value = stats.kruskal(*groups)
            rows.append({
                "test_name": "Kruskal-Wallis H-test",
                "business_question": "Are salary midpoint distributions different across role families?",
                "variables": "role_label vs salary_mid_lpa",
                "statistic": _safe_round(stat),
                "p_value": _safe_round(p_value),
                "result": _interpret_p_value(p_value),
                "interpretation": "Significant result suggests at least one role family has a different salary distribution.",
                "scipy_function": "scipy.stats.kruskal",
            })

    if {"role_label", "experience_level"}.issubset(df.columns):
        contingency = pd.crosstab(df["role_label"], df["experience_level"])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2 and contingency.values.sum() > 0:
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            rows.append({
                "test_name": "Chi-square independence test",
                "business_question": "Is experience-level mix independent of role family?",
                "variables": "role_label vs experience_level",
                "statistic": _safe_round(chi2),
                "p_value": _safe_round(p_value),
                "result": _interpret_p_value(p_value),
                "interpretation": f"Degrees of freedom: {dof}. Significant result means role and experience-level mix are associated.",
                "scipy_function": "scipy.stats.chi2_contingency",
            })

    if skill_fact is not None and not skill_fact.empty and {"job_id", "skill"}.issubset(skill_fact.columns) and "salary_mid_lpa" in df.columns:
        salary_by_job = df[["job_id", "salary_mid_lpa"]].dropna().drop_duplicates("job_id")
        top_skills = skill_fact["skill"].value_counts().head(10).index.tolist()
        for skill in top_skills:
            jobs_with_skill = set(skill_fact.loc[skill_fact["skill"] == skill, "job_id"])
            with_skill = salary_by_job[salary_by_job["job_id"].isin(jobs_with_skill)]["salary_mid_lpa"].astype(float)
            without_skill = salary_by_job[~salary_by_job["job_id"].isin(jobs_with_skill)]["salary_mid_lpa"].astype(float)
            if len(with_skill) >= 5 and len(without_skill) >= 5:
                stat, p_value = stats.mannwhitneyu(with_skill, without_skill, alternative="two-sided")
                rows.append({
                    "test_name": "Mann-Whitney U test",
                    "business_question": f"Do postings mentioning {skill} have a different salary distribution?",
                    "variables": f"salary_mid_lpa for postings with vs without {skill}",
                    "statistic": _safe_round(stat),
                    "p_value": _safe_round(p_value),
                    "result": _interpret_p_value(p_value),
                    "interpretation": "Non-parametric salary comparison for one skill against all other postings.",
                    "scipy_function": "scipy.stats.mannwhitneyu",
                })

    return pd.DataFrame(rows, columns=[
        "test_name", "business_question", "variables", "statistic", "p_value", "result", "interpretation", "scipy_function"
    ])


def build_role_skill_similarity(skill_fact: pd.DataFrame) -> pd.DataFrame:
    """Build a role-to-role similarity table using SciPy cosine distance.

    The output helps explain which roles share similar skill-demand profiles.
    For example, Data Scientist and ML Engineer may be close because both mention
    Python, statistics, machine learning, and model evaluation.
    """
    required = {"role_label", "skill"}
    if skill_fact.empty or not required.issubset(skill_fact.columns):
        return pd.DataFrame(columns=["role_a", "role_b", "cosine_similarity", "shared_top_skills", "scipy_function"])

    pivot = pd.crosstab(skill_fact["role_label"], skill_fact["skill"]).astype(float)
    if pivot.shape[0] < 2:
        return pd.DataFrame(columns=["role_a", "role_b", "cosine_similarity", "shared_top_skills", "scipy_function"])

    rows: list[dict] = []
    for role_a, role_b in combinations(pivot.index.tolist(), 2):
        vec_a = pivot.loc[role_a].values
        vec_b = pivot.loc[role_b].values
        if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
            similarity = 0.0
        else:
            similarity = 1.0 - float(cosine(vec_a, vec_b))
        shared = (
            (pivot.loc[role_a] > 0) & (pivot.loc[role_b] > 0)
        )
        shared_top = pivot.columns[shared].tolist()[:8]
        rows.append({
            "role_a": role_a,
            "role_b": role_b,
            "cosine_similarity": round(similarity, 4),
            "shared_top_skills": " | ".join(shared_top),
            "scipy_function": "scipy.spatial.distance.cosine",
        })

    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)
