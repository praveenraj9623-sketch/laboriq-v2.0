from __future__ import annotations

import duckdb
import pandas as pd
from .skill_extractor import explode_skills


def create_skill_fact_table(df: pd.DataFrame) -> pd.DataFrame:
    fact = explode_skills(df)
    if fact.empty:
        return pd.DataFrame(columns=["job_id", "posted_date", "location", "job_title", "role_label", "skill"])
    fact["posted_date"] = pd.to_datetime(fact["posted_date"], errors="coerce")
    fact["month"] = fact["posted_date"].dt.to_period("M").astype(str)
    return fact


def run_core_analytics(df: pd.DataFrame, skill_fact: pd.DataFrame) -> dict[str, pd.DataFrame]:
    con = duckdb.connect(database=":memory:")
    con.register("jobs", df)
    con.register("skill_fact", skill_fact)

    outputs = {
        "role_demand": con.execute("""
            SELECT role_label, COUNT(*) AS postings,
                   ROUND(AVG(salary_mid_lpa), 2) AS avg_salary_lpa,
                   ROUND(AVG(skill_count), 2) AS avg_skill_count
            FROM jobs
            GROUP BY role_label
            ORDER BY postings DESC
        """).df(),
        "location_demand": con.execute("""
            SELECT location, COUNT(*) AS postings, COUNT(DISTINCT company) AS companies,
                   ROUND(AVG(salary_mid_lpa), 2) AS avg_salary_lpa
            FROM jobs
            GROUP BY location
            ORDER BY postings DESC
        """).df(),
        "top_companies": con.execute("""
            SELECT company, COUNT(*) AS postings, COUNT(DISTINCT role_label) AS role_variety
            FROM jobs
            GROUP BY company
            ORDER BY postings DESC
            LIMIT 25
        """).df(),
        "salary_by_role_experience": con.execute("""
            SELECT role_label, experience_level, COUNT(*) AS postings,
                   ROUND(AVG(salary_min_lpa), 2) AS avg_min_lpa,
                   ROUND(AVG(salary_max_lpa), 2) AS avg_max_lpa,
                   ROUND(AVG(salary_mid_lpa), 2) AS avg_mid_lpa
            FROM jobs
            GROUP BY role_label, experience_level
            ORDER BY role_label, avg_mid_lpa DESC
        """).df(),
        "monthly_role_demand": con.execute("""
            SELECT STRFTIME(posted_date, '%Y-%m') AS month, role_label, COUNT(*) AS postings
            FROM jobs
            WHERE posted_date IS NOT NULL
            GROUP BY month, role_label
            ORDER BY month, role_label
        """).df(),
        "source_mix": con.execute("""
            SELECT source, COUNT(*) AS postings, COUNT(DISTINCT company) AS companies,
                   COUNT(DISTINCT location) AS locations,
                   ROUND(AVG(salary_mid_lpa), 2) AS avg_salary_lpa
            FROM jobs
            GROUP BY source
            ORDER BY postings DESC
        """).df(),
    }

    if not skill_fact.empty:
        outputs.update({
            "top_skills": con.execute("""
                SELECT skill, COUNT(*) AS mentions,
                       COUNT(DISTINCT job_id) AS unique_postings
                FROM skill_fact
                GROUP BY skill
                ORDER BY mentions DESC
                LIMIT 100
            """).df(),
            "skills_by_role": con.execute("""
                SELECT role_label, skill, COUNT(*) AS mentions
                FROM skill_fact
                GROUP BY role_label, skill
                QUALIFY ROW_NUMBER() OVER(PARTITION BY role_label ORDER BY COUNT(*) DESC) <= 15
                ORDER BY role_label, mentions DESC
            """).df(),
            "skills_by_location": con.execute("""
                SELECT location, skill, COUNT(*) AS mentions
                FROM skill_fact
                GROUP BY location, skill
                QUALIFY ROW_NUMBER() OVER(PARTITION BY location ORDER BY COUNT(*) DESC) <= 10
                ORDER BY location, mentions DESC
            """).df(),
            "monthly_skill_demand": con.execute("""
                SELECT month, skill, COUNT(*) AS mentions, COUNT(DISTINCT job_id) AS postings
                FROM skill_fact
                GROUP BY month, skill
                ORDER BY month, skill
            """).df(),
        })
    else:
        outputs.update({
            "top_skills": pd.DataFrame(),
            "skills_by_role": pd.DataFrame(),
            "skills_by_location": pd.DataFrame(),
            "monthly_skill_demand": pd.DataFrame(),
        })
    con.close()
    return outputs
