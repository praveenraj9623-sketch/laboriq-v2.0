from __future__ import annotations

import re
import pandas as pd

CANONICAL_COLUMNS = {
    "title": "job_title",
    "job title": "job_title",
    "job_title": "job_title",
    "description": "job_description",
    "job description": "job_description",
    "job_description": "job_description",
    "company": "company",
    "company_name": "company",
    "location": "location",
    "city": "location",
    "posted": "posted_date",
    "posted_date": "posted_date",
    "date": "posted_date",
    "salary_min": "salary_min_lpa",
    "salary_max": "salary_max_lpa",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower().replace("-", "_")
        key_space = key.replace("_", " ")
        if key in CANONICAL_COLUMNS:
            rename[col] = CANONICAL_COLUMNS[key]
        elif key_space in CANONICAL_COLUMNS:
            rename[col] = CANONICAL_COLUMNS[key_space]
    return df.rename(columns=rename)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9+#./\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_experience_level(title: str, description: str) -> str:
    t = f"{title} {description}".lower()
    if any(k in t for k in ["intern", "trainee", "fresher", "entry"]):
        return "Entry"
    if any(k in t for k in ["senior", "lead", "principal", "staff"]):
        return "Senior"
    if any(k in t for k in ["associate", "2+", "2 years", "3 years"]):
        return "Associate"
    return "Mid"


def clean_job_postings(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df).copy()
    required = ["job_title", "job_description"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    defaults = {
        "job_id": None,
        "posted_date": None,
        "company": "Unknown",
        "location": "Unknown",
        "employment_type": "Unknown",
        "experience_level": None,
        "salary_min_lpa": None,
        "salary_max_lpa": None,
        "source": "local_csv",
        "true_role_label": None,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["job_title"] = df["job_title"].apply(clean_text)
    df["job_description"] = df["job_description"].apply(clean_text)
    df["company"] = df["company"].fillna("Unknown").astype(str).str.strip()
    df["location"] = df["location"].fillna("Unknown").astype(str).str.strip().str.title()
    df["employment_type"] = df["employment_type"].fillna("Unknown").astype(str).str.title()
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")

    for col in ["salary_min_lpa", "salary_max_lpa"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_exp = df["experience_level"].isna() | (df["experience_level"].astype(str).str.strip() == "")
    df.loc[missing_exp, "experience_level"] = df.loc[missing_exp].apply(
        lambda r: infer_experience_level(r["job_title"], r["job_description"]), axis=1
    )
    df["experience_level"] = df["experience_level"].astype(str).str.title()

    df["salary_mid_lpa"] = df[["salary_min_lpa", "salary_max_lpa"]].mean(axis=1)
    df["full_text"] = (df["job_title"] + " " + df["job_description"]).str.strip()
    df = df.drop_duplicates(subset=["company", "job_title", "location", "job_description"])
    df = df[df["job_description"].str.len() > 20].reset_index(drop=True)
    if df["job_id"].isna().any():
        df["job_id"] = [f"LOCAL{i+1:06d}" for i in range(len(df))]
    return df
