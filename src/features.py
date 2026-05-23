from __future__ import annotations

import pandas as pd


def make_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_for_model"] = (
        out["job_title"].fillna("") + " " +
        out["job_description"].fillna("") + " " +
        out["extracted_skills"].fillna("").str.replace("|", " ", regex=False)
    )
    out["salary_mid_lpa"] = pd.to_numeric(out["salary_mid_lpa"], errors="coerce")
    out["role_label"] = out["role_label"].fillna(out.get("true_role_label", "Unknown"))
    return out
