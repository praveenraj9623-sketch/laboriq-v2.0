from __future__ import annotations

import pandas as pd
from .skill_extractor import normalize_skill_list


def recommend_skills_for_role(
    target_role: str,
    current_skills: list[str],
    skills_by_role: pd.DataFrame,
    trends: pd.DataFrame | None = None,
    top_n: int = 15,
) -> pd.DataFrame:
    current = {s.lower() for s in normalize_skill_list(current_skills)}
    role_df = skills_by_role[skills_by_role["role_label"].str.lower() == target_role.lower()].copy()
    if role_df.empty:
        return pd.DataFrame(columns=["skill", "mentions", "priority", "reason"])
    role_df = role_df.sort_values("mentions", ascending=False).head(top_n).copy()
    if trends is not None and not trends.empty:
        trend_map = trends.set_index("skill")["trend_label"].to_dict()
        score_map = trends.set_index("skill")["growth_score"].to_dict()
    else:
        trend_map, score_map = {}, {}
    rows=[]
    for _, row in role_df.iterrows():
        skill = row["skill"]
        has_skill = skill.lower() in current
        trend = trend_map.get(skill, "Stable")
        growth = float(score_map.get(skill, 1.0))
        if has_skill:
            priority = "Already Strong"
            reason = "You already list this skill. Keep project proof ready."
        elif trend == "Emerging" or growth >= 1.5:
            priority = "High"
            reason = "High role demand and emerging-growth signal."
        elif row["mentions"] >= role_df["mentions"].median():
            priority = "Medium"
            reason = "Common core skill for the target role."
        else:
            priority = "Low"
            reason = "Useful supporting skill."
        rows.append({"skill": skill, "mentions": row["mentions"], "trend_label": trend, "growth_score": round(growth, 3), "priority": priority, "reason": reason})
    priority_order = {"High":0, "Medium":1, "Low":2, "Already Strong":3}
    return pd.DataFrame(rows).sort_values(by="priority", key=lambda s: s.map(priority_order))
