from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SkillMatch:
    canonical_skill: str
    category: str
    matched_alias: str
    start: int
    end: int


class SkillExtractor:
    """Dictionary/taxonomy based skill extractor for job descriptions.

    The implementation is deliberately explainable, which is useful in labor-market
    intelligence where stakeholders need to understand why a skill was extracted.
    """

    def __init__(self, taxonomy: pd.DataFrame):
        required = {"canonical_skill", "category", "aliases"}
        missing = required - set(taxonomy.columns)
        if missing:
            raise ValueError(f"Taxonomy missing columns: {missing}")
        self.taxonomy = taxonomy.copy()
        self.patterns = []
        for _, row in self.taxonomy.iterrows():
            aliases = [row["canonical_skill"]] + str(row.get("aliases", "")).split("|")
            for alias in aliases:
                alias = alias.strip()
                if not alias:
                    continue
                pattern = re.compile(rf"(?<![A-Za-z0-9+#]){re.escape(alias)}(?![A-Za-z0-9+#])", re.IGNORECASE)
                self.patterns.append((pattern, row["canonical_skill"], row["category"], alias))

    def extract_matches(self, text: str) -> list[SkillMatch]:
        text = text or ""
        matches: dict[str, SkillMatch] = {}
        for pattern, canonical, category, alias in self.patterns:
            m = pattern.search(text)
            if m and canonical not in matches:
                matches[canonical] = SkillMatch(canonical, category, alias, m.start(), m.end())
        return sorted(matches.values(), key=lambda x: (x.category, x.canonical_skill))

    def extract_skills(self, text: str) -> list[str]:
        return [m.canonical_skill for m in self.extract_matches(text)]

    def extract_categories(self, text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for match in self.extract_matches(text):
            counts[match.category] = counts.get(match.category, 0) + 1
        return counts

    def transform_dataframe(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        out = df.copy()
        out["extracted_skills"] = out[text_col].fillna("").apply(lambda t: "|".join(self.extract_skills(t)))
        out["skill_count"] = out["extracted_skills"].apply(lambda s: 0 if not s else len(s.split("|")))
        for category in sorted(self.taxonomy["category"].unique()):
            col = "skillcat_" + re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
            out[col] = out[text_col].fillna("").apply(lambda t, c=category: self.extract_categories(t).get(c, 0))
        return out


def explode_skills(df: pd.DataFrame, skills_col: str = "extracted_skills") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        skills = [s for s in str(row.get(skills_col, "")).split("|") if s]
        for skill in skills:
            rows.append({
                "job_id": row.get("job_id"),
                "posted_date": row.get("posted_date"),
                "location": row.get("location"),
                "job_title": row.get("job_title"),
                "role_label": row.get("role_label", row.get("true_role_label")),
                "skill": skill,
            })
    return pd.DataFrame(rows)


def normalize_skill_list(skills: Iterable[str]) -> list[str]:
    return sorted({str(s).strip() for s in skills if str(s).strip()})
