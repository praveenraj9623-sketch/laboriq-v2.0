from __future__ import annotations

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class OccupationMapper:
    """Map job descriptions to occupation families and standardized role labels.

    Uses a hybrid of keyword rules and TF-IDF similarity to keep outputs explainable.
    """

    def __init__(self, occupation_taxonomy: pd.DataFrame):
        self.taxonomy = occupation_taxonomy.copy()
        required = {"occupation_family", "role_label", "role_keywords", "core_skills"}
        missing = required - set(self.taxonomy.columns)
        if missing:
            raise ValueError(f"Occupation taxonomy missing columns: {missing}")
        self.taxonomy["profile_text"] = (
            self.taxonomy["role_label"].astype(str) + " " +
            self.taxonomy["occupation_family"].astype(str) + " " +
            self.taxonomy["role_keywords"].astype(str).str.replace("|", " ", regex=False) + " " +
            self.taxonomy["core_skills"].astype(str).str.replace("|", " ", regex=False)
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.taxonomy["profile_text"])

    def map_one(self, text: str) -> dict:
        text = text or ""
        lowered = text.lower()
        scores = []
        for _, row in self.taxonomy.iterrows():
            keywords = [k.strip().lower() for k in str(row["role_keywords"]).split("|") if k.strip()]
            hit_count = sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}\b", lowered))
            scores.append(hit_count)
        best_rule_score = max(scores) if scores else 0
        if best_rule_score > 0:
            idx = scores.index(best_rule_score)
            row = self.taxonomy.iloc[idx]
            confidence = min(0.55 + best_rule_score * 0.1, 0.95)
            method = "keyword_rules"
        else:
            vector = self.vectorizer.transform([text])
            sims = cosine_similarity(vector, self.matrix).ravel()
            idx = int(sims.argmax())
            row = self.taxonomy.iloc[idx]
            confidence = float(sims[idx])
            method = "tfidf_similarity"
        return {
            "occupation_family": row["occupation_family"],
            "role_label": row["role_label"],
            "occupation_confidence": round(float(confidence), 4),
            "mapping_method": method,
            "core_skills_for_role": row["core_skills"],
        }

    def transform_dataframe(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        mapped = df[text_col].fillna("").apply(self.map_one).apply(pd.Series)
        return pd.concat([df.reset_index(drop=True), mapped.reset_index(drop=True)], axis=1)
