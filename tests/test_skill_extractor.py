import pandas as pd
from src.skill_extractor import SkillExtractor


def test_skill_extractor_finds_core_skills():
    tax = pd.DataFrame({
        "canonical_skill": ["Python", "SQL", "RAG"],
        "category": ["Programming", "Programming", "NLP & GenAI"],
        "aliases": ["python", "sql", "retrieval augmented generation|rag"],
    })
    extractor = SkillExtractor(tax)
    skills = extractor.extract_skills("We need Python, SQL and retrieval augmented generation experience.")
    assert set(skills) == {"Python", "SQL", "RAG"}
