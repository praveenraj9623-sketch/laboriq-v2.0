import pandas as pd
from src.occupation_mapper import OccupationMapper


def test_occupation_mapper_maps_nlp_role():
    occ = pd.DataFrame({
        "occupation_family": ["NLP & GenAI", "Data Analytics"],
        "role_label": ["NLP Engineer", "Data Analyst"],
        "role_keywords": ["nlp engineer|natural language processing|rag", "data analyst|dashboard|excel"],
        "core_skills": ["NLP|RAG|Python", "SQL|Excel|Power BI"],
    })
    mapper = OccupationMapper(occ)
    result = mapper.map_one("Natural language processing engineer needed for RAG and entity extraction")
    assert result["role_label"] == "NLP Engineer"
