# Architecture

## Layer 1: Data Ingestion

Sources:
- Sample CSV for reproducible demo
- Kaggle-compatible CSV loader
- Optional Adzuna API ingestion script

## Layer 2: Data Cleaning

`src/data_cleaning.py` standardizes columns, cleans HTML/text, parses dates, normalizes salary, removes duplicates, and builds `full_text`.

## Layer 3: NLP Skill Extraction

`src/skill_extractor.py` uses an explainable taxonomy-based extractor. Every extracted skill can be traced to the exact alias matched.

## Layer 4: Occupation Mapping

`src/occupation_mapper.py` maps job postings to role labels and occupation families using keyword rules first and TF-IDF similarity as a fallback.

## Layer 5: Analytics

`src/analytics.py` creates a skill fact table and runs DuckDB SQL analytics for role demand, location demand, skill demand, salary by role, and monthly demand.

## Layer 6: Modeling

`src/modeling.py` trains:
- TF-IDF + Logistic Regression role classifier
- TF-IDF + categorical + Ridge salary baseline model

## Layer 7: Forecasting

`src/forecasting.py` builds monthly skill panel data, detects emerging skills, and forecasts 6-month future skill demand with lag features. It uses `scipy.optimize.minimize` to fit a transparent regularized forecasting model.

## Layer 8: SciPy Statistical Validation

`src/scipy_insights.py` uses `scipy.stats` for labor-market hypothesis tests and `scipy.spatial.distance.cosine` for role-skill similarity analysis. The outputs are saved to `reports/statistical_tests.csv` and `reports/role_skill_similarity.csv`.

## Layer 9: Product UI

`app.py` exposes all modules as a Streamlit product dashboard, including a SciPy Statistical Insights page.
