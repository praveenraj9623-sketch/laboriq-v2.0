# Labor Market Intelligence & Skills Demand Analytics Platform

A production-style, Lightcast-inspired Data Science flagship project for labor-market analytics.

This is built as **one portfolio product with three complete modules**:

1. **Labor Market Intelligence & Skills Demand Analytics Platform**
   - Cleans job postings
   - Analyzes demand by role, location, company, salary, and skill
   - Uses DuckDB SQL and Python analytics

2. **Job Description Skill Extraction & Occupation Mapping Engine**
   - Extracts standardized skills from unstructured job descriptions
   - Maps jobs to occupation families and role labels
   - Trains an ML role classifier using TF-IDF + Logistic Regression

3. **Workforce Demand Forecasting & Emerging Skills Trend Analyzer**
   - Builds monthly skill-demand panel data
   - Detects emerging, stable, and declining skills
   - Forecasts future skill demand using lag-based regression
   - Uses SciPy optimization for transparent demand forecasting
   - Shows trends in a Streamlit dashboard

4. **SciPy Statistical Validation Layer**
   - Validates labor-market patterns with SciPy statistical tests
   - Compares salary distributions across roles and skills
   - Builds role-skill similarity using SciPy cosine distance

## Why this project matches Lightcast

Lightcast works on labor-market intelligence: job postings, skills, titles, occupations, compensation data, and workforce trends. This project mirrors that real business problem at portfolio scale.

## Quick Start

```bash
cd lightcast_flagship_project
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python pipelines/run_pipeline.py
streamlit run app.py
```

For Mac/Linux:

```bash
source .venv/bin/activate
```

## Optional live Adzuna API ingestion

Use the official Adzuna API rather than risky scraping. This gives the project a real-time labor-market data layer.

```bash
copy .env.example .env
# add ADZUNA_APP_ID and ADZUNA_APP_KEY
python scripts/fetch_lightcast_ready_adzuna_data.py
python pipelines/run_pipeline.py --external data/external/adzuna_lightcast_roles.csv
streamlit run app.py
```

Custom live pull:

```bash
python scripts/fetch_adzuna.py --queries "data scientist" "data analyst" "machine learning engineer" --location "India" --country in --pages 1 --results-per-page 25
python pipelines/run_pipeline.py --external data/external/adzuna_jobs.csv
```

The API layer includes `.env` credentials, multi-query fetching, deduplication, local caching, source tracking, and Streamlit UI controls. See `docs/adzuna_api_setup.md`.

## Output files

After running the pipeline:

- `reports/processed_job_postings.csv`
- `reports/top_skills.csv`
- `reports/skills_by_role.csv`
- `reports/skill_trends.csv`
- `reports/skill_forecasts.csv`
- `reports/model_metrics.json`
- `reports/source_mix.csv`
- `reports/statistical_tests.csv`
- `reports/role_skill_similarity.csv`
- `models/role_classifier.joblib`
- `models/salary_model.joblib`

## Interview story

“I built a Lightcast-style labor market intelligence platform. It takes raw job postings, cleans the data, extracts standardized skills from text, maps job descriptions to role families, analyzes skills and salary demand using SQL, forecasts emerging skills, and presents the insights in a stakeholder-ready dashboard.”

## Test

```bash
pytest -q
```
