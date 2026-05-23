# Interview Story for Lightcast

## 60-second answer

I studied Lightcast’s business and understood that the company works with labor-market intelligence, job postings, skills, titles, occupations, compensation data, and future-of-work insights. So I built a project that follows the same problem at a smaller scale.

My platform takes raw job postings, cleans and normalizes the data, extracts skills using an explainable skills taxonomy, maps job descriptions to occupation families, analyzes demand by role/location/salary using SQL, trains a role classifier, predicts salary baselines, detects emerging skills, forecasts future skill demand, and presents everything in a Streamlit dashboard.

## Technical explanation

- Data processing: Pandas cleaning pipeline
- SQL analytics: DuckDB queries for labor-market metrics
- NLP: taxonomy-based skill extraction and TF-IDF representation
- ML: Logistic Regression role classifier and Ridge salary model
- Forecasting: monthly skill-demand panel with SciPy-optimized lag-based regression
- SciPy validation: Spearman, Kruskal-Wallis, chi-square, Mann-Whitney U, and cosine role-skill similarity
- Product layer: Streamlit dashboard
- Production practices: modular source code, tests, Dockerfile, Makefile, reproducible sample data

## Why this is relevant

The project demonstrates the exact areas required in the job description: Python, SQL, SciPy, statistics, NLP, machine learning, large data processing, visualization, and communication of recommendations to stakeholders.
