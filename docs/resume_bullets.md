# Resume Bullets

## Labor Market Intelligence & Skills Demand Analytics Platform

**Tech:** Python, SQL, DuckDB, Pandas, NumPy, SciPy, NLP, TF-IDF, scikit-learn, Logistic Regression, Ridge Regression, Plotly, Streamlit, Adzuna API

- Built a Lightcast-inspired labor-market intelligence platform that converts raw job postings into structured insights on skill demand, job-role categories, salary ranges, location demand, and candidate skill gaps.
- Cleaned, normalized, and analyzed job-posting datasets using Python, Pandas, and DuckDB SQL to identify trends across skills, roles, locations, companies, and salary ranges.
- Developed an NLP-based skill extraction engine using a controlled skills taxonomy to standardize unstructured job-description text into skill labels.
- Built a hybrid occupation mapping engine using keyword rules and TF-IDF similarity to map job postings into role labels and occupation families.
- Trained a TF-IDF + Logistic Regression classifier to predict job-role categories from job descriptions and extracted skills.
- Built a salary intelligence baseline model using text, location, role, experience level, and skill-count features.
- Developed an emerging-skills trend analyzer that calculates monthly skill demand, growth signals, trend labels, and 6-month future demand forecasts using SciPy optimization.
- Added a SciPy statistical validation layer using Spearman correlation, Kruskal-Wallis, chi-square, Mann-Whitney U, and cosine similarity to validate salary, role, experience, and skill-demand patterns.
- Created an interactive Streamlit dashboard for role demand, top skills, location-wise hiring demand, salary analytics, skill forecasts, job exploration, and stakeholder recommendations.
