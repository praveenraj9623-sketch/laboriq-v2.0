# Live Data Architecture

## Data flow

```text
Adzuna API
   ↓
src/adzuna_client.py
   ↓
data/external/adzuna_jobs.csv + data/external/adzuna_cache/
   ↓
python pipelines/run_pipeline.py --external data/external/adzuna_jobs.csv
   ↓
cleaning → skill extraction → occupation mapping → analytics → forecasting → models
   ↓
Streamlit dashboard
```

## Why Adzuna is used

- It is an official API, not risky scraping.
- It provides fresh job posting data.
- It adds real-world API integration to the portfolio.
- It strengthens the Lightcast alignment because the project becomes a real labor-market intelligence workflow.

## Production-style features added

- Environment-based credentials through `.env`
- Multi-query extraction
- Pagination control
- API retry handling
- Rate-limit friendly sleep
- Local caching to reduce repeated API hits
- CSV persistence for reproducibility
- Deduplication by Adzuna job ID
- Source tracking through `source=adzuna_api`
- Static + live data merge through `--external`
- Dashboard page for live ingestion
