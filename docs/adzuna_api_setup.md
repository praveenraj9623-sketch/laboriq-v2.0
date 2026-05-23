# Adzuna API Live Ingestion Setup

This project uses Adzuna as the **safe live-data layer**. It avoids scraping restricted job boards and uses an official API for fresh job postings.

## 1. Get credentials

1. Go to the Adzuna Developer portal.
2. Register for API access.
3. Copy your `app_id` and `app_key`.

Adzuna's official overview says developers should register to receive an `app_key` and `app_id`, and its search documentation says the search endpoint retrieves job advertisement listings.

## 2. Configure local credentials

Windows:

```bash
copy .env.example .env
```

Mac/Linux:

```bash
cp .env.example .env
```

Open `.env` and fill:

```env
ADZUNA_APP_ID=your_real_app_id
ADZUNA_APP_KEY=your_real_app_key
ADZUNA_COUNTRY=in
ADZUNA_RESULTS_PER_PAGE=25
ADZUNA_REQUEST_SLEEP_SECONDS=1.0
```

## 3. Fetch live Lightcast-style job data

Recommended portfolio pull:

```bash
python scripts/fetch_lightcast_ready_adzuna_data.py
```

Custom pull:

```bash
python scripts/fetch_adzuna.py --queries "data scientist" "data analyst" "machine learning engineer" "ai engineer" --location "India" --country in --pages 1 --results-per-page 25
```

The script writes a CSV to:

```text
data/external/adzuna_jobs.csv
```

It also caches the result in:

```text
data/external/adzuna_cache/
```

## 4. Combine static + live data in the main pipeline

```bash
python pipelines/run_pipeline.py --external data/external/adzuna_jobs.csv
```

Or for the convenience script output:

```bash
python pipelines/run_pipeline.py --external data/external/adzuna_lightcast_roles.csv
```

## 5. Run the dashboard

```bash
streamlit run app.py
```

Open the **Live Adzuna API Ingestion** page in the sidebar to fetch data directly from the UI.

## Interview explanation

Say this:

> I used a hybrid data strategy. The static job-posting dataset gives reproducible model training and evaluation. The Adzuna API adds live labor-market signals, so the dashboard can refresh current job demand by role, location, salary and skill. I used caching and small page limits to protect API quota and make the workflow reproducible.
