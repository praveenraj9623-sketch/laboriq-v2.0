# Deployment Guide

## Local

```bash
pip install -r requirements.txt
python pipelines/run_pipeline.py
streamlit run app.py
```

## Docker

```bash
docker build -t labor-market-intelligence .
docker run -p 8501:8501 labor-market-intelligence
```

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Set `app.py` as the main file.
3. Add secrets only if using Adzuna API.
4. The sample dataset allows the app to run without external credentials.
