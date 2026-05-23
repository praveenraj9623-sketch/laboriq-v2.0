from __future__ import annotations

"""Convenience script for Praveen's Lightcast-focused portfolio demo.

It fetches live India job postings for Data Science / Analytics / AI roles from
Adzuna and saves them to data/external/adzuna_lightcast_roles.csv.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.adzuna_client import AdzunaSearchSpec, fetch_with_cache

LIGHTCAST_QUERIES = [
    "data scientist",
    "data analyst",
    "machine learning engineer",
    "artificial intelligence engineer",
    "nlp engineer",
    "data engineer",
    "business analyst",
]

if __name__ == "__main__":
    specs = [
        AdzunaSearchSpec(query=query, location="India", pages=1, max_days_old=45)
        for query in LIGHTCAST_QUERIES
    ]
    output = PROJECT_ROOT / "data/external/adzuna_lightcast_roles.csv"
    df = fetch_with_cache(specs, output_path=output, country="in", results_per_page=25)
    print(f"Saved {len(df)} rows to {output}")
    print(f"Run: python pipelines/run_pipeline.py --external {output}")
