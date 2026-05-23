from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


@dataclass
class AdzunaConfig:
    app_id: str
    app_key: str
    country: str = "in"
    results_per_page: int = 50


def load_job_postings(path: str | Path) -> pd.DataFrame:
    """Load a CSV job-postings file. Works with the sample file and most Kaggle exports.

    Expected core columns can be mapped later by `standardize_columns`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def load_taxonomy(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")
    return pd.read_csv(path)

# Backwards-compatible wrappers. The full production API client lives in src/adzuna_client.py.
from .adzuna_client import (  # noqa: E402
    AdzunaConfig,
    AdzunaSearchSpec,
    fetch_adzuna_jobs as _fetch_adzuna_jobs,
    fetch_multiple_adzuna_jobs,
    fetch_with_cache,
)


def fetch_adzuna_jobs(
    query: str,
    location: str = "India",
    pages: int = 1,
    config: Optional[AdzunaConfig] = None,
) -> pd.DataFrame:
    """Fetch job ads from the official Adzuna API using the production client."""
    spec = AdzunaSearchSpec(query=query, location=location, pages=pages)
    return _fetch_adzuna_jobs(spec, config=config)
