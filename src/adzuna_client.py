from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "external" / "adzuna_cache"


@dataclass
class AdzunaConfig:
    """
    Backward-compatible config class.

    Some older project modules import AdzunaConfig from src.adzuna_client.
    The newer Streamlit live ingestion uses AdzunaSearchSpec + fetch_with_cache,
    but we keep this class so old imports do not break.
    """
    app_id: str | None = None
    app_key: str | None = None
    country: str = "in"
    results_per_page: int = 25
    request_sleep_seconds: float = 1.0

@dataclass
class AdzunaSearchSpec:
    """
    Search specification used by the Streamlit Live Adzuna API page.
    Each object represents one query/location/page configuration.
    """
    query: str
    location: str = "India"
    pages: int = 1
    max_days_old: int = 45
    
def load_env_file() -> None:
    """
    Lightweight .env loader so this module works even if python-dotenv
    is not installed.
    """
    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def get_adzuna_credentials() -> tuple[str, str]:
    load_env_file()

    app_id = os.getenv("ADZUNA_APP_ID", "").strip()
    app_key = os.getenv("ADZUNA_APP_KEY", "").strip()

    if not app_id or not app_key:
        raise ValueError(
            "Missing Adzuna credentials. Add ADZUNA_APP_ID and ADZUNA_APP_KEY "
            "inside your .env file."
        )

    return app_id, app_key


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def clean_text(value: Any) -> str:
    text = safe_text(value)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_job_id(value: str) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return f"ADZUNA_{digest}"


def convert_api_salary_to_lpa(value: Any) -> Any:
    """
    Converts Adzuna salary value into INR LPA.

    For Adzuna India, salary values are usually annual INR when present.
    If the value already looks like LPA, we keep it as LPA.
    """
    if value is None:
        return pd.NA

    try:
        if pd.isna(value):
            return pd.NA
    except Exception:
        pass

    try:
        amount = float(value)
    except Exception:
        return pd.NA

    if amount <= 0:
        return pd.NA

    # Annual INR salary -> LPA
    if amount >= 100000:
        lpa = amount / 100000
    else:
        # Already LPA-like
        lpa = amount

    if 0.5 <= lpa <= 300:
        return round(lpa, 2)

    return pd.NA


def parse_salary_text_to_lpa(text: str) -> tuple[Any, Any, str]:
    """
    Extracts salary from job text only when explicitly present.

    Handles:
    - 8-12 LPA
    - 10 LPA
    - 6 to 9 lakhs
    - ₹8,00,000
    - INR 12,00,000
    - Rs. 15,00,000

    Returns:
    salary_min_lpa, salary_max_lpa, salary_source
    """
    text = clean_text(text).lower()

    if not text:
        return pd.NA, pd.NA, "missing"

    missing_terms = [
        "not disclosed",
        "undisclosed",
        "confidential",
        "salary not disclosed",
        "as per industry standards",
    ]

    if any(term in text for term in missing_terms):
        return pd.NA, pd.NA, "missing"

    # Example: 8-12 LPA, 8 to 12 LPA, 8 lakhs to 12 lakhs
    lpa_range_patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakhs|lakh|lacs|lac)",
        r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs|lakh|lacs|lac)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakhs|lakh|lacs|lac)?",
    ]

    for pattern in lpa_range_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            first = float(match.group(1))
            second = float(match.group(2))

            if 0.5 <= first <= 300 and 0.5 <= second <= 300:
                return round(min(first, second), 2), round(max(first, second), 2), "text_derived"

    # Example: 10 LPA, 8 lakhs
    single_lpa_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs|lakh|lacs|lac)",
        text,
        flags=re.IGNORECASE,
    )

    if single_lpa_match:
        value = float(single_lpa_match.group(1))

        if 0.5 <= value <= 300:
            return round(value, 2), round(value, 2), "text_derived"

    # Example: ₹800000, INR 1200000, Rs. 10,00,000
    inr_patterns = [
        r"(?:₹|rs\.?|inr)\s*([\d,]+)",
        r"([\d,]+)\s*(?:pa|per annum|per year)",
    ]

    values = []

    for pattern in inr_patterns:
        amounts = re.findall(pattern, text, flags=re.IGNORECASE)

        for amount in amounts:
            try:
                amount_num = float(str(amount).replace(",", ""))
            except ValueError:
                continue

            if amount_num >= 100000:
                lpa = amount_num / 100000

                if 0.5 <= lpa <= 300:
                    values.append(round(lpa, 2))

    if values:
        return min(values), max(values), "text_derived"

    return pd.NA, pd.NA, "missing"


def derive_experience_level(text: str) -> tuple[str, str]:
    """
    Derives experience level from job title and description text.

    This is not API-provided; it is text-derived.
    """
    text = clean_text(text).lower()

    if not text:
        return "Unknown", "missing"

    if any(word in text for word in ["fresher", "freshers", "graduate trainee", "intern", "entry level"]):
        return "Entry", "text_derived"

    if any(word in text for word in ["lead", "principal", "architect", "manager", "head of"]):
        return "Lead", "text_derived"

    if any(word in text for word in ["senior", "sr.", "sr "]):
        return "Senior", "text_derived"

    if any(word in text for word in ["associate", "junior", "jr.", "jr "]):
        return "Associate", "text_derived"

    # Examples:
    # 2+ years
    # 3-5 years
    # 4 to 6 years
    # minimum 5 years
    exp_patterns = [
        r"(\d+)\s*\+\s*(?:years|year|yrs|yr)",
        r"(\d+)\s*(?:-|to)\s*\d+\s*(?:years|year|yrs|yr)",
        r"(?:minimum|min\.?|at least)\s*(\d+)\s*(?:years|year|yrs|yr)",
        r"(\d+)\s*(?:years|year|yrs|yr)\s*(?:of)?\s*(?:experience|exp)",
    ]

    for pattern in exp_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            min_exp = int(match.group(1))

            if min_exp <= 1:
                return "Entry", "text_derived"
            if min_exp <= 3:
                return "Associate", "text_derived"
            if min_exp <= 6:
                return "Mid", "text_derived"
            if min_exp <= 10:
                return "Senior", "text_derived"

            return "Lead", "text_derived"

    return "Unknown", "missing"


def normalize_employment_type(item: dict[str, Any], text: str) -> str:
    contract_time = safe_text(item.get("contract_time")).lower()
    contract_type = safe_text(item.get("contract_type")).lower()
    combined = f"{contract_time} {contract_type} {text.lower()}"

    if "full_time" in contract_time or "full time" in combined or "full-time" in combined:
        return "full_time"

    if "part_time" in contract_time or "part time" in combined or "part-time" in combined:
        return "part_time"

    if "contract" in combined:
        return "contract"

    if "permanent" in combined:
        return "permanent"

    if "remote" in combined:
        return "remote"

    if "hybrid" in combined:
        return "hybrid"

    return "Unknown"


def get_nested_display_name(item: dict[str, Any], key: str) -> str:
    value = item.get(key)

    if isinstance(value, dict):
        return safe_text(value.get("display_name"))

    return safe_text(value)


def normalize_adzuna_item(
    item: dict[str, Any],
    query: str,
    source_location: str,
) -> dict[str, Any]:
    title = clean_text(item.get("title"))
    description = clean_text(item.get("description"))
    company = get_nested_display_name(item, "company") or "Unknown Company"
    location = get_nested_display_name(item, "location") or source_location or "Unknown Location"
    category = get_nested_display_name(item, "category")

    combined_text = f"{title}. {description}"

    raw_id = safe_text(item.get("id"))
    job_id = f"ADZUNA_{raw_id}" if raw_id else stable_job_id(f"{title}|{company}|{location}|{description[:120]}")

    api_salary_min_lpa = convert_api_salary_to_lpa(item.get("salary_min"))
    api_salary_max_lpa = convert_api_salary_to_lpa(item.get("salary_max"))

    text_salary_min_lpa, text_salary_max_lpa, text_salary_source = parse_salary_text_to_lpa(combined_text)

    if pd.notna(api_salary_min_lpa) or pd.notna(api_salary_max_lpa):
        salary_min_lpa = api_salary_min_lpa
        salary_max_lpa = api_salary_max_lpa
        salary_source = "api"

        if pd.isna(salary_min_lpa):
            salary_min_lpa = salary_max_lpa

        if pd.isna(salary_max_lpa):
            salary_max_lpa = salary_min_lpa
    else:
        salary_min_lpa = text_salary_min_lpa
        salary_max_lpa = text_salary_max_lpa
        salary_source = text_salary_source

    experience_level, experience_source = derive_experience_level(combined_text)

    employment_type = normalize_employment_type(item, combined_text)

    created = pd.to_datetime(item.get("created"), errors="coerce")

    return {
        "job_id": job_id,
        "posted_date": created,
        "company": company,
        "job_title": title or "Unknown Role",
        "location": location,
        "employment_type": employment_type,
        "experience_level": experience_level,
        "experience_source": experience_source,
        "salary_min_lpa": salary_min_lpa,
        "salary_max_lpa": salary_max_lpa,
        "salary_source": salary_source,
        "salary_is_predicted": item.get("salary_is_predicted", pd.NA),
        "api_salary_min_raw": item.get("salary_min", pd.NA),
        "api_salary_max_raw": item.get("salary_max", pd.NA),
        "job_description": description or title,
        "category": category,
        "source_query": query,
        "source_location": source_location,
        "redirect_url": item.get("redirect_url", ""),
        "source": "adzuna_api",
    }


def make_cache_key(
    country: str,
    query: str,
    location: str,
    page: int,
    results_per_page: int,
    max_days_old: int,
) -> str:
    raw = f"{country}|{query}|{location}|{page}|{results_per_page}|{max_days_old}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_cache_key(
    specs: list[AdzunaSearchSpec],
    country: str = "in",
    results_per_page: int = 25,
) -> str:
    """
    Backward-compatible aggregate cache key for older tests/scripts.
    """
    payload = {
        "country": country,
        "results_per_page": int(results_per_page),
        "specs": [
            {
                "query": spec.query,
                "location": spec.location,
                "pages": int(spec.pages),
                "max_days_old": int(spec.max_days_old),
            }
            for spec in specs
        ],
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def normalize_adzuna_results(
    payload: dict[str, Any],
    query: str,
    location: str,
) -> pd.DataFrame:
    """
    Backward-compatible payload normalizer for older tests/scripts.
    """
    rows = [
        normalize_adzuna_item(item=item, query=query, source_location=location)
        for item in payload.get("results", [])
    ]
    return pd.DataFrame(rows)


def fetch_adzuna_page(
    *,
    country: str,
    query: str,
    location: str,
    page: int,
    results_per_page: int,
    max_days_old: int,
) -> list[dict[str, Any]]:
    app_id, app_key = get_adzuna_credentials()

    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "what": query,
        "where": location,
        "content-type": "application/json",
        "max_days_old": max_days_old,
    }

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Adzuna API request failed: {response.status_code} - {response.text[:500]}"
        )

    payload = response.json()

    return payload.get("results", [])


def fetch_with_cache(
    specs: list[AdzunaSearchSpec],
    output_path: Path,
    country: str = "in",
    results_per_page: int = 25,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetches jobs from Adzuna, enriches salary/experience transparently,
    saves CSV, and returns dataframe.
    """
    output_path = Path(output_path)
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    sleep_seconds = float(os.getenv("ADZUNA_REQUEST_SLEEP_SECONDS", "1.0"))

    all_items: list[dict[str, Any]] = []

    for spec in specs:
        query = spec.query.strip()
        location = spec.location.strip() or "India"

        if not query:
            continue

        for page in range(1, int(spec.pages) + 1):
            cache_key = make_cache_key(
                country=country,
                query=query,
                location=location,
                page=page,
                results_per_page=int(results_per_page),
                max_days_old=int(spec.max_days_old),
            )

            cache_path = cache_dir / f"{cache_key}.json"

            if cache_path.exists() and not force_refresh:
                cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                items = cached_payload.get("results", [])
            else:
                items = fetch_adzuna_page(
                    country=country,
                    query=query,
                    location=location,
                    page=page,
                    results_per_page=int(results_per_page),
                    max_days_old=int(spec.max_days_old),
                )

                cache_path.write_text(
                    json.dumps({"results": items}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                time.sleep(sleep_seconds)

            for item in items:
                item["_source_query"] = query
                item["_source_location"] = location

            all_items.extend(items)

    rows = [
        normalize_adzuna_item(
            item=item,
            query=item.get("_source_query", ""),
            source_location=item.get("_source_location", ""),
        )
        for item in all_items
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    # Clean duplicates
    if "job_id" in df.columns:
        df = df.drop_duplicates(subset=["job_id"])
    else:
        df = df.drop_duplicates()

    # Extra duplicate protection when IDs differ across same job
    duplicate_subset = ["job_title", "company", "location"]
    if all(col in df.columns for col in duplicate_subset):
        df = df.drop_duplicates(subset=duplicate_subset)

    # Convert salary to numeric after fallback logic
    for col in ["salary_min_lpa", "salary_max_lpa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill max/min if one side is present
    if "salary_min_lpa" in df.columns and "salary_max_lpa" in df.columns:
        df["salary_min_lpa"] = df["salary_min_lpa"].fillna(df["salary_max_lpa"])
        df["salary_max_lpa"] = df["salary_max_lpa"].fillna(df["salary_min_lpa"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Adzuna fetch completed.")
    print("Rows saved:", len(df))

    if "salary_source" in df.columns:
        print("\nSalary source distribution:")
        print(df["salary_source"].value_counts(dropna=False))

    if "experience_source" in df.columns:
        print("\nExperience source distribution:")
        print(df["experience_source"].value_counts(dropna=False))

    return df

def fetch_adzuna_jobs(
    queries=None,
    location="India",
    country=None,
    pages=1,
    results_per_page=None,
    max_days_old=45,
    output_path=None,
    force_refresh=False,
    config=None,
    **kwargs,
):
    """
    Backward-compatible wrapper around fetch_with_cache.

    Older project modules import and call fetch_adzuna_jobs().
    New Streamlit live ingestion uses AdzunaSearchSpec + fetch_with_cache().
    This wrapper keeps both old and new code working.
    """

    # Support old-style config object
    if config is not None:
        country = country or getattr(config, "country", "in")
        results_per_page = results_per_page or getattr(config, "results_per_page", 25)

    # Support if first argument is an AdzunaConfig object
    if isinstance(queries, AdzunaConfig):
        config = queries
        queries = None
        country = country or getattr(config, "country", "in")
        results_per_page = results_per_page or getattr(config, "results_per_page", 25)

    # Support alternative keyword names from older scripts
    if queries is None:
        queries = kwargs.get("search_queries") or kwargs.get("query_terms") or kwargs.get("keywords")

    if queries is None:
        queries = [
            "data scientist",
            "data analyst",
            "machine learning engineer",
            "ai engineer",
            "nlp engineer",
        ]

    if isinstance(queries, str):
        queries = [queries]

    country = country or "in"
    results_per_page = results_per_page or 25

    specs = [
        AdzunaSearchSpec(
            query=str(query),
            location=location,
            pages=int(pages),
            max_days_old=int(max_days_old),
        )
        for query in queries
        if str(query).strip()
    ]

    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "external" / "adzuna_jobs.csv"

    return fetch_with_cache(
        specs=specs,
        output_path=Path(output_path),
        country=country,
        results_per_page=int(results_per_page),
        force_refresh=force_refresh,
    )
    

def fetch_multiple_adzuna_jobs(
    queries=None,
    location="India",
    country=None,
    pages=1,
    results_per_page=None,
    max_days_old=45,
    output_path=None,
    force_refresh=False,
    config=None,
    **kwargs,
):
    """
    Backward-compatible wrapper for older project modules.

    Older code imports fetch_multiple_adzuna_jobs().
    The upgraded code uses fetch_with_cache().
    This function connects the old name to the new implementation.
    """

    if config is not None:
        country = country or getattr(config, "country", "in")
        results_per_page = results_per_page or getattr(config, "results_per_page", 25)

    if queries is None:
        queries = kwargs.get("search_queries") or kwargs.get("query_terms") or kwargs.get("keywords")

    if queries is None:
        queries = [
            "data scientist",
            "data analyst",
            "business analyst",
            "data engineer",
            "machine learning engineer",
            "ai engineer",
            "nlp engineer",
            "business intelligence analyst",
        ]

    if isinstance(queries, str):
        queries = [queries]

    country = country or "in"
    results_per_page = results_per_page or 25

    specs = [
        AdzunaSearchSpec(
            query=str(query),
            location=location,
            pages=int(pages),
            max_days_old=int(max_days_old),
        )
        for query in queries
        if str(query).strip()
    ]

    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "external" / "adzuna_jobs.csv"

    return fetch_with_cache(
        specs=specs,
        output_path=Path(output_path),
        country=country,
        results_per_page=int(results_per_page),
        force_refresh=force_refresh,
    )
