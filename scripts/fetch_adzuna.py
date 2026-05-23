from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.adzuna_client import AdzunaSearchSpec, fetch_with_cache


def _parse_queries(single_query: str | None, queries: list[str] | None) -> list[str]:
    values: list[str] = []
    if single_query:
        values.append(single_query)
    if queries:
        values.extend(queries)
    cleaned = [q.strip() for q in values if q and q.strip()]
    return cleaned or ["data scientist", "data analyst", "machine learning engineer"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch live job postings from the official Adzuna API with caching and multi-query support."
    )
    parser.add_argument("--query", default=None, help="Single job-search query, e.g. 'data scientist'")
    parser.add_argument("--queries", nargs="*", default=None, help="Multiple queries, e.g. 'data scientist' 'data analyst'")
    parser.add_argument("--location", default="Chennai", help="Search location, e.g. Chennai, India, Bangalore")
    parser.add_argument("--country", default="in", help="Adzuna country code, e.g. in, gb, us")
    parser.add_argument("--pages", type=int, default=1, help="Pages per query. Keep low on free quota.")
    parser.add_argument("--results-per-page", type=int, default=25, help="Results per page. Free quota friendly default: 25")
    parser.add_argument("--max-days-old", type=int, default=30, help="Limit to jobs posted in last N days")
    parser.add_argument("--salary-min", type=int, default=None, help="Optional minimum annual salary filter")
    parser.add_argument("--full-time", action="store_true", help="Only full-time roles where supported")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore local cache and call the API again")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/external/adzuna_jobs.csv")
    args = parser.parse_args()

    queries = _parse_queries(args.query, args.queries)
    specs = [
        AdzunaSearchSpec(
            query=q,
            location=args.location,
            pages=args.pages,
            max_days_old=args.max_days_old,
            salary_min=args.salary_min,
            full_time=True if args.full_time else None,
        )
        for q in queries
    ]

    df = fetch_with_cache(
        specs=specs,
        output_path=args.output,
        country=args.country,
        results_per_page=args.results_per_page,
        force_refresh=args.force_refresh,
    )
    print(f"Saved {len(df)} deduplicated Adzuna rows to {args.output}")
    print("Next step:")
    print(f"python pipelines/run_pipeline.py --external {args.output}")
