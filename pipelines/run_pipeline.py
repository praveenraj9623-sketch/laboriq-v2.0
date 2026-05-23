from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allows running as: python pipelines/run_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_JOBS_PATH, DEFAULT_SKILLS_PATH, DEFAULT_OCCUPATIONS_PATH, REPORTS_DIR, MODELS_DIR, PROCESSED_DIR
from src.utils import ensure_dirs, save_json, get_logger
from src.data_ingestion import load_job_postings, load_taxonomy
import pandas as pd
from src.data_cleaning import clean_job_postings
from src.skill_extractor import SkillExtractor
from src.occupation_mapper import OccupationMapper
from src.analytics import create_skill_fact_table, run_core_analytics
from src.features import make_modeling_frame
from src.modeling import train_models
from src.forecasting import build_workforce_forecast
from src.scipy_insights import run_statistical_tests, build_role_skill_similarity
from src.evaluation import data_quality_report

logger = get_logger("pipeline")


def run(input_path: Path, skills_path: Path, occupations_path: Path, forecast_periods: int = 6, external_paths: list[Path] | None = None) -> None:
    ensure_dirs(REPORTS_DIR, MODELS_DIR, PROCESSED_DIR)
    logger.info("Loading data")
    raw_frames = [load_job_postings(input_path)]
    external_paths = external_paths or []
    for ext_path in external_paths:
        if ext_path.exists():
            logger.info("Appending external data: %s", ext_path)
            raw_frames.append(load_job_postings(ext_path))
        else:
            logger.warning("External data file not found and will be skipped: %s", ext_path)
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    skills_tax = load_taxonomy(skills_path)
    occ_tax = load_taxonomy(occupations_path)

    logger.info("Cleaning job postings")
    jobs = clean_job_postings(raw)
    data_quality_report(jobs).to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)

    logger.info("Extracting skills")
    extractor = SkillExtractor(skills_tax)
    jobs = extractor.transform_dataframe(jobs)

    logger.info("Mapping occupations")
    mapper = OccupationMapper(occ_tax)
    jobs = mapper.transform_dataframe(jobs)

    logger.info("Building analytics tables")
    skill_fact = create_skill_fact_table(jobs)
    analytics = run_core_analytics(jobs, skill_fact)

    logger.info("Building workforce forecast tables")
    forecast_outputs = build_workforce_forecast(analytics["monthly_skill_demand"], periods=forecast_periods)

    logger.info("Running SciPy statistical validation and role-skill similarity")
    scipy_outputs = {
        "statistical_tests": run_statistical_tests(jobs, skill_fact),
        "role_skill_similarity": build_role_skill_similarity(skill_fact),
    }

    jobs_model = make_modeling_frame(jobs)
    logger.info("Training models")
    artifacts = train_models(jobs_model, MODELS_DIR)

    logger.info("Saving outputs")
    jobs_model.to_csv(REPORTS_DIR / "processed_job_postings.csv", index=False)
    skill_fact.to_csv(REPORTS_DIR / "skill_fact_table.csv", index=False)
    for name, table in analytics.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
    for name, table in forecast_outputs.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
    for name, table in scipy_outputs.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
    save_json(artifacts.metrics, REPORTS_DIR / "model_metrics.json")
    logger.info("Pipeline complete. Open Streamlit with: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lightcast-style labor market intelligence pipeline")
    parser.add_argument("--input", type=Path, default=DEFAULT_JOBS_PATH)
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS_PATH)
    parser.add_argument("--occupations", type=Path, default=DEFAULT_OCCUPATIONS_PATH)
    parser.add_argument("--forecast-periods", type=int, default=6)
    parser.add_argument("--external", nargs="*", type=Path, default=[], help="Optional external CSV files such as data/external/adzuna_jobs.csv to append to the base dataset")
    args = parser.parse_args()
    run(args.input, args.skills, args.occupations, args.forecast_periods, args.external)
