from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_JOBS_PATH = SAMPLE_DIR / "job_postings_sample.csv"
DEFAULT_SKILLS_PATH = SAMPLE_DIR / "skills_taxonomy.csv"
DEFAULT_OCCUPATIONS_PATH = SAMPLE_DIR / "occupation_taxonomy.csv"
DEFAULT_CANDIDATES_PATH = SAMPLE_DIR / "candidate_profiles.csv"

RANDOM_STATE = 42
