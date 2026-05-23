from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/external/combined_real_job_data.csv")
OUTPUT_PATH = Path("data/external/combined_real_job_data_portfolio.csv")

# Keep this small for fast Streamlit + fast skill extraction
MAX_STATIC_ROWS = 1000

# Long job descriptions slow down skill extraction.
# 1200 characters is enough for skills + role matching.
MAX_DESCRIPTION_CHARS = 1200

if not INPUT_PATH.exists():
    raise FileNotFoundError("Missing data/external/combined_real_job_data.csv")

df = pd.read_csv(INPUT_PATH)

print("Original combined shape:", df.shape)
print("Columns:", df.columns.tolist())

if "source" not in df.columns:
    df["source"] = "unknown"

df["source"] = df["source"].fillna("unknown").astype(str)

# Keep all live Adzuna rows
adzuna_df = df[df["source"].str.contains("adzuna", case=False, na=False)].copy()

# Static data = Indian dataset / Kaggle / sample etc.
static_df = df[~df["source"].str.contains("adzuna", case=False, na=False)].copy()

# Make sure required columns exist
for col in ["job_title", "job_description", "location", "company", "salary_min_lpa"]:
    if col not in static_df.columns:
        static_df[col] = ""

for col in ["job_title", "job_description", "location", "company", "salary_min_lpa"]:
    if col not in adzuna_df.columns:
        adzuna_df[col] = ""

# Shorten long text before running skill extraction
static_df["job_description"] = (
    static_df["job_description"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.slice(0, MAX_DESCRIPTION_CHARS)
)

adzuna_df["job_description"] = (
    adzuna_df["job_description"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.slice(0, MAX_DESCRIPTION_CHARS)
)

# Build searchable text
static_df["_search_text"] = (
    static_df["job_title"].fillna("").astype(str)
    + " "
    + static_df["job_description"].fillna("").astype(str)
    + " "
    + static_df["location"].fillna("").astype(str)
    + " "
    + static_df["company"].fillna("").astype(str)
).str.lower()

keywords = [
    "data scientist",
    "data science",
    "data analyst",
    "business analyst",
    "business intelligence",
    "bi analyst",
    "data engineer",
    "machine learning",
    "ml engineer",
    "artificial intelligence",
    "ai engineer",
    "nlp",
    "natural language processing",
    "analytics",
    "sql",
    "python",
    "power bi",
    "tableau",
    "data visualization",
    "etl",
    "data warehouse",
    "big data",
    "spark",
    "cloud",
    "llm",
    "genai",
    "generative ai",
    "rag",
]

pattern = "|".join(keywords)

relevant_static = static_df[
    static_df["_search_text"].str.contains(pattern, case=False, na=False)
].copy()

print("Relevant static rows before sampling:", relevant_static.shape)
print("Adzuna live rows:", adzuna_df.shape)

# Prefer salary-available rows
if "salary_min_lpa" in relevant_static.columns:
    relevant_static["salary_min_lpa"] = pd.to_numeric(
        relevant_static["salary_min_lpa"],
        errors="coerce"
    )

    with_salary = relevant_static[relevant_static["salary_min_lpa"].notna()].copy()
    without_salary = relevant_static[relevant_static["salary_min_lpa"].isna()].copy()

    if len(with_salary) >= MAX_STATIC_ROWS:
        selected_static = with_salary.sample(n=MAX_STATIC_ROWS, random_state=42)
    else:
        remaining_needed = MAX_STATIC_ROWS - len(with_salary)
        sampled_without_salary = without_salary.sample(
            n=min(remaining_needed, len(without_salary)),
            random_state=42
        )
        selected_static = pd.concat(
            [with_salary, sampled_without_salary],
            ignore_index=True
        )
else:
    selected_static = relevant_static.sample(
        n=min(MAX_STATIC_ROWS, len(relevant_static)),
        random_state=42
    )

# Remove helper columns
for temp_col in ["_search_text"]:
    if temp_col in selected_static.columns:
        selected_static = selected_static.drop(columns=[temp_col])

    if temp_col in adzuna_df.columns:
        adzuna_df = adzuna_df.drop(columns=[temp_col])

# Combine selected static rows + all Adzuna rows
final_df = pd.concat([selected_static, adzuna_df], ignore_index=True, sort=False)

# Remove duplicates
if "job_id" in final_df.columns:
    final_df = final_df.drop_duplicates(subset=["job_id"])
else:
    final_df = final_df.drop_duplicates()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
print("Final portfolio shape:", final_df.shape)

print("\nSource distribution:")
print(final_df["source"].value_counts(dropna=False))

if "salary_min_lpa" in final_df.columns:
    print("\nSalary missing count:", final_df["salary_min_lpa"].isna().sum())
    print("Salary available count:", final_df["salary_min_lpa"].notna().sum())

print("\nPreview:")
print(final_df.head())