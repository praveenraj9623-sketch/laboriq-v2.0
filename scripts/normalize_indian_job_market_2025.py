import re
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/external/indian_job_market_2025.xlsx")
OUTPUT_PATH = Path("data/external/kaggle_jobs_normalized.csv")
FILTERED_OUTPUT_PATH = Path("data/external/indian_job_market_2025_filtered.csv")
FULL_NORMALIZED_OUTPUT_PATH = Path("data/external/indian_job_market_2025_full_normalized.csv")


ROLE_KEYWORDS = [
    "data scientist",
    "data science",
    "data analyst",
    "data analytics",
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
    "deep learning",
    "analytics",
    "power bi",
    "tableau",
    "sql",
    "python",
    "pandas",
    "numpy",
    "scikit",
    "statistics",
    "statistical",
    "forecasting",
    "predictive modeling",
    "big data",
    "spark",
    "hadoop",
    "etl",
    "data warehouse",
    "data warehousing",
    "cloud data",
    "azure data",
    "aws data",
    "gcp data",
    "llm",
    "genai",
    "generative ai",
    "rag",
    "prompt engineering",
]


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_html(text: str) -> str:
    text = safe_text(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_currency(value) -> str:
    text = safe_text(value).upper()
    if not text:
        return "INR"
    return text


def convert_salary_to_lpa(value, currency="INR"):
    """
    Converts salary values into INR LPA.

    Handles:
    - minimumSalary / maximumSalary as yearly INR amounts
    - already-LPA values
    - text like 3-5 Lacs PA, 5 LPA, Not disclosed
    """
    if pd.isna(value):
        return pd.NA

    text = str(value).lower().strip()

    if text in ["", "nan", "none", "not disclosed", "not available", "undisclosed"]:
        return pd.NA

    # Direct LPA/lakh/lac format
    if any(word in text for word in ["lpa", "lakh", "lakhs", "lac", "lacs"]):
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if nums:
            values = [float(n) for n in nums]
            return round(max(values), 2)

    # Numeric salary
    nums = re.findall(r"\d[\d,]*\.?\d*", text)
    if not nums:
        return pd.NA

    try:
        amount = float(nums[0].replace(",", ""))
    except ValueError:
        return pd.NA

    currency = normalize_currency(currency)

    # Convert foreign currency if present, but this dataset should mostly be INR
    if currency == "USD":
        amount = amount * 83
    elif currency == "EUR":
        amount = amount * 90
    elif currency == "GBP":
        amount = amount * 105

    # If annual INR salary
    if amount >= 100000:
        lpa = amount / 100000
    else:
        # If already in LPA
        lpa = amount

    if 0.5 <= lpa <= 300:
        return round(lpa, 2)

    return pd.NA


def parse_salary_text_to_min_max(text, currency="INR"):
    """
    Parses salary text like:
    3-5 Lacs PA
    8-18 Lacs PA
    Not disclosed
    """
    if pd.isna(text):
        return (pd.NA, pd.NA)

    text = str(text).lower().strip()

    if text in ["", "nan", "none", "not disclosed", "not available", "undisclosed"]:
        return (pd.NA, pd.NA)

    # LPA / lakh / lac format
    if any(word in text for word in ["lpa", "lakh", "lakhs", "lac", "lacs"]):
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if nums:
            values = [float(n) for n in nums]
            return (round(min(values), 2), round(max(values), 2))

    # Annual INR range
    nums = re.findall(r"\d[\d,]*\.?\d*", text)
    values = []

    for n in nums:
        try:
            amount = float(n.replace(",", ""))
        except ValueError:
            continue

        currency = normalize_currency(currency)

        if currency == "USD":
            amount = amount * 83
        elif currency == "EUR":
            amount = amount * 90
        elif currency == "GBP":
            amount = amount * 105

        if amount >= 100000:
            lpa = amount / 100000
        else:
            lpa = amount

        if 0.5 <= lpa <= 300:
            values.append(round(lpa, 2))

    if values:
        return (round(min(values), 2), round(max(values), 2))

    return (pd.NA, pd.NA)


def normalize_experience_from_min_max(min_exp, max_exp, raw_exp=""):
    raw_exp_text = safe_text(raw_exp).lower()

    if raw_exp_text in ["fresher", "freshers", "entry level"]:
        return "Entry"

    try:
        min_exp_num = float(min_exp)
    except Exception:
        min_exp_num = None

    if min_exp_num is None:
        nums = re.findall(r"\d+", raw_exp_text)
        if nums:
            min_exp_num = float(nums[0])

    if min_exp_num is None:
        return "Unknown"

    if min_exp_num <= 1:
        return "Entry"
    if min_exp_num <= 3:
        return "Associate"
    if min_exp_num <= 6:
        return "Mid"
    if min_exp_num <= 10:
        return "Senior"

    return "Lead"


def convert_uploaded_date(value):
    today = pd.Timestamp.today().normalize()

    if pd.isna(value):
        return today

    text = str(value).lower().strip()

    if text in ["", "nan", "none", "not disclosed"]:
        return today

    if "day" in text:
        nums = re.findall(r"\d+", text)
        days = int(nums[0]) if nums else 1
        return today - pd.Timedelta(days=days)

    if "month" in text:
        nums = re.findall(r"\d+", text)
        months = int(nums[0]) if nums else 1
        return today - pd.DateOffset(months=months)

    if "year" in text:
        nums = re.findall(r"\d+", text)
        years = int(nums[0]) if nums else 1
        return today - pd.DateOffset(years=years)

    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return today

    return parsed


def make_search_text(row):
    parts = [
        safe_text(row.get("title")),
        safe_text(row.get("tagsAndSkills")),
        safe_text(row.get("jobDescription")),
        safe_text(row.get("experience")),
        safe_text(row.get("location")),
    ]

    return " ".join(parts).lower()


def is_relevant_data_ai_analytics_job(text: str) -> bool:
    text = safe_text(text).lower()

    if not text:
        return False

    for keyword in ROLE_KEYWORDS:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, text):
            return True

    return False


def build_job_description(row):
    title = safe_text(row.get("job_title"))
    skills = safe_text(row.get("skills"))
    experience = safe_text(row.get("experience_level"))
    employment_type = safe_text(row.get("employment_type"))
    company = safe_text(row.get("company"))
    location = safe_text(row.get("location"))
    raw_description = clean_html(row.get("raw_description"))

    description = (
        f"{title} role. "
        f"Required skills: {skills}. "
        f"Experience level: {experience}. "
        f"Employment type: {employment_type}. "
        f"Company: {company}. "
        f"Location: {location}. "
        f"Job description: {raw_description}"
    )

    return re.sub(r"\s+", " ", description).strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing data/external/indian_job_market_2025.xlsx. "
            "Place the Indian Job Market 2025 Excel file inside data/external."
        )

    print("Reading Excel file. Please wait...")
    df = pd.read_excel(INPUT_PATH, sheet_name=0)

    print("\nOriginal shape:", df.shape)
    print("Original columns:")
    print(df.columns.tolist())

    expected_cols = [
        "title",
        "jobId",
        "currency",
        "jobUploaded",
        "companyName",
        "tagsAndSkills",
        "experience",
        "salary",
        "location",
        "companyId",
        "ReviewsCount",
        "AggregateRating",
        "jobDescription",
        "minimumSalary",
        "maximumSalary",
        "minimumExperience",
        "maximumExperience",
    ]

    missing_cols = [c for c in expected_cols if c not in df.columns]

    if missing_cols:
        print("\nWarning: Missing expected columns:")
        print(missing_cols)

    # Build text for filtering
    df["_search_text"] = df.apply(make_search_text, axis=1)

    # Filter for Data / AI / Analytics roles
    filtered_df = df[df["_search_text"].apply(is_relevant_data_ai_analytics_job)].copy()

    print("\nFiltered Data/AI/Analytics shape:", filtered_df.shape)

    if filtered_df.empty:
        raise ValueError(
            "Filter returned 0 rows. Check column names or update ROLE_KEYWORDS."
        )

    normalized = pd.DataFrame()

    normalized["job_id"] = filtered_df["jobId"].apply(
        lambda x: "INDIA2025_" + safe_text(x)
        if safe_text(x)
        else "INDIA2025_UNKNOWN"
    )

    normalized["posted_date"] = filtered_df["jobUploaded"].apply(convert_uploaded_date)

    normalized["company"] = filtered_df["companyName"].apply(
        lambda x: safe_text(x) or "Unknown Company"
    )

    normalized["job_title"] = filtered_df["title"].apply(
        lambda x: safe_text(x) or "Unknown Role"
    )

    normalized["location"] = filtered_df["location"].apply(
        lambda x: safe_text(x) or "Unknown Location"
    )

    normalized["employment_type"] = "full_time"

    normalized["experience_level"] = filtered_df.apply(
        lambda row: normalize_experience_from_min_max(
            row.get("minimumExperience"),
            row.get("maximumExperience"),
            row.get("experience"),
        ),
        axis=1,
    )

    currency_series = filtered_df["currency"] if "currency" in filtered_df.columns else "INR"

    # Use minimumSalary / maximumSalary first
    normalized["salary_min_lpa"] = [
        convert_salary_to_lpa(sal, cur)
        for sal, cur in zip(filtered_df["minimumSalary"], currency_series)
    ]

    normalized["salary_max_lpa"] = [
        convert_salary_to_lpa(sal, cur)
        for sal, cur in zip(filtered_df["maximumSalary"], currency_series)
    ]

    # Fallback to salary text if numeric salary fields are missing
    salary_text_parsed = [
        parse_salary_text_to_min_max(sal, cur)
        for sal, cur in zip(filtered_df["salary"], currency_series)
    ]

    salary_text_min = pd.Series([x[0] for x in salary_text_parsed], index=filtered_df.index)
    salary_text_max = pd.Series([x[1] for x in salary_text_parsed], index=filtered_df.index)

    normalized["salary_min_lpa"] = pd.to_numeric(
        normalized["salary_min_lpa"], errors="coerce"
    )

    normalized["salary_max_lpa"] = pd.to_numeric(
        normalized["salary_max_lpa"], errors="coerce"
    )

    normalized["salary_min_lpa"] = normalized["salary_min_lpa"].fillna(
        pd.to_numeric(salary_text_min.reset_index(drop=True), errors="coerce")
    )

    normalized["salary_max_lpa"] = normalized["salary_max_lpa"].fillna(
        pd.to_numeric(salary_text_max.reset_index(drop=True), errors="coerce")
    )

    normalized["skills"] = filtered_df["tagsAndSkills"].apply(safe_text)
    normalized["raw_description"] = filtered_df["jobDescription"].apply(clean_html)

    normalized["job_description"] = normalized.apply(build_job_description, axis=1)

    normalized["source"] = "indian_job_market_2025"

    final_df = normalized[
        [
            "job_id",
            "posted_date",
            "company",
            "job_title",
            "location",
            "employment_type",
            "experience_level",
            "salary_min_lpa",
            "salary_max_lpa",
            "job_description",
            "source",
        ]
    ].copy()

    final_df = final_df.drop_duplicates(subset=["job_id"])
    final_df = final_df.drop_duplicates(subset=["job_title", "company", "location"])

    # Save full normalized filtered dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    filtered_df.to_csv(FILTERED_OUTPUT_PATH, index=False)
    final_df.to_csv(OUTPUT_PATH, index=False)
    final_df.to_csv(FULL_NORMALIZED_OUTPUT_PATH, index=False)

    print("\nSaved filtered raw data:", FILTERED_OUTPUT_PATH)
    print("Saved normalized project data:", OUTPUT_PATH)
    print("Saved backup normalized data:", FULL_NORMALIZED_OUTPUT_PATH)

    print("\nFinal normalized shape:", final_df.shape)

    print("\nPreview:")
    print(final_df.head())

    print("\nSource distribution:")
    print(final_df["source"].value_counts(dropna=False))

    print("\nExperience distribution:")
    print(final_df["experience_level"].value_counts(dropna=False))

    print("\nSalary missing count:", final_df["salary_min_lpa"].isna().sum())
    print("Salary available count:", final_df["salary_min_lpa"].notna().sum())

    print("\nTop 20 job titles:")
    print(final_df["job_title"].value_counts().head(20))

    print("\nTop 20 locations:")
    print(final_df["location"].value_counts().head(20))


if __name__ == "__main__":
    main()