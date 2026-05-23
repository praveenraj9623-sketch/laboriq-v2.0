import re
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/external/kaggle_jobs_raw.csv")
OUTPUT_PATH = Path("data/external/kaggle_jobs_normalized.csv")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def convert_post_date(value):
    """
    Converts values like:
    - 17 days ago
    - 3 months ago
    - a month ago
    - normal date strings

    into real dates.
    """
    today = pd.Timestamp.today().normalize()

    if pd.isna(value):
        return today

    text = str(value).lower().strip()

    if text in ["", "nan", "none"]:
        return today

    if "day" in text:
        numbers = re.findall(r"\d+", text)
        days = int(numbers[0]) if numbers else 1
        return today - pd.Timedelta(days=days)

    if "month" in text:
        numbers = re.findall(r"\d+", text)
        months = int(numbers[0]) if numbers else 1
        return today - pd.DateOffset(months=months)

    if "year" in text:
        numbers = re.findall(r"\d+", text)
        years = int(numbers[0]) if numbers else 1
        return today - pd.DateOffset(years=years)

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return today

    return parsed


def currency_to_lpa(amount: float, currency: str) -> float:
    """
    Approximate annual salary conversion to INR LPA.
    This is for portfolio analytics, not financial reporting.
    """
    if currency == "€":
        inr = amount * 90
    elif currency == "£":
        inr = amount * 105
    elif currency == "$":
        inr = amount * 83
    elif currency == "₹":
        inr = amount
    else:
        inr = amount

    return round(inr / 100000, 2)


def parse_salary_range_to_lpa(value):
    """
    Parses salary values like:
    - €100,472
    - €100,472 - €150,708
    - $120,000
    - £80,000
    - ₹1200000
    - 12 LPA
    - 10 lakh

    Returns:
    (salary_min_lpa, salary_max_lpa)
    """
    if pd.isna(value):
        return (pd.NA, pd.NA)

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none", "unknown"]:
        return (pd.NA, pd.NA)

    # Direct LPA / lakh format
    lpa_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs|lakh)",
        text,
        flags=re.IGNORECASE,
    )

    if lpa_matches:
        values = [float(x) for x in lpa_matches]
        return (round(min(values), 2), round(max(values), 2))

    # Currency format: €100,472 / $120,000 / £80,000 / ₹1200000
    currency_matches = re.findall(
        r"([€$£₹])\s*([\d,]+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    lpa_values = []

    for currency, amount_text in currency_matches:
        amount_text = amount_text.replace(",", "")

        try:
            amount = float(amount_text)
        except ValueError:
            continue

        lpa = currency_to_lpa(amount, currency)

        # Keep realistic salary values only
        if 1 <= lpa <= 300:
            lpa_values.append(lpa)

    if lpa_values:
        return (round(min(lpa_values), 2), round(max(lpa_values), 2))

    # Plain numeric fallback
    numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)
    numeric_values = []

    for number in numbers:
        try:
            amount = float(number.replace(",", ""))
        except ValueError:
            continue

        # If amount is large, assume INR annual salary
        if amount >= 100000:
            lpa = amount / 100000
        else:
            # If amount is small, assume already LPA
            lpa = amount

        if 1 <= lpa <= 300:
            numeric_values.append(round(lpa, 2))

    if numeric_values:
        return (round(min(numeric_values), 2), round(max(numeric_values), 2))

    return (pd.NA, pd.NA)


def normalize_experience_level(value):
    text = safe_text(value).lower()

    if text in ["", "nan", "none", "unknown"]:
        return "Unknown"

    if "intern" in text or "entry" in text or "junior" in text:
        return "Entry"

    if "associate" in text:
        return "Associate"

    if "mid" in text:
        return "Mid"

    if "senior" in text:
        return "Senior"

    if "lead" in text or "principal" in text or "manager" in text:
        return "Lead"

    return safe_text(value).title()


def normalize_employment_type(value):
    text = safe_text(value).lower()

    if text in ["", "nan", "none", "unknown"]:
        return "Unknown"

    if "remote" in text:
        return "remote"

    if "hybrid" in text:
        return "hybrid"

    if "on-site" in text or "onsite" in text:
        return "on-site"

    if "full" in text:
        return "full_time"

    if "part" in text:
        return "part_time"

    if "contract" in text:
        return "contract"

    return safe_text(value)


def get_column(df: pd.DataFrame, possible_names, default_value=""):
    """
    Finds first matching column from possible names.
    If not found, returns a Series with default value.
    """
    for name in possible_names:
        if name in df.columns:
            return df[name]

    return pd.Series([default_value] * len(df), index=df.index)


def build_job_description(row) -> str:
    parts = [
        f"{safe_text(row.get('job_title'))} role",
        f"requiring skills: {safe_text(row.get('skills'))}",
        f"seniority level: {safe_text(row.get('experience_level'))}",
        f"work mode/status: {safe_text(row.get('employment_type'))}",
        f"company: {safe_text(row.get('company'))}",
        f"location: {safe_text(row.get('location'))}",
        f"industry: {safe_text(row.get('industry'))}",
    ]

    return ". ".join([p for p in parts if p and not p.endswith(": ")]) + "."


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing data/external/kaggle_jobs_raw.csv. "
            "Rename your Kaggle CSV file exactly as kaggle_jobs_raw.csv "
            "and place it inside data/external."
        )

    df = pd.read_csv(INPUT_PATH)
    df = clean_column_names(df)

    print("Original Kaggle columns:")
    print(df.columns.tolist())
    print("Shape:", df.shape)

    normalized = pd.DataFrame()

    normalized["job_id"] = [
        "KAGGLE_" + str(i + 1).zfill(6) for i in range(len(df))
    ]

    normalized["posted_date"] = get_column(
        df,
        ["post_date", "posted_date", "date", "posting_date", "published_date"],
        default_value="",
    ).apply(convert_post_date)

    normalized["company"] = get_column(
        df,
        ["company", "company_name", "employer", "organization"],
        default_value="Unknown Company",
    ).apply(lambda x: safe_text(x) or "Unknown Company")

    normalized["job_title"] = get_column(
        df,
        ["job_title", "title", "role", "position"],
        default_value="Unknown Role",
    ).apply(lambda x: safe_text(x) or "Unknown Role")

    normalized["location"] = get_column(
        df,
        ["location", "city", "company_location", "employee_residence"],
        default_value="Unknown Location",
    ).apply(lambda x: safe_text(x) or "Unknown Location")

    raw_employment = get_column(
        df,
        ["status", "employment_type", "job_type", "work_mode"],
        default_value="Unknown",
    )

    normalized["employment_type"] = raw_employment.apply(normalize_employment_type)

    raw_experience = get_column(
        df,
        ["seniority_level", "experience_level", "seniority", "level"],
        default_value="Unknown",
    )

    normalized["experience_level"] = raw_experience.apply(normalize_experience_level)

    salary_source = get_column(
        df,
        ["salary", "salary_range", "compensation", "pay", "annual_salary"],
        default_value="",
    )

    salary_parsed = salary_source.apply(parse_salary_range_to_lpa)

    normalized["salary_min_lpa"] = salary_parsed.apply(lambda x: x[0])
    normalized["salary_max_lpa"] = salary_parsed.apply(lambda x: x[1])

    normalized["salary_min_lpa"] = pd.to_numeric(
        normalized["salary_min_lpa"], errors="coerce"
    )

    normalized["salary_max_lpa"] = pd.to_numeric(
        normalized["salary_max_lpa"], errors="coerce"
    )

    normalized["skills"] = get_column(
        df,
        ["skills", "skill", "required_skills", "technologies"],
        default_value="",
    ).apply(safe_text)

    normalized["industry"] = get_column(
        df,
        ["industry", "sector", "domain"],
        default_value="",
    ).apply(safe_text)

    normalized["job_description"] = normalized.apply(build_job_description, axis=1)

    normalized["source"] = "kaggle"

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
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("Final shape:", final_df.shape)
    print("\nPreview:")
    print(final_df.head())

    print("\nSalary missing count:", final_df["salary_min_lpa"].isna().sum())
    print("Salary available count:", final_df["salary_min_lpa"].notna().sum())

    print("\nSalary sample:")
    print(final_df[["job_title", "salary_min_lpa", "salary_max_lpa"]].head(10))


if __name__ == "__main__":
    main()