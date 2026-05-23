from pathlib import Path
import pandas as pd

files = [
    Path("data/external/kaggle_jobs_normalized.csv"),
    Path("data/external/adzuna_jobs.csv"),
]

frames = []

for file in files:
    if file.exists():
        df = pd.read_csv(file)
        frames.append(df)
        print(f"Loaded {file}: {df.shape}")
    else:
        print(f"Missing: {file}")

if not frames:
    raise FileNotFoundError("No external files found.")

combined = pd.concat(frames, ignore_index=True)

if "job_id" in combined.columns:
    combined = combined.drop_duplicates(subset=["job_id"])

output = Path("data/external/combined_real_job_data.csv")
combined.to_csv(output, index=False)

print("Saved:", output)
print("Final shape:", combined.shape)
print(combined.head())