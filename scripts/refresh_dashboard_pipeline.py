from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(command, step_name):
    print("\n" + "=" * 80)
    print(f"STARTING: {step_name}")
    print("COMMAND:", " ".join(command))
    print("=" * 80)

    start_time = time.time()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    elapsed = round(time.time() - start_time, 2)

    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with return code {result.returncode}"
        )

    print(f"COMPLETED: {step_name} in {elapsed} seconds")


def main():
    print("Refreshing dashboard pipeline...")
    print("Project root:", PROJECT_ROOT)

    run_step(
        [sys.executable, "scripts/combine_external_data.py"],
        "Combine India dataset + latest Adzuna jobs",
    )

    run_step(
        [sys.executable, "scripts/create_portfolio_dataset.py"],
        "Create smaller portfolio dataset",
    )

    run_step(
        [
            sys.executable,
            "pipelines/run_pipeline.py",
            "--external",
            "data/external/combined_real_job_data_portfolio.csv",
        ],
        "Run full analytics/modeling/forecasting pipeline",
    )

    print("\n" + "=" * 80)
    print("Dashboard refresh completed successfully.")
    print("Now refresh Streamlit browser with Ctrl + R.")
    print("=" * 80)


if __name__ == "__main__":
    main()