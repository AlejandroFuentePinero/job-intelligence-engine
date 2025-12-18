# src/job_intel/pipelines/chapter3_individual_positioning.py


from pathlib import Path

from src.job_intel.positioning import run_positioning


OUT_DIR = Path("reports/chapter3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    profile, jobs_df, gaps_df = run_positioning(
        skill_text=(
            "Python, SQL, machine learning, statistics, data analysis, "
            "scikit-learn, pandas, experimentation"
        ),
        current_state="CA",
        job_title_family="data_scientist",
        salary_target=150000,
    )

    jobs_df.to_csv(OUT_DIR / "ranked_jobs.csv", index=False)
    gaps_df.to_csv(OUT_DIR / "skill_gaps.csv", index=False)


if __name__ == "__main__":
    main()
