from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models
MODELS_DIR = PROJECT_ROOT / "models"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw dataset file name
RAW_DS_JOBS_FILE = RAW_DATA_DIR / "DataScientist.csv"
RAW_DA_JOBS_FILE = RAW_DATA_DIR / "DataAnalyst.csv"

# Interim dataset file name
INTERIM_CLEANED_JOBS = INTERIM_DATA_DIR / "01_cleaned_jobs_interim.csv"
INTERIM_NORM_TITLE = INTERIM_DATA_DIR / "03_job_titles_cleaned_domain_and_family.csv"

# Chapter 0 processed dataset
CH0_PROCESSED_JOBS_FILE = PROCESSED_DATA_DIR / "jobs_ch0.csv"
CH0_DOMAIN_LOOKUP_FILE = INTERIM_DATA_DIR / "domain_lookup_ch0.csv"
