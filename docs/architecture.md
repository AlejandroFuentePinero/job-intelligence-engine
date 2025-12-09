# Job Intelligence Engine — System Architecture  
**Scope:** Chapter 0 → Salary Model (Chapter 1)  
This document maps all modules, functions, artefacts, and data flows that define the technical architecture of the Job Intelligence Engine.  
Verified against:  
- `chapter0_build_base_dataset.py`  
- `jobs_ch0.csv`  
Date: 2025-12-09

---

# 1. Overview  
The system is a modular, deterministic pipeline that transforms raw Glassdoor job postings into structured features, engineered skills, PCA components, and an XGBoost salary model.  
All components live in `src/job_intel`.

---

# 2. Pipeline Architecture (Chapter 0)

## 2.1 Pipeline Entrypoint  
**File:** `pipelines/chapter0_build_base_dataset.py`  
**Function:** `build_chapter0_base_dataset(save=True, verbose=True)`  

**Responsibilities:**  
- Load raw Analyst/Scientist CSVs  
- Align and merge columns  
- Run feature modules sequentially  
- Handle missing data  
- Drop intermediate fields  
- Save final output as `ch0_processed_jobs.csv`

---

# 3. Feature Modules (Chapter 0)

## 3.1 Title Processing  
**Module:** `features/titles.py`  
**Outputs:**  
- `job_title_raw`  
- `job_title_base`  
- `job_title_norm`  
- `job_title_family`  

---

## 3.2 Seniority Extraction  
**Module:** `features/seniority.py`  
**Outputs:**  
- `seniority_title`  
- `seniority_description`  
- `seniority_combined`

---

## 3.3 Description Cleaning  
**Module:** `features/text_cleaning.py`  
**Outputs:**  
- `job_description_clean`

---

## 3.4 Domain Mapping  
**Module:** `features/domain.py`  
**Input:** `chapter0_domain_lookup.csv`  
**Output:**  
- `domain`

---

## 3.5 Salary Parsing  
**Module:** `features/salary.py`  
**Outputs:**  
- `sal_min`  
- `sal_max`  
- `sal_mean`  
- `sal_is_hourly`

---

## 3.6 Skill Taxonomy & Extraction  
**Modules:**  
- `skills_taxonomy.py` — curated token lists + validation  
- `skill_extractor.py` — matching + extraction functions  

**Output:**  
- Multi-hot skill flags (`core_programming__basic`, `ml_ai__advanced`, etc.)

---

# 4. Chapter 0 Output  
**File produced:** `data/processed/ch0_processed_jobs.csv`  
**Contains:**  
- Company metadata  
- Title, family, seniority  
- Domain  
- Salary fields  
- Skill multi-hot flags  
- Text fields  
- 47 stable columns  

---

# 5. Chapter 1 Architecture (Salary Model)

## 5.1 Skill PCA Transformation  
**Module:** `features/skills_pca.py`  

**Components:**  
- `SKILL_COLS`: ordered list of binary skill flags  
- `PCA_MODEL`: `models/skill_pca_v1.pkl`  

**Function:**  
- `transform_skills_to_pca()`  

**Output:**  
- `skill_PC1 … skill_PC10`

---

## 5.2 Salary Model  
**Module:** `models/salary_predictor.py`  

**Artefacts:**  
- `salary_model_v4.pkl` (XGBoost)  
- `skill_pca_v1.pkl` (PCA transformer)

**Inputs:**  
- Encoded categorical fields  
- Raw skill flags  

**Function:**  
- `predict_salary(record)`  

---

# 6. File Dependency Map

| File | Depends On | Provides |
|------|------------|----------|
| `chapter0_build_base_dataset.py` | All feature modules | Full Chapter 0 dataset |
| `titles.py` | — | Title, base, family |
| `seniority.py` | titles.py, text_cleaning.py | Seniority |
| `text_cleaning.py` | — | Cleaned description |
| `domain.py` | domain lookup | Domain |
| `salary.py` | — | Salary fields |
| `skills_taxonomy.py` | — | Token lists |
| `skill_extractor.py` | taxonomy | Skill flags |
| `skills_pca.py` | PCA artefact | PCA components |
| `salary_predictor.py` | PCA + model | Salary prediction |

---

# X. End-to-End Flow

```
RAW ANALYST/SCIENTIST CSVs
        ↓
chapter0_build_base_dataset.py
        ↓
Titles → Seniority → Description Cleaning
        ↓
Domain Lookup → Salary Parsing → Skill Extraction
        ↓
ch0_processed_jobs.csv
        ↓
skills_pca.py (PCA)
        ↓
salary_predictor.py (XGBoost v4)
        ↓
Predicted Salary
```

This is the verified system architecture for Chapter 0 and the Salary Model (Chapter 1).

---

# 7. Evaluation Layer

Evaluation tools live in `src/job_intel/evaluation/` and provide reproducible,
deterministic validation of both data (Chapter 0) and models (Chapter 1).

## 7.1 Chapter 0 — Data Benchmark Validator

**File:** `build_base_dataset_benchmark.py`  
**Purpose:** Validate the output of the Chapter 0 pipeline by comparing the newly
generated dataset to a stored benchmark dataset.

**Responsibilities:**
- Regenerate Chapter 0 dataset via `build_chapter0_base_dataset()`
- Load benchmark: `data/interim/05_skills_extracted.csv`
- Compare critical columns: structure, missing-data patterns, numeric/categorical values
- Compute match ratios + mismatch counts
- Provide diagnostic barplots of match ratios

**Usage Example:**
```python
from job_intel.evaluation.build_base_dataset_benchmark import run_benchmark
run_benchmark(show_plots=True)
```

---

## 7.2 Chapter 1 — Salary Model Evaluation

**File:** `salary_model_eval.py`  
**Purpose:** Evaluate the trained XGBoost salary model (`salary_model_v4.pkl`)
with standard metrics and diagnostic visualisations.

**Responsibilities:**
- Auto-load saved model if none provided
- Predict on train/test splits
- Compute R², RMSE, MAE (train + test)
- Produce residual diagnostics:
  - Residual histogram  
  - Residuals vs predicted  
  - Predicted vs actual (y=x reference)
- Plot feature importances (if available)

**Usage Example:**
```python
from job_intel.evaluation.salary_model_eval import evaluate_salary_model
metrics = evaluate_salary_model(X_train, y_train, X_test, y_test)
print(metrics)
```

This evaluation layer ensures consistent data integrity checks 
and stable model performance throughout the project lifecycle.
