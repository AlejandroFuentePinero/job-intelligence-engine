# Job Intelligence Engine — System Architecture
Date: 2025-12-12

This file contains the updated architecture for the project.

---

# 1. Overview
The system is a modular, deterministic pipeline transforming raw Glassdoor job postings into structured features, engineered skills, PCA components, salary models, and 27 binary skill-requirement models.

All components live in `src/job_intel`.

---

# 2. Chapter 0 Architecture (Entrypoint)

## 2.1 Pipeline Entrypoint
**File:** `pipelines/chapter0_build_base_dataset.py`  
**Function:** `build_chapter0_base_dataset()`

### Responsibilities
- Load raw DS/DA job CSVs.  
- Apply all feature modules.  
- Clean and validate fields.  
- Build final Chapter 0 dataset.  
- Save: `data/processed/ch0_processed_jobs.csv`.

---

# 3. Feature Modules

## Chapter 0 Components

### 3.1 Title Processing
**Module:** `features/titles.py`  
Outputs:
- `job_title_raw`
- `job_title_base`
- `job_title_norm`
- `job_title_family`

### 3.2 Seniority Extraction
**Module:** `features/seniority.py`  
Outputs seniority from titles + descriptions.

### 3.3 Description Cleaning
**Module:** `features/text_cleaning.py`  
Output:  
- `job_description_clean`

### 3.4 Domain Mapping
**Module:** `features/domain.py`  
Using lookup:  
- Output: `domain`

### 3.5 Salary Parsing
**Module:** `features/salary.py`  
Outputs:
- `sal_min`, `sal_max`, `sal_mean`, `sal_is_hourly`

### 3.6 Skill Extraction
**Modules:**  
- `features/skills_taxonomy.py`  
- `features/skill_extractor.py`
Outputs multi-hot skill flags.

## Chapter 1 Components

### 3.7 PCA Transformer
**File:** `features/skills_pca.py`  
**Output:**  
- PCA model  
- `skill_PC1` to `skill_PC10`  
**Validation:**  
PCA correctness is verified by confirming that the salary model (XGBoost v4) reproduces the exact metrics obtained in the exploratory notebook.  
Identical metrics guarantee that PCA ordering, scaling, and component structure are consistent with the original modelling workflow.

---

# 4. Evaluation modules

### 4.1 Salary Model Evaluator
**File:** `evaluation/salary_model_eval.py`  
Outputs:
- R², RMSE, MAE  
- Diagnostics  
- Feature importances  

### 4.2 Skill Model Evaluator
**File:** `evaluation/skill_model_eval.py`  
Outputs:
- ROC AUC / PR AUC  
- Calibration  
- Feature importance  

### 4.3 Chapter 0 Benchmark Validator
**File:** `evaluation/build_base_dataset_benchmark.py`  
Compares Chapter 0 dataset to benchmark.

---

# 5. Model modules

### 5.1 Salary Model Predictor
**File:** `models/salary_predictor.py`
Function:  
- Predict salary for new records.

### 5.2 Skill Probability Builder
**File:** `models/skill_prob_matrix.py`
Function:  
- Predict skill probability for each data entry.

---

# 6. Pipeline

### 6.1 Data builder
**File:** `pipelines/chapter0_build_base_dataset.py`
Steps:
1. Load raw DS/DA job CSVs.  
2. Apply all feature modules.  
3. Clean and validate fields.  
4. Build final Chapter 0 dataset.  
5. Save: `data/processed/ch0_processed_jobs.csv`.

### 6.2 Salary Modelling Pipeline  
**File:** `pipelines/salary_model_pipeline.py`  
Steps:  
1. Build Chapter 0 dataset  
2. Encode categoricals  
3. PCA transform  
4. Train XGBoost  
5. Optional evaluation  
6. Optional save artefacts  

### 6.3 Skill Requirement Pipeline  
**File:** `pipelines/skill_model_pipeline.py`  
Steps:  
1. Build Chapter 0 dataset  
2. Encode features  
3. Train 27 LightGBM models  
4. Evaluate  
5. Save models  
6. Generate skill probability matrix  


---

# 6. Data Flow Diagram (Text)

RAW CSVs  
→ Chapter 0 Pipeline  
→ ch0_processed_jobs.csv  
→ Salary Pipeline  
   → PCA + XGBoost  
   → salary_model_v4.pkl  

→ Skill Pipeline  
   → 27 LGBM models  
   → skill_prob_matrix.csv  

---

# 7. Artefacts

- `jobs_ch0.csv` Chapter 0 pipeline output
- `skill_pca_v1.pkl` PCA model
- `salary_model_v4.pkl` Salary model
- `skill_prob_matrix.csv` Skill probability matrix
- `skill_df_v01.csv` Skill P/A dataset
- `salary_model_dfv02_pca.csv` PCA appended to the main data for salary model
- 27 × `{skill}_model.pkl` Individual skill models
- Evaluation tables and plots 
   - `skill_model_evaluation_results.csv`
   - `{feature}_fairness.csv` Fairness analyses outputs

---

# 8. End-to-End Objective
Provide a reproducible system for extracting job structure, modelling salary, inferring skill requirements, and preparing downstream semantic modelling (Chapters 2–5).

---
