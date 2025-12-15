# Job Intelligence Engine — System Architecture
Date: 2025-12-15

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

## Chapter 2 Architecture — Hidden Structure (Graphs & Ecosystems)

### 3.8 Job–Skill Bipartite Graph
**Module:** `features/graph_job_skill.py`  
Outputs:
- Weighted bipartite graph linking `job_id` nodes to 27 skill-group nodes.
- Edge weights representing predicted skill–job probabilities.
This feature transforms Chapter 1 skill probability outputs into a relational graph structure used as the foundation for Chapter 2 embedding, clustering, and ecosystem analyses.

### 3.9 Node2Vec Embedding Loader
**Module:** `features/embedding_loader.py`  
Outputs:
- `job_emb`: job embedding table (index = `job_id`, columns = `emb_0` … `emb_{d-1}`).
- `skill_emb`: skill embedding table (index = skill name, columns = `emb_0` … `emb_{d-1}`).

This utility provides a consistent, validated interface for loading persisted Node2Vec embeddings for downstream Chapter 2 clustering and analysis.

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

### 5.1 Titles SBERT + Clustering
**File:** `models/sbert_clustering_training_titles.py`  
Function:  
- Train SBERT embeddings for unique job titles and cluster them using KMeans to produce a deterministic title → domain lookup table consumed by the Chapter 0 pipeline.


### 5.2 Salary Model Predictor
**File:** `models/salary_predictor.py`
Function:  
- Predict salary for new records.

### 5.3 Skill Probability Builder
**File:** `models/skill_prob_matrix.py`
Function:  
- Predict skill probability for each data entry.

### 5.4 Node2Vec Embedding Trainer
**File:** `models/node2vec_trainer.py`  
Functions:  
- Train Node2Vec embeddings from the Chapter 2 job–skill bipartite graph.  
- Extract and return job and skill embedding tables (jobs × d, skills × d).  
- Optionally run a lightweight stability diagnostic (nearest-neighbour overlap) and persist embeddings + metadata as reusable artefacts.

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

### 6.2 Run model
**File:** `pipelines/chapter1_models.py`
Function: takes Salary Modelling Pipeline and Skill Requirement Pipeline and runs one of both of them. It links chapter 0 and chapter 2 directly through a unified model pipeline


### 6.2a Salary Modelling Pipeline  
**File:** `pipelines/salary_model_pipeline.py`  
Steps:  
1. Build Chapter 0 dataset  
2. Encode categoricals  
3. PCA transform  
4. Train XGBoost  
5. Optional evaluation  
6. Optional save artefacts  

### 6.2b Skill Requirement Pipeline  
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
- `job_embeddings_node2vec_v01.csv` node2vec embeddings
- `skill_embeddings_node2vec_v01.csv` node2vec embeddings
- `job_skill_bipartite_thres0_5.gpickle` graph

---

# 8. End-to-End Objective
Provide a reproducible system for extracting job structure, modelling salary, inferring skill requirements, and preparing downstream semantic modelling (Chapters 2–5).

---

