# Job Intelligence Engine — System Architecture
Date: 2025-12-18

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

### 3.10 Job families clustering
**Module:** `features/job_families_clustering.py`  
Outputs:
- `km_jobs_df`: job clusters table (index = `job_id`, columns = `job family`).

This utility provides a consistent, validated interface for fitting a KMean clustering to the embeddings produced by node2vec.

### 3.11 Skill Ecosystem Structure
**Module:** `features/skill_embedding_similarity.py`  
Outputs:
- `top_5_skill_neighbours`: top neighbour table (columns = `skill_1`, `skill_2`, `similarity`).

This utility provides a consistent, validated interface for building the skill similarity matrix.

### 3.12 Skill Specialisation Map
**Module:** `features/skill_specialisation_map.py`  
Outputs:
- `{group_col}_skill_specialisation.csv`: skill specialisation table (columns = `{group_col}`, `individual skill family column`*27).

This utility provides a consistent, validated interface for building the skill specialisation map.

### 3.13 User Skill Processing
**Module:** `userprofile_skill_processing.py`  
Outputs:
PCA axes derived from the user's skills. This utility provides a consistent user skill transformer needed to predict user aspiring salary.

---

## Chapter 3 Architecture — Individual Positioning (User → Ranked Jobs + Gaps)

### 3.14 Chapter 3 Artefact Loader
**Module:** `features/artefacts_ch3.py`  
**Function:** `load_ch3_artefacts()`  
Outputs:
- `jobs_df`: Chapter 2 processed jobs table (modelling-ready; includes filters + skill PCs + salary fields).
- `skill_prob_matrix`: Chapter 1 job × skill probability matrix (columns `job_id` + `{skill}_prob` for all skills).

This loader centralises Chapter 3 dependencies and ensures consistent artefact sources across all positioning modules.

### 3.15 Candidate Set Construction
**Module:** `features/candidate_selection.py`  
**Function:** `candidate_set_construction()`  
Outputs:
- `profile`: validated UserProfile dict (from `schemas.py`).
- `candidates_df`: filtered job subset based on hard constraints.

Responsibilities:
- Build a UserProfile using `build_user_profile()`.
- Apply hard filters in deterministic order:
  1) `state`, 2) `Sector`, 3) `title_rich`, 4) `job_title_family`.
- Enforce controlled failure when filters yield an empty set (raise ValueError).

### 3.16 Candidate Suitability Components
**Module:** `features/candidate_suitability.py`  
Functions:
- `add_skill_match(profile, candidates_df)`  
  Adds:
  - `skill_match_score` (raw cosine similarity in PCA space, [-1, 1])
  - `skill_match_norm` (mapped to [0, 1] via `(s + 1) / 2`, used only for aggregation)
- `add_salary_score(profile, candidates_df)`  
  Adds:
  - `salary_score` in [0, 1], one-sided target formulation (meeting/exceeding target does not reduce suitability)
- `add_suitability(profile, candidates_df, w_skill, w_salary)`  
  Adds:
  - `suitability` as weighted sum of normalized components (weights auto-normalised to sum to 1)

This module contains only component computation and does not load artefacts or perform candidate filtering.

### 3.17 Candidate Skill Gap Analysis
**Module:** `features/candidate_skill_gap.py`  
**Function:** `compute_skill_gaps(profile, candidates_df, skill_prob_matrix, top_k)`  
Output:
- `gap_df`: user-level skill gap diagnostics (one row per skill)

Method:
- Take top-K jobs by suitability.
- Join against `skill_prob_matrix` using `job_id`.
- Compute mean probability per skill across top-K jobs.
- Gap severity: `gap = mean_prob` if user lacks skill, else `0`.
This produces a calibrated, probability-based gap ranking (preferred over binary flags).


Produces calibrated, probability-based skill gaps.

---

### 3.18 Skill Rarity Weights
**Module:** `features/skill_rarity.py`  
**Function:** `compute_skill_rarity_weights(profile, skill_prob_matrix)`

Outputs:
- `rarity_weight`: inverse global prevalence per skill.
- `weight_norm`: mean-normalised rarity weights aligned to canonical skill order.

Responsibilities:
- Compute global skill prevalence across the job landscape.
- Invert prevalence so rare skills receive higher weights.
- Normalise weights for numerical stability.
- Align weights exactly to the 27-skill user vector.

Used to upweight missing *rare* skills in competitiveness scoring.

---

### 3.19 Candidate Competitiveness Index
**Module:** `features/candidate_competitiveness.py`  
**Function:** `add_competitiveness(profile, candidates_df, skill_prob_matrix, w_missing, w_salary)`

Adds to `candidates_df`:
- `expected_missing`
- `expected_missing_norm`
- `salary_pct`
- `competitiveness_index`

Method:
- Compute expected missing skill burden via dot product of:
- job-level skill probabilities
- user missing-skill indicator
- Optionally apply rarity weighting.
- Compute salary percentile within candidate set.
- Aggregate into a barrier-to-entry metric.

Competitiveness reflects **difficulty of access**, not desirability.

---

### 3.20 Competitiveness Sensitivity Analysis
**Module:** `features/competitiveness_sensitivity.py`  
**Function:** `compute_competitiveness_sensitivity(candidates_df)`

Outputs:
- Sensitivity table with:
- `w_skill`
- `w_salary`
- `spearman_rho_vs_baseline`

Method:
- Recompute competitiveness rankings across weight grid.
- Compare each to baseline ranking via Spearman correlation.
- Quantifies robustness to weighting assumptions.

---

### 3.21 Suitability Sensitivity Analysis
**Module:** `features/suitability_sensitivity.py`  
**Function:** `compute_suitability_sensitivity(candidates_df)`

Outputs:
- Sensitivity table with:
- `w_skill`
- `w_salary`
- `spearman_rho_vs_baseline`

Method mirrors competitiveness sensitivity, applied to suitability rankings.

---

### 3.22 Chapter 3 Orchestrator (Public API)
**Module:** `src/job_intel/positioning.py`  
**Function:** `run_positioning()`

Returns:
- `profile`
- `candidates_df`
- `gap_df`
- `suitability_sensitivity`
- `competitiveness_sensitivity`

Responsibilities:
- Load artefacts.
- Build user profile and candidate set.
- Compute suitability and competitiveness.
- Run sensitivity analyses.
- Compute skill gaps.

This is the **single public entrypoint** for Chapter 3.

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

### 6.3 Hidden Structures Pipeline  
**File:** `pipelines/chapter2_hidden_structures.py`  

This pipeline converts the Chapter 1 job × skill probability layer into a relational representation of the labour market.  
It first builds a weighted job–skill bipartite graph (jobs and skills as node types; probabilities as edge weights), then learns Node2Vec embeddings that place both jobs and skills into a shared geometric space capturing co-occurrence and latent similarity.

From these embeddings, the pipeline produces two interpretable Chapter 2 artefacts:  
(1) **job families**, obtained by L2-normalising job embeddings and clustering them with KMeans, and  
(2) a **skill ecosystem edge list**, obtained by L2-normalising skill embeddings, computing cosine similarity via dot products, and retaining the top-*k* neighbours per skill to form a sparse undirected skill–skill network.

All outputs are saved as versioned processed artefacts for reuse in downstream chapters (job-family aggregation, skill bundle analysis, and later recommendation logic).

### 6.4 Individual Positioning Pipeline (Chapter 3)
**File:** `pipelines/ch3_individual_positioning.py`

Runnable wrapper around `run_positioning()`.

Provides a reproducible execution path to:
- rank jobs,
- compute skill gaps,
- evaluate robustness,
- persist outputs for inspection or downstream use.
---

# 7. Schemas

## User Profile Schema (Entry Point)

Chapter 3 introduces a formal **UserProfile schema** that defines how an individual enters the Job Intelligence Engine.

The schema is implemented in `src/job_intel/schemas.py` via `build_user_profile()` and acts as the **single entrypoint** for all downstream Chapter 3 logic (suitability, skill gaps, competitiveness, sensitivity analysis, and future APIs).

**Responsibilities:**
- Validate and normalize raw user inputs (skills text, location, preferences).
- Reuse Chapter 0 skill extraction to produce the canonical 27-skill vector.
- Project user skills into the shared PCA space used by job models.
- Return a deterministic, model-ready payload with fixed shapes and semantics.

All Chapter 3 pipelines must consume the UserProfile output and must not re-implement user parsing or preprocessing logic elsewhere.

---

# 8. Artefacts

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
- `job_families_graph_embeddings.csv` clustering output chapter 2
- `skill_similarity_edges_k5_embeddings.csv` skill–skill network derived from Node2Vec embeddings
- `job_family_skill_specialisation.csv` canonical output of the “Industry / Job-Family Specialisation Map” module
- Chapter 3 artefacts (consumed inputs):
  - `data/processed/ch2_processed_df.csv` (jobs_df; modelling-ready table used for positioning)
  - `data/processed/skill_prob_matrix.csv` (job × `{skill}_prob` matrix used for calibrated gap analysis)
  - `expected_missing`
  - `expected_missing_norm`
  - `salary_pct`
  - `competitiveness_index`
  - `suitability_sensitivity`
  - `competitiveness_sensitivity`
---
