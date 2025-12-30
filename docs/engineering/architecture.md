# Job Intelligence Engine — System Architecture
Date: 2025-12-30

This file contains the updated architecture for the project, organised for fast scanning:
1) **Overview**
2) **Features** (by chapter)
3) **Models** (by chapter)
4) **Evaluation** (by chapter)
5) **Pipelines** (by chapter)
6) **Schemas**
7) **Artefacts**
8) **Assumptions & Limitations**

All components live in `src/job_intel`.

---

# 1. Overview

The system is a modular, deterministic pipeline transforming raw Glassdoor job postings into:
- cleaned, validated job records (Chapter 0),
- salary and skill-requirement models (Chapter 1),
- graph/embedding-based labour-market structure artefacts (Chapter 2),
- per-user positioning outputs (Chapter 3),
- recommendations + explanations + counterfactual upskilling/simulation (Chapter 4).

---

# 2. Feature Modules (by Chapter)

## 2.1 Chapter 0 — Feature Engineering (Raw → Cleaned, Model-Ready Jobs)

### 2.1.1 Title Processing
**Module:** `features/titles.py`  
Outputs:
- `job_title_raw`
- `job_title_base`
- `job_title_norm`
- `job_title_family`

### 2.1.2 Seniority Extraction
**Module:** `features/seniority.py`  
Purpose:
- Extract seniority signals from titles and descriptions.

### 2.1.3 Description Cleaning
**Module:** `features/text_cleaning.py`  
Output:
- `job_description_clean`

### 2.1.4 Domain Mapping
**Module:** `features/domain.py`  
Output:
- `domain` (lookup-based)

### 2.1.5 Salary Parsing
**Module:** `features/salary.py`  
Outputs:
- `sal_min`, `sal_max`, `sal_mean`, `sal_is_hourly`

### 2.1.6 Skill Extraction
**Modules:**
- `features/skills_taxonomy.py`
- `features/skill_extractor.py`  
Output:
- multi-hot skill-family flags (canonical 27-family space)

---

## 2.2 Chapter 1 — Modelling Features

### 2.2.1 PCA Transformer
**File:** `features/skills_pca.py`  
Outputs:
- PCA model artefact  
- `skill_PC1` to `skill_PC10` features

Validation principle:
- PCA correctness is validated indirectly by reproducing the exact salary-model metrics (XGBoost v4) from the exploratory notebook.

---

## 2.3 Chapter 2 — Hidden Structure (Graphs & Ecosystems)

### 2.3.1 Job–Skill Bipartite Graph
**Module:** `features/graph_job_skill.py`  
Outputs:
- weighted bipartite graph linking `job_id` nodes to 27 skill-family nodes
- edge weights representing predicted skill–job probabilities

### 2.3.2 Node2Vec Embedding Loader
**Module:** `features/embedding_loader.py`  
Outputs:
- `job_emb`: job embedding table (index = `job_id`, columns = `emb_0` … `emb_{d-1}`)
- `skill_emb`: skill embedding table (index = skill name, columns = `emb_0` … `emb_{d-1}`)

### 2.3.3 Job Families Clustering
**Module:** `features/job_families_clustering.py`  
Outputs:
- `km_jobs_df`: job clusters table (index = `job_id`, column = `job_family_id` or equivalent cluster id)

### 2.3.4 Skill Ecosystem Similarity Structure
**Module:** `features/skill_embedding_similarity.py`  
Outputs:
- skill–skill neighbour edge list (columns = `skill_1`, `skill_2`, `similarity`), top-*k* neighbours per skill

### 2.3.5 Skill Specialisation Map
**Module:** `features/skill_specialisation_map.py`  
Outputs:
- `{group_col}_skill_specialisation.csv`: specialisation table (columns = `{group_col}` + 27 skill-family columns)

### 2.3.6 User Skill Processing (PCA for User)
**Module:** `features/userprofile_skill_processing.py`  
Purpose / Output:
- derive PCA axes (`skill_PC1..skill_PC10`) from a user’s extracted skills, aligned to the PCA space used by salary + positioning components

---

## 2.4 Chapter 3 — Individual Positioning (User → Ranked Jobs + Gaps)

### 2.4.1 Chapter 3 Artefact Loader
**Module:** `features/artefacts_ch3.py`  
**Function:** `load_ch3_artefacts()`  
Outputs:
- `jobs_df`: modelling-ready jobs table
- `skill_prob_matrix`: job × `{skill}_prob` probability matrix

### 2.4.2 Candidate Set Construction
**Module:** `features/candidate_selection.py`  
**Function:** `candidate_set_construction()`  
Outputs:
- `profile`: validated UserProfile dict (from `schemas.py`)
- `candidates_df`: filtered job subset based on hard constraints

### 2.4.3 Suitability Components
**Module:** `features/candidate_suitability.py`  
Adds:
- `skill_match_score`, `skill_match_norm`
- `salary_score`
- `suitability`

### 2.4.4 Skill Gap Analysis
**Module:** `features/candidate_skill_gap.py`  
**Function:** `compute_skill_gaps(...)`  
Output:
- `gap_df`

### 2.4.5 Skill Rarity Weights
**Module:** `features/skill_rarity.py`  
Output:
- rarity weights aligned to canonical 27-skill order

### 2.4.6 Competitiveness Index
**Module:** `features/candidate_competitiveness.py`  
Adds:
- `expected_missing`
- `expected_missing_norm`
- `salary_pct`
- `competitiveness_index`

### 2.4.7 Sensitivity Analyses
**Modules:**
- `features/competitiveness_sensitivity.py`
- `features/suitability_sensitivity.py`

### 2.4.8 Chapter 3 Orchestrator (Public API)
**Module:** `positioning.py`  
**Function:** `run_positioning()`

---

## 2.5 Chapter 4 — Recommender Engine

### 2.5.1 Chapter 4 Context Loader
**Module:** `features/artefacts_ch4.py`  
**Function:** `load_ch4_context()`  
Outputs:
- `profile`, `candidates_df`, `gap_df`, `sensitivity_out`
- `jobs_df`, `skill_prob_matrix`
- `salary_model`
- `user_salary_model_features`

### 2.5.2 Hybrid Job Recommender (v1)
**Module:** `features/job_recommender.py`  
**Function:** `job_recommender()`  
Outputs:
- `tables`: `scored_universe`, `top_best_now`, `top_stretch`, plus intermediate `candidate_jobs`
- `params`, `counts`, `warnings`, `salary_summary`

### 2.5.3 Job Explanations (v1)
**Module:** `features/job_explanations.py`  
**Function:** `build_job_explanations(...)`  
Outputs:
- `tables`: `scored_universe_explained`, `top_best_explained`, `top_stretch_explained`
- `metric_glossary`, `meta`

### 2.5.4 Upskilling Recommender (v1)
**Module:** `features/upskilling_recommender.py`  
**Function:** `upskill_recommender(...)`  
Outputs:
- `job_base_upskill`, `upskill_summary`, `upskill_recommendation`
- `missing_dict`, `recommendation_dict`, `scenario_meta`

### 2.5.5 Career Simulation (v2 / experimental)
**Module:** `v2_updates/features/career_simulator.py`  
**Function:** `career_simulation(...)`

---

# 3. Model Modules (by Chapter)

## 3.1 Chapter 0 (Supporting Model)

### 3.1.1 Titles SBERT + Clustering Trainer
**File:** `models/sbert_clustering_training_title.py`  
Purpose:
- train SBERT embeddings for unique job titles and cluster them with KMeans to produce a deterministic title/domain lookup consumed by Chapter 0

## 3.2 Chapter 1

### 3.2.1 Salary Model Predictor
**File:** `models/salary_predictor.py`

### 3.2.2 Skill Probability Builder
**File:** `models/skill_prob_matrix.py`

## 3.3 Chapter 2

### 3.3.1 Node2Vec Embedding Trainer
**File:** `models/node2vec_trainer.py`

---

# 4. Evaluation Modules (by Chapter)

## 4.1 Chapter 0

### 4.1.1 Chapter 0 Benchmark Validator
**File:** `evaluation/build_base_dataset_benchmark.py`

## 4.2 Chapter 1

### 4.2.1 Salary Model Evaluator
**File:** `evaluation/salary_model_eval.py`

### 4.2.2 Skill Model Evaluator
**File:** `evaluation/skill_model_eval.py`

## 4.3 Chapter 2

### 4.3.1 Chapter 2 Integrity Evaluator
**File:** `evaluation/chapter2_integrity.py`

## 4.4 Chapter 3

### 4.4.1 Chapter 3 Positioning Pipeline Evaluator
**File:** `evaluation/chapter3_pipeline_eval.py`

## 4.5 Chapter 4

### 4.5.1 Chapter 4 Entry Point Evaluator
**File:** `evaluation/chapter4_entrypoint_eval.py`

### 4.5.2 Chapter 4 Pipeline Evaluator
**File:** `evaluation/chapter4_pipeline_eval.py`

---

# 5. Pipelines (by Chapter)

## 5.1 Chapter 0

### 5.1.1 Data Builder
**File:** `pipelines/chapter0_build_base_dataset.py`  
Steps:
1. Load raw DS/DA job CSVs
2. Apply all feature modules
3. Clean and validate fields
4. Build final Chapter 0 dataset
5. Save processed dataset

## 5.2 Chapter 1

### 5.2.1 Run Model (Unified Entrypoint)
**File:** `pipelines/chapter1_models.py`  
Purpose:
- run either the Salary Modelling Pipeline or the Skill Requirement Pipeline

### 5.2.2 Salary Modelling Pipeline
**File:** `pipelines/salary_model_pipeline.py`  
Steps:
1. Build Chapter 0 dataset
2. Encode categoricals
3. PCA transform
4. Train XGBoost
5. Optional evaluation
6. Optional save artefacts

### 5.2.3 Skill Requirement Pipeline
**File:** `pipelines/skill_model_pipeline.py`  
Steps:
1. Build Chapter 0 dataset
2. Encode features
3. Train 27 LightGBM models
4. Evaluate
5. Save models
6. Generate skill probability matrix

## 5.3 Chapter 2

### 5.3.1 Hidden Structures Pipeline
**File:** `pipelines/chapter2_hidden_structures.py`  
Purpose:
- convert Chapter 1 job × skill probabilities into graphs + embeddings + clustered job families + skill ecosystems and persist outputs

## 5.4 Chapter 3

### 5.4.1 Individual Positioning Pipeline
**File:** `pipelines/chapter3_individual_positioning.py`  
Purpose:
- runnable wrapper around `run_positioning()`

## 5.5 Chapter 4

### 5.5.1 Recommender Engine Pipeline
**File:** `pipelines/chapter4_recommender.py`  
Purpose:
- orchestrate:
  - baseline recommender
  - explanations (optional)
  - upskilling (optional)
  - career simulation (optional)

Recommended persistence:
- persist **demo config** only: `evaluation/recommender_demo.json` (and optionally a “golden” snapshot)

---

# 6. Schemas

## 6.1 User Profile Schema (Entry Point)
**Module:** `schemas.py`  
**Function:** `build_user_profile()`

---

# 7. Artefacts

## 7.1 Persisted Artefacts (Files)
(Union of artefacts listed across the project documentation.)

Chapter 0:
- `data/processed/ch0_processed_jobs.csv` (Chapter 0 processed jobs table)

Chapter 1:
- `models/skill_pca_v1.pkl`
- `models/salary_model_v4.pkl`
- `data/processed/salary_model_dfv02_pca.csv` (if persisted)
- `data/processed/skill_df_v01.csv` (if persisted)
- `data/processed/skill_prob_matrix.csv`
- `models/{skill}_model.pkl` (27 skill models)

Chapter 2:
- `data/processed/job_skill_bipartite_*.gpickle`
- `data/processed/job_embeddings_node2vec_v01.csv`
- `data/processed/skill_embeddings_node2vec_v01.csv`
- `data/processed/job_families_graph_embeddings.csv`
- `data/processed/skill_similarity_edges_k*_embeddings.csv`
- `data/processed/job_family_skill_specialisation.csv`

Evaluation outputs:
- `evaluation/skill_model_evaluation_results.csv` (if produced)
- `{feature}_fairness.csv` (if produced)
- other evaluation tables/plots created by evaluation scripts

Chapter 4 demo:
- `evaluation/recommender_demo.json`

## 7.2 Runtime Outputs (Not Persisted by Default)
Produced per user run (tables returned by APIs/pipelines):
- Chapter 3: `candidates_df`, `gap_df`, sensitivity outputs
- Chapter 4: `scored_universe`, bucket tables, explained tables, upskilling tables, simulation tables

---

# 8. Assumptions & Limitations (Project + Chapters)

Project-wide:
- inputs are job postings from a specific dataset source and timeframe; distribution shift is expected
- skill extraction is token/taxonomy-driven; out-of-vocabulary skills may be ignored or mapped imperfectly
- determinism is prioritised; exploration/diversity objectives are not first-class in v1

Chapter 0:
- cleaning/normalisation rules are heuristic; edge cases may exist in salary parsing and descriptions

Chapter 1:
- salary model performance depends on encoded categoricals + PCA representation; causal interpretation is not implied
- skill models predict “job requires skill-family” from observed text/tokens and are limited by label construction

Chapter 2:
- Node2Vec structure reflects co-occurrence in the learned job–skill graph; embeddings are not ground-truth semantics

Chapter 3:
- positioning is computed within a constrained candidate universe; suitability/competitiveness are relative to that universe
- skill gaps use mean probability across top-K jobs; K choice is a modelling assumption

Chapter 4:
- “best_now” vs “stretch” is defined by competitiveness thresholding; it is an operational proxy, not a guarantee
- upskilling and simulation are counterfactual token injections; they measure model response, not guaranteed labour-market outcomes
