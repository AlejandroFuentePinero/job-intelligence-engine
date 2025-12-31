```md
# Job Intelligence Engine — System Architecture
Date: 2025-12-30

This file is the canonical architecture reference for shipping + publication. It is organised for fast scanning:
1) **Overview**
2) **Features** (by chapter)
3) **Models** (by chapter)
4) **Evaluation** (by chapter)
5) **Pipelines** (by chapter)
6) **Schemas & Public APIs**
7) **Artefacts**
8) **Assumptions & Limitations**

All architectural components live in `src/job_intel`.

> Completeness contract: this architecture enumerates **47** project `.py` modules (excluding `__init__.py` and `config.py`) and matches the repository filenames shown in the attached screenshots.

---

# 1. Overview

The system is a modular, deterministic pipeline transforming raw Glassdoor job postings into:
- cleaned, validated job records (Chapter 0),
- salary and skill-requirement models (Chapter 1),
- graph/embedding-based labour-market structure artefacts (Chapter 2),
- per-user positioning outputs (Chapter 3),
- recommendations + explanations + counterfactual upskilling / simulation (Chapter 4).

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
- extract seniority signals from titles and descriptions

### 2.1.3 Description Cleaning
**Module:** `features/text_cleaning.py`  
Output:
- `job_description_clean`

### 2.1.4 Domain Mapping
**Module:** `features/domain.py`  
Purpose / Output:
- map to canonical `domain` via deterministic lookup

### 2.1.5 Salary Parsing
**Module:** `features/salary.py`  
Outputs:
- `sal_min`, `sal_max`, `sal_mean`, `sal_is_hourly`

### 2.1.6 Skill Extraction (Canonical 27-family space)
**Modules:**
- `features/skills_taxonomy.py`
- `features/skill_extractor.py`  
Purpose / Output:
- taxonomy-driven extraction into multi-hot skill-family flags (canonical 27 families)

---

## 2.2 Chapter 1 — Modelling Features

### 2.2.1 PCA Transformer (Skills → Low-Dim)
**Module:** `features/skills_pca.py`  
Outputs:
- PCA model artefact  
- `skill_PC1` … `skill_PC10` columns

Validation principle:
- PCA correctness is validated indirectly by reproducing the exact salary-model metrics (XGBoost v4) obtained in the exploratory workflow (same preprocessing + PCA ordering/scaling).

---

## 2.3 Chapter 2 — Hidden Structure (Graphs & Ecosystems)

### 2.3.1 Job–Skill Bipartite Graph
**Module:** `features/graph_job_skill.py`  
Outputs:
- weighted bipartite graph linking `job_id` nodes to 27 skill-family nodes
- edge weights = predicted skill–job probabilities (Chapter 1 output)

Role:
- this is the relational foundation for Node2Vec embedding, clustering, and downstream “ecosystem” analyses.

### 2.3.2 Node2Vec Embedding Loader
**Module:** `features/embedding_loader.py`  
Outputs:
- `job_emb`: index = `job_id`, columns = `emb_0` … `emb_{d-1}`
- `skill_emb`: index = skill name, columns = `emb_0` … `emb_{d-1}`

Role:
- canonical loader/validator for persisted embeddings used across Chapter 2/5-style analyses.

### 2.3.3 Job Families Clustering
**Module:** `features/job_families_clustering.py`  
Outputs:
- `km_jobs_df`: job family assignment table with `job_id` + `job_family_id` (cluster id)

Role:
- deterministic clustering of job embeddings into interpretable “job families”.

### 2.3.4 Skill Ecosystem Similarity Structure (k-NN skill graph)
**Module:** `features/skill_embedding_similarity.py`  
Outputs:
- skill–skill neighbour edge list with columns:
  - `skill_1`, `skill_2`, `similarity`
- constructed as top-*k* neighbours per skill (sparsified similarity network)

### 2.3.5 Skill Specialisation Map (Lift Tables)
**Module:** `features/skill_specialisation_map.py`  
Outputs:
- `{group_col}_skill_specialisation.csv` lift-style tables:
  - columns = `{group_col}` + 27 skill-family columns

### 2.3.6 User Skill Processing (PCA for User)
**Module:** `features/userprofile_skill_processing.py`  
Purpose / Output:
- derive user `skill_PC1..skill_PC10` from extracted user skills
- aligned to the PCA space used by the salary model + positioning/recommender components

---

## 2.4 Chapter 3 — Individual Positioning (User → Ranked Jobs + Skill Gaps)

### 2.4.1 Chapter 3 Artefact Loader
**Module:** `features/artefacts_ch3.py`  
**Function:** `load_ch3_artefacts()`  
Outputs:
- `jobs_df`: modelling-ready jobs table (Chapter 2 processed jobs table)
- `skill_prob_matrix`: job × `{skill}_prob` probability matrix (calibrated skill requirements)

Notes:
- `jobs_df` corresponds to the persisted Chapter 2 processed table (see Artefacts).

### 2.4.2 Candidate Set Construction (Hard Constraints)
**Module:** `features/candidate_selection.py`  
**Function:** `candidate_set_construction()`  
Outputs:
- `profile`: validated UserProfile dict
- `candidates_df`: filtered job subset based on hard constraints

Core responsibilities:
- build user profile (schema-valid)
- apply hard filters in deterministic order (as implemented), failing loudly on empty-candidate outcomes.

### 2.4.3 Suitability Components (Match + Salary)
**Module:** `features/candidate_suitability.py`  
Adds to `candidates_df`:
- `skill_match_score` (cosine similarity in PCA space, raw)
- `skill_match_norm` (mapped to [0,1] for aggregation)
- `salary_score` (one-sided target formulation)
- `suitability` (weighted sum; weights normalized to sum to 1)

Scope:
- computation only; no artefact loading and no filtering.

### 2.4.4 Skill Gap Analysis (Calibrated)
**Module:** `features/candidate_skill_gap.py`  
**Function:** `compute_skill_gaps(...)`  
Output:
- `gap_df`: one row per skill-family with probability-based gap severity

Method (high level):
- take top-K jobs by suitability
- join to `skill_prob_matrix` by `job_id`
- compute mean required probability per skill-family across top-K jobs
- gap severity = mean_prob if user lacks skill, else 0

### 2.4.5 Skill Rarity Weights
**Module:** `features/skill_rarity.py`  
Output:
- rarity weights aligned to canonical 27-skill order (mean-normalised for stability)

Role:
- optional weighting so rare missing skills contribute more to “difficulty of access”.

### 2.4.6 Competitiveness Index (Barrier-to-Entry)
**Module:** `features/candidate_competitiveness.py`  
Adds:
- `expected_missing`
- `expected_missing_norm`
- `salary_pct`
- `competitiveness_index`

Interpretation:
- competitiveness reflects *difficulty of access*, not desirability.

### 2.4.7 Sensitivity Analyses (Robustness to Weights)
**Modules:**
- `features/competitiveness_sensitivity.py`
- `features/suitability_sensitivity.py`  

Outputs:
- weight-grid sensitivity tables (rank stability vs baseline using Spearman correlation)

### 2.4.8 Chapter 3 Orchestrator
**Module:** `positioning.py`  
**Function:** `run_positioning()`  
Returns:
- `profile`, `candidates_df`, `gap_df`
- suitability + competitiveness sensitivity outputs

Responsibilities:
- load artefacts (via Chapter 3 loader)
- build candidate universe and compute all Chapter 3 metrics
- run sensitivity + gap analysis
- provide the single canonical Chapter 3 entrypoint for downstream chapters.

---

## 2.5 Chapter 4 — Recommender Engine (Decision Support)

### 2.5.1 Chapter 4 Context Loader
**Module:** `features/artefacts_ch4.py`  
**Function:** `load_ch4_context()`  
Outputs (canonical payload):
- Chapter 3 outputs: `profile`, `candidates_df`, `gap_df`, `sensitivity_out`
- Chapter 3 inputs: `jobs_df`, `skill_prob_matrix`
- `salary_model`
- `user_salary_model_features` (candidate-aligned salary design matrix)

Core responsibilities:
- call `run_positioning()` to obtain the candidate universe + derived user PCA
- load Chapter 3 artefacts (jobs_df, skill_prob_matrix) without rerunning Chapter 1
- build salary feature matrix by:
  - selecting required categorical codes from candidates (e.g., size/sector/state/ownership/seniority/title_rich codes)
  - broadcasting user `skill_PC1..skill_PC10` across candidate rows
- return a single “ready for Chapter 4” payload used by all downstream modules.

### 2.5.2 Hybrid Job Recommender (v1)
**Module:** `features/job_recommender.py`  
**Function:** `job_recommender()`  
Outputs:
- `tables`: `candidate_jobs`, `scored_universe`, `top_best_now`, `top_stretch`
- `params`, `counts`, `warnings`, `salary_summary`

Behaviour (high level):
- attach salary predictions (row-aligned; no implicit merges)
- apply suitability gating with fallback thresholds
- split into buckets by competitiveness threshold (`best_now` vs `stretch`)
- re-rank deterministically with stable tie-breakers:
  - score desc → suitability desc → competitiveness asc → job_id asc

### 2.5.3 Job Explanations (v1)
**Module:** `features/job_explanations.py`  
**Function:** `build_job_explanations(...)`  
Outputs:
- `tables`: explained versions of `scored_universe`, `top_best_now`, `top_stretch`
- `metric_glossary`, `meta` (e.g., probability threshold used)

Role:
- add human-readable rationale: why bucket, why rank, salary context, missing vs covered families
- include light contract validation (fail loudly if required joins/columns break).

### 2.5.4 Upskilling Recommender (v1)
**Module:** `features/upskilling_recommender.py`  
**Function:** `upskill_recommender(...)`  
Outputs:
- `job_base_upskill` (job × scenario long table with deltas + bucket movement)
- `upskill_summary`, `upskill_recommendation`
- `missing_dict`, `recommendation_dict`, `scenario_meta`

Core invariants:
- frozen-universe comparability across scenarios (same `job_id` universe)
- fail loudly if required missing-skill extraction/tokenisation plumbing breaks.

### 2.5.5 Career Simulation (v2 / experimental)
**Module:** `v2_updates/features/career_simulator.py`  
**Function:** `career_simulation(...)`

Role:
- user-driven counterfactual simulator (“what changes if I add these skills?”) on a frozen universe
- produces per-scenario deltas and unlocked jobs under strict universe consistency checks.

## 2.6 Chapter 5 — Insights & Dashboards (Streamlit App)

### 2.6.0 App Entrypoint (Root)
**File:** `app.py`  
Role:
- root Streamlit entrypoint used to launch the application locally
- delegates routing/navigation to `src/job_intel/app/engine.py`

### 2.6.1 App Engine + Navigation (App Spine)
**Module:** `app/engine.py`  
Role:
- Streamlit app entrypoint + navigation router across pages
- wires page-level `render()` functions
- centralizes shared config + “Reload assets” behaviour

### 2.6.2 Home Page (Demo-first landing)
**Module:** `app/home.py`  
Outputs:
- introduction + guidance for how to use the app
- lightweight “what to click next” structure (no computation)

### 2.6.3 Landscape Page (Market mechanics + explainability)
**Module:** `app/landscape.py`  
Views:
- Fairness (residuals) summary + group breakdown
- Global skill value ranking (GSVI)
- SHAP explainability (global bar + beeswarm; local drilldown for top-3 drivers)
Artefact dependencies:
- `data/processed/ch5_assets/*` (see Artefacts section additions below)

### 2.6.4 Recommender Page (User → best_now / stretch)
**Module:** `app/recommender.py`  
Role:
- UI for running Chapter 4 outputs and presenting recommendation buckets
- surfaces job-level rationale (as available from Chapter 4 outputs)

### 2.6.5 Upskilling + Macro Layer Page (Decision support)
**Module:** `app/upskilling_macro.py`  
Role:
- UI for counterfactual upskilling deltas (Chapter 4 outputs)
- UI for optional macro layer exploration (Chapter 5 scope)

---

# 3. Model Modules (by Chapter)

## 3.1 Chapter 0 (Supporting Model)

### 3.1.1 Titles SBERT + Clustering Trainer
**File:** `models/sbert_clustering_training_title.py`  
Purpose:
- train SBERT embeddings for unique job titles and cluster with KMeans to produce a deterministic title/domain lookup consumed by Chapter 0.

## 3.2 Chapter 1

### 3.2.1 Salary Model Predictor
**File:** `models/salary_predictor.py`  
Purpose:
- load and apply the trained salary model to new records (inference utilities used by Chapter 4).

### 3.2.2 Skill Probability Builder
**File:** `models/skill_prob_matrix.py`  
Purpose:
- build/persist the job × skill probability matrix from the 27 skill models.

## 3.3 Chapter 2

### 3.3.1 Node2Vec Embedding Trainer
**File:** `models/node2vec_trainer.py`  
Purpose:
- learn Node2Vec embeddings from the Chapter 2 job–skill bipartite graph and persist reusable embedding tables.

---

# 4. Evaluation Modules (by Chapter)

## 4.1 Chapter 0

### 4.1.1 Chapter 0 Benchmark Validator
**File:** `evaluation/build_base_dataset_benchmark.py`  
Purpose:
- compare Chapter 0 processed outputs to a benchmark/expectations (schema + basic distribution sanity).

## 4.2 Chapter 1

### 4.2.1 Salary Model Evaluator
**File:** `evaluation/salary_model_eval.py`  
Outputs / checks (typical):
- R² / RMSE / MAE
- diagnostics and sanity checks (e.g., leakage guards, feature availability)
- feature importance extracts (model-native)

### 4.2.2 Skill Model Evaluator
**File:** `evaluation/skill_model_eval.py`  
Outputs / checks (typical):
- ROC AUC / PR AUC + calibration summaries
- per-skill diagnostics and failure modes
- feature importance extracts (model-native)

## 4.3 Chapter 2

### 4.3.1 Chapter 2 Integrity Checks (Artefact Contracts)
**File:** `evaluation/chapter2_integrity.py`  
Purpose:
- lightweight, deterministic integrity checks for Chapter 2 artefacts (alignment, shapes, ranges, ID coverage) without expensive recomputation.

Examples of checks:
- `job_id` alignment across `jobs_df` / `prob_mat` / embeddings
- embedding dimensions and finiteness
- clustering coverage and expected number of clusters
- similarity edge list schema + dedup invariants
- lift/specialisation tables numeric + expected columns.

## 4.4 Chapter 3

### 4.4.1 Chapter 3 Positioning Pipeline Evaluator
**File:** `evaluation/chapter3_pipeline_eval.py`  
Purpose:
- engineering hardening + behavioural validation for Chapter 3.

Typical checks:
- determinism under repeated runs (same inputs → identical outputs)
- boundary conditions (empty candidates, minimal skill_text, single-job sets)
- artefact integrity (missing skill columns, job_id mismatches)
- behavioural sanity (suitability/competitiveness response to inputs)
- sensitivity integrity (weight grid normalization; Spearman bounds; baseline uniqueness)
- smoke tests for required columns, valid ranges, ranking order.

## 4.5 Chapter 4

### 4.5.1 Chapter 4 Context Loader Evaluator
**File:** `evaluation/chapter4_entrypoint_eval.py`  
Purpose:
- validate Chapter 4-specific risks introduced by `load_ch4_context()` without duplicating Chapter 3 tests.

Typical checks:
- salary feature matrix columns present and non-missing
- PC broadcasting correctness (user PCs constant across candidate rows)
- salary prediction smoke test (predict runs, finite outputs, correct length)
- prerequisite alignment (candidate job_ids subset of prob-matrix universe).

### 4.5.2 Chapter 4 Pipeline Evaluator
**File:** `evaluation/chapter4_pipeline_eval.py`  
Purpose:
- orchestration-level validation for the Chapter 4 recommender pipeline using the persisted demo config.

Typical checks:
- demo config loads and validates
- end-to-end pipeline smoke execution
- output contracts: required tables exist, required columns exist, non-null identifiers
- bucket integrity: no overlap, correct labels, deterministic sorting
- explanation contract (when enabled)
- upskilling contract + frozen-universe invariant (when enabled)
- simulation contract + frozen-universe invariant (when enabled).

Supporting config file (not counted as a `.py` module):
- `evaluation/recommender_demo.json`

---

# 5. Pipelines (by Chapter)

## 5.1 Chapter 0

### 5.1.1 Data Builder
**File:** `pipelines/chapter0_build_base_dataset.py`  
Purpose:
1. load raw job CSVs
2. apply Chapter 0 feature modules
3. validate + clean fields deterministically
4. persist processed dataset for reuse downstream.

## 5.2 Chapter 1

### 5.2.1 Run Model (Unified Entrypoint)
**File:** `pipelines/chapter1_models.py`  
Purpose:
- run either the salary modelling pipeline or the skill requirement pipeline via a single entrypoint.

### 5.2.2 Salary Modelling Pipeline
**File:** `pipelines/salary_model_pipeline.py`  
Purpose (high level):
- build modelling table → encode categoricals → apply PCA → train XGBoost → (optional eval) → persist artefacts.

### 5.2.3 Skill Requirement Pipeline
**File:** `pipelines/skill_model_pipeline.py`  
Purpose (high level):
- build skill dataset → train 27 models → evaluate → persist models → generate `skill_prob_matrix`.

## 5.3 Chapter 2

### 5.3.1 Hidden Structures Pipeline
**File:** `pipelines/chapter2_hidden_structures.py`  
Purpose:
- convert Chapter 1 job × skill probabilities into:
  - bipartite graph
  - Node2Vec embeddings (jobs + skills)
  - job family clustering
  - skill similarity edge list (k-NN graph)
  - skill specialisation lift tables
- persist outputs as versioned processed artefacts.

## 5.4 Chapter 3

### 5.4.1 Individual Positioning Pipeline
**File:** `pipelines/chapter3_individual_positioning.py`  
Purpose:
- runnable wrapper around `run_positioning()` for reproducible execution outside notebooks.

## 5.5 Chapter 4

### 5.5.1 Recommender Engine Pipeline
**File:** `pipelines/chapter4_recommender.py`  
Purpose:
- orchestrate baseline recommendation plus optional decision-support modules:
  - explanations
  - upskilling counterfactuals
  - career simulation scenarios

Recommended persistence:
- persist demo config only (`evaluation/recommender_demo.json`), optionally plus a small “golden” output snapshot for regression checks.

## 5.6 Chapter 5 — App Build / Asset Pipelines

### 5.6.1 Fairness Assets Builder
**File:** `pipelines/ch5_build_fairness_assets.py`  
Purpose:
- assemble deterministic, app-ready fairness artefacts (group summaries + histogram bins + box stats)
- persist outputs into `data/processed/ch5_assets/` for fast app loading

### 5.6.2 App Asset Build + Validation (Single Entrypoint)
**File:** `pipelines/ch5_app_build.py`  
Purpose:
- provide a **single deterministic build entrypoint** for Chapter 5 app assets (no training)
- call Chapter 5 asset builders (currently: fairness assets) and **validate required runtime artefacts exist**
- fail fast if any required artefact is missing (ship hygiene + smoke-test friendly)

Build scope (v1):
- rebuild (if enabled): fairness group summaries + box stats (+ optional histogram bins) via `ch5_build_fairness_assets.py`
- validate presence of required artefacts consumed by the app:
  - `data/processed/df_with_residuals.csv`
  - `data/processed/ch5_assets/fairness_group_summary_long.csv`
  - `data/processed/ch5_assets/fairness_residual_box_stats.json`
  - `data/processed/ch5_assets/skill_value_index.csv`
  - `data/processed/ch5_assets/shap_salary_explanation.npz`
  - `data/processed/skill_similarity_matrix/skill_similarity_edges_k5_embeddings.csv`
  - (optional) `src/job_intel/evaluation/recommender_demo.json`

Execution:
- `python -m src.job_intel.pipelines.ch5_app_build`

---

# 6. Schemas & Public APIs

## 6.1 User Profile Schema (Entry Point)
**Module:** `schemas.py`  
**Function:** `build_user_profile()`  

Responsibilities:
- validate + normalize raw user inputs
- reuse Chapter 0 skill extraction to produce the canonical 27-skill vector
- project user skills into the shared PCA space used downstream
- return deterministic, model-ready payload consumed by Chapter 3+.

## 6.2 Chapter 3 Public API
**Module:** `positioning.py`  
**Function:** `run_positioning()`  

Role:
- single canonical public entrypoint for positioning; consumed by Chapter 4 context loader.

---

# 7. Artefacts

## 7.1 Persisted Artefacts (Files)

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
- `data/processed/ch2_processed_df.csv` (canonical `jobs_df` consumed by Chapter 3; a.k.a. “Chapter 2 processed jobs table”)
- `data/processed/job_skill_bipartite_*.gpickle`
- `data/processed/job_embeddings_node2vec_v01.csv`
- `data/processed/skill_embeddings_node2vec_v01.csv`
- `data/processed/job_families_graph_embeddings.csv`
- `data/processed/skill_similarity_edges_k*_embeddings.csv`
- `data/processed/job_family_skill_specialisation.csv` (or equivalent `{group_col}` lift outputs)

Chapter 5 (App assets):
- `data/processed/ch5_assets/fairness_group_summary_long.csv`
- `data/processed/ch5_assets/fairness_residual_box_stats.json`
- `data/processed/ch5_assets/fairness_residual_hist_bins.csv`
- `data/processed/ch5_assets/shap_salary_explanation.npz` (saved SHAP Explanation object for app plots)
- `data/processed/ch5_assets/skill_value_index.csv`

Evaluation / reproducibility:
- `evaluation/recommender_demo.json`
- evaluation tables/plots produced by evaluators (if enabled), e.g.:
  - `evaluation/skill_model_evaluation_results.csv`
  - `{feature}_fairness.csv` (if produced)

## 7.2 Runtime Outputs (Not Persisted by Default)
Produced per user run (returned by APIs/pipelines):
- Chapter 3: `candidates_df`, `gap_df`, sensitivity outputs
- Chapter 4: `scored_universe`, bucket tables, explained tables, upskilling tables, simulation tables

---

# 8. Assumptions & Limitations (Project + Chapters)

Project-wide:
- inputs are job postings from a specific dataset source/timeframe; distribution shift is expected
- skill extraction is taxonomy/token-driven; out-of-vocabulary skills may be ignored or mapped imperfectly
- determinism is prioritized; exploration/diversity objectives are not first-class in v1

Chapter 0:
- cleaning/normalisation rules are heuristic; salary parsing and text cleaning have edge cases

Chapter 1:
- salary model is predictive, not causal
- skill models are limited by label construction and token coverage

Chapter 2:
- embeddings reflect co-occurrence in the learned job–skill graph; they are not ground-truth semantics

Chapter 3:
- positioning is computed within a constrained candidate universe; results are relative to that universe
- probability-based gaps depend on top-K design choice

Chapter 4:
- “best_now” vs “stretch” is operational (competitiveness threshold proxy), not a guarantee
- upskilling + simulation are counterfactual model responses (token injections), not guaranteed labour-market outcomes
```
