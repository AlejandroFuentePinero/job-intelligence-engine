# Job Intelligence Engine — Data Dictionary (Schema + Data Catalogue)
Date: 2026-01-01

This document is the **schema reference** for the project’s canonical processed dataset (Chapter 0) and a **lightweight catalogue** for the key derived artefacts persisted to disk.

**Scope rule (to avoid duplication with `docs/architecture.md`):**
- This file gives **column-level definitions** for the **canonical processed dataset**.
- For downstream artefacts, this file gives **short dataset-level descriptions** and **key columns only** (no full schemas).
- The **System Architecture** remains the authoritative inventory of modules, pipelines, and artefact ownership.

---

## 1) Canonical Dataset Summary (Chapter 0)

**Canonical file:** `data/processed/jobs_ch0.csv`  
*(If your repo uses `ch0_processed_jobs.csv`, treat it as the same canonical output.)*

- **Rows:** 6,162  
- **Columns:** 47  
- **Unit:** one row per job ad (Glassdoor Data Analyst + Data Scientist datasets).  
- **Primary key:** `job_id` (stable join key across chapters)

---

## 2) Column Definitions (Canonical Dataset)

### 2.1 Identifiers

| Column | Type | Definition |
|---|---|---|
| `job_id` | int | Stable identifier for the job record. Used to join all downstream artefacts. |

### 2.2 Company metadata

| Column | Type | Definition |
|---|---|---|
| `Rating` | float (nullable) | Glassdoor company rating (0–5). |
| `Size` | str | Company size category (`"Unknown"` if missing). |
| `Founded` | Int64 (nullable) | Year founded (nullable). |
| `Industry` | str | Industry label (`"Unknown"` if missing). |
| `Sector` | str | High-level sector (`"Unknown"` if missing). |
| `ownership_clean` | str | Standardised ownership category. |
| `state` | str | US state code; `"international"` for non-US postings. |

### 2.3 Role identity

| Column | Type | Definition |
|---|---|---|
| `job_title_base` | str | Cleaned title without seniority/noise tokens. |
| `job_title_norm` | str | Normalised title used for taxonomy lookup. |
| `job_title_family` | str | Normalised job family label (from title taxonomy). |
| `title_rich` | str | Enriched title representation used for modelling/navigation (captures family + meaningful modifiers). |
| `seniority_combined` | str | Seniority extracted from title + description (e.g., junior, senior, lead). |
| `domain` | str | Domain label from SBERT/nearest-neighbour lookup. |
| `role_source` | str | Dataset origin (e.g., `DA` vs `DS`). |

### 2.4 Salary features

| Column | Type | Definition |
|---|---|---|
| `sal_min` | float (nullable) | Annualised salary minimum. |
| `sal_max` | float (nullable) | Annualised salary maximum. |
| `sal_mean` | float (nullable) | Salary midpoint (target variable for the salary model). |
| `sal_is_hourly` | bool | True if original salary was hourly before annualisation. |

### 2.5 Skill flags (binary 0/1)

Each skill flag indicates whether the (rule-based) extractor detected evidence of the skill group in the job text.

#### Core Programming
- `core_programming__basic`
- `core_programming__intermediate`
- `core_programming__advanced`

#### Data Engineering & Pipelines
- `data_engineering_pipelines__basic`
- `data_engineering_pipelines__intermediate`
- `data_engineering_pipelines__advanced`

#### Machine Learning & AI
- `ml_ai__basic`
- `ml_ai__intermediate`
- `ml_ai__advanced`

#### Analytics & Statistics
- `analytics_stats__basic`
- `analytics_stats__intermediate`
- `analytics_stats__advanced`

#### BI & Visualisation
- `bi_viz__basic`
- `bi_viz__intermediate`
- `bi_viz__advanced`

#### Cloud / MLOps
- `cloud__basic`
- `cloud__intermediate`
- `cloud__advanced`

#### Databases & Storage
- `db_storage__basic`
- `db_storage__intermediate`
- `db_storage__advanced`

#### Productivity & Workflow
- `productivity_workflow__basic`
- `productivity_workflow__intermediate`
- `productivity_workflow__advanced`

#### Soft Skills
- `soft_skills__core`
- `soft_skills__leadership`

#### Domain-specific
- `domain_specific__none`

### 2.6 Text fields

| Column | Type | Definition |
|---|---|---|
| `Job Description` | str | Raw job description text. |
| `job_description_clean` | str | Normalised/cleaned description used for extraction and modelling. |
| `title_plus_description` | str | Combined text field used for skill extraction and some modelling utilities. |

---

## 3) Missing-Value Policy (Canonical Dataset)

- **Drop** rows missing **both** salary fields *and* job description (cannot support modelling or extraction).
- Fill with `"Unknown"`: `Size`, `Industry`, `Sector` (categorical robustness).
- Keep as NA: `Rating`, `Founded` (genuine missingness).
- Skill flags are always 0/1 (never NA).
- Salary fields may be NA for ads without salary information.

---

## 4) Engineered Modelling Features (Chapter 1)

These features are created inside Chapter 1 modelling tables (not required to exist in the canonical Chapter 0 dataset).

### 4.1 Categorical encodings (examples)
- `size_code`
- `sector_code`
- `state_code`
- `ownership_code`
- `seniority_code`
- `title_rich_code`

### 4.2 PCA components
- `skill_PC1` … `skill_PC10`

---

## 5) Derived Tabular Artefacts (Data Catalogue)

These artefacts are persisted for reuse, evaluation, and/or the Streamlit app.  
They are **documented here at a dataset level only**; detailed ownership and build steps live in the System Architecture.

### 5.1 Salary model outputs (Chapter 1)

- `data/processed/df_with_residuals.csv`  
  **What it is:** canonical salary modelling dataframe augmented with predictions and residuals.  
  **Key columns:** `job_id`, `sal_mean`, `sal_pred`, `residuals`, plus the model feature columns used during inference.

- `data/processed/global_shap_mean.csv`  
  **What it is:** global SHAP importance summary for the salary model (aggregated).  
  **Key columns:** `feature`, `mean_abs_shap` (and/or equivalent importance fields).

- `data/processed/ch5_assets/skill_value_index.csv` *(mirrors `data/processed/skill_value_index/skill_value_index.csv` if both exist)*  
  **What it is:** global skill value index (PCA-backprojected SHAP signal onto individual skills).  
  **Key columns:** `skill`, `value` (standardised score).

- `data/processed/fairness/*.csv` *(base fairness outputs)*  
  **What it is:** residual summaries grouped by a categorical dimension.  
  **Examples:** `state_fairness.csv`, `sector_fairness.csv`, `title_fairness.csv`, `size_fairness.csv`, `ownership_fairness.csv`, `seniority_fairness.csv`.  
  **Key columns:** `group_value` (or the grouping label), `mean_residual`, `median_residual`, `count`, `size_weighted_mean`.

### 5.2 Skill requirement model outputs (Chapter 1)

- `data/processed/skill_prob_matrix.csv`  
  **What it is:** job × skill probability matrix (27 columns, values in [0, 1]).  
  **Key columns:** `job_id` + `prob_*` skill probability columns.

- `data/processed/skill_model_evaluation_result.csv`  
  **What it is:** evaluation table (one row per skill classifier).  
  **Key columns:** `model`, ROC AUC / PR AUC (train/test), `brier_test`, plus tuned hyperparameters and feature importances.

### 5.3 Hidden-structure / market layer outputs (Chapter 2)

- `data/processed/job_embeddings_node2vec_v01.csv`  
  **What it is:** Node2Vec embedding vectors for job nodes.  
  **Key columns:** `job_id` + embedding dimensions.

- `data/processed/skill_embeddings_node2vec_v01.csv`  
  **What it is:** Node2Vec embedding vectors for skill nodes.  
  **Key columns:** `skill` (or skill-node id) + embedding dimensions.

- `data/processed/job_families_graph_embeddings.csv`  
  **What it is:** latent job-family cluster assignment from job embeddings.  
  **Key columns:** `job_id`, `job_family_id`.

- `data/processed/skill_similarity_matrix/skill_similarity_edges_k5_embeddings.csv`  
  **What it is:** undirected edge list linking each skill to its top-k neighbours by cosine similarity (embedding space).  
  **Key columns:** `skill_1`, `skill_2`, `similarity`.

- `data/processed/skill_specialisation/*.csv`  
  **What it is:** skill “lift” tables: group mean probability minus global mean probability.  
  **Examples:** `sector_skill_specialisation.csv`, `seniority_skill_specialisation.csv`, `state_skill_specialisation.csv`, `title_skill_specialisation.csv`, `title_condensed_skill_specialisation.csv`, `company_size_skill_specialisation.csv`, `ownership_skill_specialisation.csv`, `job_family_skill_specialisation.csv`.  
  **Key columns:** group label + `*_prob` lift columns.

### 5.4 Chapter 5 app-ready assets (deterministic)

These files exist to keep the app fast and avoid recomputing aggregations at runtime.

- `data/processed/ch5_assets/fairness_group_summary_long.csv`  
  **What it is:** a stacked (long-form) table of fairness summaries across multiple grouping variables.  
  **Key columns:** `group_type`, `group_value`, `n`, `mean_residual`, `median_residual`, `p10`, `p90`, `size_weighted_mean`.

- `data/processed/ch5_assets/fairness_residual_box_stats.json`  
  **What it is:** global distribution summary of residuals (for box/whisker plots).  
  **Key fields:** `n`, `mean`, `std`, `p10`, `p90`, `q1`, `median`, `q3`, `whisker_low`, `whisker_high`.

- `data/processed/ch5_assets/fairness_residual_hist_bins.csv`  
  **What it is:** histogram bins for residual distribution (pre-binned).  
  **Key columns:** `bin_left`, `bin_right`, `count`, `bin_mid`.

- `data/processed/ch5_assets/shap_salary_explanation.npz`  
  **What it is:** compact SHAP payload for the salary model (used for app plots).  
  **Key contents:** arrays required to reconstruct global SHAP views; exact array names are implementation-defined.

---

## 6) Model artefacts (non-tabular, referenced by filename only)

These are documented primarily in `docs/architecture.md` (Models / Artefacts sections). Listed here only as a pointer.

- `models/salary_model_v4.pkl` — trained salary model.
- `models/skill_pca_v1.pkl` — fitted PCA transform for skill components.
- `models/*_model.pkl` — 27 trained skill requirement models.
- `models/job_skill_bipartite_thres0_5.gpickle` — thresholded bipartite graph used for embeddings.

---

## 7) Relationship to the System Architecture

- Use this file when you need **schema-level clarity** (what columns mean, how to interpret them).
- Use `docs/architecture.md` when you need **ownership and build logic** (which module/pipeline creates which artefact, and where it lives).
