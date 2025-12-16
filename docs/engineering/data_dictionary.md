# Job Intelligence Engine — Data Dictionary  
Verified against jobs_ch0.csv  
Date: 2025-12-16

---

# 1. Dataset Summary  
**Source:** Raw Glassdoor datasets (Data Analyst + Data Scientist)  
**Final Output:** `ch0_processed_jobs.csv`  
**Rows:** 6,162  
**Columns:** 47  

---

# 2. Column Definitions (Chapter 0)

## A. Company Metadata

| Column | Type | Definition |
|--------|------|------------|
| `Rating` | float (nullable) | Glassdoor rating (0–5). |
| `Size` | str | Company size category (`Unknown` if missing). |
| `Founded` | Int64 | Nullable year founded. |
| `Industry` | str | Company industry (`Unknown`). |
| `Sector` | str | High-level sector (`Unknown`). |
| `ownership_clean` | str | Standardised ownership. |
| `state` | str | US state or `"international"`. |

---

## B. Role Identity

| Column | Type | Definition |
|--------|------|------------|
| `seniority_combined` | str | Seniority extracted from title + description. |
| `job_title_family` | str | Normalised job family. |
| `domain` | str | Domain label from SBERT lookup. |
| `role_source` | str | DA or DS dataset origin. |

---

## C. Salary Features

| Column | Type | Definition |
|--------|------|------------|
| `sal_min` | float | Annualised salary minimum. |
| `sal_max` | float | Annualised salary maximum. |
| `sal_mean` | float | Salary midpoint (target). |
| `sal_is_hourly` | bool | Whether original salary was hourly before annualisation. |

---

## D. Skill Flags (0/1)

### Core Programming  
- `core_programming__basic`  
- `core_programming__intermediate`  
- `core_programming__advanced`  

### Data Engineering & Pipelines  
- `data_engineering_pipelines__basic`  
- `data_engineering_pipelines__intermediate`  
- `data_engineering_pipelines__advanced`  

### Machine Learning & AI  
- `ml_ai__basic`  
- `ml_ai__intermediate`  
- `ml_ai__advanced`  

### Analytics & Statistics  
- `analytics_stats__basic`  
- `analytics_stats__intermediate`  
- `analytics_stats__advanced`  

### BI & Visualisation  
- `bi_viz__basic`  
- `bi_viz__intermediate`  
- `bi_viz__advanced`  

### Cloud / MLOps  
- `cloud__basic`  
- `cloud__intermediate`  
- `cloud__advanced`  

### Databases & Storage  
- `db_storage__basic`  
- `db_storage__intermediate`  
- `db_storage__advanced`  

### Productivity & Workflow  
- `productivity_workflow__basic`  
- `productivity_workflow__intermediate`  
- `productivity_workflow__advanced`  

### Soft Skills  
- `soft_skills__core`  
- `soft_skills__leadership`  

### Domain-Specific  
- `domain_specific__none`  

---

## E. Text Fields

| Column | Meaning |
|--------|---------|
| `Job Description` | Raw job description. |
| `job_description_clean` | Cleaned description for extraction. |
| `job_title_base` | Cleaned title without noise or seniority. |
| `job_title_norm` | Normalised version for lookup. |
| `title_plus_description` | Combined text used for skills. |

---

# 3. Missing Value Policy

- Drop rows missing **both** salary and description  
- Fill with `"Unknown"`: Size, Industry, Sector  
- Keep NA: Rating, Founded  
- Skill flags: always 0/1  
- Salary fields: allow NA  

---

# 4. Salary model (Chapter 1) Features (Engineered)

## Categorical Encodings  
- `size_code`  
- `sector_code`  
- `state_code`  
- `ownership_code`  
- `seniority_code`  
- `title_rich_code`

## PCA Components  
- `skill_PC1` … `skill_PC10`

## Optional One-Hot Dummies  
Generated during experimentation only.



---

# 5. Salary Fairness Outputs (Chapter 1)

The Salary Response Model produces residuals for each job record, defined as:

\[
\text{residual} = \text{sal\_mean} - \text{sal\_pred}
\]

These residuals represent how much a job pays **above** or **below** the model’s expected salary after controlling for job family, seniority, sector, state, company size, ownership, and PCA skill components.

Fairness is assessed by grouping residuals across six key categorical variables:

- **state**  
- **sector**  
- **job family** (enriched title representation)  
- **company size**  
- **ownership type**  
- **seniority level**

Each grouping produces a fairness summary table with an identical structure and is exported as an individual CSV file.

---

## A. Residual Columns (added to modelling dataset)

| Column | Type | Definition |
|--------|------|------------|
| `sal_pred` | float | Predicted salary from the Salary Response Model. |
| `residuals` | float | Observed minus predicted salary for each job. |

---

## B. Unified Fairness Table Structure

All fairness outputs share the same schema:

| Column | Definition |
|--------|------------|
| `mean_residual` | Average residual for the category. |
| `median_residual` | Median residual. |
| `count` | Number of job records in the category. |
| `size_weighted_mean` | Normalised mean residual adjusted by the category’s proportional size. |

This common format ensures interpretability and comparability across all categorical dimensions.

---

## C. Exported Fairness Files

Each file contains the unified structure above, indexed by the grouping category:

| File Name | Grouping Variable | Description |
|-----------|-------------------|-------------|
| `state_fairness.csv` | `state` | Model-adjusted over/under-payment across U.S. states. |
| `sector_fairness.csv` | `sector` | Structural salary deviations across company sectors. |
| `family_fairness.csv` | `title_rich_code` | Fairness across normalised job families. |
| `size_fairness.csv` | `Size` | Pay deviations by company size category. |
| `ownership_fairness.csv` | `ownership_clean` | Salary behaviour across ownership classes. |
| `seniority_fairness.csv` | `seniority_combined` | Fairness across seniority levels. |

All files contain the same four fairness metrics.

---

## D. Storage Location

All fairness summary tables are stored in:
data/processed/


These outputs are interpretability artefacts for Chapter 1.  
They are **not** inputs for downstream modelling pipelines.

---

---
---

# 6. Skill Requirement Models — Outputs (Chapter 1)

This section documents the outputs generated by the 27 binary skill requirement models:
1. The **Skill Probability Matrix** (job × skill probability table).  
2. The **Skill Evaluation Table** (one row per skill model).  
Both outputs have fixed, reproducible structure and serve as analytical artefacts for later chapters.

---

## A. Skill Probability Matrix

The probability matrix stores model-estimated likelihoods that each skill group is required for each job.  
It replaces sparse binary skill flags with smooth, calibrated probabilities.

### Structure

| Column | Type | Description |
|--------|------|-------------|
| `prob_core_programming__basic` | float | P(basic programming skill required). |
| `prob_core_programming__intermediate` | float | Continuous probability estimate. |
| … | … | All skill groups follow this naming format. |

There are **27 probability columns**, each corresponding to one skill group.

### Dimensions

- **Rows:** 6,162 jobs  
- **Columns:** 27 probability features  
- **Range:** 0–1 for all values  

### Use Cases

- Job–skill embeddings  
- Skill-demand landscapes  
- Similarity and clustering  
- Recommendation logic  
- Competitiveness modelling  

This is the canonical, continuous skill layer for downstream chapters.

---

## B. Skill Model Evaluation Table

The evaluation table records performance metrics, hyperparameters, and feature importances for each of the 27 skill classifiers.  
Each row corresponds to one skill model.

### Structure  

| Column | Type | Definition |
|--------|------|------------|
| `model` | str | Skill group name. |
| `pos_frac_train` | float | Positive class prevalence in training set. |
| `pos_frac_test` | float | Positive class prevalence in test set. |
| `roc_auc_train` | float | ROC AUC on training set. |
| `roc_auc_test` | float | ROC AUC on test set. |
| `pr_auc_train` | float | PR AUC on training set. |
| `pr_auc_test` | float | PR AUC on test set. |
| `brier_test` | float | Brier score on test set (calibration metric). |
| `learning_rate` | float | Final model hyperparameter. |
| `n_estimators` | int | Number of boosting rounds. |
| `num_leaves` | int | Tree leaf count used in best model. |
| `colsample_bytree` | float | Column subsampling hyperparameter. |
| `subsample` | float | Row subsampling hyperparameter. |

### Encoded Feature Importances  
The table includes one importance column **per predictor variable**.  
These show the *gain-based LightGBM importance* for each feature in each model.

Categorical predictors:

- `size_code`  
- `sector_code`  
- `state_code`  
- `ownership_code`  
- `seniority_code`  
- `title_rich_code`  

Binary skill indicators (all 27):

- `core_programming__basic`  
- `core_programming__intermediate`  
- `core_programming__advanced`  
- `data_engineering_pipelines__basic`  
- `data_engineering_pipelines__intermediate`  
- `data_engineering_pipelines__advanced`  
- `ml_ai__basic`  
- `ml_ai__intermediate`  
- `ml_ai__advanced`  
- `analytics_stats__basic`  
- `analytics_stats__intermediate`  
- `analytics_stats__advanced`  
- `bi_viz__basic`  
- `bi_viz__intermediate`  
- `bi_viz__advanced`  
- `cloud__basic`  
- `cloud__intermediate`  
- `cloud__advanced`  
- `db_storage__basic`  
- `db_storage__intermediate`  
- `db_storage__advanced`  
- `productivity_workflow__basic`  
- `productivity_workflow__intermediate`  
- `productivity_workflow__advanced`  
- `soft_skills__core`  
- `soft_skills__leadership`  
- `domain_specific__none`  

### Notes

- Importance values are **not** model quality metrics — they indicate how much each feature contributed to reducing loss.  
- All skill models share the same set of predictors, allowing direct comparison of feature roles across skill groups.  
- This table fully captures both *performance* and *behaviour* of every skill classifier.

---

## C. Summary

The skill requirement modelling stage yields two structured, reusable artefacts:

1. **Skill Probability Matrix** (6,162 × 27) — continuous estimates of skill demand.  
2. **Skill Evaluation Table** (27 rows) — performance metrics, hyperparameters, and predictor importances.

Together, these outputs form the analytical backbone for Chapters 2–4, including embeddings, clustering, recommendation systems, and job–skill landscape modelling.

---
# 7. Skill Value Index — Outputs (Chapter 1)

This section documents the Global Skill Value Index derived from the Salary Response Model.

---

## A. Skill Value Index Table

The Skill Value Index provides a standardised, model-implied measure of the association between individual skills and predicted salary.

### File
- `skill_value_index.csv`

### Storage Location
- `data/processed/`

### Structure

| Column | Type | Definition |
|--------|------|------------|
| `skill` | str | Skill identifier corresponding to Chapter 0 binary skill flags (e.g. `ml_ai__advanced`). |
| `value` | float | Standardised global skill value score (z-score). |

### Notes

- Values are computed by back-projecting PCA component importance (from SHAP) onto individual skills using PCA loadings.  
- Scores reflect **relative importance within the salary model**, not causal effects.  
- The index is global (no sector, state, or title stratification).  
- This table is an interpretability artefact and is **not used as an input** to downstream modelling pipelines.
---


# 8. Hidden Structure — Outputs (Chapter 2)

This section documents the structural artefacts produced in Chapter 2. These outputs convert Chapter 1’s tabular job + skill-probability signals into graph-based representations, embeddings, job families, and skill ecosystem summaries.

---

## A. Job–Skill Bipartite Graph (Artefact)

A weighted bipartite graph where:
- **Job nodes** represent individual job records (`job_id`)
- **Skill nodes** represent skill groups (27 skill probabilities)
- **Edges** exist when `P(skill | job) ≥ threshold`
- **Edge weights** equal `P(skill | job)` (continuous in [0, 1])

### File (optional)
- `models/job_skill_bipartite_thres{threshold}.pkl`

### Notes
- Used as the canonical structure for Node2Vec embedding training.
- Identifiers are stable and designed to join back to the Chapter 1 modelling dataframe via `job_id`.

---

## B. Node2Vec Embeddings (Jobs and Skills)

Node embeddings learned from random walks over the bipartite graph.

### Outputs
- **Job embeddings:** one vector per `job_id`
- **Skill embeddings:** one vector per skill node (skill group)

### Structure (conceptual)
| Field | Type | Definition |
|------|------|------------|
| `job_id` | int | Stable job identifier (joins to Chapter 1 modelling dataframe). |
| `embedding_dim_*` | float | Embedding vector components (e.g., 64 dimensions). |

| Field | Type | Definition |
|------|------|------------|
| `skill_id` | str | Skill identifier (e.g., `ml_ai__advanced_prob`). |
| `embedding_dim_*` | float | Embedding vector components (e.g., 64 dimensions). |

### Notes
- Embeddings are latent features capturing graph neighbourhood structure.
- These vectors are **not directly interpretable**; they are inputs to clustering and similarity analysis.

---

## C. Job Families (Cluster Assignments)

Unsupervised job ecosystem labels created by clustering job embeddings (KMeans on L2-normalised vectors).

### File
- `data/processed/job_families_graph_embeddings.csv`

### Structure
| Column | Type | Definition |
|--------|------|------------|
| `job_id` | int | Stable job identifier. |
| `job_family_id` | int | Cluster label representing latent job ecosystem membership. |

### Notes
- Every job receives exactly one job family assignment (complete partition).
- Job families are provisional structural constructs intended for aggregation and navigation.

---

## D. Skill Similarity Edges (Skill Ecosystem Graph)

Undirected edge list connecting each skill to its top-*k* nearest neighbouring skills in embedding space.

### File
- `data/processed/skill_similarity_edges_k5_embeddings.csv`

### Structure
| Column | Type | Definition |
|--------|------|------------|
| `skill_1` | str | Skill identifier (lexicographically smaller of the pair). |
| `skill_2` | str | Skill identifier (lexicographically larger of the pair). |
| `similarity` | float | Cosine similarity between skill embeddings (after L2-normalisation). |

### Notes
- Constructed by: cosine similarity matrix → top-*k* neighbours per skill → deduplicate to undirected edges.
- Provides the backbone for skill bundle discovery and gateway-skill interpretation.

---

## E. Skill Specialisation Maps (Lift vs Global Mean)

Skill ecosystem summaries computed by group (e.g., sector, title, seniority) showing relative skill over/under-representation.

Each map is computed as:

\[
\text{lift}_{g,s} = \overline{P(s \mid g)} - \overline{P(s)}
\]

where \( g \) is a group category and \( s \) is a skill probability column.

### Files
- `data/processed/sector_skill_specialisation.csv`
- `data/processed/title_skill_specialisation.csv`
- `data/processed/seniority_skill_specialisation.csv`
- `data/processed/ownership_skill_specialisation.csv`
- `data/processed/state_skill_specialisation.csv`
- `data/processed/company_size_skill_specialisation.csv`
- `data/processed/title_condensed_skill_specialisation.csv`

### Structure
| Field | Type | Definition |
|------|------|------------|
| `<group_label>` (index) | str | Group category label (e.g., sector name, title name). |
| `*_prob` | float | Lift value for a given skill: (group mean − global mean). |

### Notes
- Positive values indicate skills over-represented in the group relative to the market baseline.
- Negative values indicate skills under-represented in the group.
- These are interpretability artefacts supporting ecosystem analysis; they are not model inputs.

---
