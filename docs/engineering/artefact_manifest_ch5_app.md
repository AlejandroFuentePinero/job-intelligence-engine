# Chapter 5 — Artefact Manifest (App Runtime + Build Outputs)

This manifest lists **every persisted file** the Chapter 5 Streamlit app expects at runtime, plus the **minimum upstream artefacts** required for the embedded Chapter 4 pipeline to execute.

Paths below are shown **relative to repo root** unless otherwise stated.

---

## 0) Conventions

- `PROCESSED_DATA_DIR` → `data/processed/`
- Chapter 5 assets directory → `data/processed/ch5_assets/`
- “Producer” means the script/pipeline responsible for persisting the artefact (where known from code).
- “Consumer” means the app page/module that loads it.

---

## 1) App Runtime Config (Demo Inputs)

### A1 — Demo persona config
- **Path:** `src/job_intel/evaluation/recommender_demo.json`
- **Producer:** authored config (no build script)
- **Consumer:** `src/job_intel/app/recommender.py` → `_default_demo_path()` / “Load demo persona”
- **Required shape (minimal):**
  - `{"user_inputs": {...}, "pipeline_params": {...}}`
- **Notes:** used only to populate UI defaults; pipeline still runs on “Run recommender”.

---

## 2) Chapter 5 Persisted App Assets (fast load; no training)

### B1 — Fairness group summary table (long)
- **Path:** `data/processed/ch5_assets/fairness_group_summary_long.csv`
- **Producer:** `pipelines/ch5_build_fairness_assets.py` (your `ch5_build_fairness_assets.py`)
- **Consumer:** `src/job_intel/app/landscape.py` → `_load_fairness_tables()`
- **Required columns (used by UI):**
  - `group_type` (e.g., `location`, `sector`, `company_size`, `ownership`, `seniority`, `job_title`)
  - `group_value` (label value)
  - metrics selectable in UI:
    - `mean_residual`
    - `size_weighted_mean`
    - `median_residual`
    - `n`

### B2 — Fairness residual box summary stats
- **Path:** `data/processed/ch5_assets/fairness_residual_box_stats.json`
- **Producer:** `pipelines/ch5_build_fairness_assets.py`
- **Consumer:** `src/job_intel/app/landscape.py` → `_load_fairness_tables()` (expander JSON)
- **Required keys:** flexible; displayed verbatim. (Typical: `n`, `mean`, `std`, `p10`, `p90`, `q1`, `median`, `q3`, `whisker_low`, `whisker_high`.)

### B3 — Skill value index (GSVI)
- **Path:** `data/processed/ch5_assets/skill_value_index.csv`
- **Producer:** precomputed upstream (builder not in the reviewed script set)
- **Consumer:** `src/job_intel/app/landscape.py` → `_load_skill_value_index()`
- **Required columns:**
  - `skill` (string)
  - `value` (numeric)

### B4 — Salary-model SHAP explanation bundle
- **Path:** `data/processed/ch5_assets/shap_salary_explanation.npz`
- **Producer:** precomputed upstream (builder not in the reviewed script set)
- **Consumer:** `src/job_intel/app/landscape.py` → `_load_shap_explanation()`
- **Required NPZ keys (minimum):**
  - `values` : array `(n_rows, n_features)`
  - `feature_names` : array `(n_features,)`
- **Optional keys (supported):**
  - `data` : array `(n_rows, n_features)`
  - `base_values` : scalar or `(n_rows,)`

### B5 — (Optional) Residual histogram bins (not currently used by UI)
- **Path:** `data/processed/ch5_assets/fairness_residual_hist_bins.csv`
- **Producer:** `pipelines/ch5_build_fairness_assets.py`
- **Consumer:** none in current `landscape.py` (histogram is computed directly from residual series)
- **Notes:** keep for reproducibility / possible v2 speedup.

---

## 3) Upstream Required Data for Chapter 5 Pages

### C1 — Residuals table (source for fairness build + histogram)
- **Path:** `data/processed/df_with_residuals.csv`
- **Producer:** upstream Chapter 1/5 build step (not in reviewed script set)
- **Consumers:**
  - `pipelines/ch5_build_fairness_assets.py` (input)
  - `src/job_intel/app/landscape.py` → `_load_residuals_series()` (histogram)
- **Required columns:**
  - `residuals` (numeric)

### C2 — Chapter 1 salary-model PCA dataframe (lookup labels for categorical codes)
- **Path:** `CH1_PROCESSED_SALARY_MODEL_PCA_DF` (configured; typically under `data/processed/`)
- **Producer:** Chapter 1 processing pipeline
- **Consumer:** `src/job_intel/app/landscape.py` → `_load_ch1_lookup_tables()`
- **Required columns (used to map codes → labels):**
  - `size_code`, `Size`
  - `sector_code`, `Sector`
  - `state_code`, `state`
  - `ownership_code`, `ownership_clean`
  - `seniority_code`, `seniority_combined`
  - `title_rich_code`, `title_rich`

---

## 4) Chapter 2 Artefacts Used by the App (Macro co-learning only)

### D1 — Skill similarity edges (kNN graph from embeddings)
- **Path:** `data/processed/skill_similarity_matrix/skill_similarity_edges_k5_embeddings.csv`
- **Producer:** Chapter 2 embedding/graph workflow (persisted upstream)
- **Consumer:** `src/job_intel/app/upskilling_macro.py` → `_load_skill_similarity_edges()`
- **Required columns:**
  - `skill_1` (string)
  - `skill_2` (string)
  - `similarity` (numeric)

---

## 5) Runtime Outputs (not persisted; produced on demand)

These are produced by `run_recommender_pipeline()` when the user clicks **Run recommender**.

- **Consumer:** `src/job_intel/app/recommender.py`
- **Key runtime tables used by UI:**
  - `candidate_jobs` (for job detail + positioning summary)
  - `top_best_explained` / `top_stretch_explained` (preferred)
  - fallback: `top_best_now` / `top_stretch`
- **Notes:** no additional persistence required for v1.

---

## 6) Minimal “File Exists” Checklist (v1 ship gate)

Required for app to run without missing-asset errors:

- `src/job_intel/evaluation/recommender_demo.json` *(demo only; app still runs without if user enters inputs manually, but UI demo button will fail)*
- `data/processed/df_with_residuals.csv`
- `data/processed/ch5_assets/fairness_group_summary_long.csv`
- `data/processed/ch5_assets/fairness_residual_box_stats.json`
- `data/processed/ch5_assets/skill_value_index.csv`
- `data/processed/ch5_assets/shap_salary_explanation.npz`
- `data/processed/skill_similarity_matrix/skill_similarity_edges_k5_embeddings.csv`

Optional / reproducibility:
- `data/processed/ch5_assets/fairness_residual_hist_bins.csv`

---
